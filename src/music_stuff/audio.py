"""Audio loading and preprocessing entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import wave


SUPPORTED_FFMPEG_SUFFIXES = {".mp3", ".flac"}
SUPPORTED_WAV_SUFFIXES = {".wav"}


@dataclass(frozen=True)
class PreparedAudio:
    """Normalized audio metadata passed into analysis backends."""

    path: Path
    sample_rate: int | None = None
    duration_seconds: float | None = None
    samples: tuple[float, ...] = ()


@dataclass
class AudioPreprocessor:
    """Prepare input audio before model-based transcription."""

    ffmpeg_binary: str | None = None
    compressed_sample_rate: int = 22050

    def prepare(self, input_path: Path) -> PreparedAudio:
        """Validate and decode an audio file into normalized mono samples."""
        input_path = input_path.expanduser()
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        suffix = input_path.suffix.lower()

        if suffix in SUPPORTED_WAV_SUFFIXES:
            return self._prepare_wav(input_path)
        if suffix in SUPPORTED_FFMPEG_SUFFIXES:
            return self._prepare_with_ffmpeg(input_path)
        raise ValueError("Supported audio formats are WAV, MP3, and FLAC.")

    def _prepare_wav(self, input_path: Path) -> PreparedAudio:
        with wave.open(str(input_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)

        decoded = _decode_pcm(raw, sample_width)
        mono = _downmix(decoded, channels)
        duration = len(mono) / sample_rate if sample_rate else None
        return PreparedAudio(
            path=input_path,
            sample_rate=sample_rate,
            duration_seconds=duration,
            samples=tuple(mono),
        )

    def _prepare_with_ffmpeg(self, input_path: Path) -> PreparedAudio:
        ffmpeg = self._resolve_ffmpeg_binary()
        command = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(self.compressed_sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ]
        try:
            result = subprocess.run(command, capture_output=True, check=True)
        except FileNotFoundError as exc:
            raise ValueError(_missing_ffmpeg_message()) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            message = f"Could not decode {input_path.name} with ffmpeg."
            if detail:
                message = f"{message} {detail}"
            raise ValueError(message) from exc

        mono = _decode_pcm(result.stdout, sample_width=2)
        duration = len(mono) / self.compressed_sample_rate if self.compressed_sample_rate else None
        return PreparedAudio(
            path=input_path,
            sample_rate=self.compressed_sample_rate,
            duration_seconds=duration,
            samples=tuple(mono),
        )

    def _resolve_ffmpeg_binary(self) -> str:
        if self.ffmpeg_binary:
            return self.ffmpeg_binary

        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg

        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise ValueError(_missing_ffmpeg_message()) from exc
        return imageio_ffmpeg.get_ffmpeg_exe()


def _decode_pcm(raw: bytes, sample_width: int) -> list[float]:
    """Decode little-endian PCM samples to floats in roughly [-1.0, 1.0]."""
    if sample_width not in {1, 2, 3, 4}:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    max_value = float(1 << (sample_width * 8 - 1))
    values: list[float] = []
    for index in range(0, len(raw), sample_width):
        chunk = raw[index : index + sample_width]
        if len(chunk) < sample_width:
            break
        if sample_width == 1:
            integer = int.from_bytes(chunk, "little", signed=False) - 128
        else:
            integer = int.from_bytes(chunk, "little", signed=True)
        values.append(max(-1.0, min(1.0, integer / max_value)))
    return values


def _downmix(samples: list[float], channels: int) -> list[float]:
    if channels <= 0:
        raise ValueError("WAV file must contain at least one channel.")
    if channels == 1:
        return samples

    mono: list[float] = []
    for index in range(0, len(samples), channels):
        frame = samples[index : index + channels]
        if len(frame) == channels:
            mono.append(sum(frame) / channels)
    return mono


def _missing_ffmpeg_message() -> str:
    return (
        "MP3/FLAC input requires ffmpeg. Install ffmpeg on PATH, or install "
        "the optional imageio-ffmpeg package."
    )
