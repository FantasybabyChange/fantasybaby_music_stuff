"""Audio loading and preprocessing entry points."""

from __future__ import annotations

__all__ = ["PreparedAudio", "AudioPreprocessor"]

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any
import wave


SUPPORTED_FFMPEG_SUFFIXES = {".mp3", ".flac"}
SUPPORTED_WAV_SUFFIXES = {".wav"}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedAudio:
    """Normalized audio metadata passed into analysis backends."""

    path: Path
    sample_rate: int | None = None
    duration_seconds: float | None = None
    samples: Any = ()


@dataclass
class AudioPreprocessor:
    """Prepare input audio before model-based transcription."""

    ffmpeg_binary: str | None = None
    compressed_sample_rate: int = 22050
    max_duration_seconds: float | None = None

    def prepare(self, input_path: Path) -> PreparedAudio:
        """Validate and decode an audio file into normalized mono samples."""
        input_path = input_path.expanduser()
        LOGGER.info("Preparing audio: %s", input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        suffix = input_path.suffix.lower()

        if suffix in SUPPORTED_WAV_SUFFIXES:
            return self._prepare_wav(input_path)
        if suffix in SUPPORTED_FFMPEG_SUFFIXES:
            return self._prepare_with_ffmpeg(input_path)
        raise ValueError("Supported audio formats are WAV, MP3, and FLAC.")

    def _prepare_wav(self, input_path: Path) -> PreparedAudio:
        LOGGER.info("Decoding WAV with built-in reader")
        with wave.open(str(input_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)

        decoded = _decode_pcm(raw, sample_width)
        mono = _downmix(decoded, channels)
        duration = len(mono) / sample_rate if sample_rate else None
        LOGGER.info(
            "Prepared WAV: sample_rate=%s channels=%s duration=%.2fs samples=%s",
            sample_rate,
            channels,
            duration or 0.0,
            len(mono),
        )
        return PreparedAudio(
            path=input_path,
            sample_rate=sample_rate,
            duration_seconds=duration,
            samples=mono,
        )

    def _prepare_with_ffmpeg(self, input_path: Path) -> PreparedAudio:
        ffmpeg = self._resolve_ffmpeg_binary()
        LOGGER.info(
            "Decoding compressed audio with ffmpeg: sample_rate=%s max_duration=%s",
            self.compressed_sample_rate,
            self.max_duration_seconds,
        )
        LOGGER.debug("Using ffmpeg binary: %s", ffmpeg)
        command = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(input_path),
        ]
        if self.max_duration_seconds:
            command.extend(["-t", str(self.max_duration_seconds)])
        command.extend([
            "-ac",
            "1",
            "-ar",
            str(self.compressed_sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ])
        try:
            result = subprocess.run(command, capture_output=True, check=True, timeout=120)
        except FileNotFoundError as exc:
            raise ValueError(_missing_ffmpeg_message()) from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"ffmpeg timed out after {exc.timeout}s while decoding {input_path.name}.") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            message = f"Could not decode {input_path.name} with ffmpeg."
            if detail:
                message = f"{message} {detail}"
            raise ValueError(message) from exc

        mono = _decode_pcm(result.stdout, sample_width=2)
        duration = len(mono) / self.compressed_sample_rate if self.compressed_sample_rate else None
        LOGGER.info(
            "Prepared compressed audio: sample_rate=%s duration=%.2fs samples=%s",
            self.compressed_sample_rate,
            duration or 0.0,
            len(mono),
        )
        return PreparedAudio(
            path=input_path,
            sample_rate=self.compressed_sample_rate,
            duration_seconds=duration,
            samples=mono,
        )

    def _resolve_ffmpeg_binary(self) -> str:
        if self.ffmpeg_binary:
            return self.ffmpeg_binary

        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            LOGGER.debug("Resolved ffmpeg from PATH: %s", system_ffmpeg)
            return system_ffmpeg

        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise ValueError(_missing_ffmpeg_message()) from exc
        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        LOGGER.debug("Resolved ffmpeg from imageio-ffmpeg: %s", bundled_ffmpeg)
        return bundled_ffmpeg


def _decode_pcm(raw: bytes, sample_width: int) -> "np.ndarray":
    """Decode little-endian PCM samples to floats in roughly [-1.0, 1.0]."""
    import numpy as np

    if sample_width not in {1, 2, 3, 4}:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if sample_width == 2:
        return np.clip(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, -1.0, 1.0)
    if sample_width == 1:
        return np.clip((np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0, -1.0, 1.0)
    if sample_width == 4:
        return np.clip(np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0, -1.0, 1.0).astype(np.float32)

    # 3-byte samples: manually decode each 3-byte little-endian signed int
    sample_count = len(raw) // 3
    values = np.empty(sample_count, dtype=np.float32)
    for i in range(sample_count):
        chunk = raw[i * 3 : (i + 1) * 3]
        integer = int.from_bytes(chunk, "little", signed=True)
        values[i] = max(-1.0, min(1.0, integer / 8388608.0))
    return values


def _downmix(samples: "np.ndarray", channels: int) -> "np.ndarray":
    """Mix multi-channel samples down to mono by averaging channels."""
    import numpy as np

    if channels <= 0:
        raise ValueError("WAV file must contain at least one channel.")
    if channels == 1:
        return samples
    return samples.reshape(-1, channels).mean(axis=1)


def _missing_ffmpeg_message() -> str:
    return (
        "MP3/FLAC input requires ffmpeg. Install ffmpeg on PATH, or install "
        "the optional imageio-ffmpeg package."
    )
