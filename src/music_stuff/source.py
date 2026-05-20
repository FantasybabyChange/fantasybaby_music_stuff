"""Optional source separation for voice and accompaniment stems."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import logging
import os
from pathlib import Path
import shutil
import subprocess
import wave


LOGGER = logging.getLogger(__name__)

HUMAN_VOICE = "human_voice"
ACCOMPANIMENT = "accompaniment"
MIXED = "mixed"
SOURCE_LABELS = {
    HUMAN_VOICE: "人声",
    ACCOMPANIMENT: "伴奏旋律",
    MIXED: "原始混音",
}


@dataclass(frozen=True)
class SourceStem:
    """A separated audio stem to analyze."""

    kind: str
    label: str
    path: Path


@dataclass(frozen=True)
class SourceSeparationResult:
    """Result of a source separation attempt."""

    backend: str
    status: str
    stems: tuple[SourceStem, ...] = field(default_factory=tuple)
    message: str | None = None

    @property
    def is_available(self) -> bool:
        return self.status == "ok" and bool(self.stems)


@dataclass
class DemucsSourceSeparator:
    """Separate vocals from accompaniment when Demucs is installed.

    Demucs is intentionally optional because it downloads model weights and can
    be slow on CPU-only machines. The pipeline falls back to mixed audio when
    this separator is unavailable or fails.
    """

    enabled: bool = True
    model_name: str = "htdemucs"
    timeout_seconds: int = 900
    max_duration_seconds: float | None = 60.0

    def separate(self, input_path: Path, output_dir: Path) -> SourceSeparationResult:
        if not self.enabled:
            return SourceSeparationResult(
                backend="demucs",
                status="disabled",
                message="Source separation is disabled.",
            )
        if importlib.util.find_spec("demucs") is None:
            return SourceSeparationResult(
                backend="demucs",
                status="unavailable",
                message="Demucs is not installed. Install the source-separation extra to enable it.",
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            demucs_input = self._prepare_demucs_input(input_path, output_dir)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            LOGGER.warning("Could not prepare audio for Demucs: %s", detail)
            return SourceSeparationResult(backend="demucs", status="failed", message=detail or str(exc))
        LOGGER.info("Separating audio sources with Demucs: input=%s output=%s", demucs_input, output_dir)
        try:
            self._separate_with_model(demucs_input, output_dir)
        except Exception as exc:
            LOGGER.warning("Demucs source separation failed: %s", exc)
            return SourceSeparationResult(backend="demucs", status="failed", message=str(exc))

        stems = self._find_stems(output_dir)
        if not stems:
            LOGGER.warning("Demucs finished but no expected stems were found in %s", output_dir)
            return SourceSeparationResult(
                backend="demucs",
                status="failed",
                message="Demucs did not produce vocals/no_vocals stems.",
            )
        LOGGER.info("Source separation complete: stems=%s", ", ".join(stem.label for stem in stems))
        return SourceSeparationResult(backend="demucs", status="ok", stems=tuple(stems))

    def _find_stems(self, output_dir: Path) -> list[SourceStem]:
        vocals = sorted(output_dir.rglob("vocals.wav"))
        no_vocals = sorted(output_dir.rglob("no_vocals.wav"))
        stems: list[SourceStem] = []
        if vocals:
            stems.append(SourceStem(kind=HUMAN_VOICE, label=SOURCE_LABELS[HUMAN_VOICE], path=vocals[-1]))
        if no_vocals:
            stems.append(SourceStem(kind=ACCOMPANIMENT, label=SOURCE_LABELS[ACCOMPANIMENT], path=no_vocals[-1]))
        return stems

    def _subprocess_env(self, output_dir: Path | None = None) -> dict[str, str]:
        env = os.environ.copy()
        ffmpeg = self._resolve_ffmpeg_binary(output_dir)
        if ffmpeg:
            ffmpeg_dir = str(Path(ffmpeg).parent)
            env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
            LOGGER.debug("Added ffmpeg directory to Demucs PATH: %s", ffmpeg_dir)
        return env

    def _resolve_ffmpeg_binary(self, output_dir: Path | None = None) -> str | None:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg
        try:
            import imageio_ffmpeg
        except ImportError:
            return None
        bundled_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if output_dir is None or bundled_ffmpeg.name.lower() == "ffmpeg.exe":
            return str(bundled_ffmpeg)

        ffmpeg_shim = output_dir / "ffmpeg.exe"
        if not ffmpeg_shim.exists():
            shutil.copy2(bundled_ffmpeg, ffmpeg_shim)
        return str(ffmpeg_shim)

    def _prepare_demucs_input(self, input_path: Path, output_dir: Path) -> Path:
        ffmpeg = self._resolve_ffmpeg_binary()
        if ffmpeg is None:
            return input_path

        converted_path = output_dir / "demucs_input.wav"
        command = [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(input_path),
        ]
        if self.max_duration_seconds:
            command.extend(["-t", str(self.max_duration_seconds)])
        command.extend([
            "-ac",
            "2",
            "-ar",
            "44100",
            str(converted_path),
        ])
        LOGGER.info("Converting audio for Demucs: input=%s output=%s", input_path, converted_path)
        subprocess.run(command, capture_output=True, text=True, check=True)
        return converted_path

    def _separate_with_model(self, input_path: Path, output_dir: Path) -> None:
        import numpy as np
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        model = get_model(self.model_name)
        model.cpu()
        model.eval()

        wav, sample_rate = self._load_wav_tensor(input_path)
        if sample_rate != model.samplerate:
            raise ValueError(f"Expected {model.samplerate} Hz Demucs input, got {sample_rate} Hz.")
        if wav.shape[0] != model.audio_channels:
            raise ValueError(f"Expected {model.audio_channels} audio channels, got {wav.shape[0]}.")

        ref = wav.mean(0)
        center = ref.mean()
        scale = ref.std().clamp_min(1e-8)
        normalized = (wav - center) / scale
        with torch.no_grad():
            sources = apply_model(
                model,
                normalized[None],
                device="cpu",
                shifts=1,
                split=True,
                overlap=0.25,
                progress=False,
                num_workers=0,
            )[0]
        sources = sources * scale + center

        sources_by_name = dict(zip(model.sources, sources))
        vocals = sources_by_name.get("vocals")
        if vocals is None:
            raise ValueError(f"Demucs model did not produce a vocals source: {model.sources}")

        no_vocals = torch.zeros_like(vocals)
        for source_name, source_audio in sources_by_name.items():
            if source_name != "vocals":
                no_vocals += source_audio

        stem_dir = output_dir / self.model_name / input_path.stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        self._save_wav_tensor(vocals, stem_dir / "vocals.wav", model.samplerate, np)
        self._save_wav_tensor(no_vocals, stem_dir / "no_vocals.wav", model.samplerate, np)

    def _load_wav_tensor(self, input_path: Path):
        import numpy as np
        import torch

        with wave.open(str(input_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            raw = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            raise ValueError(f"Expected 16-bit WAV input for Demucs, got {sample_width * 8}-bit.")
        audio = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        audio = audio.reshape(-1, channels).T
        return torch.from_numpy(audio), sample_rate

    def _save_wav_tensor(self, audio, output_path: Path, sample_rate: int, np) -> None:
        clipped = audio.detach().cpu().clamp(-1.0, 1.0)
        pcm = (clipped.T.numpy() * 32767.0).astype("<i2")
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(clipped.shape[0])
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
