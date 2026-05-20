"""Optional source separation for voice and accompaniment stems."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import logging
from pathlib import Path
import subprocess
import sys


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
        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            self.model_name,
            "--two-stems",
            "vocals",
            "-o",
            str(output_dir),
            str(input_path),
        ]
        LOGGER.info("Separating audio sources with Demucs: input=%s output=%s", input_path, output_dir)
        try:
            subprocess.run(command, capture_output=True, text=True, check=True, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            LOGGER.warning("Demucs source separation timed out after %ss", self.timeout_seconds)
            return SourceSeparationResult(backend="demucs", status="failed", message=str(exc))
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            LOGGER.warning("Demucs source separation failed: %s", detail)
            return SourceSeparationResult(backend="demucs", status="failed", message=detail or str(exc))

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
