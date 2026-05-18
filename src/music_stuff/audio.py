"""Audio loading and preprocessing entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreparedAudio:
    """Normalized audio metadata passed into analysis backends."""

    path: Path
    sample_rate: int | None = None
    duration_seconds: float | None = None


class AudioPreprocessor:
    """Prepare input audio before model-based transcription."""

    def prepare(self, input_path: Path) -> PreparedAudio:
        """Validate, decode, and normalize an audio file.

        Future implementation:
        - use ffmpeg or soundfile to accept wav/mp3/m4a
        - normalize sample rate
        - optionally split vocals/accompaniment before melody extraction
        """
        raise NotImplementedError("Audio preprocessing backend is not wired yet.")
