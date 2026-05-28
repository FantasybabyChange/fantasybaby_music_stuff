"""Shared data contracts for the transcription pipeline."""

from __future__ import annotations

__all__ = [
    "NoteEvent",
    "Melody",
    "RhythmEstimate",
    "KeyEstimate",
    "ChordSegment",
    "AnalysisResult",
    "TranscriptionResult",
]

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class NoteEvent:
    """A single symbolic note after pitch detection and rhythm cleanup."""

    pitch: int
    start: float
    end: float
    velocity: int = 80


@dataclass(frozen=True)
class Melody:
    """A monophonic or lightly polyphonic melodic line."""

    notes: tuple[NoteEvent, ...]
    source: str
    source_kind: str = "mixed"
    source_label: str = "原始混音"
    source_confidence: float | None = None


@dataclass(frozen=True)
class RhythmEstimate:
    """Estimated beat grid for rhythmic quantization."""

    tempo_bpm: float
    beat_times: tuple[float, ...] = field(default_factory=tuple)
    beat_offset: float = 0.0
    meter: str = "4/4"
    confidence: float | None = None


@dataclass(frozen=True)
class KeyEstimate:
    """Estimated tonal center and mode."""

    tonic: str
    mode: str
    confidence: float | None = None

    @property
    def label(self) -> str:
        return f"{self.tonic} {self.mode}"


@dataclass(frozen=True)
class ChordSegment:
    """A chord label attached to a time range."""

    symbol: str
    start: float
    end: float
    roman_numeral: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class AnalysisResult:
    """Musical analysis that can be serialized to JSON later."""

    key: KeyEstimate | None = None
    tempo_bpm: float | None = None
    meter: str | None = None
    chords: tuple[ChordSegment, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TranscriptionResult:
    """Top-level result produced by the pipeline."""

    melody: Melody
    analysis: AnalysisResult
    output_dir: Path
    jianpu_path: Path | None = None
    midi_path: Path | None = None
    musicxml_path: Path | None = None
    analysis_path: Path | None = None
