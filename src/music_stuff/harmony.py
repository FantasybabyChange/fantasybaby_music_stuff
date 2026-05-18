"""Key and chord analysis interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from music_stuff.audio import PreparedAudio
from music_stuff.models import AnalysisResult, ChordSegment, KeyEstimate, Melody


@dataclass
class KeyAnalyzer:
    """Estimate tonic and mode from audio and/or melody notes."""

    def analyze(self, audio: PreparedAudio, melody: Melody) -> KeyEstimate:
        raise NotImplementedError("Key analysis is not implemented yet.")


@dataclass
class ChordAnalyzer:
    """Infer core chord symbols across the song timeline."""

    segment_beats: int = 4

    def analyze(
        self,
        audio: PreparedAudio,
        melody: Melody,
        key: KeyEstimate | None = None,
    ) -> tuple[ChordSegment, ...]:
        raise NotImplementedError("Chord analysis is not implemented yet.")


def build_analysis(
    *,
    key: KeyEstimate | None,
    tempo_bpm: float | None,
    chords: tuple[ChordSegment, ...],
    meter: str | None = None,
) -> AnalysisResult:
    """Collect analysis outputs into one stable result object."""
    return AnalysisResult(key=key, tempo_bpm=tempo_bpm, meter=meter, chords=chords)
