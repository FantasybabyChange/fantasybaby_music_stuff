"""Key and chord analysis interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from music_stuff.audio import PreparedAudio
from music_stuff.models import AnalysisResult, ChordSegment, KeyEstimate, Melody


@dataclass
class KeyAnalyzer:
    """Estimate tonic and mode from audio and/or melody notes."""

    def analyze(self, audio: PreparedAudio, melody: Melody) -> KeyEstimate:
        if not melody.notes:
            return KeyEstimate(tonic="C", mode="major", confidence=0.0)

        major_scale = {0, 2, 4, 5, 7, 9, 11}
        minor_scale = {0, 2, 3, 5, 7, 8, 10}
        pitch_classes = [note.pitch % 12 for note in melody.notes]
        best_score = -1
        best_tonic = 0
        best_mode = "major"
        for tonic in range(12):
            major_score = sum(1 for pc in pitch_classes if (pc - tonic) % 12 in major_scale)
            minor_score = sum(1 for pc in pitch_classes if (pc - tonic) % 12 in minor_scale)
            if major_score > best_score:
                best_score = major_score
                best_tonic = tonic
                best_mode = "major"
            if minor_score > best_score:
                best_score = minor_score
                best_tonic = tonic
                best_mode = "minor"

        confidence = best_score / len(pitch_classes)
        return KeyEstimate(tonic=_NOTE_NAMES[best_tonic], mode=best_mode, confidence=confidence)


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
        return ()


def build_analysis(
    *,
    key: KeyEstimate | None,
    tempo_bpm: float | None,
    chords: tuple[ChordSegment, ...],
    meter: str | None = None,
) -> AnalysisResult:
    """Collect analysis outputs into one stable result object."""
    return AnalysisResult(key=key, tempo_bpm=tempo_bpm, meter=meter, chords=chords)


_NOTE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
