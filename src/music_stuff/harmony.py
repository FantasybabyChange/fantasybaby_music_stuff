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
        pitch_classes = [note.pitch % 12 for note in melody.notes]
        scores = []
        for tonic in range(12):
            score = sum(1 for pitch_class in pitch_classes if (pitch_class - tonic) % 12 in major_scale)
            scores.append(score)

        tonic_index = max(range(12), key=scores.__getitem__)
        confidence = scores[tonic_index] / len(pitch_classes)
        return KeyEstimate(tonic=_NOTE_NAMES[tonic_index], mode="major", confidence=confidence)


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
