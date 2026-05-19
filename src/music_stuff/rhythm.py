"""Tempo detection and rhythmic quantization interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from music_stuff.audio import PreparedAudio
from music_stuff.models import Melody


@dataclass
class RhythmQuantizer:
    """Turn raw note timings into readable score timings."""

    smallest_grid: str = "1/16"
    default_tempo_bpm: float = 120.0

    def estimate_tempo(self, audio: PreparedAudio) -> float:
        return self.default_tempo_bpm

    def quantize(self, melody: Melody, tempo_bpm: float) -> Melody:
        grid_seconds = (60.0 / tempo_bpm) / 4.0
        quantized_notes = []
        for note in melody.notes:
            start = round(note.start / grid_seconds) * grid_seconds
            end = round(note.end / grid_seconds) * grid_seconds
            if end <= start:
                end = start + grid_seconds
            quantized_notes.append(
                type(note)(
                    pitch=note.pitch,
                    start=max(0.0, start),
                    end=end,
                    velocity=note.velocity,
                )
            )
        return Melody(notes=tuple(quantized_notes), source=melody.source)
