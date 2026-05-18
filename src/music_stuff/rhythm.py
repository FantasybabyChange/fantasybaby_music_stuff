"""Tempo detection and rhythmic quantization interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from music_stuff.audio import PreparedAudio
from music_stuff.models import Melody


@dataclass
class RhythmQuantizer:
    """Turn raw note timings into readable score timings."""

    smallest_grid: str = "1/16"

    def estimate_tempo(self, audio: PreparedAudio) -> float:
        raise NotImplementedError("Tempo estimation is not implemented yet.")

    def quantize(self, melody: Melody, tempo_bpm: float) -> Melody:
        raise NotImplementedError("Rhythm quantization is not implemented yet.")
