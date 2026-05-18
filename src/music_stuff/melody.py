"""Melody transcription interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from music_stuff.audio import PreparedAudio
from music_stuff.models import Melody


class MelodyTranscriber(Protocol):
    """Convert audio into symbolic melody notes."""

    def transcribe(self, audio: PreparedAudio) -> Melody:
        """Return melody notes extracted from the prepared audio."""


@dataclass
class BasicPitchMelodyTranscriber:
    """Placeholder for a future Spotify Basic Pitch integration."""

    model_name: str = "basic-pitch"

    def transcribe(self, audio: PreparedAudio) -> Melody:
        raise NotImplementedError("Basic Pitch integration is not implemented yet.")
