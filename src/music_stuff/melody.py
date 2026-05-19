"""Melody transcription interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from music_stuff.audio import PreparedAudio
from music_stuff.models import Melody, NoteEvent


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


@dataclass
class SimplePitchMelodyTranscriber:
    """Extract a monophonic melody from WAV samples with autocorrelation.

    This is intentionally lightweight: it gives the project a working first
    feature without requiring a model download. Polyphonic commercial mixes will
    still need a stronger backend such as Basic Pitch later.
    """

    frame_size: int = 2048
    hop_size: int = 512
    min_frequency: float = 80.0
    max_frequency: float = 1000.0
    rms_threshold: float = 0.015
    correlation_threshold: float = 0.35
    min_note_duration: float = 0.08

    def transcribe(self, audio: PreparedAudio) -> Melody:
        if not audio.samples or not audio.sample_rate:
            return Melody(notes=(), source=str(audio.path))

        frames = self._frame_pitches(audio.samples, audio.sample_rate)
        notes = self._frames_to_notes(frames)
        return Melody(notes=tuple(notes), source=str(audio.path))

    def _frame_pitches(
        self,
        samples: tuple[float, ...],
        sample_rate: int,
    ) -> list[tuple[float, float, int | None]]:
        pitches: list[tuple[float, float, int | None]] = []
        if len(samples) < self.frame_size:
            padded = samples + (0.0,) * (self.frame_size - len(samples))
            pitch = self._estimate_frame_pitch(padded, sample_rate)
            return [(0.0, len(samples) / sample_rate, pitch)]

        for start in range(0, len(samples) - self.frame_size + 1, self.hop_size):
            frame = samples[start : start + self.frame_size]
            frame_start = start / sample_rate
            frame_end = (start + self.frame_size) / sample_rate
            pitches.append((frame_start, frame_end, self._estimate_frame_pitch(frame, sample_rate)))
        return pitches

    def _estimate_frame_pitch(self, frame: tuple[float, ...], sample_rate: int) -> int | None:
        mean = sum(frame) / len(frame)
        centered = [sample - mean for sample in frame]
        rms = math.sqrt(sum(sample * sample for sample in centered) / len(centered))
        if rms < self.rms_threshold:
            return None

        min_lag = max(1, int(sample_rate / self.max_frequency))
        max_lag = min(len(centered) // 2, int(sample_rate / self.min_frequency))
        best_lag = 0
        best_score = 0.0
        for lag in range(min_lag, max_lag + 1):
            score = 0.0
            energy = 0.0
            for index in range(0, len(centered) - lag):
                left = centered[index]
                right = centered[index + lag]
                score += left * right
                energy += left * left + right * right
            normalized = (2.0 * score / energy) if energy else 0.0
            if normalized > best_score:
                best_score = normalized
                best_lag = lag

        if not best_lag or best_score < self.correlation_threshold:
            return None
        frequency = sample_rate / best_lag
        return round(69 + 12 * math.log2(frequency / 440.0))

    def _frames_to_notes(self, frames: list[tuple[float, float, int | None]]) -> list[NoteEvent]:
        notes: list[NoteEvent] = []
        current_pitch: int | None = None
        current_start = 0.0
        current_end = 0.0

        for frame_start, frame_end, pitch in frames:
            if pitch == current_pitch:
                current_end = frame_end
                continue

            if current_pitch is not None and current_end - current_start >= self.min_note_duration:
                notes.append(NoteEvent(pitch=current_pitch, start=current_start, end=current_end))

            current_pitch = pitch
            current_start = frame_start
            current_end = frame_end

        if current_pitch is not None and current_end - current_start >= self.min_note_duration:
            notes.append(NoteEvent(pitch=current_pitch, start=current_start, end=current_end))
        return notes
