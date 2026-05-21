"""Melody transcription interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from collections.abc import Callable
from typing import Any, Protocol

from music_stuff.audio import PreparedAudio
from music_stuff.models import Melody, NoteEvent


LOGGER = logging.getLogger(__name__)


def _amplitude_to_velocity(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 80
    if numeric <= 1.0:
        numeric *= 127.0
    return max(1, min(127, int(round(numeric))))


class MelodyTranscriber(Protocol):
    """Convert audio into symbolic melody notes."""

    def transcribe(self, audio: PreparedAudio) -> Melody:
        """Return melody notes extracted from the prepared audio."""


@dataclass
class BasicPitchMelodyTranscriber:
    """Use Spotify Basic Pitch when it is installed in the runtime."""

    model_name: str = "basic-pitch"
    min_midi_pitch: int = 48
    max_midi_pitch: int = 88
    min_note_duration: float = 0.08
    melody_frame_seconds: float = 0.10
    preferred_midi_pitch: int = 76
    continuity_weight: float = 0.4
    predict_func: Callable[[str], tuple[Any, Any, Any]] | None = None

    def transcribe(self, audio: PreparedAudio) -> Melody:
        predict = self.predict_func or self._load_predict()
        LOGGER.info("Extracting melody with Basic Pitch: input=%s", audio.path)
        _model_output, _midi_data, note_events = predict(str(audio.path))
        candidates = self._note_events_to_notes(note_events)
        notes = self._select_melody_line(candidates)
        LOGGER.info("Basic Pitch melody extraction complete: notes=%s", len(notes))
        return Melody(notes=tuple(notes), source=str(audio.path))

    def is_available(self) -> bool:
        try:
            self._load_predict()
        except ImportError:
            return False
        return True

    def _load_predict(self):
        from basic_pitch.inference import predict

        return predict

    def _note_events_to_notes(self, note_events: Any) -> list[NoteEvent]:
        notes: list[NoteEvent] = []
        for event in note_events:
            parsed = self._parse_note_event(event)
            if parsed is None:
                continue
            start, end, pitch, amplitude = parsed
            if pitch < self.min_midi_pitch or pitch > self.max_midi_pitch:
                continue
            if end <= start or end - start < self.min_note_duration:
                continue
            notes.append(NoteEvent(pitch=pitch, start=start, end=end, velocity=amplitude))
        return sorted(notes, key=lambda note: (note.start, note.pitch))

    def _select_melody_line(self, candidates: list[NoteEvent]) -> list[NoteEvent]:
        if not candidates:
            return []

        start = min(note.start for note in candidates)
        end = max(note.end for note in candidates)
        frames: list[tuple[float, float, NoteEvent | None]] = []
        previous_pitch: int | None = None
        cursor = start
        while cursor < end:
            frame_end = min(end, cursor + self.melody_frame_seconds)
            active = [
                note
                for note in candidates
                if note.start < frame_end and note.end > cursor
            ]
            note = self._choose_melody_note(active, previous_pitch)
            if note is not None:
                previous_pitch = note.pitch
            frames.append((cursor, frame_end, note))
            cursor = frame_end

        return self._melody_frames_to_notes(frames)

    def _choose_melody_note(
        self,
        active_notes: list[NoteEvent],
        previous_pitch: int | None,
    ) -> NoteEvent | None:
        best_note: NoteEvent | None = None
        best_score = 0.0
        for note in active_notes:
            register_score = 1.0 - min(abs(note.pitch - self.preferred_midi_pitch) / 24.0, 0.75)
            continuity_score = 1.0
            if previous_pitch is not None:
                continuity_score = 1.0 - min(abs(note.pitch - previous_pitch) / 12.0, 0.85)
            score = note.velocity * (register_score + self.continuity_weight * continuity_score)
            if score > best_score:
                best_score = score
                best_note = note
        return best_note

    def _melody_frames_to_notes(
        self,
        frames: list[tuple[float, float, NoteEvent | None]],
    ) -> list[NoteEvent]:
        notes: list[NoteEvent] = []
        current_pitch: int | None = None
        current_start = 0.0
        current_end = 0.0
        current_velocity = 80

        for frame_start, frame_end, note in frames:
            pitch = note.pitch if note is not None else None
            velocity = note.velocity if note is not None else 80
            if pitch == current_pitch:
                current_end = frame_end
                current_velocity = max(current_velocity, velocity)
                continue

            if current_pitch is not None and current_end - current_start >= self.min_note_duration:
                notes.append(
                    NoteEvent(
                        pitch=current_pitch,
                        start=current_start,
                        end=current_end,
                        velocity=current_velocity,
                    )
                )

            current_pitch = pitch
            current_start = frame_start
            current_end = frame_end
            current_velocity = velocity

        if current_pitch is not None and current_end - current_start >= self.min_note_duration:
            notes.append(
                NoteEvent(
                    pitch=current_pitch,
                    start=current_start,
                    end=current_end,
                    velocity=current_velocity,
                )
            )
        return notes

    def _parse_note_event(self, event: Any) -> tuple[float, float, int, int] | None:
        if isinstance(event, dict):
            start = event.get("start_time_s", event.get("start", event.get("onset")))
            end = event.get("end_time_s", event.get("end", event.get("offset")))
            pitch = event.get("pitch_midi", event.get("pitch", event.get("midi_pitch")))
            amplitude = event.get("amplitude", event.get("velocity", 80))
        else:
            try:
                start, end, pitch, amplitude = event[:4]
            except (TypeError, ValueError):
                return None

        if start is None or end is None or pitch is None:
            return None
        velocity = _amplitude_to_velocity(amplitude)
        return float(start), float(end), int(round(float(pitch))), velocity


@dataclass
class AutoMelodyTranscriber:
    """Prefer Basic Pitch, then fall back to the local mixed-audio backend."""

    basic_pitch: BasicPitchMelodyTranscriber = field(default_factory=BasicPitchMelodyTranscriber)
    fallback: "MixedAudioMelodyTranscriber" = field(default_factory=lambda: MixedAudioMelodyTranscriber())

    def transcribe(self, audio: PreparedAudio) -> Melody:
        try:
            if self.basic_pitch.is_available():
                return self.basic_pitch.transcribe(audio)
        except Exception as exc:
            LOGGER.warning("Basic Pitch failed; falling back to mixed-audio backend: %s", exc)
        LOGGER.info("Using mixed-audio melody backend")
        return self.fallback.transcribe(audio)


@dataclass
class MixedAudioMelodyTranscriber:
    """Extract the most prominent melody candidate from mixed music audio.

    This backend uses librosa's harmonic separation and spectral pitch tracking
    when available. It is still a heuristic, but it is much better suited to
    mixed recordings than raw autocorrelation on the waveform.
    """

    fallback: "SimplePitchMelodyTranscriber" = field(default_factory=lambda: SimplePitchMelodyTranscriber())
    frame_size: int = 2048
    hop_size: int = 512
    min_midi_pitch: int = 48
    max_midi_pitch: int = 88
    preferred_midi_pitch: int = 72
    min_note_duration: float = 0.12
    silence_threshold_ratio: float = 0.08
    continuity_weight: float = 0.35

    def transcribe(self, audio: PreparedAudio) -> Melody:
        try:
            import librosa
            import numpy as np
        except ImportError:
            LOGGER.warning("librosa is not available; falling back to simple pitch transcriber")
            return self.fallback.transcribe(audio)

        if not audio.samples or not audio.sample_rate:
            LOGGER.warning("No audio samples available for mixed-audio melody extraction: %s", audio.path)
            return Melody(notes=(), source=str(audio.path))

        y = np.asarray(audio.samples, dtype=np.float32)
        if y.size == 0:
            return Melody(notes=(), source=str(audio.path))

        LOGGER.info(
            "Extracting mixed-audio melody: samples=%s sample_rate=%s frame_size=%s hop_size=%s",
            len(y),
            audio.sample_rate,
            self.frame_size,
            self.hop_size,
        )
        harmonic = librosa.effects.harmonic(y)
        pitches, magnitudes = librosa.piptrack(
            y=harmonic,
            sr=audio.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            fmin=librosa.midi_to_hz(self.min_midi_pitch),
            fmax=librosa.midi_to_hz(self.max_midi_pitch),
        )
        frames = self._select_pitch_track(pitches, magnitudes, audio.sample_rate, librosa, np)
        notes = self._frames_to_notes(frames)
        LOGGER.info(
            "Mixed-audio melody extraction complete: frames=%s voiced_frames=%s notes=%s",
            len(frames),
            sum(1 for _start, _end, pitch in frames if pitch is not None),
            len(notes),
        )
        return Melody(notes=tuple(notes), source=str(audio.path))

    def _select_pitch_track(self, pitches, magnitudes, sample_rate: int, librosa, np):
        frame_count = pitches.shape[1]
        max_magnitude = float(np.max(magnitudes)) if magnitudes.size else 0.0
        silence_threshold = max_magnitude * self.silence_threshold_ratio
        frames: list[tuple[float, float, int | None]] = []
        previous_pitch: int | None = None

        for frame_index in range(frame_count):
            frame_magnitudes = magnitudes[:, frame_index]
            if not frame_magnitudes.size or float(np.max(frame_magnitudes)) < silence_threshold:
                frames.append(self._frame_time(frame_index, sample_rate, None))
                continue

            candidate_indices = np.argwhere(frame_magnitudes > 0).flatten()
            best_pitch: int | None = None
            best_score = 0.0
            for index in candidate_indices:
                frequency = float(pitches[index, frame_index])
                if frequency <= 0.0:
                    continue
                midi = int(round(float(librosa.hz_to_midi(frequency))))
                if midi < self.min_midi_pitch or midi > self.max_midi_pitch:
                    continue

                magnitude = float(frame_magnitudes[index])
                register_score = 1.0 - min(abs(midi - self.preferred_midi_pitch) / 24.0, 0.75)
                continuity_score = 1.0
                if previous_pitch is not None:
                    continuity_score = 1.0 - min(abs(midi - previous_pitch) / 12.0, 0.85)
                score = magnitude * (register_score + self.continuity_weight * continuity_score)
                if score > best_score:
                    best_score = score
                    best_pitch = midi

            if best_pitch is not None:
                previous_pitch = best_pitch
            frames.append(self._frame_time(frame_index, sample_rate, best_pitch))
        return self._smooth_pitch_track(frames)

    def _frame_time(
        self,
        frame_index: int,
        sample_rate: int,
        pitch: int | None,
    ) -> tuple[float, float, int | None]:
        start = frame_index * self.hop_size / sample_rate
        end = (frame_index * self.hop_size + self.frame_size) / sample_rate
        return start, end, pitch

    def _smooth_pitch_track(
        self,
        frames: list[tuple[float, float, int | None]],
    ) -> list[tuple[float, float, int | None]]:
        smoothed = list(frames)
        for index in range(1, len(frames) - 1):
            previous_pitch = frames[index - 1][2]
            pitch = frames[index][2]
            next_pitch = frames[index + 1][2]
            if pitch is not None and previous_pitch == next_pitch and previous_pitch is not None:
                if abs(pitch - previous_pitch) > 7:
                    start, end, _pitch = frames[index]
                    smoothed[index] = (start, end, previous_pitch)
        return smoothed

    def _frames_to_notes(self, frames: list[tuple[float, float, int | None]]) -> list[NoteEvent]:
        notes = self.fallback._frames_to_notes(frames)
        return [note for note in notes if note.end - note.start >= self.min_note_duration]


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
    correlation_stride: int = 4
    min_midi_pitch: int = 48
    max_midi_pitch: int = 88

    def transcribe(self, audio: PreparedAudio) -> Melody:
        if not audio.samples or not audio.sample_rate:
            LOGGER.warning("No audio samples available for melody extraction: %s", audio.path)
            return Melody(notes=(), source=str(audio.path))

        LOGGER.info(
            "Extracting melody: samples=%s sample_rate=%s frame_size=%s hop_size=%s",
            len(audio.samples),
            audio.sample_rate,
            self.frame_size,
            self.hop_size,
        )
        frames = self._frame_pitches(audio.samples, audio.sample_rate)
        notes = self._frames_to_notes(frames)
        voiced_frames = sum(1 for _start, _end, pitch in frames if pitch is not None)
        LOGGER.info(
            "Melody extraction complete: frames=%s voiced_frames=%s notes=%s",
            len(frames),
            voiced_frames,
            len(notes),
        )
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
        stride = max(1, self.correlation_stride)
        for lag in range(min_lag, max_lag + 1):
            score = 0.0
            energy = 0.0
            for index in range(0, len(centered) - lag, stride):
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
        pitch = round(69 + 12 * math.log2(frequency / 440.0))
        if pitch < self.min_midi_pitch or pitch > self.max_midi_pitch:
            return None
        return pitch

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
