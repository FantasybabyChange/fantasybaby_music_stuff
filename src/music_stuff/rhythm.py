"""Tempo detection and rhythmic quantization interfaces."""

from __future__ import annotations

__all__ = ["RhythmQuantizer"]

from dataclasses import dataclass
import logging
import math
from statistics import median

from music_stuff.audio import PreparedAudio
from music_stuff.constants import MERGE_GAP_RATIO
from music_stuff.models import Melody, NoteEvent, RhythmEstimate


LOGGER = logging.getLogger(__name__)


@dataclass
class RhythmQuantizer:
    """Turn raw note timings into readable score timings."""

    smallest_grid: str = "1/16"
    default_tempo_bpm: float = 120.0
    subdivisions_per_beat: int = 4
    default_meter: str = "4/4"

    def analyze(self, audio: PreparedAudio) -> RhythmEstimate:
        """Estimate tempo and beat positions from the prepared audio."""
        if not len(audio.samples) or not audio.sample_rate:
            LOGGER.warning("No audio samples available for rhythm analysis: %s", audio.path)
            return self._fallback_estimate()

        try:
            import librosa
            import numpy as np
        except ImportError:
            LOGGER.warning("librosa is not available; using default tempo %.1f", self.default_tempo_bpm)
            return self._fallback_estimate()

        y = np.asarray(audio.samples, dtype=np.float32)
        if y.size == 0:
            return self._fallback_estimate()

        LOGGER.info("Estimating rhythm with librosa beat tracker: samples=%s sample_rate=%s", len(y), audio.sample_rate)
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=audio.sample_rate, trim=False)
        except Exception as exc:
            LOGGER.warning("Beat tracking failed; using default tempo %.1f: %s", self.default_tempo_bpm, exc)
            return self._fallback_estimate()

        tempo_bpm = self._tempo_to_float(tempo)
        if not math.isfinite(tempo_bpm) or tempo_bpm <= 0:
            LOGGER.warning("Beat tracker returned invalid tempo %r; using default tempo %.1f", tempo, self.default_tempo_bpm)
            return self._fallback_estimate()
        tempo_bpm = self._normalize_tempo_range(tempo_bpm)

        beat_times = tuple(float(value) for value in librosa.frames_to_time(beat_frames, sr=audio.sample_rate))
        beat_offset = self._estimate_beat_offset(beat_times, tempo_bpm)
        confidence = min(1.0, len(beat_times) / max(1.0, audio.duration_seconds * tempo_bpm / 60.0))
        LOGGER.info(
            "Rhythm estimate complete: tempo=%.2f bpm beats=%s offset=%.3fs confidence=%.2f",
            tempo_bpm,
            len(beat_times),
            beat_offset,
            confidence,
        )
        return RhythmEstimate(
            tempo_bpm=tempo_bpm,
            beat_times=beat_times,
            beat_offset=beat_offset,
            meter=self.default_meter,
            confidence=confidence,
        )

    def estimate_tempo(self, audio: PreparedAudio) -> float:
        return self.analyze(audio).tempo_bpm

    def quantize(self, melody: Melody, rhythm: RhythmEstimate | float) -> Melody:
        estimate = rhythm if isinstance(rhythm, RhythmEstimate) else RhythmEstimate(tempo_bpm=float(rhythm))
        beat_seconds = 60.0 / estimate.tempo_bpm
        grid_seconds = beat_seconds / self.subdivisions_per_beat
        offset = estimate.beat_offset
        quantized_notes: list[NoteEvent] = []
        for note in melody.notes:
            start = self._snap_to_grid(note.start, grid_seconds, offset)
            end = self._snap_to_grid(note.end, grid_seconds, offset)
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
        quantized_notes = self._clean_quantized_notes(quantized_notes, grid_seconds)
        return Melody(
            notes=tuple(quantized_notes),
            source=melody.source,
            source_kind=melody.source_kind,
            source_label=melody.source_label,
            source_confidence=melody.source_confidence,
        )

    def _fallback_estimate(self) -> RhythmEstimate:
        return RhythmEstimate(
            tempo_bpm=self.default_tempo_bpm,
            beat_offset=0.0,
            meter=self.default_meter,
            confidence=0.0,
        )

    def _tempo_to_float(self, tempo: object) -> float:
        if hasattr(tempo, "item"):
            return float(tempo.item())
        try:
            return float(tempo)
        except TypeError:
            return float(tempo[0])

    def _estimate_beat_offset(self, beat_times: tuple[float, ...], tempo_bpm: float) -> float:
        if not beat_times:
            return 0.0
        if len(beat_times) >= 2:
            intervals = [
                beat_times[index + 1] - beat_times[index]
                for index in range(len(beat_times) - 1)
                if beat_times[index + 1] > beat_times[index]
            ]
            beat_seconds = median(intervals) if intervals else 60.0 / tempo_bpm
        else:
            beat_seconds = 60.0 / tempo_bpm

        offset = beat_times[0] % beat_seconds
        return max(0.0, offset)

    def _snap_to_grid(self, seconds: float, grid_seconds: float, offset: float) -> float:
        grid_index = round((seconds - offset) / grid_seconds)
        return max(0.0, offset + grid_index * grid_seconds)

    def _clean_quantized_notes(self, notes: list[NoteEvent], grid_seconds: float) -> list[NoteEvent]:
        cleaned: list[NoteEvent] = []
        merge_gap = grid_seconds * MERGE_GAP_RATIO
        for note in sorted(notes, key=lambda item: (item.start, item.end, item.pitch)):
            start = max(0.0, note.start)
            end = max(note.end, start + grid_seconds)
            current = NoteEvent(
                pitch=note.pitch,
                start=start,
                end=end,
                velocity=note.velocity,
            )
            if not cleaned:
                cleaned.append(current)
                continue

            previous = cleaned[-1]
            gap = current.start - previous.end
            if current.pitch == previous.pitch and gap <= merge_gap:
                cleaned[-1] = NoteEvent(
                    pitch=previous.pitch,
                    start=previous.start,
                    end=max(previous.end, current.end),
                    velocity=max(previous.velocity, current.velocity),
                )
                continue

            if current.start < previous.end:
                adjusted_start = previous.end
                if current.end <= adjusted_start:
                    continue
                current = NoteEvent(
                    pitch=current.pitch,
                    start=adjusted_start,
                    end=current.end,
                    velocity=current.velocity,
                )

            cleaned.append(current)
        return cleaned

    def _normalize_tempo_range(self, tempo_bpm: float) -> float:
        normalized = tempo_bpm
        while normalized > 160.0:
            normalized /= 2.0
        while normalized < 60.0:
            normalized *= 2.0
        if not math.isclose(normalized, tempo_bpm):
            LOGGER.info("Normalized tempo from %.2f bpm to %.2f bpm", tempo_bpm, normalized)
        return normalized
