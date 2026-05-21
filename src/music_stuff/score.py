"""Score and analysis export interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import logging
import math
from pathlib import Path

from music_stuff.models import AnalysisResult, Melody, NoteEvent


LOGGER = logging.getLogger(__name__)

_UNITS_PER_BEAT = 4
_SMALLEST_BEAT = Fraction(1, _UNITS_PER_BEAT)
_DOT_ABOVE = "\u0307"
_DOT_BELOW = "\u0323"
_UNDERLINE = "\u0332"
_DOUBLE_UNDERLINE = "\u0333"


@dataclass(frozen=True)
class _JianpuEvent:
    pitch: int
    duration_beats: Fraction


class ScoreExporter:
    """Export symbolic results into files users can inspect."""

    def export_midi(self, melody: Melody, output_path: Path) -> Path:
        raise NotImplementedError("MIDI export is not implemented yet.")

    def export_musicxml(
        self,
        melody: Melody,
        analysis: AnalysisResult,
        output_path: Path,
    ) -> Path:
        raise NotImplementedError("MusicXML export is not implemented yet.")

    def export_analysis_json(self, analysis: AnalysisResult, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(analysis), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def export_jianpu(
        self,
        melody: Melody,
        analysis: AnalysisResult,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(format_jianpu(melody, analysis), encoding="utf-8")
        LOGGER.info("Wrote Jianpu score: %s", output_path)
        return output_path


def format_jianpu(melody: Melody, analysis: AnalysisResult) -> str:
    """Render the selected melody as standard numbered musical notation."""
    tempo_bpm = analysis.tempo_bpm or 120.0
    meter = analysis.meter or "4/4"
    key_tonic = analysis.key.tonic if analysis.key else "C"
    key_mode = analysis.key.mode if analysis.key else "major"
    beat_seconds = 60.0 / tempo_bpm
    measure_beats = _measure_beats(meter)
    display_offset = _display_offset_seconds(melody.notes, beat_seconds, melody.source_kind)
    display_notes = _shift_notes(melody.notes, display_offset)
    events = _melody_events_with_rests(display_notes, beat_seconds)
    measures = _render_measures(events, key_tonic, measure_beats)

    lines = [
        "标准简谱（主旋律）",
        f"Source: {melody.source}",
        f"Main melody source: {melody.source_label}",
        f"Key: 1={key_tonic} ({key_mode})",
        f"Tempo: ♩={tempo_bpm:g}",
        f"Meter: {meter}",
        "Legend: 0=休止, -=延音, 下划线=八分音符, 双下划线=十六分音符, 数字上/下点=高/低八度",
        "Note: 当前输出为自动识别的单声部主旋律，复杂混音仍可能需要人工校正。",
    ]
    if display_offset > 0:
        lines.append(f"Melody entry: {display_offset:.2f}s in original audio; notation starts there.")

    lines.extend(
        [
            "",
            "标准简谱：",
        ]
    )
    lines.extend(measures)
    lines.append("")
    return "\n".join(lines)


def _measure_beats(meter: str) -> Fraction:
    try:
        numerator_text, denominator_text = meter.split("/", 1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except (AttributeError, ValueError):
        return Fraction(4, 1)
    if numerator <= 0 or denominator <= 0:
        return Fraction(4, 1)
    return Fraction(numerator * 4, denominator)


def _display_offset_seconds(notes: tuple[NoteEvent, ...], beat_seconds: float, source_kind: str = "mixed") -> float:
    if not notes:
        return 0.0
    first_start = min(note.start for note in notes)
    trim_threshold = beat_seconds * 8
    if first_start >= trim_threshold:
        return first_start
    if source_kind != "human_voice":
        return 0.0

    phrases = _melody_phrases(notes, gap_seconds=max(2.5, beat_seconds * 4))
    if len(phrases) < 2 or phrases[0][0].start <= beat_seconds * 2:
        return 0.0

    scored = [(phrase, _phrase_score(phrase)) for phrase in phrases]
    best_phrase, best_score = max(scored, key=lambda item: item[1])
    early_best = max((score for phrase, score in scored if phrase[0].start < best_phrase[0].start), default=0.0)
    if best_phrase[0].start >= trim_threshold and early_best < best_score * 0.8:
        return best_phrase[0].start
    return 0.0


def _melody_phrases(notes: tuple[NoteEvent, ...], gap_seconds: float) -> list[list[NoteEvent]]:
    phrases: list[list[NoteEvent]] = []
    current: list[NoteEvent] = []
    last_end: float | None = None
    for note in sorted(notes, key=lambda item: item.start):
        if current and last_end is not None and note.start - last_end > gap_seconds:
            phrases.append(current)
            current = []
        current.append(note)
        last_end = max(last_end or note.end, note.end)
    if current:
        phrases.append(current)
    return phrases


def _phrase_score(phrase: list[NoteEvent]) -> float:
    voiced_duration = math.fsum(max(0.0, note.end - note.start) for note in phrase)
    return voiced_duration + len(phrase) * 0.15


def _shift_notes(notes: tuple[NoteEvent, ...], offset: float) -> tuple[NoteEvent, ...]:
    if offset <= 0:
        return notes
    return tuple(
        NoteEvent(
            pitch=note.pitch,
            start=max(0.0, note.start - offset),
            end=max(0.0, note.end - offset),
            velocity=note.velocity,
        )
        for note in notes
    )


def _melody_events_with_rests(
    notes: tuple[NoteEvent, ...],
    beat_seconds: float,
) -> list[_JianpuEvent]:
    events: list[_JianpuEvent] = []
    cursor = Fraction(0, 1)
    for note in sorted(notes, key=lambda item: (item.start, item.pitch)):
        if note.end <= note.start:
            continue

        start = _seconds_to_beats(note.start, beat_seconds)
        end = _seconds_to_beats(note.end, beat_seconds)
        if end <= start:
            end = start + _SMALLEST_BEAT
        if start < cursor:
            if end <= cursor:
                continue
            start = cursor

        gap = start - cursor
        if gap >= _SMALLEST_BEAT:
            events.append(_JianpuEvent(pitch=-1, duration_beats=gap))

        duration = max(_SMALLEST_BEAT, end - start)
        events.append(_JianpuEvent(pitch=note.pitch, duration_beats=duration))
        cursor = start + duration
    return events


def _seconds_to_beats(seconds: float, beat_seconds: float) -> Fraction:
    if beat_seconds <= 0:
        return Fraction(0, 1)
    units = round((seconds / beat_seconds) * _UNITS_PER_BEAT)
    return Fraction(max(0, units), _UNITS_PER_BEAT)


def _render_measures(
    events: list[_JianpuEvent],
    key_tonic: str,
    measure_beats: Fraction,
    measures_per_line: int = 2,
) -> list[str]:
    if not events:
        return ["| (no melody detected) ||"]

    measures: list[list[str]] = [[]]
    position = Fraction(0, 1)
    for event in events:
        remaining = event.duration_beats
        continuation = False
        while remaining > 0:
            if position >= measure_beats:
                measures.append([])
                position = Fraction(0, 1)

            room = measure_beats - position
            segment = min(remaining, room)
            base = _event_base_token(event.pitch, key_tonic, continuation)
            measures[-1].extend(_duration_tokens(base, segment))
            position += segment
            remaining -= segment
            continuation = True

            if position >= measure_beats and remaining > 0:
                measures.append([])
                position = Fraction(0, 1)

    while len(measures) > 1 and not measures[-1]:
        measures.pop()

    lines: list[str] = []
    for index in range(0, len(measures), measures_per_line):
        group = measures[index : index + measures_per_line]
        rendered_measures = [" ".join(measure).strip() for measure in group]
        is_last_group = index + measures_per_line >= len(measures)
        ending = " ||" if is_last_group else " |"
        lines.append("| " + " | ".join(rendered_measures) + ending)
    return lines


def _event_base_token(pitch: int, key_tonic: str, continuation: bool) -> str:
    if pitch < 0:
        return "0" if not continuation else "-"
    if continuation:
        return "-"
    return _pitch_to_jianpu(pitch, key_tonic)


def _duration_tokens(base: str, duration: Fraction) -> list[str]:
    if duration <= 0:
        return []
    if duration in {Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1, 1), Fraction(3, 2)}:
        return [_mark_duration(base, duration)]

    tokens = [_mark_duration(base, min(duration, Fraction(1, 1)))]
    remaining = duration - min(duration, Fraction(1, 1))
    while remaining > 0:
        piece = min(remaining, Fraction(1, 1))
        tokens.append(_mark_duration("-", piece))
        remaining -= piece
    return tokens


def _mark_duration(base: str, duration: Fraction) -> str:
    if duration == Fraction(1, 4):
        return base + _DOUBLE_UNDERLINE
    if duration == Fraction(1, 2):
        return base + _UNDERLINE
    if duration == Fraction(3, 4):
        return base + _UNDERLINE + "."
    if duration == Fraction(3, 2):
        return base + "."
    return base


def _pitch_to_jianpu(pitch: int, key_tonic: str) -> str:
    tonic_pitch_class = _TONIC_TO_PC.get(key_tonic, 0)
    relative_pc = (pitch % 12 - tonic_pitch_class) % 12
    octave = (pitch - (60 + tonic_pitch_class)) // 12

    degree = _MAJOR_DEGREES.get(relative_pc)
    if degree is None and (relative_pc - 1) % 12 in _MAJOR_DEGREES:
        degree = "#" + _MAJOR_DEGREES[(relative_pc - 1) % 12]
    if degree is None and (relative_pc + 1) % 12 in _MAJOR_DEGREES:
        degree = "b" + _MAJOR_DEGREES[(relative_pc + 1) % 12]
    if degree is None:
        degree = "?"

    if octave > 0:
        return degree + (_DOT_ABOVE * octave)
    if octave < 0:
        return degree + (_DOT_BELOW * abs(octave))
    return degree


_TONIC_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
_MAJOR_DEGREES = {0: "1", 2: "2", 4: "3", 5: "4", 7: "5", 9: "6", 11: "7"}
