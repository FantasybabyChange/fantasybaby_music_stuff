"""Score and analysis export interfaces."""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import json
import logging
import math
from pathlib import Path

from music_stuff.models import AnalysisResult, Melody, NoteEvent


LOGGER = logging.getLogger(__name__)


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
    """Render a readable numbered-notation draft."""
    tempo_bpm = analysis.tempo_bpm or 120.0
    meter = analysis.meter or "4/4"
    key_tonic = analysis.key.tonic if analysis.key else "C"
    beat_seconds = 60.0 / tempo_bpm
    display_offset = _display_offset_seconds(melody.notes, beat_seconds, melody.source_kind)
    display_notes = _shift_notes(melody.notes, display_offset)
    events = _melody_events_with_rests(display_notes, beat_seconds)
    tokens = [_format_event(event, key_tonic, beat_seconds) for event in events]

    lines = [
        "Jianpu melody draft",
        f"Source: {melody.source}",
        f"Main melody source: {melody.source_label}",
        f"Key: 1={key_tonic}",
        f"Tempo: {tempo_bpm:g} bpm",
        f"Meter: {meter}",
        "Legend: 1-7=scale degrees, #/b=accidentals, '=higher octave, ,=lower octave, /=short, -=hold, 0=rest",
        "Note: This exports one selected main melody only; multi-track vocal/accompaniment scores will come later.",
    ]
    if display_offset > 0:
        lines.append(f"Melody entry: {display_offset:.2f}s in original audio; detailed draft starts there.")
    lines.extend([
        "",
        "Melody outline (pitch only, repeated notes collapsed):",
    ])
    lines.extend(_wrap_outline(_melody_outline(display_notes, key_tonic)))
    lines.extend([
        "",
        "Detailed rhythm draft:",
    ])
    lines.extend(_wrap_tokens(tokens))
    lines.append("")
    return "\n".join(lines)


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
) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    cursor = 0.0
    rest_threshold = beat_seconds
    for note in sorted(notes, key=lambda item: item.start):
        if note.start - cursor >= rest_threshold:
            events.append(NoteEvent(pitch=-1, start=cursor, end=note.start, velocity=0))
        events.append(note)
        cursor = max(cursor, note.end)
    return events


def _melody_outline(notes: tuple[NoteEvent, ...], key_tonic: str) -> list[str]:
    outline: list[str] = []
    last_token: str | None = None
    for note in sorted(notes, key=lambda item: item.start):
        token = _pitch_to_jianpu(note.pitch, key_tonic)
        if token == last_token:
            continue
        outline.append(token)
        last_token = token
    return outline


def _wrap_outline(tokens: list[str], per_line: int = 24) -> list[str]:
    if not tokens:
        return ["(no melody detected)"]
    return [
        " ".join(tokens[index : index + per_line])
        for index in range(0, len(tokens), per_line)
    ]


def _format_event(event: NoteEvent, key_tonic: str, beat_seconds: float) -> tuple[str, float]:
    duration_beats = max(0.25, (event.end - event.start) / beat_seconds)
    if event.pitch < 0:
        return f"0{_duration_suffix(duration_beats)}", _quantize_beats(duration_beats)
    return (
        f"{_pitch_to_jianpu(event.pitch, key_tonic)}{_duration_suffix(duration_beats)}",
        _quantize_beats(duration_beats),
    )


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
        return degree + ("'" * octave)
    if octave < 0:
        return degree + ("," * abs(octave))
    return degree


def _duration_suffix(duration_beats: float) -> str:
    quantized = _quantize_beats(duration_beats)
    if quantized <= 0.25:
        return "//"
    if math.isclose(quantized, 0.5):
        return "/"
    if math.isclose(quantized, 0.75):
        return "/."
    if math.isclose(quantized, 1.0):
        return ""

    holds: list[str] = []
    remaining = quantized - 1.0
    while remaining >= 0.999:
        holds.append("-")
        remaining -= 1.0
    if remaining >= 0.74:
        holds.append("-/.")
    elif remaining >= 0.49:
        holds.append("-/")
    elif remaining >= 0.24:
        holds.append("-//")
    return " " + " ".join(holds) if holds else ""


def _quantize_beats(duration_beats: float) -> float:
    return max(0.25, round(float(Fraction(duration_beats).limit_denominator(4)) * 4) / 4)


def _wrap_tokens(tokens: list[tuple[str, float]], measures_per_line: int = 4) -> list[str]:
    if not tokens:
        return ["(no melody detected)"]

    lines: list[str] = []
    current: list[str] = []
    measure_beats = 0.0
    measure_count = 0
    for token, duration_beats in tokens:
        current.append(token)
        measure_beats += duration_beats
        if measure_beats >= 3.99:
            current.append("|")
            measure_beats = 0.0
            measure_count += 1
        if measure_count >= measures_per_line:
            lines.append(" ".join(current).rstrip())
            current = []
            measure_count = 0
    if current:
        lines.append(" ".join(current).rstrip())
    return lines


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
