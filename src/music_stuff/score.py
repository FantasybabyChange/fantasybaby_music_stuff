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
    """Render a compact numbered-notation score."""
    tempo_bpm = analysis.tempo_bpm or 120.0
    key_tonic = analysis.key.tonic if analysis.key else "C"
    beat_seconds = 60.0 / tempo_bpm
    events = _melody_events_with_rests(melody.notes, beat_seconds)
    tokens = [_format_event(event, key_tonic, beat_seconds) for event in events]

    lines = [
        "简谱",
        f"Source: {melody.source}",
        f"Key: 1={key_tonic}",
        f"Tempo: {tempo_bpm:g} bpm",
        "Meter: 4/4",
        "",
    ]
    lines.extend(_wrap_tokens(tokens))
    lines.append("")
    return "\n".join(lines)


def _melody_events_with_rests(
    notes: tuple[NoteEvent, ...],
    beat_seconds: float,
) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    cursor = 0.0
    rest_threshold = beat_seconds / 4.0
    for note in sorted(notes, key=lambda item: item.start):
        if note.start - cursor >= rest_threshold:
            events.append(NoteEvent(pitch=-1, start=cursor, end=note.start, velocity=0))
        events.append(note)
        cursor = max(cursor, note.end)
    return events


def _format_event(event: NoteEvent, key_tonic: str, beat_seconds: float) -> str:
    duration_beats = max(0.25, (event.end - event.start) / beat_seconds)
    if event.pitch < 0:
        return f"0{_duration_suffix(duration_beats)}"
    return f"{_pitch_to_jianpu(event.pitch, key_tonic)}{_duration_suffix(duration_beats)}"


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
    quantized = float(Fraction(duration_beats).limit_denominator(4))
    if math.isclose(quantized, 1.0):
        return ""
    if quantized > 1.0 and math.isclose(quantized, round(quantized)):
        return " " + " ".join("-" for _ in range(int(round(quantized)) - 1))
    if math.isclose(quantized, 0.5):
        return "/"
    if math.isclose(quantized, 0.25):
        return "//"
    return f"({quantized:g})"


def _wrap_tokens(tokens: list[str], per_line: int = 16) -> list[str]:
    if not tokens:
        return ["(no melody detected)"]

    lines: list[str] = []
    current: list[str] = []
    for index, token in enumerate(tokens, start=1):
        current.append(token)
        if index % 4 == 0:
            current.append("|")
        if index % per_line == 0:
            lines.append(" ".join(current).rstrip())
            current = []
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
