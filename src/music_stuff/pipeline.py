"""High-level orchestration for audio-to-score transcription."""

from __future__ import annotations

__all__ = ["PipelinePlan", "MusicTranscriptionPipeline"]

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from time import perf_counter

from typing import TypeVar

from collections.abc import Callable

from music_stuff.audio import AudioPreprocessor, PreparedAudio
from music_stuff.constants import (
    MELODY_SCORE_DENSITY_DIVISOR,
    MELODY_SCORE_DENSITY_WEIGHT,
    MELODY_SCORE_DURATION_DIVISOR,
    MELODY_SCORE_DURATION_WEIGHT,
    MELODY_SCORE_MINIMUM,
    VOICE_VS_ACCOMPANIMENT_THRESHOLD,
)
from music_stuff.harmony import ChordAnalyzer, KeyAnalyzer, build_analysis
from music_stuff.melody import (
    AutoMelodyTranscriber,
    BasicPitchMelodyTranscriber,
    MelodyTranscriber,
    MixedAudioMelodyTranscriber,
)
from music_stuff.models import Melody, TranscriptionResult
from music_stuff.rhythm import RhythmQuantizer
from music_stuff.score import ScoreExporter
from music_stuff.source import (
    ACCOMPANIMENT,
    HUMAN_VOICE,
    MIXED,
    SOURCE_LABELS,
    DemucsSourceSeparator,
    SourceSeparationResult,
    SourceStem,
)


LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")


@dataclass
class PipelinePlan:
    """Human-readable list of stages in the current pipeline."""

    stages: tuple[str, ...] = (
        "prepare audio",
        "separate sources",
        "extract melody",
        "estimate rhythm",
        "quantize rhythm",
        "estimate key",
        "infer core chords",
        "export Jianpu score and analysis JSON",
    )


@dataclass
class MusicTranscriptionPipeline:
    """Coordinate the future transcription backends."""

    audio_preprocessor: AudioPreprocessor = field(default_factory=AudioPreprocessor)
    source_separator: DemucsSourceSeparator = field(default_factory=DemucsSourceSeparator)
    melody_transcriber: MelodyTranscriber = field(default_factory=AutoMelodyTranscriber)
    rhythm_quantizer: RhythmQuantizer = field(default_factory=RhythmQuantizer)
    key_analyzer: KeyAnalyzer = field(default_factory=KeyAnalyzer)
    chord_analyzer: ChordAnalyzer = field(default_factory=ChordAnalyzer)
    score_exporter: ScoreExporter = field(default_factory=ScoreExporter)

    def plan(self) -> PipelinePlan:
        return PipelinePlan()

    def transcribe(self, input_path: Path, output_dir: Path) -> TranscriptionResult:
        """Run the full transcription flow.

        The melody stage prefers Basic Pitch for mixed recordings and falls back
        to the local pitch tracker when that optional backend is unavailable.
        """
        LOGGER.info("Starting transcription: input=%s output=%s", input_path, output_dir)
        started = perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        audio = self._time_stage("prepare audio", lambda: self.audio_preprocessor.prepare(input_path))
        separation = self._time_stage(
            "separate sources",
            lambda: self.source_separator.separate(input_path, output_dir / "stems"),
        )
        raw_melody = self._time_stage(
            "extract melody",
            lambda: self._extract_source_aware_melody(audio, separation),
        )
        rhythm = self._time_stage("estimate rhythm", lambda: self.rhythm_quantizer.analyze(audio))
        melody = self._time_stage("quantize rhythm", lambda: self.rhythm_quantizer.quantize(raw_melody, rhythm))
        key = self._time_stage("estimate key", lambda: self.key_analyzer.analyze(audio, melody))
        chords = self._time_stage("infer chords", lambda: self.chord_analyzer.analyze(audio, melody, key))
        analysis = build_analysis(key=key, tempo_bpm=rhythm.tempo_bpm, meter=rhythm.meter, chords=chords)

        jianpu_path = self._time_stage(
            "export Jianpu",
            lambda: self.score_exporter.export_jianpu(
                melody,
                analysis,
                output_dir / "melody.jianpu.txt",
            ),
        )
        analysis_path = self._time_stage(
            "export analysis JSON",
            lambda: self.score_exporter.export_analysis_json(
                analysis,
                output_dir / "analysis.json",
            ),
        )
        LOGGER.info(
            "Transcription complete: notes=%s key=%s tempo=%s elapsed=%.2fs",
            len(melody.notes),
            analysis.key.label if analysis.key else "unknown",
            analysis.tempo_bpm,
            perf_counter() - started,
        )

        return TranscriptionResult(
            melody=melody,
            analysis=analysis,
            output_dir=output_dir,
            jianpu_path=jianpu_path,
            analysis_path=analysis_path,
        )

    def _time_stage(self, stage: str, action: Callable[[], _T]) -> _T:
        started = perf_counter()
        LOGGER.info("Stage started: %s", stage)
        result = action()
        LOGGER.info("Stage finished: %s elapsed=%.2fs", stage, perf_counter() - started)
        return result

    def _extract_source_aware_melody(self, mixed_audio: PreparedAudio, separation: SourceSeparationResult) -> Melody:
        if not separation.is_available:
            LOGGER.info(
                "Source separation not used: backend=%s status=%s message=%s",
                separation.backend,
                separation.status,
                separation.message or "",
            )

        candidates: list[tuple[SourceStem, Melody, float]] = []
        for stem in separation.stems:
            try:
                stem_audio = self.audio_preprocessor.prepare(stem.path)
                melody = self._transcribe_stem_melody(stem_audio, stem)
            except Exception as exc:
                LOGGER.warning("Skipping separated stem %s after melody extraction failed: %s", stem.label, exc)
                continue
            melody = self._tag_melody_source(melody, stem.kind, stem.label, self._score_melody_candidate(melody))
            candidates.append((stem, melody, melody.source_confidence or 0.0))

        selected = self._select_melody_candidate(candidates)
        if selected is not None:
            LOGGER.info(
                "Selected melody source: %s confidence=%.2f notes=%s",
                selected.source_label,
                selected.source_confidence or 0.0,
                len(selected.notes),
            )
            return selected

        LOGGER.info("Using original mixed audio for melody extraction")
        melody = self.melody_transcriber.transcribe(mixed_audio)
        return self._tag_melody_source(melody, MIXED, SOURCE_LABELS[MIXED], self._score_melody_candidate(melody))

    def _select_melody_candidate(self, candidates: list[tuple[SourceStem, Melody, float]]) -> Melody | None:
        useful = [
            (stem, melody, confidence)
            for stem, melody, confidence in candidates
            if melody.notes and self._voiced_duration(melody) >= 1.0
        ]
        if not useful:
            return None

        voice = next((melody for stem, melody, _confidence in useful if stem.kind == HUMAN_VOICE), None)
        accompaniment = next((melody for stem, melody, _confidence in useful if stem.kind == ACCOMPANIMENT), None)
        if voice and (not accompaniment or (voice.source_confidence or 0.0) >= (accompaniment.source_confidence or 0.0) * VOICE_VS_ACCOMPANIMENT_THRESHOLD):
            return voice
        if accompaniment:
            return accompaniment
        return max((melody for _stem, melody, _confidence in useful), key=lambda item: item.source_confidence or 0.0)

    def _tag_melody_source(self, melody: Melody, kind: str, label: str, confidence: float) -> Melody:
        return Melody(
            notes=melody.notes,
            source=melody.source,
            source_kind=kind,
            source_label=label,
            source_confidence=confidence,
        )

    def _transcribe_stem_melody(self, stem_audio: PreparedAudio, stem: SourceStem) -> Melody:
        if stem.kind != HUMAN_VOICE or not isinstance(self.melody_transcriber, AutoMelodyTranscriber):
            return self.melody_transcriber.transcribe(stem_audio)

        vocal_transcriber = AutoMelodyTranscriber(
            basic_pitch=BasicPitchMelodyTranscriber(
                min_midi_pitch=43,
                max_midi_pitch=84,
                preferred_midi_pitch=64,
                continuity_weight=0.55,
            ),
            fallback=MixedAudioMelodyTranscriber(
                min_midi_pitch=43,
                max_midi_pitch=84,
                preferred_midi_pitch=64,
            ),
        )
        return vocal_transcriber.transcribe(stem_audio)

    def _score_melody_candidate(self, melody: Melody) -> float:
        if not melody.notes:
            return 0.0
        voiced = self._voiced_duration(melody)
        density = min(1.0, len(melody.notes) / MELODY_SCORE_DENSITY_DIVISOR)
        duration_score = min(1.0, voiced / MELODY_SCORE_DURATION_DIVISOR)
        return max(MELODY_SCORE_MINIMUM, min(1.0, (duration_score * MELODY_SCORE_DURATION_WEIGHT) + (density * MELODY_SCORE_DENSITY_WEIGHT)))

    def _voiced_duration(self, melody: Melody) -> float:
        return math.fsum(max(0.0, note.end - note.start) for note in melody.notes)
