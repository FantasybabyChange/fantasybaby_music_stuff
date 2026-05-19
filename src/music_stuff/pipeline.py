"""High-level orchestration for audio-to-score transcription."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from time import perf_counter

from music_stuff.audio import AudioPreprocessor
from music_stuff.harmony import ChordAnalyzer, KeyAnalyzer, build_analysis
from music_stuff.melody import MelodyTranscriber, SimplePitchMelodyTranscriber
from music_stuff.models import TranscriptionResult
from music_stuff.rhythm import RhythmQuantizer
from music_stuff.score import ScoreExporter


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelinePlan:
    """Human-readable list of stages in the current pipeline."""

    stages: tuple[str, ...] = (
        "prepare audio",
        "extract melody",
        "estimate tempo",
        "quantize rhythm",
        "estimate key",
        "infer core chords",
        "export Jianpu score and analysis JSON",
    )


@dataclass
class MusicTranscriptionPipeline:
    """Coordinate the future transcription backends."""

    audio_preprocessor: AudioPreprocessor = field(default_factory=AudioPreprocessor)
    melody_transcriber: MelodyTranscriber = field(default_factory=SimplePitchMelodyTranscriber)
    rhythm_quantizer: RhythmQuantizer = field(default_factory=RhythmQuantizer)
    key_analyzer: KeyAnalyzer = field(default_factory=KeyAnalyzer)
    chord_analyzer: ChordAnalyzer = field(default_factory=ChordAnalyzer)
    score_exporter: ScoreExporter = field(default_factory=ScoreExporter)

    def plan(self) -> PipelinePlan:
        return PipelinePlan()

    def transcribe(self, input_path: Path, output_dir: Path) -> TranscriptionResult:
        """Run the full transcription flow.

        The first concrete backend is intentionally lightweight and targets clear
        monophonic WAV recordings. Later versions can swap in model-based melody
        extraction without changing this orchestration.
        """
        LOGGER.info("Starting transcription: input=%s output=%s", input_path, output_dir)
        started = perf_counter()
        audio = self._time_stage("prepare audio", lambda: self.audio_preprocessor.prepare(input_path))
        raw_melody = self._time_stage("extract melody", lambda: self.melody_transcriber.transcribe(audio))
        tempo_bpm = self._time_stage("estimate tempo", lambda: self.rhythm_quantizer.estimate_tempo(audio))
        melody = self._time_stage("quantize rhythm", lambda: self.rhythm_quantizer.quantize(raw_melody, tempo_bpm))
        key = self._time_stage("estimate key", lambda: self.key_analyzer.analyze(audio, melody))
        chords = self._time_stage("infer chords", lambda: self.chord_analyzer.analyze(audio, melody, key))
        analysis = build_analysis(key=key, tempo_bpm=tempo_bpm, chords=chords)

        output_dir.mkdir(parents=True, exist_ok=True)
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

    def _time_stage(self, stage: str, action):
        started = perf_counter()
        LOGGER.info("Stage started: %s", stage)
        result = action()
        LOGGER.info("Stage finished: %s elapsed=%.2fs", stage, perf_counter() - started)
        return result
