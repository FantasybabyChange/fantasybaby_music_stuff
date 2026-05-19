"""High-level orchestration for audio-to-score transcription."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from music_stuff.audio import AudioPreprocessor
from music_stuff.harmony import ChordAnalyzer, KeyAnalyzer, build_analysis
from music_stuff.melody import MelodyTranscriber, SimplePitchMelodyTranscriber
from music_stuff.models import TranscriptionResult
from music_stuff.rhythm import RhythmQuantizer
from music_stuff.score import ScoreExporter


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
        audio = self.audio_preprocessor.prepare(input_path)
        raw_melody = self.melody_transcriber.transcribe(audio)
        tempo_bpm = self.rhythm_quantizer.estimate_tempo(audio)
        melody = self.rhythm_quantizer.quantize(raw_melody, tempo_bpm)
        key = self.key_analyzer.analyze(audio, melody)
        chords = self.chord_analyzer.analyze(audio, melody, key)
        analysis = build_analysis(key=key, tempo_bpm=tempo_bpm, chords=chords)

        output_dir.mkdir(parents=True, exist_ok=True)
        jianpu_path = self.score_exporter.export_jianpu(
            melody,
            analysis,
            output_dir / "melody.jianpu.txt",
        )
        analysis_path = self.score_exporter.export_analysis_json(
            analysis,
            output_dir / "analysis.json",
        )

        return TranscriptionResult(
            melody=melody,
            analysis=analysis,
            output_dir=output_dir,
            jianpu_path=jianpu_path,
            analysis_path=analysis_path,
        )
