from pathlib import Path

from music_stuff.audio import PreparedAudio
from music_stuff.models import Melody, NoteEvent
from music_stuff.pipeline import MusicTranscriptionPipeline
from music_stuff.source import ACCOMPANIMENT, HUMAN_VOICE, SOURCE_LABELS, SourceSeparationResult, SourceStem


class FakeAudioPreprocessor:
    def prepare(self, input_path):
        return PreparedAudio(
            path=Path(input_path),
            sample_rate=22050,
            duration_seconds=2.0,
            samples=(0.1, -0.1) * 22050,
        )


class FakeSourceSeparator:
    def separate(self, _input_path, output_dir):
        return SourceSeparationResult(
            backend="fake",
            status="ok",
            stems=(
                SourceStem(HUMAN_VOICE, SOURCE_LABELS[HUMAN_VOICE], output_dir / "vocals.wav"),
                SourceStem(ACCOMPANIMENT, SOURCE_LABELS[ACCOMPANIMENT], output_dir / "no_vocals.wav"),
            ),
        )


class FakeMelodyTranscriber:
    def transcribe(self, audio):
        if Path(audio.path).name == "vocals.wav":
            return Melody(
                notes=(
                    NoteEvent(pitch=69, start=0.0, end=0.7),
                    NoteEvent(pitch=71, start=0.7, end=1.5),
                ),
                source=str(audio.path),
            )
        return Melody(
            notes=(NoteEvent(pitch=60, start=0.0, end=1.5),),
            source=str(audio.path),
        )


def test_pipeline_marks_selected_melody_as_human_voice(tmp_path):
    input_path = tmp_path / "song.mp3"
    input_path.write_bytes(b"demo")
    output_dir = tmp_path / "out"
    pipeline = MusicTranscriptionPipeline(
        audio_preprocessor=FakeAudioPreprocessor(),
        source_separator=FakeSourceSeparator(),
        melody_transcriber=FakeMelodyTranscriber(),
    )

    result = pipeline.transcribe(input_path, output_dir)

    assert result.melody.source_kind == HUMAN_VOICE
    assert result.melody.source_label == "人声"
    assert "Main melody source: 人声" in result.jianpu_path.read_text(encoding="utf-8")
