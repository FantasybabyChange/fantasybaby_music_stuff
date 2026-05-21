import math
import wave

from music_stuff.cli import main
from music_stuff.models import AnalysisResult, KeyEstimate, Melody, NoteEvent
from music_stuff.score import format_jianpu


def test_format_jianpu_uses_numbered_scale_degrees():
    melody = Melody(
        notes=(
            NoteEvent(pitch=60, start=0.0, end=0.5),
            NoteEvent(pitch=62, start=0.5, end=1.0),
            NoteEvent(pitch=64, start=1.0, end=1.5),
        ),
        source="demo.wav",
    )
    analysis = AnalysisResult(key=KeyEstimate("C", "major"), tempo_bpm=120.0)

    score = format_jianpu(melody, analysis)

    assert "Key: 1=C" in score
    assert "Legend:" in score
    assert "标准简谱：" in score
    assert "| 1 2 3" in score


def test_format_jianpu_uses_readable_duration_marks():
    melody = Melody(
        notes=(
            NoteEvent(pitch=60, start=0.0, end=0.375),
            NoteEvent(pitch=62, start=0.375, end=1.0),
            NoteEvent(pitch=64, start=1.0, end=2.125),
        ),
        source="demo.wav",
    )
    analysis = AnalysisResult(key=KeyEstimate("C", "major"), tempo_bpm=120.0)

    score = format_jianpu(melody, analysis)

    assert "(0.75)" not in score
    assert "(1.25)" not in score
    detailed = score.split("标准简谱：", 1)[1]
    assert "/" not in detailed
    assert "\u0332." in score
    assert "-" in score


def test_format_jianpu_uses_standard_measure_bars_and_holds():
    melody = Melody(
        notes=(
            NoteEvent(pitch=60, start=0.0, end=2.0),
            NoteEvent(pitch=72, start=2.0, end=2.25),
            NoteEvent(pitch=48, start=2.25, end=2.5),
        ),
        source="demo.wav",
    )
    analysis = AnalysisResult(key=KeyEstimate("C", "major"), tempo_bpm=120.0)

    score = format_jianpu(melody, analysis)

    assert "| 1 - - - |" in score
    assert "1\u0307\u0332" in score
    assert "1\u0323\u0332" in score
    assert score.rstrip().endswith("||")


def test_format_jianpu_trims_long_leading_silence_from_display():
    melody = Melody(
        notes=(
            NoteEvent(pitch=60, start=37.0, end=37.5),
            NoteEvent(pitch=62, start=37.5, end=38.0),
        ),
        source="vocal.wav",
    )
    analysis = AnalysisResult(key=KeyEstimate("C", "major"), tempo_bpm=120.0)

    score = format_jianpu(melody, analysis)

    assert "Melody entry: 37.00s" in score
    detailed = score.split("标准简谱：", 1)[1].strip()
    assert detailed.startswith("| 1")
    assert not detailed.startswith("| 0")


def test_format_jianpu_trims_sparse_vocal_false_starts():
    melody = Melody(
        notes=(
            NoteEvent(pitch=54, start=4.5, end=4.7),
            NoteEvent(pitch=54, start=8.1, end=8.3),
            NoteEvent(pitch=63, start=24.8, end=25.1),
            NoteEvent(pitch=56, start=25.5, end=25.8),
            NoteEvent(pitch=55, start=25.9, end=26.2),
            NoteEvent(pitch=46, start=37.5, end=37.8),
            NoteEvent(pitch=55, start=37.9, end=38.1),
            NoteEvent(pitch=53, start=38.1, end=38.6),
            NoteEvent(pitch=51, start=38.6, end=39.0),
            NoteEvent(pitch=49, start=39.4, end=39.6),
            NoteEvent(pitch=50, start=39.6, end=40.0),
            NoteEvent(pitch=48, start=40.1, end=40.5),
            NoteEvent(pitch=51, start=40.5, end=41.0),
        ),
        source="vocals.wav",
        source_kind="human_voice",
        source_label="人声",
    )
    analysis = AnalysisResult(key=KeyEstimate("C", "major"), tempo_bpm=86.0)

    score = format_jianpu(melody, analysis)

    assert "Melody entry: 37.50s" in score
    detailed = score.split("标准简谱：", 1)[1].strip()
    assert not detailed.startswith("| 0")


def test_transcribe_wav_writes_jianpu(tmp_path):
    audio_path = tmp_path / "tone.wav"
    output_dir = tmp_path / "out"
    _write_sine_wave(audio_path, frequency=440.0, duration=0.5)

    exit_code = main(["transcribe", str(audio_path), "--out", str(output_dir)])

    assert exit_code == 0
    jianpu_path = output_dir / "melody.jianpu.txt"
    assert jianpu_path.exists()
    assert "Key:" in jianpu_path.read_text(encoding="utf-8")


def _write_sine_wave(path, *, frequency: float, duration: float) -> None:
    sample_rate = 8000
    amplitude = 12000
    total_samples = int(sample_rate * duration)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for index in range(total_samples):
            sample = int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(sample.to_bytes(2, "little", signed=True))
        wav_file.writeframes(bytes(frames))
