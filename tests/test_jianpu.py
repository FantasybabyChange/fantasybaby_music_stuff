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
    assert "1 2 3" in score


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
