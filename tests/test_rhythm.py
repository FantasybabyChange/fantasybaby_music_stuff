import numpy as np

from music_stuff.audio import PreparedAudio
from music_stuff.models import Melody, NoteEvent, RhythmEstimate
from music_stuff.rhythm import RhythmQuantizer


def test_analyze_estimates_tempo_from_click_track():
    sample_rate = 22050
    tempo_bpm = 100.0
    beat_seconds = 60.0 / tempo_bpm
    duration_seconds = 8.0
    samples = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
    pulse = np.hanning(100).astype(np.float32)
    for beat_time in np.arange(0.3, duration_seconds, beat_seconds):
        start = int(beat_time * sample_rate)
        samples[start : start + len(pulse)] += pulse

    audio = PreparedAudio(
        path="click.wav",
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        samples=tuple(float(value) for value in samples),
    )

    rhythm = RhythmQuantizer().analyze(audio)

    assert 95.0 <= rhythm.tempo_bpm <= 105.0
    assert rhythm.beat_times
    assert rhythm.meter == "4/4"


def test_quantize_uses_real_tempo_and_beat_offset():
    melody = Melody(
        notes=(
            NoteEvent(pitch=60, start=0.13, end=0.72),
            NoteEvent(pitch=62, start=0.74, end=1.33),
        ),
        source="demo.wav",
    )
    rhythm = RhythmEstimate(
        tempo_bpm=100.0,
        beat_times=(0.1, 0.7, 1.3),
        beat_offset=0.1,
        meter="4/4",
    )

    quantized = RhythmQuantizer().quantize(melody, rhythm)

    assert quantized.notes[0].start == 0.1
    assert quantized.notes[0].end == 0.7
    assert quantized.notes[1].start == 0.7
    assert quantized.notes[1].end == 1.3


def test_normalize_tempo_halves_likely_double_time():
    assert RhythmQuantizer()._normalize_tempo_range(172.0) == 86.0
