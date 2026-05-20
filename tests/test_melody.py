import math

from music_stuff.audio import PreparedAudio
from music_stuff.melody import AutoMelodyTranscriber, BasicPitchMelodyTranscriber, MixedAudioMelodyTranscriber
from music_stuff.models import Melody, NoteEvent


def test_mixed_audio_transcriber_prefers_prominent_upper_melody():
    sample_rate = 11025
    duration = 1.0
    samples = []
    for index in range(int(sample_rate * duration)):
        time = index / sample_rate
        bass = 0.18 * math.sin(2 * math.pi * 220.0 * time)
        melody = 0.75 * math.sin(2 * math.pi * 440.0 * time)
        samples.append(bass + melody)

    audio = PreparedAudio(
        path="synthetic_mix.wav",
        sample_rate=sample_rate,
        duration_seconds=duration,
        samples=tuple(samples),
    )

    result = MixedAudioMelodyTranscriber().transcribe(audio)

    assert result.notes
    assert max(note.pitch for note in result.notes) >= 68
    assert min(note.pitch for note in result.notes) >= 55


def test_basic_pitch_transcriber_converts_note_events():
    def fake_predict(_path):
        return None, None, [
            (0.0, 0.5, 60, 0.8, []),
            {"start_time_s": 0.5, "end_time_s": 1.0, "pitch_midi": 64, "amplitude": 0.5},
            (1.0, 1.02, 67, 0.7, []),
        ]

    audio = PreparedAudio(path="demo.wav", sample_rate=11025, duration_seconds=1.0)
    result = BasicPitchMelodyTranscriber(predict_func=fake_predict).transcribe(audio)

    assert [note.pitch for note in result.notes] == [60, 64]
    assert result.notes[0].velocity == 102


def test_basic_pitch_transcriber_selects_single_melody_line_from_overlap():
    def fake_predict(_path):
        return None, None, [
            (0.0, 1.0, 48, 0.9, []),
            (0.0, 1.0, 72, 0.6, []),
            (1.0, 2.0, 50, 0.9, []),
            (1.0, 2.0, 74, 0.6, []),
        ]

    audio = PreparedAudio(path="polyphonic_demo.wav", sample_rate=11025, duration_seconds=2.0)
    result = BasicPitchMelodyTranscriber(predict_func=fake_predict).transcribe(audio)

    assert [note.pitch for note in result.notes] == [72, 74]


def test_auto_transcriber_falls_back_when_basic_pitch_unavailable():
    class MissingBasicPitch(BasicPitchMelodyTranscriber):
        def is_available(self) -> bool:
            return False

    class FallbackTranscriber:
        def transcribe(self, audio):
            return Melody(notes=(NoteEvent(pitch=69, start=0.0, end=1.0),), source=str(audio.path))

    audio = PreparedAudio(path="demo.wav", sample_rate=11025, duration_seconds=1.0)
    result = AutoMelodyTranscriber(
        basic_pitch=MissingBasicPitch(),
        fallback=FallbackTranscriber(),
    ).transcribe(audio)

    assert result.notes[0].pitch == 69
