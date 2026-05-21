import subprocess
import sys

import pytest

from music_stuff.audio import AudioPreprocessor


def test_prepare_mp3_uses_ffmpeg_decoder(tmp_path, monkeypatch):
    audio_path = tmp_path / "tone.mp3"
    audio_path.write_bytes(b"fake mp3 bytes")
    raw_pcm = b"".join(
        sample.to_bytes(2, "little", signed=True)
        for sample in (0, 16384, -16384, 8192)
    )
    calls = []

    def fake_run(command, *, capture_output, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=raw_pcm, stderr=b"")

    monkeypatch.setattr("music_stuff.audio.subprocess.run", fake_run)

    prepared = AudioPreprocessor(ffmpeg_binary="ffmpeg-test").prepare(audio_path)

    assert prepared.sample_rate == 22050
    assert prepared.duration_seconds == len(prepared.samples) / 22050
    assert prepared.samples[1] == 0.5
    assert calls[0][0] == "ffmpeg-test"
    assert "-i" in calls[0]
    assert "-t" not in calls[0]


def test_prepare_flac_uses_ffmpeg_decoder(tmp_path, monkeypatch):
    audio_path = tmp_path / "tone.flac"
    audio_path.write_bytes(b"fake flac bytes")

    def fake_run(command, *, capture_output, check):
        return subprocess.CompletedProcess(command, 0, stdout=b"\x00\x00", stderr=b"")

    monkeypatch.setattr("music_stuff.audio.subprocess.run", fake_run)

    prepared = AudioPreprocessor(ffmpeg_binary="ffmpeg-test").prepare(audio_path)

    assert prepared.sample_rate == 22050
    assert prepared.samples == (0.0,)


def test_prepare_compressed_audio_can_still_be_limited_when_requested(tmp_path, monkeypatch):
    audio_path = tmp_path / "tone.mp3"
    audio_path.write_bytes(b"fake mp3 bytes")
    calls = []

    def fake_run(command, *, capture_output, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"\x00\x00", stderr=b"")

    monkeypatch.setattr("music_stuff.audio.subprocess.run", fake_run)

    AudioPreprocessor(ffmpeg_binary="ffmpeg-test", max_duration_seconds=12.5).prepare(audio_path)

    assert "-t" in calls[0]
    assert "12.5" in calls[0]


def test_prepare_mp3_reports_missing_ffmpeg(tmp_path, monkeypatch):
    audio_path = tmp_path / "tone.mp3"
    audio_path.write_bytes(b"fake mp3 bytes")
    monkeypatch.setattr("music_stuff.audio.shutil.which", lambda _name: None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)

    with pytest.raises(ValueError, match="requires ffmpeg"):
        AudioPreprocessor().prepare(audio_path)
