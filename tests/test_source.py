from pathlib import Path
import subprocess

from music_stuff.source import DemucsSourceSeparator


def test_demucs_separator_defaults_to_high_quality_settings():
    separator = DemucsSourceSeparator()

    assert separator.model_name == "htdemucs_ft"
    assert separator.shifts == 4
    assert separator.overlap == 0.5
    assert separator.split is True


def test_demucs_separator_reports_unavailable_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr("music_stuff.source.importlib.util.find_spec", lambda _name: None)

    result = DemucsSourceSeparator().separate(tmp_path / "song.mp3", tmp_path / "stems")

    assert result.status == "unavailable"
    assert not result.stems


def test_demucs_subprocess_env_adds_imageio_ffmpeg_to_path(monkeypatch):
    monkeypatch.setattr("music_stuff.source.shutil.which", lambda _name: None)

    class FakeImageioFfmpeg:
        @staticmethod
        def get_ffmpeg_exe():
            return r"C:\tools\ffmpeg\ffmpeg.exe"

    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", FakeImageioFfmpeg)

    env = DemucsSourceSeparator()._subprocess_env()

    assert env["PATH"].startswith(r"C:\tools\ffmpeg")


def test_demucs_auto_device_prefers_cuda_when_available():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCuda()

    assert DemucsSourceSeparator()._resolve_torch_device(FakeTorch) == "cuda"
    assert DemucsSourceSeparator(device="cpu")._resolve_torch_device(FakeTorch) == "cpu"


def test_demucs_prepare_input_uses_timeout(tmp_path, monkeypatch):
    audio_path = tmp_path / "song.mp3"
    audio_path.write_bytes(b"demo")
    calls = []

    def fake_run(command, *, capture_output, text, check, timeout):
        calls.append((command, timeout))
        output_path = Path(command[-1])
        output_path.write_bytes(b"wav")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    separator = DemucsSourceSeparator(timeout_seconds=123)
    monkeypatch.setattr(separator, "_resolve_ffmpeg_binary", lambda: "ffmpeg-test")
    monkeypatch.setattr("music_stuff.source.subprocess.run", fake_run)

    prepared = separator._prepare_demucs_input(audio_path, tmp_path / "stems")

    assert prepared.name == "demucs_input.wav"
    assert calls[0][1] == 123
