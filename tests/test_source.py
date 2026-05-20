from music_stuff.source import DemucsSourceSeparator


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
