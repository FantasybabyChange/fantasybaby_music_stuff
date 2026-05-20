from music_stuff.source import DemucsSourceSeparator


def test_demucs_separator_reports_unavailable_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr("music_stuff.source.importlib.util.find_spec", lambda _name: None)

    result = DemucsSourceSeparator().separate(tmp_path / "song.mp3", tmp_path / "stems")

    assert result.status == "unavailable"
    assert not result.stems
