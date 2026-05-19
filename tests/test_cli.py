from music_stuff.cli import main


def test_plan_command_prints_pipeline_stages(capsys):
    exit_code = main(["plan"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "prepare audio" in output
    assert "infer core chords" in output


def test_transcribe_dry_run_does_not_require_audio_file(capsys):
    exit_code = main(["transcribe", "demo.wav", "--out", "output/demo", "--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "demo.wav" in output
    assert "score" in output.lower()


def test_ui_dry_run_prints_local_url(capsys):
    exit_code = main(["ui", "--port", "8123", "--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "http://127.0.0.1:8123" in output


def test_log_file_option_creates_parent_directory(tmp_path):
    log_path = tmp_path / "logs" / "music-stuff.log"

    exit_code = main(["--log-file", str(log_path), "ui", "--dry-run"])

    assert exit_code == 0
    assert log_path.exists()
