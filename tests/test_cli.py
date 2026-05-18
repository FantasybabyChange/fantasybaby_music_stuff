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
