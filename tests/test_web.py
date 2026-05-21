from music_stuff.models import AnalysisResult, Melody, NoteEvent
from music_stuff.web import _loaded_player_payload, _melody_player_payload, _safe_filename, render_page


def test_render_page_contains_upload_workbench():
    page = render_page()

    assert "Music Stuff" in page
    assert "audio-file" in page
    assert "upload-form" in page
    assert "WAV" in page
    assert "MP3" in page
    assert "FLAC" in page
    assert "preview" in page


def test_render_page_escapes_result_text():
    page = render_page(result_text="<script>alert(1)</script>", file_name="demo.mp3")

    assert "&lt;script&gt;" in page
    assert "<script>alert(1)</script>" not in page
    assert "data-copy-target" in page


def test_render_page_includes_download_links():
    page = render_page(
        result_text="1 2 3",
        file_name="demo.mp3",
        jianpu_href="/artifacts/abc123abc123/melody.jianpu.txt",
        analysis_href="/artifacts/abc123abc123/analysis.json",
    )

    assert "/artifacts/abc123abc123/melody.jianpu.txt" in page
    assert "/artifacts/abc123abc123/analysis.json" in page


def test_render_page_includes_autoplay_melody_player():
    payload = {
        "source": "demo.wav",
        "sourceKind": "mixed",
        "sourceLabel": "mixed",
        "sourceConfidence": None,
        "noteCount": 2,
        "durationSeconds": 1.0,
        "notes": [
            {"pitch": 60, "start": 0.0, "end": 0.5, "velocity": 80},
            {"pitch": 62, "start": 0.5, "end": 1.0, "velocity": 80},
        ],
    }

    page = render_page(result_text="1 2", file_name="demo.wav", player_payload=payload)

    assert "melody-player-data" in page
    assert "melody-play" in page
    assert "playMelody(true)" in page
    assert "createBiquadFilter" in page
    assert "createDynamicsCompressor" in page
    assert '"triangle"' in page
    assert '"pitch": 60' in page


def test_melody_player_payload_uses_sorted_note_events():
    melody = Melody(
        notes=(
            NoteEvent(pitch=64, start=0.45, end=1.0),
            NoteEvent(pitch=60, start=0.0, end=0.51),
        ),
        source="demo.wav",
    )

    payload = _melody_player_payload(melody, AnalysisResult(tempo_bpm=120.0))

    assert payload["noteCount"] == 2
    assert payload["durationSeconds"] == 1.0
    assert payload["tempoBpm"] == 120.0
    assert payload["notes"][0]["pitch"] == 60
    assert payload["notes"][1]["pitch"] == 64
    assert payload["notes"][0]["start"] == 0.0
    assert payload["notes"][0]["end"] == 0.5
    assert payload["notes"][1]["start"] == 0.5


def test_melody_player_payload_preserves_only_clear_rests():
    melody = Melody(
        notes=(
            NoteEvent(pitch=60, start=0.0, end=0.5),
            NoteEvent(pitch=62, start=0.62, end=1.0),
            NoteEvent(pitch=64, start=2.0, end=2.5),
        ),
        source="demo.wav",
    )

    payload = _melody_player_payload(melody, AnalysisResult(tempo_bpm=120.0))

    assert payload["notes"][1]["start"] == 0.5
    assert payload["notes"][2]["start"] == 1.875


def test_loaded_legacy_player_payload_is_retimed_from_analysis(tmp_path):
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text('{"tempo_bpm": 120.0}', encoding="utf-8")
    payload = {
        "source": "demo.wav",
        "sourceKind": "mixed",
        "sourceLabel": "mixed",
        "sourceConfidence": None,
        "noteCount": 2,
        "durationSeconds": 1.0,
        "notes": [
            {"pitch": 60, "start": 0.0, "end": 0.51, "velocity": 80},
            {"pitch": 64, "start": 0.45, "end": 1.0, "velocity": 80},
        ],
    }

    retimed = _loaded_player_payload(payload, analysis_path)

    assert retimed["tempoBpm"] == 120.0
    assert retimed["notes"][0]["end"] == 0.5
    assert retimed["notes"][1]["start"] == 0.5


def test_safe_filename_preserves_supported_suffix():
    assert _safe_filename("我的 旋律.mp3") == "audio.mp3"
    assert _safe_filename("../demo tune.flac") == "demo_tune.flac"
