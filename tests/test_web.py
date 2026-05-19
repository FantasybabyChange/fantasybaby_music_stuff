from music_stuff.web import _safe_filename, render_page


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


def test_safe_filename_preserves_supported_suffix():
    assert _safe_filename("我的 旋律.mp3") == "audio.mp3"
    assert _safe_filename("../demo tune.flac") == "demo_tune.flac"
