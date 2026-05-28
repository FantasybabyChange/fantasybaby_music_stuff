"""Small local web UI for melody extraction."""

from __future__ import annotations

__all__ = ["UIConfig", "UploadedAudio", "RunSummary", "run_ui", "render_page"]

import cgi
from dataclasses import dataclass
import functools
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import logging
import mimetypes
from pathlib import Path
import re
from string import Template
from urllib.parse import unquote, urlparse
import uuid
import webbrowser

from music_stuff.models import AnalysisResult, Melody, NoteEvent
from music_stuff.pipeline import MusicTranscriptionPipeline
from music_stuff.source import (
    COMPUTE_MODE_AUTO,
    COMPUTE_MODE_BALANCED,
    COMPUTE_MODE_CPU,
    COMPUTE_MODE_GPU,
    DEFAULT_COMPUTE_MODE,
    build_demucs_separator,
    normalize_compute_mode,
)


SUPPORTED_UPLOAD_SUFFIXES = {".wav", ".mp3", ".flac"}
PLAYER_ARTIFACT_NAME = "melody.player.json"
DOWNLOADABLE_ARTIFACTS = {"melody.jianpu.txt", "analysis.json", PLAYER_ARTIFACT_NAME}
LOGGER = logging.getLogger(__name__)
ASSETS_DIR = Path(__file__).with_name("assets")
FANTASYBABY_LOGO_NAME = "fantasybaby-logo.jpg"
FANTASYBABY_LOGO_HREF = f"/assets/{FANTASYBABY_LOGO_NAME}"
FANTASYBABY_VIDEO_NAME = "fantasybaby-brand-video.mp4"
FANTASYBABY_VIDEO_HREF = f"/assets/{FANTASYBABY_VIDEO_NAME}"
STATIC_ASSETS = {
    FANTASYBABY_LOGO_NAME: ASSETS_DIR / FANTASYBABY_LOGO_NAME,
    FANTASYBABY_VIDEO_NAME: ASSETS_DIR / FANTASYBABY_VIDEO_NAME,
}
COMPUTE_MODE_OPTIONS = (
    (COMPUTE_MODE_BALANCED, "均衡", "GPU中负载 + CPU分析"),
    (COMPUTE_MODE_GPU, "GPU", "高质量"),
    (COMPUTE_MODE_CPU, "CPU", "不占GPU"),
    (COMPUTE_MODE_AUTO, "自动", "按设备选择"),
)
_TEMPLATES_DIR = Path(__file__).with_name("templates")


@functools.lru_cache(maxsize=1)
def _load_index_template() -> Template:
    """Load the HTML template from the templates directory, caching it in memory."""
    return Template((_TEMPLATES_DIR / "index.html").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class UIConfig:
    host: str = "127.0.0.1"
    port: int = 8767
    output_dir: Path = Path("output/ui")
    open_browser: bool = False


@dataclass(frozen=True)
class UploadedAudio:
    original_name: str
    safe_name: str
    content: bytes
    compute_mode: str


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    label: str
    updated_at: str
    note_count: int | None = None
    duration_seconds: float | None = None
    can_play: bool = False


def run_ui(config: UIConfig) -> None:
    """Start the local transcription UI."""
    handler = _build_handler(config.output_dir)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    url = f"http://{config.host}:{server.server_port}"
    LOGGER.info("Starting UI server: url=%s output_dir=%s", url, config.output_dir.resolve())
    print(f"Music Stuff UI running at {url}")
    print("Press Ctrl+C to stop.")
    if config.open_browser:
        webbrowser.open(url)
    server.serve_forever()


def render_page(
    *,
    result_text: str | None = None,
    message: str | None = None,
    artifact_path: Path | None = None,
    file_name: str | None = None,
    compute_mode: str | None = None,
    jianpu_href: str | None = None,
    analysis_href: str | None = None,
    player_payload: dict[str, object] | None = None,
    history_runs: tuple[RunSummary, ...] = (),
) -> str:
    result_html = _render_result(
        result_text=result_text,
        artifact_path=artifact_path,
        file_name=file_name,
        jianpu_href=jianpu_href,
        analysis_href=analysis_href,
        player_payload=player_payload,
    )
    history_html = _render_history(history_runs)
    message_html = f'<p class="message">{html.escape(message)}</p>' if message else ""
    compute_mode_html = _render_compute_mode_options(normalize_compute_mode(compute_mode))
    template = _load_index_template()
    return template.safe_substitute(
        logo_href=html.escape(FANTASYBABY_LOGO_HREF),
        video_href=html.escape(FANTASYBABY_VIDEO_HREF),
        compute_mode_html=compute_mode_html,
        message_html=message_html,
        history_html=history_html,
        result_html=result_html,
    )


def _build_handler(output_dir: Path) -> type[BaseHTTPRequestHandler]:
    output_root = output_dir.resolve()

    class MusicStuffHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            LOGGER.info("HTTP GET %s", parsed.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_html(render_page(history_runs=_list_run_summaries(output_root)))
                return
            if parsed.path.startswith("/runs/"):
                self._serve_run_page(parsed.path)
                return
            if parsed.path.startswith("/artifacts/"):
                self._serve_artifact(parsed.path)
                return
            if parsed.path.startswith("/assets/"):
                self._serve_asset(parsed.path)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            LOGGER.info("HTTP POST %s", parsed.path)
            if parsed.path != "/transcribe":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            selected_compute_mode = DEFAULT_COMPUTE_MODE
            try:
                upload = self._read_upload()
                selected_compute_mode = upload.compute_mode
                _ensure_compute_mode_available(upload.compute_mode)
                run_id = uuid.uuid4().hex[:12]
                run_dir = output_root / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                input_path = run_dir / upload.safe_name
                input_path.write_bytes(upload.content)
                LOGGER.info(
                    "Upload received: run_id=%s file=%s bytes=%s saved_as=%s compute_mode=%s",
                    run_id,
                    upload.original_name,
                    len(upload.content),
                    input_path,
                    upload.compute_mode,
                )

                pipeline = MusicTranscriptionPipeline(
                    source_separator=build_demucs_separator(upload.compute_mode)
                )
                result = pipeline.transcribe(input_path, run_dir)
                result_text = result.jianpu_path.read_text(encoding="utf-8") if result.jianpu_path else ""
                player_payload = _melody_player_payload(result.melody, result.analysis)
                (run_dir / PLAYER_ARTIFACT_NAME).write_text(
                    json.dumps(player_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self._send_html(
                    render_page(
                        result_text=result_text,
                        artifact_path=result.jianpu_path,
                        file_name=upload.original_name,
                        compute_mode=upload.compute_mode,
                        jianpu_href=_artifact_href(run_id, "melody.jianpu.txt"),
                        analysis_href=_artifact_href(run_id, "analysis.json"),
                        player_payload=player_payload,
                        history_runs=_list_run_summaries(output_root),
                    )
                )
                LOGGER.info("Upload transcription rendered: run_id=%s", run_id)
            except Exception as exc:
                LOGGER.exception("Upload transcription failed")
                self._send_html(
                    render_page(
                        message=str(exc),
                        compute_mode=selected_compute_mode,
                        history_runs=_list_run_summaries(output_root),
                    ),
                    status=HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _serve_run_page(self, request_path: str) -> None:
            parts = [unquote(part) for part in request_path.split("/") if part]
            if len(parts) != 2 or parts[0] != "runs":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            run_id = parts[1]
            if not re.fullmatch(r"[a-f0-9]{12}", run_id):
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            run_dir = (output_root / run_id).resolve()
            jianpu_path = run_dir / "melody.jianpu.txt"
            analysis_path = run_dir / "analysis.json"
            player_path = run_dir / PLAYER_ARTIFACT_NAME
            if not run_dir.is_relative_to(output_root) or not jianpu_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            result_text = jianpu_path.read_text(encoding="utf-8")
            player_payload = None
            if player_path.exists():
                player_payload = _loaded_player_payload(
                    json.loads(player_path.read_text(encoding="utf-8-sig")),
                    analysis_path,
                )
            LOGGER.info("Serving run page: run_id=%s", run_id)
            self._send_html(
                render_page(
                    result_text=result_text,
                    artifact_path=jianpu_path,
                    file_name=run_id,
                    jianpu_href=_artifact_href(run_id, "melody.jianpu.txt"),
                    analysis_href=_artifact_href(run_id, "analysis.json") if analysis_path.exists() else None,
                    player_payload=player_payload,
                    history_runs=_list_run_summaries(output_root),
                )
            )

        def _serve_artifact(self, request_path: str) -> None:
            parts = [unquote(part) for part in request_path.split("/") if part]
            if len(parts) != 3 or parts[0] != "artifacts":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            run_id, file_name = parts[1], parts[2]
            if not re.fullmatch(r"[a-f0-9]{12}", run_id):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if file_name not in DOWNLOADABLE_ARTIFACTS:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            target = (output_root / run_id / file_name).resolve()
            if not target.is_relative_to(output_root) or not target.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            content = target.read_bytes()
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            LOGGER.info("Serving artifact: %s bytes=%s", target, len(content))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
            self.end_headers()
            self.wfile.write(content)

        def _serve_asset(self, request_path: str) -> None:
            parts = [unquote(part) for part in request_path.split("/") if part]
            if len(parts) != 2 or parts[0] != "assets":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            asset_path = STATIC_ASSETS.get(parts[1])
            if asset_path is None or not asset_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            content = asset_path.read_bytes()
            content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
            LOGGER.info("Serving asset: %s bytes=%s", asset_path, len(content))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(content)

        def _read_upload(self) -> UploadedAudio:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            field = form["audio"] if "audio" in form else None
            if field is None or not field.filename:
                raise ValueError("Please choose an audio file.")
            compute_mode = normalize_compute_mode(_form_value(form, "compute_mode"))

            original_name = Path(field.filename).name
            suffix = Path(original_name).suffix.lower()
            if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
                raise ValueError("Only WAV, MP3, and FLAC audio are supported.")

            content = field.file.read()
            if not content:
                raise ValueError("Audio file is empty.")
            return UploadedAudio(
                original_name=original_name,
                safe_name=_safe_filename(original_name),
                content=content,
                compute_mode=compute_mode,
            )

    return MusicStuffHandler


def _render_compute_mode_options(selected_mode: str) -> str:
    options: list[str] = []
    for mode, title, meta in COMPUTE_MODE_OPTIONS:
        checked = " checked" if mode == selected_mode else ""
        options.append(
            f"""<label class="mode-option" for="compute-mode-{html.escape(mode)}">
              <input id="compute-mode-{html.escape(mode)}" type="radio" name="compute_mode" value="{html.escape(mode)}"{checked}>
              <span class="mode-card">
                <span class="mode-title">{html.escape(title)}</span>
                <span class="mode-meta">{html.escape(meta)}</span>
              </span>
            </label>"""
        )
    return f"""<fieldset class="compute-mode" aria-label="compute mode">
          <legend>&#35745;&#31639;&#27169;&#24335;</legend>
          <div class="mode-grid">{"".join(options)}</div>
        </fieldset>"""


def _render_history(history_runs: tuple[RunSummary, ...]) -> str:
    if not history_runs:
        return """<section class="history" aria-label="history">
          <h2>&#21382;&#21490;&#31616;&#35889;</h2>
          <p class="history-empty">&#37325;&#21551;&#21518;&#65292;&#24050;&#29983;&#25104;&#30340;&#31616;&#35889;&#20250;&#20986;&#29616;&#22312;&#36825;&#37324;&#12290;</p>
        </section>"""

    items: list[str] = []
    for run in history_runs:
        bits = [run.updated_at]
        if run.note_count is not None:
            bits.append(f"{run.note_count} notes")
        if run.duration_seconds is not None:
            bits.append(f"{run.duration_seconds:.1f}s")
        if run.can_play:
            bits.append("playable")
        meta = " · ".join(bits)
        items.append(
            f"""<li class="history-item">
              <a href="/runs/{html.escape(run.run_id)}" title="{html.escape(run.label)}">
                <span class="history-title">{html.escape(run.label)}</span>
                <span class="history-meta">{html.escape(meta)}</span>
              </a>
            </li>"""
        )

    return f"""<section class="history" aria-label="history">
      <h2>&#21382;&#21490;&#31616;&#35889;</h2>
      <ul class="history-list">{"".join(items)}</ul>
    </section>"""


def _list_run_summaries(output_root: Path, *, limit: int = 12) -> tuple[RunSummary, ...]:
    if not output_root.exists():
        return ()

    runs: list[tuple[float, RunSummary]] = []
    for run_dir in output_root.iterdir():
        if not run_dir.is_dir() or not re.fullmatch(r"[a-f0-9]{12}", run_dir.name):
            continue

        jianpu_path = run_dir / "melody.jianpu.txt"
        if not jianpu_path.exists():
            continue

        player_path = run_dir / PLAYER_ARTIFACT_NAME
        updated_timestamp = jianpu_path.stat().st_mtime
        payload = _read_player_payload(player_path)
        runs.append(
            (
                updated_timestamp,
                RunSummary(
                    run_id=run_dir.name,
                    label=_run_label(run_dir, payload),
                    updated_at=_format_timestamp(updated_timestamp),
                    note_count=_optional_int(payload.get("noteCount")) if payload else None,
                    duration_seconds=_optional_float(payload.get("durationSeconds")) if payload else None,
                    can_play=bool(payload and payload.get("notes")),
                ),
            )
        )

    runs.sort(key=lambda item: item[0], reverse=True)
    return tuple(run for _timestamp, run in runs[:limit])


def _read_player_payload(player_path: Path) -> dict[str, object] | None:
    if not player_path.exists():
        return None
    try:
        payload = json.loads(player_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _run_label(run_dir: Path, payload: dict[str, object] | None) -> str:
    for child in sorted(run_dir.iterdir(), key=lambda item: item.stat().st_mtime):
        if child.is_file() and child.suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES:
            return child.name

    source = payload.get("source") if payload else None
    if isinstance(source, str) and source:
        return Path(source).name or run_dir.name
    return run_dir.name


def _format_timestamp(timestamp: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_result(
    *,
    result_text: str | None,
    artifact_path: Path | None,
    file_name: str | None,
    jianpu_href: str | None = None,
    analysis_href: str | None = None,
    player_payload: dict[str, object] | None = None,
) -> str:
    if not result_text:
        return """<div class="result-head">
          <div><h2>&#31616;&#35889;&#32467;&#26524;</h2><p>&#22788;&#29702;&#23436;&#25104;&#21518;&#20250;&#26174;&#31034;&#22312;&#36825;&#37324;&#12290;</p></div>
        </div>
        <div class="empty"><div><strong>&#31561;&#24453;&#38899;&#39057;</strong><span>&#36873;&#25321;&#25991;&#20214;&#24182;&#24320;&#22987;&#25552;&#21462;&#12290;</span></div></div>"""

    source = html.escape(file_name or "uploaded audio")
    actions = ['<button type="button" data-copy-target="#jianpu-result">&#22797;&#21046;&#31616;&#35889;</button>']
    if jianpu_href:
        actions.append(f'<a href="{html.escape(jianpu_href)}" download>&#19979;&#36733;&#31616;&#35889;</a>')
    if analysis_href:
        actions.append(f'<a href="{html.escape(analysis_href)}" download>&#19979;&#36733;&#20998;&#26512;</a>')
    if artifact_path:
        actions.append(f'<span class="artifact">{html.escape(str(artifact_path))}</span>')
    player_html = _render_player(player_payload)

    return f"""<div class="result-head">
      <div><h2>&#31616;&#35889;&#32467;&#26524;</h2><p>{source}</p></div>
      <div class="actions">{"".join(actions)}</div>
    </div>
    {player_html}
    <pre id="jianpu-result">{html.escape(result_text)}</pre>"""


def _render_player(player_payload: dict[str, object] | None) -> str:
    if not player_payload:
        return ""
    encoded_payload = html.escape(json.dumps(player_payload, ensure_ascii=False), quote=False)
    note_count = int(player_payload.get("noteCount", 0))
    duration = float(player_payload.get("durationSeconds", 0.0))
    tempo_bpm = player_payload.get("tempoBpm")
    tempo_text = f"&#65292;{float(tempo_bpm):.1f} bpm" if tempo_bpm else ""
    return f"""<section class="player" aria-label="Jianpu melody player">
      <div class="player-meta">
        <p class="player-title">&#31616;&#35889;&#26059;&#24459;&#25773;&#25918;&#22120;</p>
        <p id="melody-player-status" class="player-status">{note_count} &#20010;&#38899;&#65292;{duration:.1f}s{tempo_text}</p>
      </div>
      <div class="player-controls">
        <button id="melody-play" type="button" title="&#25773;&#25918;" aria-label="&#25773;&#25918;">&#9654;</button>
        <button id="melody-restart" type="button" title="&#37325;&#22836;&#25773;&#25918;" aria-label="&#37325;&#22836;&#25773;&#25918;">&#8635;</button>
        <button id="melody-stop" type="button" title="&#20572;&#27490;" aria-label="&#20572;&#27490;">&#9632;</button>
      </div>
      <script id="melody-player-data" type="application/json">{encoded_payload}</script>
    </section>"""


def _melody_player_payload(melody: Melody, analysis: AnalysisResult | None = None) -> dict[str, object]:
    tempo_bpm = _player_tempo_bpm(analysis)
    notes = _score_time_player_notes(melody, tempo_bpm)
    duration = max((float(note["end"]) for note in notes), default=0.0)
    return {
        "source": melody.source,
        "sourceKind": melody.source_kind,
        "sourceLabel": melody.source_label,
        "sourceConfidence": melody.source_confidence,
        "tempoBpm": tempo_bpm,
        "noteCount": len(notes),
        "durationSeconds": round(duration, 4),
        "notes": notes,
    }


def _loaded_player_payload(payload: dict[str, object], analysis_path: Path) -> dict[str, object]:
    if payload.get("tempoBpm") is not None:
        return payload
    notes = payload.get("notes")
    if not isinstance(notes, list):
        return payload

    melody_notes: list[NoteEvent] = []
    for item in notes:
        if not isinstance(item, dict):
            continue
        try:
            melody_notes.append(
                NoteEvent(
                    pitch=int(item["pitch"]),
                    start=float(item["start"]),
                    end=float(item["end"]),
                    velocity=int(item.get("velocity", 80)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    melody = Melody(
        notes=tuple(melody_notes),
        source=str(payload.get("source", "")),
        source_kind=str(payload.get("sourceKind", "mixed")),
        source_label=str(payload.get("sourceLabel", "")),
        source_confidence=payload.get("sourceConfidence") if isinstance(payload.get("sourceConfidence"), float) else None,
    )
    return _melody_player_payload(melody, _analysis_from_json(analysis_path))


def _analysis_from_json(path: Path) -> AnalysisResult | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return AnalysisResult(tempo_bpm=data.get("tempo_bpm"))


def _player_tempo_bpm(analysis: AnalysisResult | None) -> float | None:
    if analysis is None or analysis.tempo_bpm is None:
        return None
    try:
        tempo_bpm = float(analysis.tempo_bpm)
    except (TypeError, ValueError):
        return None
    if tempo_bpm <= 0:
        return None
    return tempo_bpm


def _score_time_player_notes(melody: Melody, tempo_bpm: float | None) -> list[dict[str, float | int]]:
    source_notes = [
        note
        for note in sorted(melody.notes, key=lambda item: (item.start, item.pitch))
        if note.pitch >= 0 and note.end > note.start
    ]
    if not source_notes:
        return []

    beat_seconds = 60.0 / tempo_bpm if tempo_bpm else None
    grid_seconds = beat_seconds / 4.0 if beat_seconds else None
    rest_threshold = beat_seconds or 0.5
    cursor_source = source_notes[0].start
    cursor_play = 0.0
    player_notes: list[dict[str, float | int]] = []

    for note in source_notes:
        gap = note.start - cursor_source
        if gap >= rest_threshold:
            cursor_play += _round_to_grid(gap, grid_seconds)

        duration = _round_to_grid(note.end - note.start, grid_seconds)
        duration = max(grid_seconds or 0.08, duration)
        start = cursor_play
        end = start + duration
        player_notes.append(
            {
                "pitch": note.pitch,
                "start": round(start, 4),
                "end": round(end, 4),
                "velocity": note.velocity,
            }
        )
        cursor_play = end
        cursor_source = max(cursor_source, note.end)
    return player_notes


def _round_to_grid(seconds: float, grid_seconds: float | None) -> float:
    if grid_seconds is None or grid_seconds <= 0:
        return max(0.05, seconds)
    grid_count = max(1, round(seconds / grid_seconds))
    return grid_count * grid_seconds


def _safe_filename(file_name: str) -> str:
    name = Path(file_name).name
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    if not safe_stem:
        safe_stem = "audio"
    return f"{safe_stem}{suffix}"


def _form_value(form: cgi.FieldStorage, name: str) -> str | None:
    if name not in form:
        return None
    field = form[name]
    if isinstance(field, list):
        field = field[0] if field else None
    if field is None:
        return None
    value = getattr(field, "value", None)
    return str(value) if value is not None else None


def _ensure_compute_mode_available(compute_mode: str) -> None:
    if normalize_compute_mode(compute_mode) != COMPUTE_MODE_GPU:
        return
    if importlib.util.find_spec("torch") is None:
        raise ValueError("GPU 模式需要安装带 CUDA 支持的 PyTorch。")

    import torch

    if not torch.cuda.is_available():
        raise ValueError("已选择 GPU 模式，但 PyTorch 当前看不到可用 CUDA 显卡。")


def _artifact_href(run_id: str, file_name: str) -> str:
    return f"/artifacts/{run_id}/{file_name}"
