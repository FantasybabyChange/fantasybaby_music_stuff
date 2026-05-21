"""Small local web UI for melody extraction."""

from __future__ import annotations

import cgi
from dataclasses import dataclass
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlparse
import uuid
import webbrowser

from music_stuff.models import AnalysisResult, Melody, NoteEvent
from music_stuff.pipeline import MusicTranscriptionPipeline


SUPPORTED_UPLOAD_SUFFIXES = {".wav", ".mp3", ".flac"}
PLAYER_ARTIFACT_NAME = "melody.player.json"
DOWNLOADABLE_ARTIFACTS = {"melody.jianpu.txt", "analysis.json", PLAYER_ARTIFACT_NAME}
LOGGER = logging.getLogger(__name__)


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
    jianpu_href: str | None = None,
    analysis_href: str | None = None,
    player_payload: dict[str, object] | None = None,
) -> str:
    result_html = _render_result(
        result_text=result_text,
        artifact_path=artifact_path,
        file_name=file_name,
        jianpu_href=jianpu_href,
        analysis_href=analysis_href,
        player_payload=player_payload,
    )
    message_html = f'<p class="message">{html.escape(message)}</p>' if message else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Music Stuff &#31616;&#35889;&#25552;&#21462;</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17201a;
      --muted: #5d6a62;
      --line: #d9ded9;
      --panel: #ffffff;
      --accent: #1f7a63;
      --accent-dark: #145743;
      --soft: #eef5f1;
      font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: linear-gradient(180deg, #f7f5ef 0%, #eef3ef 100%);
    }}
    .shell {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }}
    header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0;
      font-size: 32px;
      line-height: 1.1;
      font-weight: 760;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .format-list {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .format-list span {{
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.75);
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    main {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
      align-items: stretch;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 16px 40px rgba(35, 45, 38, 0.08);
    }}
    .upload {{
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .dropzone {{
      position: relative;
      display: grid;
      place-items: center;
      min-height: 230px;
      border: 1px dashed #aeb8af;
      border-radius: 8px;
      background:
        linear-gradient(90deg, rgba(31, 122, 99, 0.09) 1px, transparent 1px) 0 0 / 28px 28px,
        linear-gradient(180deg, rgba(185, 131, 36, 0.08), rgba(255, 255, 255, 0.82));
      text-align: center;
      overflow: hidden;
    }}
    .dropzone::before {{
      content: "";
      position: absolute;
      inset: auto 22px 42px;
      height: 64px;
      background:
        repeating-linear-gradient(0deg, transparent 0 11px, rgba(23, 32, 26, 0.18) 11px 12px),
        linear-gradient(90deg, transparent, rgba(31, 122, 99, 0.16), transparent);
      border-radius: 8px;
      opacity: 0.9;
    }}
    .dropzone input {{
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
    }}
    .dropcopy {{
      position: relative;
      z-index: 1;
      padding: 20px;
    }}
    .dropcopy strong {{
      display: block;
      font-size: 20px;
      margin-bottom: 8px;
    }}
    .dropcopy span {{
      color: var(--muted);
      font-size: 14px;
    }}
    audio {{
      width: 100%;
      height: 40px;
      display: none;
    }}
    audio.is-visible {{
      display: block;
    }}
    .selected {{
      min-height: 20px;
      color: var(--accent-dark);
      font-size: 14px;
      word-break: break-word;
    }}
    button {{
      width: 100%;
      height: 44px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      font-size: 15px;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-dark); }}
    button:disabled {{
      background: #8aa79e;
      cursor: wait;
    }}
    .message {{
      margin: 0;
      padding: 12px 14px;
      border-radius: 8px;
      background: #fff6df;
      color: #6d4a09;
      border: 1px solid #ead7a8;
      font-size: 14px;
    }}
    .result {{
      display: flex;
      flex-direction: column;
      min-height: 560px;
      overflow: hidden;
    }}
    .result-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--soft);
    }}
    .result-head h2 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
    }}
    .result-head p {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .actions button,
    .actions a {{
      width: auto;
      min-height: 36px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 8px;
      padding: 8px 11px;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--accent-dark);
      font-size: 13px;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
    }}
    .actions button:hover,
    .actions a:hover {{
      border-color: #9ab6ad;
      background: #f7fbf8;
    }}
    .player {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #fffaf0;
    }}
    .player-meta {{
      min-width: 0;
    }}
    .player-title {{
      margin: 0;
      font-size: 14px;
      font-weight: 760;
    }}
    .player-status {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .player-controls {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .player-controls button {{
      width: 40px;
      height: 36px;
      display: inline-grid;
      place-items: center;
      padding: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--accent-dark);
      font-size: 15px;
      font-weight: 800;
    }}
    .player-controls button:hover {{
      border-color: #9ab6ad;
      background: #f7fbf8;
    }}
    .artifact {{
      color: var(--muted);
      font-size: 12px;
      max-width: 260px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    pre {{
      flex: 1;
      margin: 0;
      padding: 24px;
      overflow: auto;
      background: #fffdf8;
      color: #1e261f;
      font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 17px;
      line-height: 1.8;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .empty {{
      flex: 1;
      display: grid;
      place-items: center;
      padding: 40px;
      color: var(--muted);
      text-align: center;
      background:
        repeating-linear-gradient(0deg, #fffdf8 0 28px, #f6f1e8 28px 29px);
    }}
    .empty strong {{
      display: block;
      margin-bottom: 8px;
      color: var(--ink);
      font-size: 20px;
    }}
    @media (max-width: 860px) {{
      header {{
        align-items: start;
        flex-direction: column;
      }}
      .format-list {{
        justify-content: flex-start;
      }}
      main {{
        grid-template-columns: 1fr;
      }}
      .result {{
        min-height: 430px;
      }}
      .player {{
        grid-template-columns: 1fr;
      }}
      .player-controls {{
        justify-content: flex-start;
      }}
      h1 {{
        font-size: 28px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>&#20027;&#26059;&#24459;&#31616;&#35889;&#25552;&#21462;</h1>
        <p class="subtitle">&#19978;&#20256;&#19968;&#27573;&#28165;&#26224;&#26059;&#24459;&#38899;&#39057;&#65292;&#29983;&#25104;&#21487;&#22797;&#21046;&#30340; numbered notation&#12290;</p>
      </div>
      <div class="format-list" aria-label="supported formats">
        <span>WAV</span><span>MP3</span><span>FLAC</span>
      </div>
    </header>
    <main>
      <form id="upload-form" class="panel upload" action="/transcribe" method="post" enctype="multipart/form-data">
        <label class="dropzone">
          <input id="audio-file" name="audio" type="file" accept=".wav,.mp3,.flac,audio/wav,audio/mpeg,audio/flac" required>
          <span class="dropcopy">
            <strong>&#36873;&#25321;&#38899;&#39057;&#25991;&#20214;</strong>
            <span>&#25903;&#25345; WAV&#12289;MP3&#12289;FLAC&#65307;&#21333;&#26059;&#24459;&#24405;&#38899;&#25928;&#26524;&#26368;&#22909;&#12290;</span>
          </span>
        </label>
        <div id="selected" class="selected">&#23578;&#26410;&#36873;&#25321;&#25991;&#20214;</div>
        <audio id="preview" controls></audio>
        {message_html}
        <button id="submit-button" type="submit">&#25552;&#21462;&#20027;&#26059;&#24459;&#24182;&#29983;&#25104;&#31616;&#35889;</button>
      </form>
      <section class="panel result">
        {result_html}
      </section>
    </main>
  </div>
  <script>
    const form = document.getElementById("upload-form");
    const input = document.getElementById("audio-file");
    const selected = document.getElementById("selected");
    const preview = document.getElementById("preview");
    const submit = document.getElementById("submit-button");
    input.addEventListener("change", () => {{
      if (!input.files.length) {{
        selected.textContent = "\\u5c1a\\u672a\\u9009\\u62e9\\u6587\\u4ef6";
        preview.removeAttribute("src");
        preview.classList.remove("is-visible");
        return;
      }}
      selected.textContent = input.files[0].name;
      preview.src = URL.createObjectURL(input.files[0]);
      preview.classList.add("is-visible");
    }});
    form.addEventListener("submit", () => {{
      submit.disabled = true;
      submit.textContent = "\\u6b63\\u5728\\u63d0\\u53d6...";
    }});
    document.querySelectorAll("[data-copy-target]").forEach((button) => {{
      button.addEventListener("click", async () => {{
        const target = document.querySelector(button.dataset.copyTarget);
        if (!target) return;
        await navigator.clipboard.writeText(target.textContent);
        button.textContent = "\\u5df2\\u590d\\u5236";
        setTimeout(() => {{ button.textContent = "\\u590d\\u5236\\u7b80\\u8c31"; }}, 1400);
      }});
    }});

    const playerDataNode = document.getElementById("melody-player-data");
    if (playerDataNode) {{
      const playerData = JSON.parse(playerDataNode.textContent);
      const playButton = document.getElementById("melody-play");
      const restartButton = document.getElementById("melody-restart");
      const stopButton = document.getElementById("melody-stop");
      const status = document.getElementById("melody-player-status");
      let audioContext = null;
      let scheduled = [];
      let activeMaster = null;
      let stopTimer = null;

      const clearScheduled = () => {{
        scheduled.forEach((node) => {{
          try {{ node.stop(); }} catch (_error) {{}}
        }});
        scheduled = [];
        if (activeMaster) {{
          try {{ activeMaster.disconnect(); }} catch (_error) {{}}
          activeMaster = null;
        }}
        if (stopTimer) {{
          clearTimeout(stopTimer);
          stopTimer = null;
        }}
      }};

      const setStatus = (text) => {{
        if (status) status.textContent = text;
      }};

      const pitchToFrequency = (pitch) => 440 * Math.pow(2, (pitch - 69) / 12);

      const scheduleTone = (note, start, duration, master) => {{
        const frequency = pitchToFrequency(note.pitch);
        const voice = audioContext.createGain();
        const toneFilter = audioContext.createBiquadFilter();
        toneFilter.type = "lowpass";
        toneFilter.frequency.setValueAtTime(Math.min(6200, frequency * 8), start);
        toneFilter.Q.setValueAtTime(0.7, start);
        voice.gain.setValueAtTime(0.0001, start);
        voice.gain.exponentialRampToValueAtTime(0.62, start + 0.018);
        voice.gain.linearRampToValueAtTime(0.48, start + Math.min(0.16, duration * 0.45));
        voice.gain.setTargetAtTime(0.0001, start + Math.max(0.04, duration - 0.08), 0.045);
        voice.connect(toneFilter).connect(master);

        const vibrato = audioContext.createOscillator();
        const vibratoDepth = audioContext.createGain();
        vibrato.type = "sine";
        vibrato.frequency.setValueAtTime(5.2, start);
        vibratoDepth.gain.setValueAtTime(Math.min(3.0, frequency * 0.004), start);
        vibrato.connect(vibratoDepth);

        [
          ["triangle", 1.0, 0.74],
          ["sine", 2.01, 0.18],
          ["sine", 3.0, 0.08],
        ].forEach(([type, ratio, level]) => {{
          const oscillator = audioContext.createOscillator();
          const partialGain = audioContext.createGain();
          oscillator.type = type;
          oscillator.frequency.setValueAtTime(frequency * ratio, start);
          partialGain.gain.setValueAtTime(level, start);
          vibratoDepth.connect(oscillator.frequency);
          oscillator.connect(partialGain).connect(voice);
          oscillator.start(start);
          oscillator.stop(start + duration + 0.12);
          scheduled.push(oscillator);
        }});

        if (duration >= 0.12) {{
          const noise = audioContext.createBufferSource();
          const noiseGain = audioContext.createGain();
          const noiseFilter = audioContext.createBiquadFilter();
          const sampleCount = Math.max(1, Math.floor(audioContext.sampleRate * 0.035));
          const buffer = audioContext.createBuffer(1, sampleCount, audioContext.sampleRate);
          const channel = buffer.getChannelData(0);
          for (let index = 0; index < sampleCount; index += 1) {{
            channel[index] = (Math.random() * 2 - 1) * Math.pow(1 - index / sampleCount, 2);
          }}
          noise.buffer = buffer;
          noiseFilter.type = "bandpass";
          noiseFilter.frequency.setValueAtTime(Math.min(4600, frequency * 6), start);
          noiseFilter.Q.setValueAtTime(1.2, start);
          noiseGain.gain.setValueAtTime(0.025, start);
          noiseGain.gain.exponentialRampToValueAtTime(0.0001, start + 0.045);
          noise.connect(noiseFilter).connect(noiseGain).connect(master);
          noise.start(start);
          noise.stop(start + 0.055);
          scheduled.push(noise);
        }}

        vibrato.start(start);
        vibrato.stop(start + duration + 0.12);
        scheduled.push(vibrato);
      }};

      const playMelody = async (restart = true) => {{
        if (!playerData.notes.length) {{
          setStatus("\\u6ca1\\u6709\\u53ef\\u64ad\\u653e\\u7684\\u65cb\\u5f8b");
          return;
        }}
        clearScheduled();
        audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
        await audioContext.resume();
        const now = audioContext.currentTime + 0.08;
        const firstStart = restart ? playerData.notes[0].start : 0;
        const master = audioContext.createGain();
        const compressor = audioContext.createDynamicsCompressor();
        master.gain.value = 0.16;
        compressor.threshold.value = -22;
        compressor.knee.value = 24;
        compressor.ratio.value = 3;
        compressor.attack.value = 0.006;
        compressor.release.value = 0.18;
        master.connect(compressor).connect(audioContext.destination);
        activeMaster = master;

        playerData.notes.forEach((note) => {{
          const start = now + Math.max(0, note.start - firstStart);
          const duration = Math.max(0.08, note.end - note.start);
          scheduleTone(note, start, duration, master);
        }});

        const totalMs = Math.max(200, (playerData.durationSeconds - firstStart) * 1000 + 180);
        setStatus("\\u6b63\\u5728\\u64ad\\u653e " + playerData.noteCount + " \\u4e2a\\u97f3");
        stopTimer = setTimeout(() => {{
          scheduled = [];
          if (activeMaster) {{
            try {{ activeMaster.disconnect(); }} catch (_error) {{}}
            activeMaster = null;
          }}
          setStatus("\\u64ad\\u653e\\u5b8c\\u6210");
        }}, totalMs);
      }};

      playButton?.addEventListener("click", () => {{
        playMelody(true).catch(() => setStatus("\\u64ad\\u653e\\u88ab\\u6d4f\\u89c8\\u5668\\u62e6\\u622a\\uff0c\\u8bf7\\u518d\\u70b9\\u4e00\\u6b21"));
      }});
      restartButton?.addEventListener("click", () => {{
        playMelody(true).catch(() => setStatus("\\u64ad\\u653e\\u5931\\u8d25"));
      }});
      stopButton?.addEventListener("click", () => {{
        clearScheduled();
        setStatus("\\u5df2\\u505c\\u6b62");
      }});
      window.addEventListener("load", () => {{
        playMelody(true).catch(() => setStatus("\\u81ea\\u52a8\\u64ad\\u653e\\u88ab\\u6d4f\\u89c8\\u5668\\u62e6\\u622a\\uff0c\\u8bf7\\u70b9\\u51fb\\u64ad\\u653e"));
      }});
    }}
  </script>
</body>
</html>"""


def _build_handler(output_dir: Path) -> type[BaseHTTPRequestHandler]:
    output_root = output_dir.resolve()

    class MusicStuffHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            LOGGER.info("HTTP GET %s", parsed.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_html(render_page())
                return
            if parsed.path.startswith("/runs/"):
                self._serve_run_page(parsed.path)
                return
            if parsed.path.startswith("/artifacts/"):
                self._serve_artifact(parsed.path)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            LOGGER.info("HTTP POST %s", parsed.path)
            if parsed.path != "/transcribe":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                upload = self._read_upload()
                run_id = uuid.uuid4().hex[:12]
                run_dir = output_root / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                input_path = run_dir / upload.safe_name
                input_path.write_bytes(upload.content)
                LOGGER.info(
                    "Upload received: run_id=%s file=%s bytes=%s saved_as=%s",
                    run_id,
                    upload.original_name,
                    len(upload.content),
                    input_path,
                )

                result = MusicTranscriptionPipeline().transcribe(input_path, run_dir)
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
                        jianpu_href=_artifact_href(run_id, "melody.jianpu.txt"),
                        analysis_href=_artifact_href(run_id, "analysis.json"),
                        player_payload=player_payload,
                    )
                )
                LOGGER.info("Upload transcription rendered: run_id=%s", run_id)
            except Exception as exc:
                LOGGER.exception("Upload transcription failed")
                self._send_html(render_page(message=str(exc)), status=HTTPStatus.BAD_REQUEST)

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
            )

    return MusicStuffHandler


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


def _artifact_href(run_id: str, file_name: str) -> str:
    return f"/artifacts/{run_id}/{file_name}"
