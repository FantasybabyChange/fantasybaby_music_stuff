"""Command-line interface for music-stuff."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from music_stuff import __version__
from music_stuff.pipeline import MusicTranscriptionPipeline
from music_stuff.web import UIConfig, run_ui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-stuff",
        description="Generate melody, key, chord, and score artifacts from audio.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity for CLI and UI workflows.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional file path for writing workflow logs.",
    )

    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Show the planned pipeline stages.")
    plan_parser.set_defaults(func=_handle_plan)

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe WAV audio into melody and Jianpu artifacts.",
    )
    transcribe_parser.add_argument("input", type=Path, help="Path to an audio file.")
    transcribe_parser.add_argument(
        "--out",
        type=Path,
        default=Path("output"),
        help="Directory for generated Jianpu and JSON files.",
    )
    transcribe_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pipeline stages without running audio analysis.",
    )
    transcribe_parser.set_defaults(func=_handle_transcribe)

    ui_parser = subparsers.add_parser(
        "ui",
        help="Start the local web UI for uploading audio and viewing Jianpu.",
    )
    ui_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local web UI.",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the local web UI.",
    )
    ui_parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/ui"),
        help="Directory for uploaded audio and generated UI artifacts.",
    )
    ui_parser.add_argument(
        "--open",
        action="store_true",
        help="Open the UI in the default browser after starting.",
    )
    ui_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the UI URL without starting the server.",
    )
    ui_parser.set_defaults(func=_handle_ui)

    return parser


def _handle_plan(_args: argparse.Namespace) -> int:
    pipeline = MusicTranscriptionPipeline()
    for index, stage in enumerate(pipeline.plan().stages, start=1):
        print(f"{index}. {stage}")
    return 0


def _handle_transcribe(args: argparse.Namespace) -> int:
    pipeline = MusicTranscriptionPipeline()

    if args.dry_run:
        print(f"Input: {args.input}")
        print(f"Output: {args.out}")
        for index, stage in enumerate(pipeline.plan().stages, start=1):
            print(f"{index}. {stage}")
        return 0

    try:
        result = pipeline.transcribe(args.input, args.out)
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Artifacts written to {args.out}")
    if result.jianpu_path:
        print(f"Jianpu: {result.jianpu_path}")
    return 0


def _handle_ui(args: argparse.Namespace) -> int:
    config = UIConfig(
        host=args.host,
        port=args.port,
        output_dir=args.out,
        open_browser=args.open,
    )
    url = f"http://{config.host}:{config.port}"
    if args.dry_run:
        print(f"UI: {url}")
        print(f"Output: {config.output_dir}")
        return 0

    try:
        run_ui(config)
    except KeyboardInterrupt:
        print("\nUI stopped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level, args.log_file)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


def _configure_logging(level_name: str, log_file: Path | None = None) -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level_name),
        handlers=handlers,
        force=True,
    )
    for handler in handlers:
        handler.setFormatter(formatter)


if __name__ == "__main__":
    raise SystemExit(main())
