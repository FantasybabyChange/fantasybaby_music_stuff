"""Command-line interface for music-stuff."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from music_stuff import __version__
from music_stuff.pipeline import MusicTranscriptionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-stuff",
        description="Generate melody, key, chord, and score artifacts from audio.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Show the planned pipeline stages.")
    plan_parser.set_defaults(func=_handle_plan)

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe audio into symbolic music artifacts.",
    )
    transcribe_parser.add_argument("input", type=Path, help="Path to an audio file.")
    transcribe_parser.add_argument(
        "--out",
        type=Path,
        default=Path("output"),
        help="Directory for generated MIDI, MusicXML, and JSON files.",
    )
    transcribe_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pipeline stages without running audio analysis.",
    )
    transcribe_parser.set_defaults(func=_handle_transcribe)

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
        pipeline.transcribe(args.input, args.out)
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}", file=sys.stderr)
        return 2

    print(f"Artifacts written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
