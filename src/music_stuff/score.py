"""Score and analysis export interfaces."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from music_stuff.models import AnalysisResult, Melody


class ScoreExporter:
    """Export symbolic results into files users can inspect."""

    def export_midi(self, melody: Melody, output_path: Path) -> Path:
        raise NotImplementedError("MIDI export is not implemented yet.")

    def export_musicxml(
        self,
        melody: Melody,
        analysis: AnalysisResult,
        output_path: Path,
    ) -> Path:
        raise NotImplementedError("MusicXML export is not implemented yet.")

    def export_analysis_json(self, analysis: AnalysisResult, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(analysis), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path
