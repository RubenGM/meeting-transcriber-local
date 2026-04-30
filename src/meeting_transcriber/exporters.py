from __future__ import annotations

import json
import re
from pathlib import Path

from meeting_transcriber.types import ConversationTurn


def export_markdown_text(turns: list[ConversationTurn]) -> str:
    lines = ["# Transcripcion", ""]
    for turn in turns:
        lines.append(f"[{_clock_time(turn.start)}] **{turn.speaker}:** {turn.text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_plain_text(turns: list[ConversationTurn]) -> str:
    lines = []
    for turn in turns:
        lines.append(f"[{_clock_time(turn.start)}] {turn.speaker}: {turn.text}")
    return "\n".join(lines) + "\n"


def export_json_text(turns: list[ConversationTurn]) -> str:
    payload = {
        "turns": [
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": turn.speaker,
                "text": turn.text,
            }
            for turn in turns
        ]
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def export_srt_text(turns: list[ConversationTurn]) -> str:
    blocks = []
    for index, turn in enumerate(turns, start=1):
        blocks.append(
            f"{index}\n"
            f"{_srt_time(turn.start)} --> {_srt_time(turn.end)}\n"
            f"{turn.speaker}: {turn.text}"
        )
    return "\n\n".join(blocks) + "\n"


def write_all_exports(
    output_dir: Path,
    turns: list[ConversationTurn],
    *,
    basename: str = "transcript",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{basename}.md").write_text(export_markdown_text(turns), encoding="utf-8")
    (output_dir / f"{basename}.txt").write_text(export_plain_text(turns), encoding="utf-8")
    (output_dir / f"{basename}.json").write_text(export_json_text(turns), encoding="utf-8")
    (output_dir / f"{basename}.srt").write_text(export_srt_text(turns), encoding="utf-8")


def build_processing_output_dir(
    base_output_dir: Path,
    audio_path: Path,
    *,
    start_seconds: float | None,
    end_seconds: float | None,
) -> Path:
    audio_name = _safe_path_part(audio_path.stem)
    range_name = f"{_path_time(start_seconds or 0)}_to_{_path_time(end_seconds)}"
    candidate = base_output_dir / audio_name / range_name
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        suffixed = candidate.with_name(f"{candidate.name}_{suffix}")
        if not suffixed.exists():
            return suffixed
        suffix += 1


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return cleaned or "audio"


def _path_time(seconds: float | None) -> str:
    if seconds is None:
        return "end"
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}-{minutes:02d}-{secs:02d}"


def _clock_time(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _srt_time(seconds: float) -> str:
    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = whole_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
