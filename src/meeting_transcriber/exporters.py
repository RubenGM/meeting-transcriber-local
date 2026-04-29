from __future__ import annotations

import json
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
