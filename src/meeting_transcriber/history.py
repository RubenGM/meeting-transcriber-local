from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HistoryEntry:
    start_seconds: float | None
    end_seconds: float | None
    output_dir: Path


@dataclass(frozen=True)
class AnalysisHistory:
    entries: dict[str, list[HistoryEntry]]

    def entries_for(self, audio_path: Path) -> list[HistoryEntry]:
        return self.entries.get(str(audio_path), [])


def load_history(path: Path) -> AnalysisHistory:
    if not path.exists():
        return AnalysisHistory(entries={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[str, list[HistoryEntry]] = {}
    for audio_path, items in payload.get("entries", {}).items():
        entries[audio_path] = [
            HistoryEntry(
                start_seconds=item["start_seconds"],
                end_seconds=item["end_seconds"],
                output_dir=Path(item["output_dir"]),
            )
            for item in items
        ]
    return AnalysisHistory(entries=entries)


def save_history(path: Path, history: AnalysisHistory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": {
            audio_path: [
                {
                    "start_seconds": entry.start_seconds,
                    "end_seconds": entry.end_seconds,
                    "output_dir": str(entry.output_dir),
                }
                for entry in entries
            ]
            for audio_path, entries in history.entries.items()
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def add_history_entry(path: Path, audio_path: Path, entry: HistoryEntry) -> None:
    history = load_history(path)
    entries = dict(history.entries)
    audio_key = str(audio_path)
    entries.setdefault(audio_key, []).append(entry)
    save_history(path, AnalysisHistory(entries=entries))

