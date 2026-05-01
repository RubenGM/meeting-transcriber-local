from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HistoryEntry:
    start_seconds: float | None
    end_seconds: float | None
    output_dir: Path
    elapsed_seconds: float | None = None
    id: str | None = None
    hidden: bool = False
    superseded_by: str | None = None
    merge_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RangeRecommendation:
    start_seconds: float
    end_seconds: float
    speed: float | None
    estimated_elapsed_seconds: float | None


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
        loaded_items = []
        for index, item in enumerate(items, start=1):
            loaded_items.append(
                HistoryEntry(
                    start_seconds=item["start_seconds"],
                    end_seconds=item["end_seconds"],
                    output_dir=Path(item["output_dir"]),
                    elapsed_seconds=item.get("elapsed_seconds"),
                    id=str(item.get("id") or index),
                    hidden=bool(item.get("hidden", False)),
                    superseded_by=item.get("superseded_by"),
                    merge_source_ids=tuple(str(value) for value in item.get("merge_source_ids", [])),
                )
            )
        entries[audio_path] = loaded_items
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
                    "elapsed_seconds": entry.elapsed_seconds,
                    "id": entry.id,
                    "hidden": entry.hidden,
                    "superseded_by": entry.superseded_by,
                    "merge_source_ids": list(entry.merge_source_ids),
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
    audio_entries = list(entries.setdefault(audio_key, []))
    audio_entries.append(_with_id(entry, _next_entry_id(audio_entries)))
    entries[audio_key] = audio_entries
    save_history(path, AnalysisHistory(entries=entries))


def visible_entries_for(history: AnalysisHistory, audio_path: Path) -> list[HistoryEntry]:
    return [entry for entry in history.entries_for(audio_path) if not entry.hidden]


def add_merged_history_entry(
    path: Path,
    audio_path: Path,
    merged_entry: HistoryEntry,
    source_ids: list[str | None] | tuple[str | None, ...],
) -> HistoryEntry:
    history = load_history(path)
    entries = {audio_key: list(items) for audio_key, items in history.entries.items()}
    audio_key = str(audio_path)
    audio_entries = entries.get(audio_key, [])
    clean_source_ids = tuple(str(source_id) for source_id in source_ids if source_id is not None)
    merged = _with_id(
        HistoryEntry(
            start_seconds=merged_entry.start_seconds,
            end_seconds=merged_entry.end_seconds,
            output_dir=merged_entry.output_dir,
            elapsed_seconds=merged_entry.elapsed_seconds,
            merge_source_ids=clean_source_ids,
        ),
        _next_entry_id(audio_entries),
    )
    updated_entries = [
        _hide_entry(entry, merged.id) if entry.id in clean_source_ids else entry
        for entry in audio_entries
    ]
    updated_entries.append(merged)
    entries[audio_key] = updated_entries
    save_history(path, AnalysisHistory(entries=entries))
    return merged


def remove_history_entry(path: Path, audio_path: Path, index: int) -> HistoryEntry:
    history = load_history(path)
    entries = {audio_key: list(items) for audio_key, items in history.entries.items()}
    audio_key = str(audio_path)
    audio_entries = entries.get(audio_key, [])
    if index < 0 or index >= len(audio_entries):
        raise IndexError("Entrada de historial no encontrada")
    removed = audio_entries.pop(index)
    if audio_entries:
        entries[audio_key] = audio_entries
    elif audio_key in entries:
        del entries[audio_key]
    save_history(path, AnalysisHistory(entries=entries))
    return removed


def reanalysis_range(entry: HistoryEntry) -> tuple[float, float | None]:
    return (float(entry.start_seconds or 0.0), entry.end_seconds)


def _next_entry_id(entries: list[HistoryEntry]) -> str:
    numeric_ids = []
    for entry in entries:
        if entry.id is None:
            continue
        try:
            numeric_ids.append(int(entry.id))
        except ValueError:
            continue
    return str((max(numeric_ids) if numeric_ids else len(entries)) + 1)


def _with_id(entry: HistoryEntry, entry_id: str) -> HistoryEntry:
    return HistoryEntry(
        start_seconds=entry.start_seconds,
        end_seconds=entry.end_seconds,
        output_dir=entry.output_dir,
        elapsed_seconds=entry.elapsed_seconds,
        id=entry.id or entry_id,
        hidden=entry.hidden,
        superseded_by=entry.superseded_by,
        merge_source_ids=entry.merge_source_ids,
    )


def _hide_entry(entry: HistoryEntry, superseded_by: str | None) -> HistoryEntry:
    return HistoryEntry(
        start_seconds=entry.start_seconds,
        end_seconds=entry.end_seconds,
        output_dir=entry.output_dir,
        elapsed_seconds=entry.elapsed_seconds,
        id=entry.id,
        hidden=True,
        superseded_by=superseded_by,
        merge_source_ids=entry.merge_source_ids,
    )


def output_dir_reference_count(history: AnalysisHistory, output_dir: Path) -> int:
    return sum(
        1
        for entries in history.entries.values()
        for entry in entries
        if entry.output_dir == output_dir
    )


def completed_ranges(
    entries: list[HistoryEntry],
    total_duration_seconds: float,
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for entry in entries:
        start = entry.start_seconds or 0.0
        end = total_duration_seconds if entry.end_seconds is None else entry.end_seconds
        start = max(0.0, min(start, total_duration_seconds))
        end = max(0.0, min(end, total_duration_seconds))
        if end > start:
            ranges.append((float(start), float(end)))

    ranges.sort()
    merged: list[tuple[float, float]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def missing_ranges(
    entries: list[HistoryEntry],
    total_duration_seconds: float,
) -> list[tuple[float, float]]:
    if total_duration_seconds <= 0:
        return []
    ranges = completed_ranges(entries, total_duration_seconds)
    missing: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in ranges:
        if start > cursor:
            missing.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total_duration_seconds:
        missing.append((cursor, total_duration_seconds))
    return missing


def coverage_seconds(entries: list[HistoryEntry], total_duration_seconds: float) -> float:
    return sum(end - start for start, end in completed_ranges(entries, total_duration_seconds))


def recent_processing_speed(
    entries: list[HistoryEntry],
    *,
    limit: int = 3,
    total_duration_seconds: float | None = None,
) -> float | None:
    samples: list[tuple[float, float]] = []
    for entry in entries:
        audio_seconds = _entry_audio_seconds(entry, total_duration_seconds)
        if audio_seconds is None or audio_seconds <= 0:
            continue
        if entry.elapsed_seconds is None or entry.elapsed_seconds <= 0:
            continue
        samples.append((audio_seconds, entry.elapsed_seconds))

    recent_samples = samples[-limit:]
    total_audio = sum(audio_seconds for audio_seconds, _elapsed in recent_samples)
    total_elapsed = sum(elapsed for _audio_seconds, elapsed in recent_samples)
    if total_audio <= 0 or total_elapsed <= 0:
        return None
    return total_audio / total_elapsed


def recommend_next_range(
    entries: list[HistoryEntry],
    *,
    total_duration_seconds: float,
    target_wait_seconds: float,
    min_audio_seconds: float = 120.0,
    max_audio_seconds: float = 1800.0,
) -> RangeRecommendation | None:
    if total_duration_seconds <= 0:
        return None

    ranges = completed_ranges(entries, total_duration_seconds)
    gap = _first_gap(ranges, total_duration_seconds)
    if gap is None:
        return None

    gap_start, gap_end = gap
    gap_seconds = gap_end - gap_start
    speed = recent_processing_speed(entries, total_duration_seconds=total_duration_seconds)
    if speed is None:
        recommended_audio_seconds = min(300.0, gap_seconds)
    else:
        recommended_audio_seconds = target_wait_seconds * speed

    recommended_audio_seconds = max(min_audio_seconds, recommended_audio_seconds)
    recommended_audio_seconds = min(max_audio_seconds, recommended_audio_seconds, gap_seconds)
    end_seconds = gap_start + recommended_audio_seconds
    estimated_elapsed = recommended_audio_seconds / speed if speed is not None and speed > 0 else None
    return RangeRecommendation(
        start_seconds=gap_start,
        end_seconds=end_seconds,
        speed=speed,
        estimated_elapsed_seconds=estimated_elapsed,
    )


def _entry_audio_seconds(
    entry: HistoryEntry,
    total_duration_seconds: float | None = None,
) -> float | None:
    start = entry.start_seconds or 0.0
    if entry.end_seconds is None:
        if total_duration_seconds is None:
            return None
        return max(0.0, total_duration_seconds - start)
    return max(0.0, entry.end_seconds - start)


def _first_gap(
    ranges: list[tuple[float, float]],
    total_duration_seconds: float,
) -> tuple[float, float] | None:
    cursor = 0.0
    for start, end in ranges:
        if start > cursor:
            return cursor, start
        cursor = max(cursor, end)
    if cursor < total_duration_seconds:
        return cursor, total_duration_seconds
    return None
