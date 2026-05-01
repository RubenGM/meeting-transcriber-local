from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AppMode(Enum):
    SIMPLE = "simple"
    ADVANCED = "advanced"


@dataclass(frozen=True)
class SimpleModeSettings:
    target_wait_seconds: float = 15 * 60.0
    chunk_overlap_seconds: float = 15.0
    min_chunk_seconds: float = 5 * 60.0
    max_chunk_seconds: float = 30 * 60.0
    normalize_audio: bool = True
    auto_apply_high_confidence_names: bool = True


@dataclass(frozen=True)
class SimpleRunSummary:
    chunks_total: int
    chunks_completed: int
    chunks_failed: int
    output_dirs: tuple[str, ...]
    final_output_dir: str | None = None
    final_transcript_path: str | None = None
    html_report_path: str | None = None
    normalized_audio_path: str | None = None
