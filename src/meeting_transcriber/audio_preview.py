from __future__ import annotations

import re
from pathlib import Path


def preview_clip_path(
    cache_dir: Path,
    audio_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
) -> Path:
    return cache_dir / (
        f"{_safe_path_part(audio_path.stem)}_"
        f"{_path_time(start_seconds)}_to_{_path_time(end_seconds)}.wav"
    )


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return cleaned or "audio"


def _path_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    whole_seconds = milliseconds // 1000
    millis = milliseconds % 1000
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = whole_seconds % 60
    return f"{hours:02d}-{minutes:02d}-{secs:02d}_{millis:03d}"
