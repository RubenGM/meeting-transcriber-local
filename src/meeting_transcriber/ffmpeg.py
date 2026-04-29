from __future__ import annotations

from pathlib import Path


def resolve_ffmpeg_path(configured_path: Path | None) -> Path:
    if configured_path is not None and str(configured_path).strip() != "":
        return configured_path
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is not installed, and no ffmpeg path was configured."
        ) from exc
    return Path(imageio_ffmpeg.get_ffmpeg_exe())

