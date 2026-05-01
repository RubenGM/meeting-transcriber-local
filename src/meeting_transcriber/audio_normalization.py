from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from meeting_transcriber.deepfilternet import (
    default_deepfilternet_dir,
    enhance_with_deepfilternet,
    resolve_deep_filter_binary,
)
from meeting_transcriber.progress import ProgressEvent


ProgressCallback = Callable[[ProgressEvent], None]


def speech_normalization_filter() -> str:
    return ",".join(
        (
            "highpass=f=80",
            "lowpass=f=8000",
            "afftdn=nf=-25",
            "dynaudnorm=f=150:g=15",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
        )
    )


def normalize_audio_for_speech(
    ffmpeg_path: Path,
    source_audio: Path,
    output_audio: Path,
    *,
    progress: ProgressCallback | None = None,
    duration_seconds: float | None = None,
) -> Path:
    deepfilter = resolve_deep_filter_binary(default_deepfilternet_dir())
    if deepfilter is not None:
        result = enhance_with_deepfilternet(
            deep_filter_binary=deepfilter,
            ffmpeg_path=ffmpeg_path,
            source_audio=source_audio,
            output_audio=output_audio,
            progress=progress,
            duration_seconds=duration_seconds,
        )
        if result.ok:
            return output_audio
        _report_normalization(
            progress,
            "DeepFilterNet no pudo completar; usando fallback FFmpeg",
            completed=1,
            total=2,
            duration_seconds=duration_seconds,
        )
    return normalize_audio_with_ffmpeg(
        ffmpeg_path,
        source_audio,
        output_audio,
        progress=progress,
        duration_seconds=duration_seconds,
    )


def normalize_audio_with_ffmpeg(
    ffmpeg_path: Path,
    source_audio: Path,
    output_audio: Path,
    *,
    progress: ProgressCallback | None = None,
    duration_seconds: float | None = None,
) -> Path:
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    _report_normalization(
        progress,
        "FFmpeg: aplicando filtros de voz",
        completed=1,
        total=2,
        duration_seconds=duration_seconds,
    )
    command = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(source_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        speech_normalization_filter(),
        str(output_audio),
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"No se pudo normalizar el audio: {_first_useful_line(result.stderr)}")
    _report_normalization(
        progress,
        "Audio normalizado con FFmpeg",
        completed=2,
        total=2,
        duration_seconds=duration_seconds,
    )
    return output_audio


def _first_useful_line(stderr: str) -> str:
    for line in reversed(stderr.splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith("["):
            return stripped[:500]
    return "ffmpeg no devolvio detalle"


def _report_normalization(
    progress: ProgressCallback | None,
    message: str,
    *,
    completed: int,
    total: int,
    duration_seconds: float | None = None,
) -> None:
    if progress is None:
        return
    progress(
        ProgressEvent(
            stage="normalization_progress",
            message=message,
            completed=completed,
            total=total,
            duration_seconds=duration_seconds,
        )
    )
