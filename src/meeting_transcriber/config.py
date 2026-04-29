from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from meeting_transcriber.types import ProcessingConfig
from meeting_transcriber.diarization_models import DEFAULT_DIARIZATION_MODEL
from meeting_transcriber.diarization_quality import DEFAULT_DIARIZATION_QUALITY
from meeting_transcriber.whisper_models import DEFAULT_WHISPER_MODEL


@dataclass(frozen=True)
class UiState:
    last_audio_dir: Path | None = None


def default_config_dir() -> Path:
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata is None:
            return Path.home() / "AppData" / "Roaming" / "MeetingTranscriber"
        return Path(appdata) / "MeetingTranscriber"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "meeting-transcriber"


def save_config(path: Path, config: ProcessingConfig, ui_state: UiState | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    for key in ("ffmpeg_path",):
        if payload[key] is not None:
            payload[key] = str(payload[key])
    if ui_state is not None and ui_state.last_audio_dir is not None:
        payload["last_audio_dir"] = str(ui_state.last_audio_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_ui_state(path: Path) -> UiState:
    if not path.exists():
        return UiState()
    payload = json.loads(path.read_text(encoding="utf-8"))
    last_audio_dir = payload.get("last_audio_dir")
    if isinstance(last_audio_dir, str) and last_audio_dir:
        return UiState(last_audio_dir=Path(last_audio_dir))
    return UiState()


def load_config(path: Path) -> ProcessingConfig | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    diarization_model = _migrate_diarization_model(payload)
    return ProcessingConfig(
        whisper_model=_migrate_whisper_model(payload),
        diarization_model=diarization_model or DEFAULT_DIARIZATION_MODEL,
        huggingface_token=payload.get("huggingface_token"),
        ffmpeg_path=Path(payload["ffmpeg_path"]) if payload.get("ffmpeg_path") is not None else None,
        language=payload["language"],
        min_speakers=payload["min_speakers"],
        max_speakers=payload["max_speakers"],
        device=payload["device"],
        compute_type=payload["compute_type"],
        export_speaker_audio=payload["export_speaker_audio"],
        start_seconds=_optional_float(payload.get("start_seconds")),
        end_seconds=_optional_float(payload.get("end_seconds")),
        diarization_quality=str(payload.get("diarization_quality") or DEFAULT_DIARIZATION_QUALITY),
    )


def _migrate_whisper_model(payload: dict[str, object]) -> str:
    if isinstance(payload.get("whisper_model"), str):
        return str(payload["whisper_model"])
    return DEFAULT_WHISPER_MODEL


def _migrate_diarization_model(payload: dict[str, object]) -> str:
    if isinstance(payload.get("diarization_model"), str):
        return str(payload["diarization_model"])
    return DEFAULT_DIARIZATION_MODEL


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    raise ValueError("Expected optional numeric config value")
