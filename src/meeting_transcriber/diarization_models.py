from __future__ import annotations

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

DIARIZATION_MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    (DEFAULT_DIARIZATION_MODEL, "Automático"),
)


def diarization_model_labels() -> list[str]:
    return [label for _model_id, label in DIARIZATION_MODEL_OPTIONS]


def diarization_model_id_from_display_name(value: str) -> str:
    normalized = value.strip()
    for model_id, label in DIARIZATION_MODEL_OPTIONS:
        if normalized == label or normalized == model_id:
            return model_id
    raise ValueError(f"Modelo de diarizacion no soportado: {value}")


def diarization_model_label_from_id(model_id: str) -> str:
    for option_id, label in DIARIZATION_MODEL_OPTIONS:
        if model_id == option_id:
            return label
    raise ValueError(f"Modelo de diarizacion no soportado: {model_id}")

