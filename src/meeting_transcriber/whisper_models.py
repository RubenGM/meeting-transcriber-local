from __future__ import annotations

DEFAULT_WHISPER_MODEL = "small"

WHISPER_MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    ("base", "Rápido"),
    ("small", "Equilibrado"),
    ("medium", "Preciso"),
    ("large-v3", "Máxima precisión"),
)


def whisper_model_labels() -> list[str]:
    return [label for _model_id, label in WHISPER_MODEL_OPTIONS]


def model_id_from_display_name(value: str) -> str:
    normalized = value.strip()
    for model_id, label in WHISPER_MODEL_OPTIONS:
        if normalized == label or normalized == model_id:
            return model_id
    raise ValueError(f"Modelo Whisper no soportado: {value}")


def model_label_from_id(model_id: str) -> str:
    for option_id, label in WHISPER_MODEL_OPTIONS:
        if model_id == option_id:
            return label
    raise ValueError(f"Modelo Whisper no soportado: {model_id}")

