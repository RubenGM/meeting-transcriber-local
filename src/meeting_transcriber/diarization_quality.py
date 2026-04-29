from __future__ import annotations

DEFAULT_DIARIZATION_QUALITY = "precise"

DIARIZATION_QUALITY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("fast", "Rapida"),
    (DEFAULT_DIARIZATION_QUALITY, "Precisa"),
    ("strict", "Muy precisa"),
)


def diarization_quality_labels() -> list[str]:
    return [label for _quality_id, label in DIARIZATION_QUALITY_OPTIONS]


def diarization_quality_id_from_display_name(value: str) -> str:
    normalized = value.strip()
    for quality_id, label in DIARIZATION_QUALITY_OPTIONS:
        if normalized == quality_id or normalized == label:
            return quality_id
    raise ValueError(f"Calidad de diarizacion no soportada: {value}")


def diarization_quality_label_from_id(quality_id: str) -> str:
    for option_id, label in DIARIZATION_QUALITY_OPTIONS:
        if quality_id == option_id:
            return label
    return diarization_quality_label_from_id(DEFAULT_DIARIZATION_QUALITY)


def uses_word_alignment(quality_id: str) -> bool:
    return quality_id in {"precise", "strict"}
