from __future__ import annotations

AUTO_LANGUAGE_LABEL = "Detección automática"

LANGUAGE_OPTIONS: tuple[tuple[str | None, str], ...] = (
    (None, AUTO_LANGUAGE_LABEL),
    ("ca", "Català"),
    ("es", "Español"),
    ("en", "English"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("it", "Italiano"),
    ("pt", "Português"),
)


def language_display_names() -> list[str]:
    return [name for _code, name in LANGUAGE_OPTIONS]


def code_from_display_name(value: str) -> str | None:
    normalized = value.strip()
    if normalized in ("", "auto"):
        return None
    for code, display_name in LANGUAGE_OPTIONS:
        if normalized == display_name or normalized == code:
            return code
    raise ValueError(f"Idioma no soportado: {value}")


def display_name_from_code(code: str | None) -> str:
    if code in (None, "", "auto"):
        return AUTO_LANGUAGE_LABEL
    for option_code, display_name in LANGUAGE_OPTIONS:
        if code == option_code:
            return display_name
    raise ValueError(f"Codigo de idioma no soportado: {code}")

