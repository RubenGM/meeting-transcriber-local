from __future__ import annotations

from meeting_transcriber.progress import format_seconds


def parse_optional_timestamp(value: str) -> float | None:
    stripped = value.strip()
    if stripped == "":
        return None
    if stripped.startswith("-"):
        raise ValueError("El tiempo no puede ser negativo")
    parts = stripped.split(":")
    if len(parts) > 3:
        raise ValueError("Usa segundos, MM:SS o HH:MM:SS")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("El tiempo debe usar numeros") from exc
    if any(number < 0 for number in numbers):
        raise ValueError("El tiempo no puede ser negativo")
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def validate_time_range(start_seconds: float | None, end_seconds: float | None) -> None:
    if start_seconds is not None and end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("El fin debe ser posterior al inicio")


def hms_to_seconds(
    hours: str,
    minutes: str,
    seconds: str,
    *,
    blank_when_zero: bool,
) -> float | None:
    parsed_hours = _parse_non_negative_int(hours, "horas")
    parsed_minutes = _parse_non_negative_int(minutes, "minutos")
    parsed_seconds = _parse_non_negative_int(seconds, "segundos")
    total = parsed_hours * 3600 + parsed_minutes * 60 + parsed_seconds
    if blank_when_zero and total == 0:
        return None
    return float(total)


def format_optional_range(start_seconds: float | None, end_seconds: float | None) -> str:
    if start_seconds is None and end_seconds is None:
        return "Todo el archivo"
    start = format_seconds(start_seconds or 0)
    end = "fin" if end_seconds is None else format_seconds(end_seconds)
    return f"{start} -> {end}"


def _parse_non_negative_int(value: str, label: str) -> int:
    stripped = value.strip()
    if stripped == "":
        return 0
    try:
        parsed = int(stripped)
    except ValueError as exc:
        raise ValueError(f"El campo {label} debe ser un numero entero") from exc
    if parsed < 0:
        raise ValueError(f"El campo {label} no puede ser negativo")
    return parsed
