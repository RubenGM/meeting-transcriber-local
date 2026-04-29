from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from meeting_transcriber.types import ConversationTurn


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    message: str
    seconds: float | None = None
    duration_seconds: float | None = None
    elapsed_seconds: float | None = None
    text_chars: int | None = None
    segments: int | None = None
    speakers: int | None = None
    summary: str | None = None
    text: str | None = None
    completed: int | None = None
    total: int | None = None


def format_progress_event(event: ProgressEvent) -> str:
    if event.stage == "transcription" and event.seconds is not None:
        return format_transcription_progress(event)
    if event.stage == "transcription_segment":
        return format_segment_preview(event)
    if event.stage == "diarization_progress":
        return format_diarization_progress(event)
    if event.summary:
        return f"{event.message}: {event.summary}"
    return event.message


def format_transcription_progress(event: ProgressEvent) -> str:
    segments = event.segments if event.segments is not None else 0
    text_chars = event.text_chars if event.text_chars is not None else 0
    seconds = event.seconds if event.seconds is not None else 0.0
    duration = event.duration_seconds
    elapsed = event.elapsed_seconds
    speed = _speed(seconds, elapsed)
    eta = _eta(seconds, duration, speed)
    timing = f"{format_seconds(seconds)} procesados"
    if duration is not None and duration > 0:
        timing = f"{format_seconds(seconds)} / {format_seconds(duration)}"
    if speed is not None:
        timing = f"{timing}, {speed:.1f}x"
    if eta is not None:
        timing = f"{timing}, ETA {format_seconds(eta)}"
    return (
        f"Transcribiendo audio: {timing}, "
        f"{segments} segmentos, {text_chars} caracteres"
    )


def format_segment_preview(event: ProgressEvent) -> str:
    seconds = event.seconds if event.seconds is not None else 0.0
    text = event.text.strip() if event.text is not None else ""
    return f"[{format_seconds(seconds)}] {text}"


def format_diarization_progress(event: ProgressEvent) -> str:
    if event.completed is None or event.total is None or event.total <= 0:
        return event.message
    percent = min(100.0, max(0.0, event.completed / event.total * 100.0))
    return f"{event.message}: {event.completed}/{event.total} ({percent:.0f}%)"


def format_speaker_summary(turns: list[ConversationTurn]) -> str:
    durations: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for turn in turns:
        durations[turn.speaker] += max(0.0, turn.end - turn.start)
        counts[turn.speaker] += 1
    parts = [
        f"{speaker}: {format_seconds(durations[speaker])} en {counts[speaker]} turnos"
        for speaker in sorted(durations)
    ]
    return " | ".join(parts)


def format_seconds(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _speed(seconds: float, elapsed: float | None) -> float | None:
    if elapsed is None or elapsed <= 0 or seconds <= 0:
        return None
    return seconds / elapsed


def _eta(seconds: float, duration: float | None, speed: float | None) -> float | None:
    if duration is None or duration <= 0 or speed is None or speed <= 0 or seconds <= 0:
        return None
    remaining = max(0.0, duration - seconds)
    return remaining / speed
