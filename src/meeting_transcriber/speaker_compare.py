from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from meeting_transcriber.speaker_memory import SpeakerMemory, identity_names
from meeting_transcriber.speaker_names import speaker_labels
from meeting_transcriber.types import ConversationTurn


@dataclass(frozen=True)
class SpeakerComparisonRow:
    speaker: str
    suggested_name: str | None
    already_known_name: bool
    turn_count: int
    total_seconds: float
    sample: str


@dataclass(frozen=True)
class SpeakerComparisonSummary:
    known_names: tuple[str, ...]
    rows: tuple[SpeakerComparisonRow, ...]


def summarize_speaker_comparison(
    turns: list[ConversationTurn],
    memory: SpeakerMemory,
    audio_path: Path,
    *,
    suggested_names: dict[str, str] | None = None,
) -> SpeakerComparisonSummary:
    suggested_names = suggested_names or {}
    known_names = tuple(identity_names(memory, audio_path))
    known_name_set = set(known_names)
    rows: list[SpeakerComparisonRow] = []
    for speaker in speaker_labels(turns):
        speaker_turns = [turn for turn in turns if turn.speaker == speaker]
        rows.append(
            SpeakerComparisonRow(
                speaker=speaker,
                suggested_name=suggested_names.get(speaker),
                already_known_name=speaker in known_name_set,
                turn_count=len(speaker_turns),
                total_seconds=sum(max(0.0, turn.end - turn.start) for turn in speaker_turns),
                sample=_first_sample(speaker_turns),
            )
        )
    return SpeakerComparisonSummary(known_names=known_names, rows=tuple(rows))


def _first_sample(turns: list[ConversationTurn]) -> str:
    for turn in turns:
        text = turn.text.strip()
        if text:
            return text[:120] + ("..." if len(text) > 120 else "")
    return ""
