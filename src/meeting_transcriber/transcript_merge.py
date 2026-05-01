from __future__ import annotations

from dataclasses import dataclass

from meeting_transcriber.types import ConversationTurn


@dataclass(frozen=True)
class MergeRow:
    left: ConversationTurn | None
    right: ConversationTurn | None
    start: float
    end: float
    chosen_speaker: str
    chosen_text: str

    @property
    def is_identical(self) -> bool:
        return (
            self.left is not None
            and self.right is not None
            and not self.has_speaker_difference
            and not self.has_text_difference
        )

    @property
    def has_speaker_difference(self) -> bool:
        if self.left is None or self.right is None:
            return True
        return self.left.speaker.strip() != self.right.speaker.strip()

    @property
    def has_text_difference(self) -> bool:
        if self.left is None or self.right is None:
            return True
        return _normalize_text(self.left.text) != _normalize_text(self.right.text)


@dataclass(frozen=True)
class DraftMergeRow:
    start: float
    end: float
    speaker: str
    text: str


def align_turns_for_merge(
    left_turns: list[ConversationTurn],
    right_turns: list[ConversationTurn],
) -> list[MergeRow]:
    rows: list[MergeRow] = []
    used_right: set[int] = set()
    for left in left_turns:
        right_index = _best_overlapping_turn(left, right_turns, used_right)
        if right_index is None:
            rows.append(_row_from_pair(left, None))
            continue
        used_right.add(right_index)
        rows.append(_row_from_pair(left, right_turns[right_index]))

    for index, right in enumerate(right_turns):
        if index not in used_right:
            rows.append(_row_from_pair(None, right))

    return sorted(rows, key=lambda row: (row.start, row.end))


def merged_turns_from_drafts(drafts: list[DraftMergeRow]) -> list[ConversationTurn]:
    turns = []
    for draft in drafts:
        text = draft.text.strip()
        speaker = draft.speaker.strip() or "Persona"
        if not text:
            continue
        turns.append(ConversationTurn(draft.start, draft.end, speaker, text))
    return sorted(turns, key=lambda turn: (turn.start, turn.end))


def _best_overlapping_turn(
    left: ConversationTurn,
    right_turns: list[ConversationTurn],
    used_right: set[int],
) -> int | None:
    best_index = None
    best_overlap = 0.0
    for index, right in enumerate(right_turns):
        if index in used_right:
            continue
        overlap = min(left.end, right.end) - max(left.start, right.start)
        if overlap > best_overlap:
            best_index = index
            best_overlap = overlap
    return best_index


def _row_from_pair(left: ConversationTurn | None, right: ConversationTurn | None) -> MergeRow:
    if left is None and right is None:
        raise ValueError("Una fila de fusion necesita al menos un turno")
    start_candidates = [turn.start for turn in (left, right) if turn is not None]
    end_candidates = [turn.end for turn in (left, right) if turn is not None]
    chosen = _default_choice(left, right)
    return MergeRow(
        left=left,
        right=right,
        start=min(start_candidates),
        end=max(end_candidates),
        chosen_speaker=chosen.speaker,
        chosen_text=chosen.text,
    )


def _default_choice(left: ConversationTurn | None, right: ConversationTurn | None) -> ConversationTurn:
    if left is None:
        assert right is not None
        return right
    if right is None:
        return left
    if _is_generic_speaker(left.speaker) and not _is_generic_speaker(right.speaker):
        return right
    if _is_generic_speaker(right.speaker) and not _is_generic_speaker(left.speaker):
        return left
    return right if len(right.text.strip()) > len(left.text.strip()) else left


def _is_generic_speaker(speaker: str) -> bool:
    value = speaker.strip()
    return value.startswith("Persona ") or value.startswith("SPEAKER_") or value == "Sin diarizar"


def _normalize_text(text: str) -> str:
    return " ".join(text.split())
