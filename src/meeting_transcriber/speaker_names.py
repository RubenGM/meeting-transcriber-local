from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace

from meeting_transcriber.types import ConversationTurn


_NAME = r"[A-ZÀ-Ý][A-Za-zÀ-ÿ'·.-]+(?:\s+(?:de|del|de la|d'|l'|[A-ZÀ-Ý][A-Za-zÀ-ÿ'·.-]+)){0,3}"
_INTRO_PATTERNS = (
    re.compile(rf"\b(?:jo\s+)?(?:em\s+dic|me\s+llamo|soc|sóc|soy)\s+(?:el|la|l'|en|na)?\s*({_NAME})", re.IGNORECASE),
    re.compile(rf"\b(?:bon dia,\s*)?(?:jo\s+)?(?:soc|sóc|soy)\s+(?:el|la|l'|en|na)?\s*({_NAME})", re.IGNORECASE),
    re.compile(rf"\b(?:el|la|l'|en|na)\s+({_NAME}),\s+(?:director|directora|t[eè]cnic|t[eè]cnica|regidor|regidora)", re.IGNORECASE),
    re.compile(rf"\b({_NAME}),\s+(?:director|directora|t[eè]cnic|t[eè]cnica|regidor|regidora)", re.IGNORECASE),
)

_BAD_STARTS = {
    "Bon",
    "Com",
    "Hola",
    "Jo",
    "La Generalitat",
    "Molt",
    "Primer",
}


def speaker_labels(turns: list[ConversationTurn]) -> list[str]:
    labels: list[str] = []
    for turn in turns:
        if turn.speaker not in labels:
            labels.append(turn.speaker)
    return labels


def suggest_names_by_speaker(turns: list[ConversationTurn]) -> dict[str, list[str]]:
    candidates: dict[str, Counter[str]] = defaultdict(Counter)
    for turn in turns:
        for name in extract_name_candidates(turn.text):
            candidates[turn.speaker][name] += 1
    return {
        speaker: [name for name, _count in counter.most_common()]
        for speaker, counter in candidates.items()
    }


def all_name_candidates(turns: list[ConversationTurn]) -> list[str]:
    counter: Counter[str] = Counter()
    for turn in turns:
        counter.update(extract_name_candidates(turn.text))
    return [name for name, _count in counter.most_common()]


def extract_name_candidates(text: str) -> list[str]:
    found: list[str] = []
    for pattern in _INTRO_PATTERNS:
        for match in pattern.finditer(text):
            name = _clean_name(match.group(1))
            if _looks_like_name(name) and name not in found:
                found.append(name)
    return found


def rename_speakers(turns: list[ConversationTurn], names: dict[str, str]) -> list[ConversationTurn]:
    renamed: list[ConversationTurn] = []
    for turn in turns:
        speaker = names.get(turn.speaker, "").strip() or turn.speaker
        renamed.append(replace(turn, speaker=speaker))
    return renamed


def _clean_name(value: str) -> str:
    name = value.strip(" ,.;:")
    name = re.sub(r"\s+", " ", name)
    for suffix in (" soc", " sóc", " soy", " director", " directora", " tècnic", " tècnica"):
        index = name.lower().find(suffix)
        if index > 0:
            name = name[:index].strip()
    return name


def _looks_like_name(name: str) -> bool:
    if len(name) < 2:
        return False
    if name in _BAD_STARTS:
        return False
    words = name.split()
    if len(words) > 5:
        return False
    return any(word[:1].isupper() for word in words)
