from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from meeting_transcriber.speaker_names import speaker_labels
from meeting_transcriber.types import ConversationTurn


@dataclass(frozen=True)
class SpeakerIdentity:
    name: str
    sample_ranges: tuple[tuple[float, float], ...]
    embeddings: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class SpeakerMemory:
    audios: dict[str, list[SpeakerIdentity]]

    def identities_for(self, audio_path: Path) -> list[SpeakerIdentity]:
        return self.audios.get(str(audio_path), [])


def load_speaker_memory(path: Path) -> SpeakerMemory:
    if not path.exists():
        return SpeakerMemory(audios={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    audios: dict[str, list[SpeakerIdentity]] = {}
    for audio_path, identities in payload.get("audios", {}).items():
        audios[audio_path] = [
            SpeakerIdentity(
                name=str(item["name"]),
                sample_ranges=tuple(
                    (float(start), float(end))
                    for start, end in item.get("sample_ranges", [])
                ),
                embeddings=tuple(
                    tuple(float(value) for value in embedding)
                    for embedding in item.get("embeddings", [])
                ),
            )
            for item in identities
        ]
    return SpeakerMemory(audios=audios)


def save_speaker_memory(path: Path, memory: SpeakerMemory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audios": {
            audio_path: [
                {
                    "name": identity.name,
                    "sample_ranges": [
                        [start, end] for start, end in identity.sample_ranges
                    ],
                    "embeddings": [
                        list(embedding) for embedding in identity.embeddings
                    ],
                }
                for identity in identities
            ]
            for audio_path, identities in memory.audios.items()
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def add_identity_samples(
    memory: SpeakerMemory,
    audio_path: Path,
    turns: list[ConversationTurn],
) -> SpeakerMemory:
    ranges_by_name: dict[str, list[tuple[float, float]]] = {}
    for turn in turns:
        name = turn.speaker.strip()
        if not name or name.startswith("SPEAKER_") or name.startswith("Persona "):
            continue
        if turn.end <= turn.start:
            continue
        ranges_by_name.setdefault(name, []).append((turn.start, turn.end))

    existing_by_name = {identity.name: identity for identity in memory.identities_for(audio_path)}
    updated: list[SpeakerIdentity] = []
    for name in sorted(set(existing_by_name) | set(ranges_by_name)):
        existing = existing_by_name.get(name)
        sample_ranges = tuple(ranges_by_name.get(name, []))
        if existing is not None:
            sample_ranges = tuple(dict.fromkeys((*existing.sample_ranges, *sample_ranges)))
            updated.append(
                SpeakerIdentity(
                    name=name,
                    sample_ranges=sample_ranges,
                    embeddings=existing.embeddings,
                )
            )
        else:
            updated.append(SpeakerIdentity(name=name, sample_ranges=sample_ranges))

    audios = dict(memory.audios)
    audios[str(audio_path)] = updated
    return SpeakerMemory(audios=audios)


def remember_validated_turns(
    path: Path,
    audio_path: Path,
    turns: list[ConversationTurn],
) -> SpeakerMemory:
    memory = load_speaker_memory(path)
    updated = add_identity_samples(memory, audio_path, turns)
    save_speaker_memory(path, updated)
    return updated


def identity_names(memory: SpeakerMemory, audio_path: Path) -> list[str]:
    return [identity.name for identity in memory.identities_for(audio_path)]


def build_unique_name_mapping(
    memory: SpeakerMemory,
    audio_path: Path,
    turns: list[ConversationTurn],
) -> dict[str, str]:
    labels = speaker_labels(turns)
    names = identity_names(memory, audio_path)
    if len(labels) != len(names):
        return {}
    if any(not label.startswith("Persona ") for label in labels):
        return {}
    return dict(zip(labels, names))
