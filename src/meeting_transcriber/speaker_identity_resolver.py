from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from meeting_transcriber.speaker_fingerprints import cosine_similarity
from meeting_transcriber.speaker_memory import SpeakerMemory, build_unique_name_mapping
from meeting_transcriber.speaker_names import rename_speakers, speaker_labels
from meeting_transcriber.types import ConversationTurn


MEMORY_THRESHOLD = 0.85
RUN_THRESHOLD = 0.82
DECISIONS_FILENAME = "speaker_identity_decisions.json"


@dataclass(frozen=True)
class SpeakerIdentityDecision:
    source_speaker: str
    resolved_name: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class ChunkSpeakerResolution:
    output_dir: Path
    mapping: dict[str, str]
    decisions: tuple[SpeakerIdentityDecision, ...]


@dataclass(frozen=True)
class ChunkSpeakerEvidence:
    output_dir: Path
    turns: list[ConversationTurn]
    embeddings: dict[str, tuple[float, ...]]
    start_seconds: float = 0.0
    end_seconds: float = 0.0


@dataclass(frozen=True)
class _Observation:
    chunk_index: int
    label: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class _Cluster:
    name: str | None
    embedding: tuple[float, ...]
    observations: tuple[_Observation, ...]


def resolve_chunk_speaker_identities(
    audio_path: Path,
    memory: SpeakerMemory,
    chunks: list[ChunkSpeakerEvidence],
) -> list[ChunkSpeakerResolution]:
    memory_candidates = _memory_embedding_candidates(memory, audio_path)
    cluster_names = _final_cluster_names(_cluster_observations(chunks), memory_candidates)
    resolutions: list[ChunkSpeakerResolution] = []

    next_person_index = _next_available_generated_index(set(cluster_names.values()))
    for chunk_index, chunk in enumerate(chunks):
        mapping: dict[str, str] = {}
        decisions: list[SpeakerIdentityDecision] = []
        used_names: set[str] = set()
        labels = speaker_labels(chunk.turns)

        for label in labels:
            decision = _cluster_decision(chunk_index, label, cluster_names, used_names)
            if decision is None:
                decision = _unique_memory_name(label, memory, audio_path, chunk.turns, used_names)
            if decision is None:
                resolved_name = _next_generated_name(used_names, next_person_index)
                next_person_index = _generated_index_after(resolved_name)
                decision = SpeakerIdentityDecision(
                    source_speaker=label,
                    resolved_name=resolved_name,
                    confidence=0.0,
                    reason="sin coincidencia fiable; nombre estable generado",
                )

            mapping[label] = decision.resolved_name
            used_names.add(decision.resolved_name)
            decisions.append(decision)

        resolutions.append(
            ChunkSpeakerResolution(
                output_dir=chunk.output_dir,
                mapping=mapping,
                decisions=tuple(decisions),
            )
        )
    return resolutions


def apply_chunk_speaker_resolution(
    turns: list[ConversationTurn],
    resolution: ChunkSpeakerResolution,
) -> list[ConversationTurn]:
    return rename_speakers(turns, resolution.mapping)


def write_identity_decisions(output_dir: Path, decisions: tuple[SpeakerIdentityDecision, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "decisions": [
            {
                "source_speaker": decision.source_speaker,
                "resolved_name": decision.resolved_name,
                "confidence": decision.confidence,
                "reason": decision.reason,
            }
            for decision in decisions
        ]
    }
    (output_dir / DECISIONS_FILENAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _memory_embedding_candidates(
    memory: SpeakerMemory,
    audio_path: Path,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    return {
        identity.name: identity.embeddings
        for identity in memory.identities_for(audio_path)
        if identity.embeddings
    }


def _match_memory(
    label: str,
    embedding: tuple[float, ...] | None,
    candidates: dict[str, tuple[tuple[float, ...], ...]],
    used_names: set[str],
) -> SpeakerIdentityDecision | None:
    if embedding is None:
        return None
    name, score = _best_match(embedding, candidates, used_names)
    if name is None or score < MEMORY_THRESHOLD:
        return None
    return SpeakerIdentityDecision(label, name, score, "coincidencia alta con memoria validada")


def _cluster_decision(
    chunk_index: int,
    label: str,
    cluster_names: dict[tuple[int, str], tuple[str, float, str]],
    used_names: set[str],
) -> SpeakerIdentityDecision | None:
    cluster = cluster_names.get((chunk_index, label))
    if cluster is None:
        return None
    name, score, reason = cluster
    if name in used_names:
        return None
    return SpeakerIdentityDecision(label, name, score, reason)


def _cluster_observations(chunks: list[ChunkSpeakerEvidence]) -> list[_Cluster]:
    clusters: list[_Cluster] = []
    for chunk_index, chunk in enumerate(chunks):
        for label, embedding in chunk.embeddings.items():
            observation = _Observation(chunk_index, label, embedding)
            best_index: int | None = None
            best_score = RUN_THRESHOLD
            for index, cluster in enumerate(clusters):
                score = cosine_similarity(embedding, cluster.embedding)
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index is None:
                clusters.append(_Cluster(None, embedding, (observation,)))
                continue
            cluster = clusters[best_index]
            clusters[best_index] = _Cluster(
                cluster.name,
                _merge_embedding(cluster.embedding, embedding),
                (*cluster.observations, observation),
            )
    return clusters


def _final_cluster_names(
    clusters: list[_Cluster],
    memory_candidates: dict[str, tuple[tuple[float, ...], ...]],
) -> dict[tuple[int, str], tuple[str, float, str]]:
    result: dict[tuple[int, str], tuple[str, float, str]] = {}
    used_generated: set[str] = set()
    next_index = 1
    for cluster in clusters:
        memory_name, memory_score = _best_match(cluster.embedding, memory_candidates, set())
        if memory_name is not None and memory_score >= MEMORY_THRESHOLD:
            name = memory_name
            confidence = memory_score
            reason = "coincidencia alta con memoria validada en pasada final"
        else:
            name = _next_generated_name(used_generated, next_index)
            next_index = _generated_index_after(name)
            used_generated.add(name)
            confidence = 0.82 if len(cluster.observations) > 1 else 0.0
            reason = (
                "coincidencia alta con otra porcion en pasada final"
                if len(cluster.observations) > 1
                else "sin coincidencia fiable; nombre estable generado"
            )
        for observation in cluster.observations:
            result[(observation.chunk_index, observation.label)] = (name, confidence, reason)
    return result


def _unique_memory_name(
    label: str,
    memory: SpeakerMemory,
    audio_path: Path,
    turns: list[ConversationTurn],
    used_names: set[str],
) -> SpeakerIdentityDecision | None:
    mapping = build_unique_name_mapping(memory, audio_path, turns)
    name = mapping.get(label)
    if name is None or name in used_names:
        return None
    return SpeakerIdentityDecision(label, name, 0.65, "reutilizacion conservadora por numero exacto de hablantes")


def _best_match(
    embedding: tuple[float, ...],
    candidates: dict[str, tuple[tuple[float, ...], ...]],
    used_names: set[str],
) -> tuple[str | None, float]:
    best_name: str | None = None
    best_score = 0.0
    for name, embeddings in candidates.items():
        if name in used_names:
            continue
        for candidate in embeddings:
            score = cosine_similarity(embedding, candidate)
            if score > best_score:
                best_name = name
                best_score = score
    return best_name, best_score


def _next_generated_name(used_names: set[str], start_index: int) -> str:
    index = start_index
    while f"Persona {index}" in used_names:
        index += 1
    return f"Persona {index}"


def _generated_index_after(name: str) -> int:
    try:
        return int(name.split()[-1]) + 1
    except (IndexError, ValueError):
        return 1


def _next_available_generated_index(names: set[str]) -> int:
    index = 1
    while f"Persona {index}" in names:
        index += 1
    return index


def _merge_embedding(
    existing: tuple[float, ...] | None,
    new: tuple[float, ...],
) -> tuple[float, ...]:
    if existing is None or len(existing) != len(new):
        return new
    return tuple((left + right) / 2.0 for left, right in zip(existing, new))
