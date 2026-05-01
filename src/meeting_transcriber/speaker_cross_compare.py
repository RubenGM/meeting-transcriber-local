from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from meeting_transcriber.speaker_fingerprints import cosine_similarity
from meeting_transcriber.speaker_names import speaker_labels
from meeting_transcriber.types import ConversationTurn


@dataclass(frozen=True)
class SpeakerSource:
    entry_id: str
    output_dir: Path
    range_label: str


@dataclass(frozen=True)
class SpeakerProfile:
    source: SpeakerSource
    label: str
    display_name: str
    total_seconds: float
    turn_count: int
    sample: str
    sample_start: float | None = None
    sample_end: float | None = None
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SpeakerMatch:
    base: SpeakerProfile
    candidate: SpeakerProfile | None
    score: float | None
    status: str
    name_status: str
    ranked_candidates: tuple[SpeakerCandidateScore, ...] = ()


@dataclass(frozen=True)
class SpeakerCandidateScore:
    profile: SpeakerProfile
    score: float


@dataclass(frozen=True)
class NameCoherenceRow:
    cluster_id: str
    names_by_source: dict[str, str]
    diagnosis: str


def build_speaker_profiles(
    source: SpeakerSource,
    turns: list[ConversationTurn],
    *,
    embeddings: dict[str, tuple[float, ...]] | None = None,
) -> list[SpeakerProfile]:
    embeddings = embeddings or {}
    profiles = []
    for speaker in speaker_labels(turns):
        speaker_turns = [turn for turn in turns if turn.speaker == speaker]
        profiles.append(
            SpeakerProfile(
                source=source,
                label=speaker,
                display_name=speaker,
                total_seconds=sum(max(0.0, turn.end - turn.start) for turn in speaker_turns),
                turn_count=len(speaker_turns),
                sample=_first_sample(speaker_turns),
                sample_start=speaker_turns[0].start if speaker_turns else None,
                sample_end=speaker_turns[0].end if speaker_turns else None,
                embedding=embeddings.get(speaker),
            )
        )
    return profiles


def compare_speaker_profiles(
    base_profiles: list[SpeakerProfile],
    candidate_profiles: list[SpeakerProfile],
) -> list[SpeakerMatch]:
    matches = []
    for base in base_profiles:
        if base.embedding is None:
            matches.append(SpeakerMatch(base, None, None, "Sin huellas disponibles", "pendiente"))
            continue
        candidates_with_embeddings = [candidate for candidate in candidate_profiles if candidate.embedding is not None]
        if not candidates_with_embeddings:
            matches.append(SpeakerMatch(base, None, None, "Sin huellas disponibles", "pendiente"))
            continue
        ranked_candidates = _rank_candidates(base, candidates_with_embeddings)
        candidate = ranked_candidates[0].profile
        score = ranked_candidates[0].score
        matches.append(
            SpeakerMatch(
                base=base,
                candidate=candidate,
                score=score,
                status=_score_status(score),
                name_status=_name_status(base, candidate, score),
                ranked_candidates=tuple(ranked_candidates),
            )
        )
    return matches


def explain_speaker_match(match: SpeakerMatch, *, compared_count: int) -> str:
    if match.candidate is None or match.score is None:
        return (
            f"{match.base.display_name} no se ha podido comparar porque faltan huellas de voz "
            "en la salida base o en las referencias seleccionadas."
        )
    return (
        f"{match.base.display_name} se ha comparado contra {compared_count} voces. "
        f"La voz mas parecida es {match.candidate.display_name} en {match.candidate.source.range_label} "
        f"con similitud {_format_score(match.score)}. "
        f"Diagnostico: {match.name_status}. "
        f"Sugerencia: {_suggested_action(match)}"
    )


def name_coherence_matrix(profiles: list[SpeakerProfile], *, threshold: float = 0.85) -> list[NameCoherenceRow]:
    clusters: list[list[SpeakerProfile]] = []
    for profile in profiles:
        cluster = _matching_cluster(profile, clusters, threshold)
        if cluster is None:
            clusters.append([profile])
        else:
            cluster.append(profile)
    rows = []
    for index, cluster in enumerate(clusters, start=1):
        names_by_source = {profile.source.range_label: profile.display_name for profile in cluster}
        rows.append(
            NameCoherenceRow(
                cluster_id=f"Cluster {index}",
                names_by_source=names_by_source,
                diagnosis=_cluster_diagnosis(cluster),
            )
        )
    return rows


def _best_candidate(
    base: SpeakerProfile,
    candidates: list[SpeakerProfile],
) -> tuple[SpeakerProfile | None, float | None]:
    ranked = _rank_candidates(base, candidates)
    if not ranked:
        return None, None
    return ranked[0].profile, ranked[0].score


def _rank_candidates(base: SpeakerProfile, candidates: list[SpeakerProfile]) -> list[SpeakerCandidateScore]:
    ranked = []
    assert base.embedding is not None
    for candidate in candidates:
        assert candidate.embedding is not None
        score = cosine_similarity(base.embedding, candidate.embedding)
        ranked.append(SpeakerCandidateScore(candidate, score))
    return sorted(ranked, key=lambda candidate: candidate.score, reverse=True)


def _score_status(score: float) -> str:
    if score >= 0.85:
        return "Coincidencia alta"
    if score >= 0.65:
        return "Coincidencia media"
    return "Coincidencia baja"


def _name_status(base: SpeakerProfile, candidate: SpeakerProfile, score: float) -> str:
    if base.display_name == candidate.display_name and score >= 0.85:
        return "ok"
    if base.display_name == candidate.display_name:
        return "Mismo nombre con voz distinta"
    if score >= 0.85:
        return "Conflicto de nombre"
    return "pendiente"


def _suggested_action(match: SpeakerMatch) -> str:
    if match.candidate is None:
        return "genera huellas o selecciona otra referencia."
    if match.status == "Coincidencia alta" and match.base.display_name != match.candidate.display_name:
        return f"aplicar {match.candidate.display_name} si la muestra de audio lo confirma."
    if match.status == "Coincidencia alta":
        return "mantener el nombre actual."
    if match.name_status == "Mismo nombre con voz distinta":
        return "revisar manualmente antes de guardar."
    return "revisar manualmente; la similitud no es suficientemente alta."


def _format_score(score: float) -> str:
    return f"{score:.2f}"


def _matching_cluster(
    profile: SpeakerProfile,
    clusters: list[list[SpeakerProfile]],
    threshold: float,
) -> list[SpeakerProfile] | None:
    if profile.embedding is None:
        return None
    for cluster in clusters:
        for candidate in cluster:
            if candidate.embedding is None:
                continue
            if cosine_similarity(profile.embedding, candidate.embedding) >= threshold:
                return cluster
    return None


def _cluster_diagnosis(cluster: list[SpeakerProfile]) -> str:
    names = {profile.display_name for profile in cluster}
    generic_names = {name for name in names if _is_generic_name(name)}
    specific_names = names - generic_names
    if len(specific_names) == 1 and generic_names:
        return "falta aplicar nombre"
    if len(specific_names) > 1:
        return "posible mezcla/conflicto"
    return "ok"


def _is_generic_name(name: str) -> bool:
    return name.startswith("Persona ") or name.startswith("SPEAKER_")


def _first_sample(turns: list[ConversationTurn]) -> str:
    for turn in turns:
        text = turn.text.strip()
        if text:
            return text[:120] + ("..." if len(text) > 120 else "")
    return ""
