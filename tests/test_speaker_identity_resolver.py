import json
from pathlib import Path

from meeting_transcriber.speaker_identity_resolver import (
    DECISIONS_FILENAME,
    ChunkSpeakerEvidence,
    apply_chunk_speaker_resolution,
    resolve_chunk_speaker_identities,
    write_identity_decisions,
)
from meeting_transcriber.speaker_memory import SpeakerIdentity, SpeakerMemory
from meeting_transcriber.types import ConversationTurn


def test_resolver_prefers_validated_memory_embeddings():
    memory = SpeakerMemory(
        audios={
            "/audio/a.m4a": [
                SpeakerIdentity("Ruben", ((0, 5),), embeddings=((1.0, 0.0),)),
            ]
        }
    )
    chunk = ChunkSpeakerEvidence(
        output_dir=Path("/out/1"),
        turns=[ConversationTurn(0, 4, "Persona 1", "Hola")],
        embeddings={"Persona 1": (0.99, 0.01)},
    )

    resolution = resolve_chunk_speaker_identities(Path("/audio/a.m4a"), memory, [chunk])[0]

    assert resolution.mapping == {"Persona 1": "Ruben"}
    assert resolution.decisions[0].confidence >= 0.85


def test_resolver_reuses_run_profile_across_chunks():
    memory = SpeakerMemory(audios={})
    first = ChunkSpeakerEvidence(
        Path("/out/1"),
        [ConversationTurn(0, 4, "Persona 1", "Hola")],
        {"Persona 1": (1.0, 0.0)},
    )
    second = ChunkSpeakerEvidence(
        Path("/out/2"),
        [ConversationTurn(10, 14, "Persona 3", "Seguim")],
        {"Persona 3": (0.98, 0.02)},
    )

    resolutions = resolve_chunk_speaker_identities(Path("/audio/a.m4a"), memory, [first, second])

    assert resolutions[1].mapping == {"Persona 3": "Persona 1"}


def test_resolver_final_pass_renames_early_chunk_when_later_memory_match_identifies_cluster():
    memory = SpeakerMemory(
        audios={
            "/audio/a.m4a": [
                SpeakerIdentity("Ruben", ((10, 14),), embeddings=((0.98, 0.02),)),
            ]
        }
    )
    first = ChunkSpeakerEvidence(
        Path("/out/1"),
        [ConversationTurn(0, 4, "Persona 1", "Hola")],
        {"Persona 1": (1.0, 0.0)},
    )
    second = ChunkSpeakerEvidence(
        Path("/out/2"),
        [ConversationTurn(10, 14, "Persona 3", "Seguim")],
        {"Persona 3": (0.98, 0.02)},
    )

    resolutions = resolve_chunk_speaker_identities(Path("/audio/a.m4a"), memory, [first, second])

    assert resolutions[0].mapping == {"Persona 1": "Ruben"}
    assert resolutions[1].mapping == {"Persona 3": "Ruben"}


def test_apply_resolution_renames_turns():
    memory = SpeakerMemory(audios={"/audio/a.m4a": [SpeakerIdentity("Nuria", ((0, 2),), ())]})
    chunk = ChunkSpeakerEvidence(
        Path("/out/1"),
        [ConversationTurn(0, 4, "Persona 1", "Bon dia")],
        {},
    )
    resolution = resolve_chunk_speaker_identities(Path("/audio/a.m4a"), memory, [chunk])[0]

    updated = apply_chunk_speaker_resolution(chunk.turns, resolution)

    assert updated[0].speaker == "Nuria"


def test_write_identity_decisions_creates_auditable_sidecar(tmp_path):
    memory = SpeakerMemory(audios={})
    chunk = ChunkSpeakerEvidence(
        tmp_path,
        [ConversationTurn(0, 4, "Persona 1", "Hola")],
        {},
    )
    resolution = resolve_chunk_speaker_identities(Path("/audio/a.m4a"), memory, [chunk])[0]

    write_identity_decisions(tmp_path, resolution.decisions)

    payload = json.loads((tmp_path / DECISIONS_FILENAME).read_text(encoding="utf-8"))
    assert payload["decisions"][0]["source_speaker"] == "Persona 1"
