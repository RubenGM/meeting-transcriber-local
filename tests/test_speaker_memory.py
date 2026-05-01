import tempfile
import unittest
from pathlib import Path

from meeting_transcriber.speaker_memory import (
    SpeakerIdentity,
    SpeakerMemory,
    add_identity_samples,
    build_unique_name_mapping,
    build_embedding_name_mapping,
    format_speaker_memory_status,
    identity_names,
    load_speaker_memory,
    remember_validated_turns,
    save_speaker_memory,
    speaker_memory_status,
)
from meeting_transcriber.types import ConversationTurn


class SpeakerMemoryTests(unittest.TestCase):
    def test_add_identity_samples_persists_validated_speaker_for_audio(self):
        turns = [
            ConversationTurn(start=0, end=5, speaker="Ruben", text="Hola."),
            ConversationTurn(start=10, end=20, speaker="Ruben", text="Seguimos."),
            ConversationTurn(start=20, end=22, speaker="Nuria", text="Bon dia."),
        ]

        memory = add_identity_samples(SpeakerMemory(audios={}), Path("/audio/a.m4a"), turns)

        identities = memory.identities_for(Path("/audio/a.m4a"))
        self.assertEqual([identity.name for identity in identities], ["Nuria", "Ruben"])
        self.assertEqual(identities[1].sample_ranges, ((0, 5), (10, 20)))

    def test_save_and_load_round_trips_memory(self):
        memory = SpeakerMemory(
            audios={
                "/audio/a.m4a": [
                    SpeakerIdentity(name="Ruben", sample_ranges=((0, 5),), embeddings=())
                ]
            }
        )

        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "speaker_memory.json"
            save_speaker_memory(path, memory)
            loaded = load_speaker_memory(path)

        self.assertEqual(loaded, memory)

    def test_remember_validated_turns_saves_memory_file(self):
        turns = [ConversationTurn(start=0, end=5, speaker="Ruben", text="Hola.")]

        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "memory.json"
            remember_validated_turns(path, Path("/audio/a.m4a"), turns)
            loaded = load_speaker_memory(path)

        self.assertEqual(loaded.identities_for(Path("/audio/a.m4a"))[0].name, "Ruben")

    def test_remember_validated_turns_saves_embeddings_by_name(self):
        turns = [ConversationTurn(start=0, end=5, speaker="Ruben", text="Hola.")]

        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "memory.json"
            remember_validated_turns(
                path,
                Path("/audio/a.m4a"),
                turns,
                embeddings_by_name={"Ruben": ((0.9, 0.1),)},
            )
            loaded = load_speaker_memory(path)

        self.assertEqual(
            loaded.identities_for(Path("/audio/a.m4a"))[0].embeddings,
            ((0.9, 0.1),),
        )

    def test_identity_names_returns_known_names_for_audio(self):
        memory = SpeakerMemory(
            audios={
                "/audio/a.m4a": [
                    SpeakerIdentity(name="Ruben", sample_ranges=((0, 5),)),
                    SpeakerIdentity(name="Nuria", sample_ranges=((10, 15),)),
                ]
            }
        )

        self.assertEqual(identity_names(memory, Path("/audio/a.m4a")), ["Ruben", "Nuria"])

    def test_speaker_memory_status_counts_names_and_embeddings(self):
        memory = SpeakerMemory(
            audios={
                "/audio/a.m4a": [
                    SpeakerIdentity(name="Ruben", sample_ranges=((0, 5),), embeddings=((1.0, 0.0),)),
                    SpeakerIdentity(
                        name="Nuria",
                        sample_ranges=((10, 15),),
                        embeddings=((0.0, 1.0), (0.1, 0.9)),
                    ),
                ]
            }
        )

        status = speaker_memory_status(memory, Path("/audio/a.m4a"))

        self.assertEqual(status.names, ("Ruben", "Nuria"))
        self.assertEqual(status.embedding_count, 3)
        self.assertTrue(status.has_embeddings)

    def test_format_speaker_memory_status_shows_missing_memory(self):
        status = speaker_memory_status(SpeakerMemory(audios={}), Path("/audio/a.m4a"))

        self.assertEqual(
            format_speaker_memory_status(status),
            "Memoria: sin nombres validados para este audio.",
        )

    def test_build_unique_name_mapping_only_maps_when_counts_match(self):
        memory = SpeakerMemory(
            audios={
                "/audio/a.m4a": [
                    SpeakerIdentity(name="Ruben", sample_ranges=((0, 5),)),
                    SpeakerIdentity(name="Nuria", sample_ranges=((10, 15),)),
                ]
            }
        )
        turns = [
            ConversationTurn(start=20, end=25, speaker="Persona 1", text="Hola."),
            ConversationTurn(start=25, end=30, speaker="Persona 2", text="Bon dia."),
        ]

        self.assertEqual(
            build_unique_name_mapping(memory, Path("/audio/a.m4a"), turns),
            {"Persona 1": "Ruben", "Persona 2": "Nuria"},
        )

    def test_build_unique_name_mapping_refuses_ambiguous_count_mismatch(self):
        memory = SpeakerMemory(
            audios={
                "/audio/a.m4a": [
                    SpeakerIdentity(name="Ruben", sample_ranges=((0, 5),)),
                ]
            }
        )
        turns = [
            ConversationTurn(start=20, end=25, speaker="Persona 1", text="Hola."),
            ConversationTurn(start=25, end=30, speaker="Persona 2", text="Bon dia."),
        ]

        self.assertEqual(build_unique_name_mapping(memory, Path("/audio/a.m4a"), turns), {})

    def test_build_embedding_name_mapping_uses_stored_embeddings(self):
        memory = SpeakerMemory(
            audios={
                "/audio/a.m4a": [
                    SpeakerIdentity(name="Ruben", sample_ranges=((0, 5),), embeddings=((1.0, 0.0),)),
                    SpeakerIdentity(name="Nuria", sample_ranges=((10, 15),), embeddings=((0.0, 1.0),)),
                ]
            }
        )

        self.assertEqual(
            build_embedding_name_mapping(
                memory,
                Path("/audio/a.m4a"),
                {"Persona 3": (0.05, 0.95)},
                threshold=0.8,
            ),
            {"Persona 3": "Nuria"},
        )


if __name__ == "__main__":
    unittest.main()
