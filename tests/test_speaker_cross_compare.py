import unittest
from pathlib import Path

from meeting_transcriber.speaker_cross_compare import (
    SpeakerSource,
    build_speaker_profiles,
    compare_speaker_profiles,
    name_coherence_matrix,
)
from meeting_transcriber.types import ConversationTurn


class SpeakerCrossCompareTests(unittest.TestCase):
    def test_build_speaker_profiles_groups_turns_and_metrics(self):
        source = SpeakerSource("a", Path("/out/a"), "00:00 -> 00:05")
        turns = [
            ConversationTurn(0, 4, "Persona 1", "Hola."),
            ConversationTurn(5, 8, "Persona 1", "Seguim."),
            ConversationTurn(10, 12, "Marta", "Bon dia."),
        ]

        profiles = build_speaker_profiles(source, turns, embeddings={"Persona 1": (1.0, 0.0)})

        self.assertEqual([profile.label for profile in profiles], ["Persona 1", "Marta"])
        self.assertEqual(profiles[0].total_seconds, 7.0)
        self.assertEqual(profiles[0].turn_count, 2)
        self.assertEqual(profiles[0].sample, "Hola.")
        self.assertEqual(profiles[0].embedding, (1.0, 0.0))

    def test_compare_profiles_detects_same_voice_different_name(self):
        base_source = SpeakerSource("a", Path("/out/a"), "A")
        ref_source = SpeakerSource("b", Path("/out/b"), "B")
        base = build_speaker_profiles(
            base_source,
            [ConversationTurn(0, 5, "Persona 1", "Hola.")],
            embeddings={"Persona 1": (1.0, 0.0)},
        )
        refs = build_speaker_profiles(
            ref_source,
            [ConversationTurn(0, 5, "Marta", "Hola.")],
            embeddings={"Marta": (0.99, 0.01)},
        )

        matches = compare_speaker_profiles(base, refs)

        self.assertEqual(matches[0].candidate.label, "Marta")
        self.assertEqual(matches[0].status, "Coincidencia alta")
        self.assertEqual(matches[0].name_status, "Conflicto de nombre")

    def test_compare_profiles_detects_same_name_different_voice(self):
        source = SpeakerSource("a", Path("/out/a"), "A")
        base = build_speaker_profiles(
            source,
            [ConversationTurn(0, 5, "Marta", "Hola.")],
            embeddings={"Marta": (1.0, 0.0)},
        )
        refs = build_speaker_profiles(
            SpeakerSource("b", Path("/out/b"), "B"),
            [ConversationTurn(0, 5, "Marta", "Adeu.")],
            embeddings={"Marta": (0.0, 1.0)},
        )

        matches = compare_speaker_profiles(base, refs)

        self.assertEqual(matches[0].status, "Coincidencia baja")
        self.assertEqual(matches[0].name_status, "Mismo nombre con voz distinta")

    def test_compare_profiles_reports_missing_embeddings(self):
        source = SpeakerSource("a", Path("/out/a"), "A")
        base = build_speaker_profiles(source, [ConversationTurn(0, 5, "Persona 1", "Hola.")])
        refs = build_speaker_profiles(source, [ConversationTurn(0, 5, "Marta", "Hola.")])

        matches = compare_speaker_profiles(base, refs)

        self.assertEqual(matches[0].status, "Sin huellas disponibles")
        self.assertIsNone(matches[0].candidate)

    def test_name_coherence_matrix_groups_by_candidate_voice(self):
        source_a = SpeakerSource("a", Path("/out/a"), "A")
        source_b = SpeakerSource("b", Path("/out/b"), "B")
        profiles = [
            *build_speaker_profiles(
                source_a,
                [ConversationTurn(0, 5, "Núria", "Hola.")],
                embeddings={"Núria": (1.0, 0.0)},
            ),
            *build_speaker_profiles(
                source_b,
                [ConversationTurn(0, 5, "Persona 1", "Hola.")],
                embeddings={"Persona 1": (0.99, 0.01)},
            ),
        ]

        rows = name_coherence_matrix(profiles)

        self.assertEqual(rows[0].diagnosis, "falta aplicar nombre")
        self.assertEqual(rows[0].names_by_source, {"A": "Núria", "B": "Persona 1"})


if __name__ == "__main__":
    unittest.main()
