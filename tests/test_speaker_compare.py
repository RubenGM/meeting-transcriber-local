import unittest
from pathlib import Path

from meeting_transcriber.speaker_compare import summarize_speaker_comparison
from meeting_transcriber.speaker_memory import SpeakerIdentity, SpeakerMemory
from meeting_transcriber.types import ConversationTurn


class SpeakerCompareTests(unittest.TestCase):
    def test_summarize_speaker_comparison_marks_known_names_and_suggestions(self):
        turns = [
            ConversationTurn(start=0, end=4, speaker="Persona 1", text="Hola."),
            ConversationTurn(start=4, end=10, speaker="Persona 1", text="Seguimos."),
            ConversationTurn(start=10, end=13, speaker="Jordi", text="Soc en Jordi."),
        ]
        memory = SpeakerMemory(
            audios={
                "/audio/a.m4a": [
                    SpeakerIdentity(name="Jordi", sample_ranges=((20, 30),)),
                    SpeakerIdentity(name="Nuria", sample_ranges=((30, 40),)),
                ]
            }
        )

        summary = summarize_speaker_comparison(
            turns,
            memory,
            Path("/audio/a.m4a"),
            suggested_names={"Persona 1": "Nuria"},
        )

        self.assertEqual(summary.known_names, ("Jordi", "Nuria"))
        self.assertEqual(len(summary.rows), 2)
        self.assertEqual(summary.rows[0].speaker, "Persona 1")
        self.assertEqual(summary.rows[0].suggested_name, "Nuria")
        self.assertFalse(summary.rows[0].already_known_name)
        self.assertEqual(summary.rows[0].turn_count, 2)
        self.assertEqual(summary.rows[0].total_seconds, 10.0)
        self.assertEqual(summary.rows[1].speaker, "Jordi")
        self.assertTrue(summary.rows[1].already_known_name)


if __name__ == "__main__":
    unittest.main()
