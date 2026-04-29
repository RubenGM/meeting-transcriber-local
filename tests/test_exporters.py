import json
import tempfile
import unittest
from pathlib import Path

from meeting_transcriber.exporters import (
    export_json_text,
    export_markdown_text,
    export_srt_text,
    write_all_exports,
)
from meeting_transcriber.types import ConversationTurn


class ExporterTests(unittest.TestCase):
    def test_export_markdown_groups_turns_with_timestamps_and_speakers(self):
        turns = [
            ConversationTurn(start=1.2, end=3.4, speaker="Persona 1", text="Hola."),
            ConversationTurn(start=65.0, end=70.0, speaker="Persona 2", text="Seguimos."),
        ]

        content = export_markdown_text(turns)

        self.assertEqual(
            content,
            "# Transcripcion\n\n"
            "[00:00:01] **Persona 1:** Hola.\n\n"
            "[00:01:05] **Persona 2:** Seguimos.\n",
        )

    def test_export_json_contains_exact_turn_data(self):
        turns = [
            ConversationTurn(start=1.25, end=3.5, speaker="Persona 1", text="Hola."),
        ]

        data = json.loads(export_json_text(turns))

        self.assertEqual(
            data,
            {
                "turns": [
                    {
                        "start": 1.25,
                        "end": 3.5,
                        "speaker": "Persona 1",
                        "text": "Hola.",
                    }
                ]
            },
        )

    def test_export_srt_formats_subtitle_blocks(self):
        turns = [
            ConversationTurn(start=1.2, end=3.45, speaker="Persona 1", text="Hola."),
            ConversationTurn(start=65.0, end=70.0, speaker="Persona 2", text="Seguimos."),
        ]

        content = export_srt_text(turns)

        self.assertEqual(
            content,
            "1\n"
            "00:00:01,200 --> 00:00:03,450\n"
            "Persona 1: Hola.\n\n"
            "2\n"
            "00:01:05,000 --> 00:01:10,000\n"
            "Persona 2: Seguimos.\n",
        )

    def test_write_all_exports_accepts_custom_basename(self):
        turns = [ConversationTurn(start=1.2, end=3.45, speaker="Sin diarizar", text="Hola.")]

        with tempfile.TemporaryDirectory() as temp_dir:
            write_all_exports(Path(temp_dir), turns, basename="transcript_raw")

            self.assertTrue((Path(temp_dir) / "transcript_raw.md").exists())
            self.assertTrue((Path(temp_dir) / "transcript_raw.txt").exists())
            self.assertTrue((Path(temp_dir) / "transcript_raw.json").exists())
            self.assertTrue((Path(temp_dir) / "transcript_raw.srt").exists())


if __name__ == "__main__":
    unittest.main()
