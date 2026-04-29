import json
import unittest
from pathlib import Path
from unittest.mock import patch

from meeting_transcriber.speaker_ai import (
    _run_opencode,
    build_speaker_identification_prompt,
    parse_speaker_mapping_response,
)
from meeting_transcriber.types import ConversationTurn


class SpeakerAiTests(unittest.TestCase):
    def test_prompt_requests_json_and_includes_transcript(self):
        prompt = build_speaker_identification_prompt(
            [ConversationTurn(start=1, end=2, speaker="Persona 1", text="Jo em dic Núria.")]
        )

        self.assertIn("Devuelve SOLO un JSON valido", prompt)
        self.assertIn("Persona 1", prompt)
        self.assertIn("Jo em dic Núria.", prompt)

    def test_parse_mapping_accepts_structured_response(self):
        response = json.dumps(
            {
                "speakers": {
                    "Persona 1": {
                        "name": "Núria Riquelme",
                        "confidence": "alta",
                        "evidence": "Jo em dic Núria Riquelme",
                    },
                    "Persona 2": {"name": None, "confidence": "baja", "evidence": "sin pista"},
                }
            }
        )

        self.assertEqual(
            parse_speaker_mapping_response(response),
            {"Persona 1": "Núria Riquelme"},
        )

    def test_parse_mapping_accepts_fenced_json(self):
        response = '```json\n{"speakers":{"Persona 3":"Jordi Roure"}}\n```'

        self.assertEqual(
            parse_speaker_mapping_response(response),
            {"Persona 3": "Jordi Roure"},
        )

    def test_opencode_message_is_not_consumed_as_file_argument(self):
        captured = {}

        def fake_run(command, stdin_text, cancelled, label):
            captured["command"] = command
            captured["stdin_text"] = stdin_text
            captured["cancelled"] = cancelled
            captured["label"] = label
            return "{}"

        with patch("meeting_transcriber.speaker_ai._run_cancellable", fake_run):
            _run_opencode("opencode", Path("/tmp/prompt.md"), "/tmp/out", None)

        command = captured["command"]
        self.assertLess(
            command.index("Lee el archivo adjunto y responde SOLO con el JSON solicitado."),
            command.index("--file"),
        )
        self.assertEqual(command[command.index("--file") + 1], "/tmp/prompt.md")


if __name__ == "__main__":
    unittest.main()
