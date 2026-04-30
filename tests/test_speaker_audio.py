import tempfile
import unittest
from pathlib import Path

from meeting_transcriber.audio import _parse_ffmpeg_duration, build_speaker_extract_plan
from meeting_transcriber.types import ConversationTurn


class SpeakerAudioTests(unittest.TestCase):
    def test_parse_ffmpeg_duration_from_metadata_output(self):
        stderr = "Input #0\n  Duration: 01:02:03.45, start: 0.000000, bitrate: 128 kb/s\n"

        self.assertEqual(_parse_ffmpeg_duration(stderr), 3723.45)

    def test_build_speaker_extract_plan_groups_segments_by_speaker(self):
        audio = Path("meeting.wav")
        turns = [
            ConversationTurn(start=0.0, end=2.0, speaker="Persona 1", text="A"),
            ConversationTurn(start=2.0, end=5.0, speaker="Persona 2", text="B"),
            ConversationTurn(start=5.0, end=6.5, speaker="Persona 1", text="C"),
        ]
        with tempfile.TemporaryDirectory() as dirname:
            output_dir = Path(dirname)

            plan = build_speaker_extract_plan(audio, turns, output_dir)

            self.assertEqual(sorted(plan.keys()), ["Persona 1", "Persona 2"])
            self.assertEqual(plan["Persona 1"].output_path, output_dir / "Persona_1.wav")
            self.assertEqual(
                [(s.start, s.end) for s in plan["Persona 1"].segments],
                [(0.0, 2.0), (5.0, 6.5)],
            )
            self.assertEqual(plan["Persona 2"].output_path, output_dir / "Persona_2.wav")


if __name__ == "__main__":
    unittest.main()
