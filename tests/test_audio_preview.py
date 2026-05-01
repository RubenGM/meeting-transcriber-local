import unittest
from pathlib import Path

from meeting_transcriber.audio_preview import preview_clip_path


class AudioPreviewTests(unittest.TestCase):
    def test_preview_clip_path_is_stable_and_safe(self):
        path = preview_clip_path(
            Path("/tmp/cache"),
            Path("/audio/Taula Institucional 18 03 26.m4a"),
            start_seconds=1.234,
            end_seconds=5.678,
        )

        self.assertEqual(
            path,
            Path("/tmp/cache") / "Taula_Institucional_18_03_26_00-00-01_234_to_00-00-05_678.wav",
        )


if __name__ == "__main__":
    unittest.main()
