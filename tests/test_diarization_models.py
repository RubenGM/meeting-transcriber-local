import unittest

from meeting_transcriber.diarization_models import (
    DEFAULT_DIARIZATION_MODEL,
    diarization_model_id_from_display_name,
    diarization_model_label_from_id,
)


class DiarizationModelTests(unittest.TestCase):
    def test_default_diarization_model_is_automatic(self):
        self.assertEqual(DEFAULT_DIARIZATION_MODEL, "pyannote/speaker-diarization-community-1")

    def test_diarization_model_id_from_display_name_returns_repo_id(self):
        self.assertEqual(
            diarization_model_id_from_display_name("Automático"),
            "pyannote/speaker-diarization-community-1",
        )

    def test_diarization_model_id_from_display_name_accepts_existing_repo_id(self):
        self.assertEqual(
            diarization_model_id_from_display_name("pyannote/speaker-diarization-community-1"),
            "pyannote/speaker-diarization-community-1",
        )

    def test_diarization_model_label_from_id_returns_human_label(self):
        self.assertEqual(
            diarization_model_label_from_id("pyannote/speaker-diarization-community-1"),
            "Automático",
        )


if __name__ == "__main__":
    unittest.main()
