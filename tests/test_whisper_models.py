import unittest

from meeting_transcriber.whisper_models import (
    DEFAULT_WHISPER_MODEL,
    model_id_from_display_name,
    model_label_from_id,
)


class WhisperModelTests(unittest.TestCase):
    def test_default_model_is_balanced_for_non_technical_users(self):
        self.assertEqual(DEFAULT_WHISPER_MODEL, "small")

    def test_model_id_from_display_name_returns_faster_whisper_model_name(self):
        self.assertEqual(model_id_from_display_name("Rápido"), "base")
        self.assertEqual(model_id_from_display_name("Equilibrado"), "small")
        self.assertEqual(model_id_from_display_name("Preciso"), "medium")

    def test_model_id_from_display_name_accepts_existing_model_ids(self):
        self.assertEqual(model_id_from_display_name("small"), "small")

    def test_model_label_from_id_returns_human_label(self):
        self.assertEqual(model_label_from_id("small"), "Equilibrado")


if __name__ == "__main__":
    unittest.main()
