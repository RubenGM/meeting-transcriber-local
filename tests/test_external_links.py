import unittest

from meeting_transcriber.external_links import HUGGINGFACE_TOKENS_URL, PYANNOTE_MODEL_URL


class ExternalLinksTests(unittest.TestCase):
    def test_pyannote_model_url_points_to_exact_gated_model(self):
        self.assertEqual(
            PYANNOTE_MODEL_URL,
            "https://huggingface.co/pyannote/speaker-diarization-community-1",
        )

    def test_huggingface_tokens_url_points_to_token_settings(self):
        self.assertEqual(HUGGINGFACE_TOKENS_URL, "https://huggingface.co/settings/tokens")


if __name__ == "__main__":
    unittest.main()
