import os
import unittest
from unittest.mock import patch

from meeting_transcriber.hf_auth import apply_huggingface_token


class HuggingFaceAuthTests(unittest.TestCase):
    def test_apply_huggingface_token_sets_common_hub_env_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            apply_huggingface_token("hf_example")

            self.assertEqual(os.environ["HF_TOKEN"], "hf_example")
            self.assertEqual(os.environ["HUGGING_FACE_HUB_TOKEN"], "hf_example")

    def test_apply_huggingface_token_ignores_empty_token(self):
        with patch.dict(os.environ, {}, clear=True):
            apply_huggingface_token(None)

            self.assertNotIn("HF_TOKEN", os.environ)


if __name__ == "__main__":
    unittest.main()
