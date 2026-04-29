import os
import tempfile
import unittest
from pathlib import Path

from meeting_transcriber.runtime import configure_runtime_environment


class RuntimeTests(unittest.TestCase):
    def test_configure_runtime_environment_sets_matplotlib_cache_when_missing(self):
        with tempfile.TemporaryDirectory() as dirname:
            previous = os.environ.pop("MPLCONFIGDIR", None)
            previous_hf = os.environ.pop("HF_HOME", None)
            try:
                result = configure_runtime_environment(Path(dirname))

                self.assertEqual(result.matplotlib_cache, Path(dirname) / ".cache" / "matplotlib")
                self.assertEqual(result.huggingface_cache, Path(dirname) / "models" / "huggingface")
                self.assertEqual(os.environ["MPLCONFIGDIR"], str(result.matplotlib_cache))
                self.assertEqual(os.environ["HF_HOME"], str(result.huggingface_cache))
                self.assertTrue(result.matplotlib_cache.exists())
                self.assertTrue(result.huggingface_cache.exists())
            finally:
                if previous is not None:
                    os.environ["MPLCONFIGDIR"] = previous
                else:
                    os.environ.pop("MPLCONFIGDIR", None)
                if previous_hf is not None:
                    os.environ["HF_HOME"] = previous_hf
                else:
                    os.environ.pop("HF_HOME", None)


if __name__ == "__main__":
    unittest.main()
