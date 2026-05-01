import json
import tempfile
import unittest
from pathlib import Path

from meeting_transcriber.config import UiState, load_config, load_ui_state, save_config
from meeting_transcriber.types import ProcessingConfig
from meeting_transcriber.diarization_models import DEFAULT_DIARIZATION_MODEL
from meeting_transcriber.whisper_models import DEFAULT_WHISPER_MODEL


class ConfigTests(unittest.TestCase):
    def test_load_config_migrates_old_manual_model_paths_to_automatic_models(self):
        old_payload = {
            "whisper_model_path": "/tmp/project/models/whisper",
            "diarization_model_path": "/tmp/project/models/diarization",
            "ffmpeg_path": "/tmp/project/.venv/bin/ffmpeg",
            "language": "ca",
            "min_speakers": None,
            "max_speakers": None,
            "device": "cpu",
            "compute_type": "int8",
            "export_speaker_audio": False,
        }
        with tempfile.TemporaryDirectory() as dirname:
            config_path = Path(dirname) / "config.json"
            config_path.write_text(json.dumps(old_payload), encoding="utf-8")

            config = load_config(config_path)

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.whisper_model, DEFAULT_WHISPER_MODEL)
        self.assertEqual(config.diarization_model, DEFAULT_DIARIZATION_MODEL)
        self.assertEqual(config.language, "ca")
        self.assertFalse(config.normalize_audio)

    def test_save_config_persists_last_audio_dir(self):
        processing_config = ProcessingConfig(
            whisper_model=DEFAULT_WHISPER_MODEL,
            diarization_model=DEFAULT_DIARIZATION_MODEL,
            huggingface_token=None,
            ffmpeg_path=None,
            language="ca",
            min_speakers=None,
            max_speakers=None,
            device="cpu",
            compute_type="int8",
            export_speaker_audio=False,
        )
        with tempfile.TemporaryDirectory() as dirname:
            config_path = Path(dirname) / "config.json"

            save_config(config_path, processing_config, UiState(last_audio_dir=Path("/tmp/audios")))
            state = load_ui_state(config_path)

        self.assertEqual(state.last_audio_dir, Path("/tmp/audios"))

    def test_save_and_load_round_trips_audio_normalization(self):
        processing_config = ProcessingConfig(
            whisper_model=DEFAULT_WHISPER_MODEL,
            diarization_model=DEFAULT_DIARIZATION_MODEL,
            huggingface_token=None,
            ffmpeg_path=None,
            language="ca",
            min_speakers=None,
            max_speakers=None,
            device="cpu",
            compute_type="int8",
            export_speaker_audio=False,
            normalize_audio=True,
        )
        with tempfile.TemporaryDirectory() as dirname:
            config_path = Path(dirname) / "config.json"

            save_config(config_path, processing_config)
            loaded = load_config(config_path)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(loaded.normalize_audio)


if __name__ == "__main__":
    unittest.main()
