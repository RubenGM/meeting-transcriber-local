import unittest

from meeting_transcriber.diarization import (
    _annotation_from_diarization_output,
    _friendly_runtime_error,
    explain_diarization_error,
)


class DiarizationTests(unittest.TestCase):
    def test_explain_diarization_error_mentions_token_for_gated_repo(self):
        message = explain_diarization_error(Exception("401 Cannot access gated repo"))

        self.assertIn("Token HF", message)
        self.assertIn("Abrir modelo pyannote", message)

    def test_annotation_from_diarize_output_uses_exclusive_diarization(self):
        exclusive = FakeAnnotation()
        output = FakeDiarizeOutput(exclusive_speaker_diarization=exclusive)

        self.assertIs(_annotation_from_diarization_output(output), exclusive)

    def test_annotation_from_diarize_output_accepts_legacy_annotation(self):
        annotation = FakeAnnotation()

        self.assertIs(_annotation_from_diarization_output(annotation), annotation)

    def test_friendly_runtime_error_hides_nvrtc_kernel_dump(self):
        message = _friendly_runtime_error(
            RuntimeError("#ifdef __HIPCC__\nnvrtc: error: failed to open libnvrtc-builtins.so.13.0")
        )

        self.assertIn("NVRTC", message)
        self.assertNotIn("#ifdef", message)


class FakeAnnotation:
    def itertracks(self, yield_label=False):
        return iter(())


class FakeDiarizeOutput:
    def __init__(self, exclusive_speaker_diarization):
        self.exclusive_speaker_diarization = exclusive_speaker_diarization
        self.speaker_diarization = FakeAnnotation()
