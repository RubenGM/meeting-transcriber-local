import unittest

from pathlib import Path

from meeting_transcriber.speaker_fingerprints import (
    best_embedding_match,
    cosine_similarity,
    extract_speaker_embeddings,
    is_cuda_embedding_error,
    match_speaker_embeddings,
    _coerce_embedding_output,
)
from meeting_transcriber.types import ConversationTurn


class SpeakerFingerprintTests(unittest.TestCase):
    def test_cosine_similarity_scores_identical_vectors_as_one(self):
        self.assertAlmostEqual(cosine_similarity((1, 0, 0), (1, 0, 0)), 1.0)

    def test_best_embedding_match_returns_name_above_threshold(self):
        candidates = {"Ruben": ((1, 0),), "Nuria": ((0, 1),)}

        self.assertEqual(best_embedding_match((0.95, 0.05), candidates, threshold=0.8), "Ruben")

    def test_best_embedding_match_returns_none_below_threshold(self):
        candidates = {"Ruben": ((1, 0),)}

        self.assertIsNone(best_embedding_match((0.1, 0.2), candidates, threshold=0.8))

    def test_extract_speaker_embeddings_averages_samples_by_speaker(self):
        turns = [
            ConversationTurn(start=0, end=3, speaker="Persona 1", text="Hola."),
            ConversationTurn(start=4, end=8, speaker="Persona 1", text="Seguimos."),
            ConversationTurn(start=9, end=10, speaker="Persona 2", text="Curt."),
        ]

        def extractor(_audio_path: Path, start: float, _end: float) -> tuple[float, ...] | None:
            if start == 0:
                return (1.0, 0.0)
            if start == 4:
                return (0.5, 0.5)
            return (0.0, 1.0)

        result = extract_speaker_embeddings(Path("/audio/a.m4a"), turns, extractor, min_duration_seconds=2.0)

        self.assertEqual(result, {"Persona 1": (0.75, 0.25)})

    def test_match_speaker_embeddings_assigns_each_known_name_once(self):
        speaker_embeddings = {
            "Persona 1": (0.98, 0.02),
            "Persona 2": (0.96, 0.04),
            "Persona 3": (0.0, 1.0),
        }
        candidates = {
            "Ruben": ((1.0, 0.0),),
            "Nuria": ((0.0, 1.0),),
        }

        self.assertEqual(
            match_speaker_embeddings(speaker_embeddings, candidates, threshold=0.8),
            {"Persona 1": "Ruben", "Persona 3": "Nuria"},
        )

    def test_cuda_embedding_error_detects_cublas_failures(self):
        error = RuntimeError("CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling cublasCreate")

        self.assertTrue(is_cuda_embedding_error(error))

    def test_cuda_embedding_error_ignores_regular_failures(self):
        self.assertFalse(is_cuda_embedding_error(RuntimeError("audio file not found")))

    def test_coerce_embedding_output_averages_rows_from_nested_output(self):
        self.assertEqual(_coerce_embedding_output([[1.0, 0.0], [0.5, 0.5]]), (0.75, 0.25))

    def test_coerce_embedding_output_accepts_objects_with_tolist(self):
        class FakeArray:
            def tolist(self):
                return [0.25, 0.75]

        self.assertEqual(_coerce_embedding_output(FakeArray()), (0.25, 0.75))


if __name__ == "__main__":
    unittest.main()
