import unittest

from meeting_transcriber.speaker_fingerprints import cosine_similarity, best_embedding_match


class SpeakerFingerprintTests(unittest.TestCase):
    def test_cosine_similarity_scores_identical_vectors_as_one(self):
        self.assertAlmostEqual(cosine_similarity((1, 0, 0), (1, 0, 0)), 1.0)

    def test_best_embedding_match_returns_name_above_threshold(self):
        candidates = {"Ruben": ((1, 0),), "Nuria": ((0, 1),)}

        self.assertEqual(best_embedding_match((0.95, 0.05), candidates, threshold=0.8), "Ruben")

    def test_best_embedding_match_returns_none_below_threshold(self):
        candidates = {"Ruben": ((1, 0),)}

        self.assertIsNone(best_embedding_match((0.1, 0.2), candidates, threshold=0.8))


if __name__ == "__main__":
    unittest.main()
