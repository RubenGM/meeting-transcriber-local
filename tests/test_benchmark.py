import unittest

from meeting_transcriber.benchmark import (
    BenchmarkAttempt,
    CudaStatus,
    RuntimeCandidate,
    cuda_status_message,
    recommend_runtime,
    runtime_candidates,
)


class BenchmarkTests(unittest.TestCase):
    def test_runtime_candidates_include_cpu_fallback_from_stronger_to_safer(self):
        candidates = runtime_candidates(cuda_available=False)

        self.assertEqual(candidates[0], RuntimeCandidate("cpu", "float32"))
        self.assertIn(RuntimeCandidate("cpu", "float32"), candidates)
        self.assertEqual(candidates[-1], RuntimeCandidate("cpu", "int8"))

    def test_runtime_candidates_try_cuda_from_maximum_to_safer_options(self):
        candidates = runtime_candidates(cuda_available=True)

        self.assertEqual(candidates[0], RuntimeCandidate("cuda", "float32"))
        self.assertEqual(candidates[1], RuntimeCandidate("cuda", "float16"))
        self.assertEqual(candidates[2], RuntimeCandidate("cuda", "int8_float16"))
        self.assertIn(RuntimeCandidate("cpu", "int8"), candidates)

    def test_recommend_runtime_returns_fastest_successful_attempt(self):
        result = recommend_runtime(
            [
                BenchmarkAttempt(
                    RuntimeCandidate("cuda", "float16"),
                    ok=False,
                    error="memoria insuficiente",
                ),
                BenchmarkAttempt(
                    RuntimeCandidate("cpu", "int8"),
                    ok=True,
                    elapsed_seconds=10.0,
                    audio_seconds=30.0,
                ),
                BenchmarkAttempt(
                    RuntimeCandidate("cuda", "int8"),
                    ok=True,
                    elapsed_seconds=5.0,
                    audio_seconds=30.0,
                ),
            ]
        )

        self.assertEqual(result, RuntimeCandidate("cuda", "int8"))

    def test_cuda_status_message_explains_skipped_cuda(self):
        message = cuda_status_message(CudaStatus(False, "driver NVIDIA no accesible"))

        self.assertIn("CUDA no disponible", message)
        self.assertIn("driver NVIDIA no accesible", message)
        self.assertIn("CPU", message)


if __name__ == "__main__":
    unittest.main()
