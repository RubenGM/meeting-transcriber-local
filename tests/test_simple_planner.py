from meeting_transcriber.app_modes import SimpleModeSettings
from meeting_transcriber.benchmark import BenchmarkAttempt, BenchmarkResult, RuntimeCandidate
from meeting_transcriber.simple_planner import (
    build_simple_processing_plan,
    choose_chunk_seconds,
    choose_whisper_model,
)
from meeting_transcriber.types import ProcessingConfig


def _config() -> ProcessingConfig:
    return ProcessingConfig(
        whisper_model="small",
        diarization_model="pyannote/speaker-diarization-community-1",
        huggingface_token=None,
        ffmpeg_path=None,
        language="ca",
        min_speakers=None,
        max_speakers=None,
        device="cpu",
        compute_type="int8",
        export_speaker_audio=False,
    )


def test_choose_whisper_model_scales_with_runtime_speed():
    assert choose_whisper_model("cuda", 13.0) == "large-v3"
    assert choose_whisper_model("cuda", 7.0) == "medium"
    assert choose_whisper_model("cuda", 2.0) == "small"
    assert choose_whisper_model("cpu", 3.0) == "small"
    assert choose_whisper_model("cpu", 1.0) == "base"


def test_choose_chunk_seconds_clamps_to_settings():
    settings = SimpleModeSettings(target_wait_seconds=600, min_chunk_seconds=300, max_chunk_seconds=1800)

    assert choose_chunk_seconds(None, settings) == 600
    assert choose_chunk_seconds(0.1, settings) == 300
    assert choose_chunk_seconds(10.0, settings) == 1800


def test_build_simple_processing_plan_applies_benchmark_recommendation_and_normalization():
    result = BenchmarkResult(
        attempts=(
            BenchmarkAttempt(RuntimeCandidate("cuda", "float16"), ok=True, elapsed_seconds=10, audio_seconds=120),
        ),
        recommendation=RuntimeCandidate("cuda", "float16"),
    )

    plan = build_simple_processing_plan(_config(), result)

    assert plan.config.device == "cuda"
    assert plan.config.compute_type == "float16"
    assert plan.config.whisper_model == "large-v3"
    assert plan.config.normalize_audio is True
    assert plan.config.diarization_quality == "precise"

