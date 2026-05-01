from __future__ import annotations

from dataclasses import dataclass, replace

from meeting_transcriber.app_modes import SimpleModeSettings
from meeting_transcriber.benchmark import BenchmarkResult, RuntimeCandidate
from meeting_transcriber.types import ProcessingConfig


@dataclass(frozen=True)
class SimpleProcessingPlan:
    config: ProcessingConfig
    chunk_seconds: float
    overlap_seconds: float
    explanation: str


def choose_whisper_model(device: str, observed_speed: float | None) -> str:
    if device == "cuda":
        if observed_speed is not None and observed_speed >= 12.0:
            return "large-v3"
        if observed_speed is not None and observed_speed >= 5.0:
            return "medium"
        return "small"
    if observed_speed is not None and observed_speed >= 2.0:
        return "small"
    return "base"


def choose_chunk_seconds(
    observed_speed: float | None,
    settings: SimpleModeSettings,
) -> float:
    if observed_speed is None:
        candidate = 10 * 60.0
    else:
        candidate = settings.target_wait_seconds * observed_speed
    return max(settings.min_chunk_seconds, min(settings.max_chunk_seconds, candidate))


def build_simple_processing_plan(
    base_config: ProcessingConfig,
    benchmark_result: BenchmarkResult,
    settings: SimpleModeSettings | None = None,
) -> SimpleProcessingPlan:
    settings = settings or SimpleModeSettings()
    recommendation = benchmark_result.recommendation or RuntimeCandidate(
        base_config.device,
        base_config.compute_type,
    )
    observed_speed = _best_successful_speed(benchmark_result)
    whisper_model = choose_whisper_model(recommendation.device, observed_speed)
    chunk_seconds = choose_chunk_seconds(observed_speed, settings)
    config = replace(
        base_config,
        whisper_model=whisper_model,
        device=recommendation.device,
        compute_type=recommendation.compute_type,
        diarization_quality="precise",
        normalize_audio=settings.normalize_audio,
        start_seconds=None,
        end_seconds=None,
    )
    speed_text = "sin velocidad medida" if observed_speed is None else f"{observed_speed:.1f}x"
    explanation = (
        "Simple: "
        f"{recommendation.device} / {recommendation.compute_type}, "
        f"modelo {whisper_model}, tandas de {int(chunk_seconds // 60)} min, {speed_text}"
    )
    return SimpleProcessingPlan(
        config=config,
        chunk_seconds=chunk_seconds,
        overlap_seconds=settings.chunk_overlap_seconds,
        explanation=explanation,
    )


def _best_successful_speed(result: BenchmarkResult) -> float | None:
    speeds = [attempt.speed for attempt in result.attempts if attempt.ok and attempt.speed is not None]
    if not speeds:
        return None
    return max(speed for speed in speeds if speed is not None)

