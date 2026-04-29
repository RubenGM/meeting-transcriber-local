from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Callable
import warnings

from meeting_transcriber.audio import extract_audio_range
from meeting_transcriber.cancellation import CancelCheck, CancelledError, raise_if_cancelled
from meeting_transcriber.cuda_runtime import configure_cuda_runtime
from meeting_transcriber.ffmpeg import resolve_ffmpeg_path
from meeting_transcriber.progress import ProgressEvent
from meeting_transcriber.transcription import transcribe_audio
from meeting_transcriber.types import ProcessingConfig

ProgressCallback = Callable[[ProgressEvent], None]

BENCHMARK_SECONDS = 30.0


class BenchmarkSetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeCandidate:
    device: str
    compute_type: str

    @property
    def label(self) -> str:
        return f"{self.device} / {self.compute_type}"


@dataclass(frozen=True)
class BenchmarkAttempt:
    candidate: RuntimeCandidate
    ok: bool
    elapsed_seconds: float | None = None
    audio_seconds: float | None = None
    error: str | None = None

    @property
    def speed(self) -> float | None:
        if not self.ok or not self.elapsed_seconds or not self.audio_seconds:
            return None
        if self.elapsed_seconds <= 0:
            return None
        return self.audio_seconds / self.elapsed_seconds


@dataclass(frozen=True)
class BenchmarkResult:
    attempts: tuple[BenchmarkAttempt, ...]
    recommendation: RuntimeCandidate | None


@dataclass(frozen=True)
class CudaStatus:
    available: bool
    reason: str | None = None


def runtime_candidates(cuda_available: bool) -> list[RuntimeCandidate]:
    candidates = [RuntimeCandidate("cpu", "float32"), RuntimeCandidate("cpu", "int8")]
    if cuda_available:
        candidates = [
            RuntimeCandidate("cuda", "float32"),
            RuntimeCandidate("cuda", "float16"),
            RuntimeCandidate("cuda", "int8_float16"),
            RuntimeCandidate("cuda", "int8"),
            *candidates,
        ]
    return candidates


def is_cuda_available() -> bool:
    return detect_cuda().available


def detect_cuda() -> CudaStatus:
    ctranslate2_reason = _detect_ctranslate2_cuda_reason()
    if ctranslate2_reason is None:
        return CudaStatus(True)

    torch_reason = _detect_torch_cuda_reason()
    if torch_reason is not None:
        return CudaStatus(False, torch_reason)
    return CudaStatus(False, ctranslate2_reason)


def _detect_ctranslate2_cuda_reason() -> str | None:
    try:
        configure_cuda_runtime()
        import ctranslate2
    except ImportError:
        return "ctranslate2 no esta instalado"
    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:
        return f"ctranslate2 no pudo comprobar CUDA: {_friendly_error(exc)}"
    if count > 0:
        return None
    return "ctranslate2 no detecta ninguna GPU CUDA utilizable"


def _detect_torch_cuda_reason() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    captured_warnings: list[warnings.WarningMessage]
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        try:
            available = bool(torch.cuda.is_available())
        except Exception as exc:
            return f"PyTorch no pudo inicializar CUDA: {_friendly_error(exc)}"
    if available:
        return None
    for warning in captured_warnings:
        message = str(warning.message)
        if "NVIDIA driver" in message:
            return "CUDA no esta disponible: el driver NVIDIA parece demasiado antiguo para esta instalacion"
        if "CUDA" in message or "NVML" in message:
            return f"CUDA no esta disponible: {message}"
    return None


def cuda_status_message(status: CudaStatus) -> str:
    if status.available:
        return "CUDA disponible: se probaran opciones de GPU"
    reason = status.reason or "no se detecto una GPU CUDA utilizable"
    if reason.lower().startswith("cuda no "):
        return f"{reason}. Se probaran opciones CPU."
    return f"CUDA no disponible: {reason}. Se probaran opciones CPU."


def run_transcription_benchmark(
    audio_path: Path,
    config: ProcessingConfig,
    *,
    seconds: float = BENCHMARK_SECONDS,
    progress: ProgressCallback | None = None,
    cancelled: CancelCheck | None = None,
) -> BenchmarkResult:
    reporter = progress if progress is not None else _ignore_progress
    raise_if_cancelled(cancelled)
    if not audio_path.is_file():
        raise BenchmarkSetupError("Selecciona primero un archivo de audio valido.")
    cuda_status = detect_cuda()
    raise_if_cancelled(cancelled)
    reporter(
        ProgressEvent(
            stage="benchmark",
            message=cuda_status_message(cuda_status),
        )
    )
    attempts: list[BenchmarkAttempt] = []

    with TemporaryDirectory() as temp_dir:
        clip_path = Path(temp_dir) / "benchmark.wav"
        start = config.start_seconds or 0.0
        end = start + seconds
        reporter(
            ProgressEvent(
                stage="benchmark",
                message=f"Preparando prueba corta de {int(seconds)} segundos",
            )
        )
        try:
            raise_if_cancelled(cancelled)
            extract_audio_range(
                resolve_ffmpeg_path(config.ffmpeg_path),
                audio_path,
                clip_path,
                start,
                end,
            )
        except Exception as exc:
            raise BenchmarkSetupError(
                "No se pudo preparar el recorte de prueba. Revisa que el archivo de audio "
                "se pueda abrir y que el rango elegido exista dentro del audio."
            ) from exc

        for candidate in runtime_candidates(cuda_status.available):
            raise_if_cancelled(cancelled)
            reporter(
                ProgressEvent(
                    stage="benchmark",
                    message=f"Probando {candidate.label}",
                )
            )
            attempt = _try_candidate(clip_path, config, candidate, seconds, cancelled)
            attempts.append(attempt)
            if attempt.ok and attempt.speed is not None:
                reporter(
                    ProgressEvent(
                        stage="benchmark",
                        message=f"{candidate.label}: OK, {attempt.speed:.1f}x",
                    )
                )
            else:
                reporter(
                    ProgressEvent(
                        stage="benchmark",
                        message=f"{candidate.label}: no usable ({attempt.error})",
                    )
                )

    recommendation = recommend_runtime(attempts)
    return BenchmarkResult(tuple(attempts), recommendation)


def recommend_runtime(attempts: list[BenchmarkAttempt] | tuple[BenchmarkAttempt, ...]) -> RuntimeCandidate | None:
    successful = [attempt for attempt in attempts if attempt.ok and attempt.speed is not None]
    if not successful:
        return None
    return max(successful, key=lambda attempt: attempt.speed or 0.0).candidate


def _try_candidate(
    clip_path: Path,
    config: ProcessingConfig,
    candidate: RuntimeCandidate,
    audio_seconds: float,
    cancelled: CancelCheck | None = None,
) -> BenchmarkAttempt:
    candidate_config = replace(
        config,
        device=candidate.device,
        compute_type=candidate.compute_type,
        start_seconds=None,
        end_seconds=None,
    )
    started_at = time.monotonic()
    try:
        raise_if_cancelled(cancelled)
        transcribe_audio(clip_path, candidate_config, cancelled=cancelled)
    except CancelledError:
        _clear_cuda_cache()
        raise
    except Exception as exc:
        _clear_cuda_cache()
        return BenchmarkAttempt(
            candidate=candidate,
            ok=False,
            error=_friendly_error(exc),
        )
    elapsed = time.monotonic() - started_at
    _clear_cuda_cache()
    return BenchmarkAttempt(
        candidate=candidate,
        ok=True,
        elapsed_seconds=elapsed,
        audio_seconds=audio_seconds,
    )


def _friendly_error(error: Exception) -> str:
    detail = str(error).strip() or type(error).__name__
    if "out of memory" in detail.lower():
        return "memoria insuficiente"
    return detail


def _clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _ignore_progress(_event: ProgressEvent) -> None:
    return None
