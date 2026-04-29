from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from meeting_transcriber.cuda_runtime import configure_cuda_runtime
from meeting_transcriber.external_links import PYANNOTE_MODEL_URL
from meeting_transcriber.hf_auth import apply_huggingface_token
from meeting_transcriber.progress import ProgressEvent
from meeting_transcriber.types import DiarizationSegment, ProcessingConfig

ProgressCallback = Callable[[ProgressEvent], None]


class DiarizationPreflightError(RuntimeError):
    pass


def load_diarization_pipeline(config: ProcessingConfig) -> Any:
    try:
        configure_cuda_runtime()
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Install project dependencies before diarizing."
        ) from exc

    try:
        apply_huggingface_token(config.huggingface_token)
        pipeline = Pipeline.from_pretrained(
            config.diarization_model,
            token=config.huggingface_token,
        )
    except Exception as exc:
        raise DiarizationPreflightError(explain_diarization_error(exc)) from exc

    if pipeline is None:
        raise DiarizationPreflightError(explain_diarization_error(None))
    _configure_pipeline_runtime(pipeline, config)
    return pipeline


def diarize_audio(
    audio_path: Path,
    config: ProcessingConfig,
    pipeline: Any | None = None,
    progress: ProgressCallback | None = None,
) -> list[DiarizationSegment]:
    diarization_pipeline = pipeline if pipeline is not None else load_diarization_pipeline(config)
    try:
        return _diarize_with_pipeline(audio_path, config, diarization_pipeline, progress)
    except Exception as exc:
        if config.device == "cuda" and _looks_like_cuda_runtime_error(exc):
            _report_diarization_step(progress, "fallback_cpu", "CUDA fallo; reintentando separacion de voces en CPU")
            _move_pipeline_to_cpu(diarization_pipeline)
            try:
                return _diarize_with_pipeline(audio_path, config, diarization_pipeline, progress)
            except Exception as cpu_exc:
                raise RuntimeError(_friendly_runtime_error(cpu_exc)) from cpu_exc
        raise RuntimeError(_friendly_runtime_error(exc)) from exc


def _diarize_with_pipeline(
    audio_path: Path,
    config: ProcessingConfig,
    diarization_pipeline: Any,
    progress: ProgressCallback | None,
) -> list[DiarizationSegment]:
    params: dict[str, int] = {}
    if config.min_speakers is not None:
        params["min_speakers"] = config.min_speakers
    if config.max_speakers is not None:
        params["max_speakers"] = config.max_speakers

    diarization = diarization_pipeline(
        str(audio_path),
        hook=_pyannote_progress_hook(progress),
        **params,
    )
    annotation = _annotation_from_diarization_output(diarization)
    return [
        DiarizationSegment(
            start=float(turn.start),
            end=float(turn.end),
            speaker=str(speaker),
        )
        for turn, _track, speaker in annotation.itertracks(yield_label=True)
    ]


def _pyannote_progress_hook(progress: ProgressCallback | None) -> Callable[..., None] | None:
    if progress is None:
        return None

    def report(step_name: str, _artifact: object = None, **kwargs: object) -> None:
        message = _diarization_step_label(step_name)
        completed = _optional_int(kwargs.get("completed"))
        total = _optional_int(kwargs.get("total"))
        progress(
            ProgressEvent(
                stage="diarization_progress",
                message=message,
                completed=completed,
                total=total,
            )
        )

    return report


def _report_diarization_step(
    progress: ProgressCallback | None,
    _step_name: str,
    message: str,
) -> None:
    if progress is None:
        return
    progress(ProgressEvent(stage="diarization_progress", message=message))


def _diarization_step_label(step_name: str) -> str:
    labels = {
        "segmentation": "Separando voces: analizando actividad de voz",
        "speaker_counting": "Separando voces: estimando hablantes simultaneos",
        "embeddings": "Separando voces: comparando huellas de voz",
        "discrete_diarization": "Separando voces: reconstruyendo turnos",
    }
    return labels.get(step_name, f"Separando voces: {step_name.replace('_', ' ')}")


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _annotation_from_diarization_output(diarization: Any) -> Any:
    if hasattr(diarization, "itertracks"):
        return diarization
    exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
    if exclusive is not None and hasattr(exclusive, "itertracks"):
        return exclusive
    speaker_diarization = getattr(diarization, "speaker_diarization", None)
    if speaker_diarization is not None and hasattr(speaker_diarization, "itertracks"):
        return speaker_diarization
    raise TypeError(
        "La diarizacion no devolvio un formato compatible. "
        f"Tipo recibido: {type(diarization).__name__}"
    )


def _configure_pipeline_runtime(pipeline: Any, config: ProcessingConfig) -> None:
    if config.device == "cuda":
        try:
            import torch

            pipeline.to(torch.device("cuda"))
        except Exception:
            # Diarization can still run on CPU if moving pyannote to CUDA fails.
            pass

    if config.diarization_quality == "strict" and hasattr(pipeline, "instantiate"):
        try:
            pipeline.instantiate(
                {
                    "segmentation": {"min_duration_off": 0.0},
                    "clustering": {"threshold": 0.5, "Fa": 0.07, "Fb": 0.8},
                }
            )
        except Exception:
            # Some pretrained pipelines do not accept manual hyper-parameters.
            pass


def _move_pipeline_to_cpu(pipeline: Any) -> None:
    try:
        import torch

        pipeline.to(torch.device("cpu"))
    except Exception:
        pass


def _looks_like_cuda_runtime_error(error: Exception) -> bool:
    detail = str(error).lower()
    return (
        "cuda" in detail
        or "nvrtc" in detail
        or "libnvrtc" in detail
        or "cublas" in detail
        or "out of memory" in detail
    )


def _friendly_runtime_error(error: Exception) -> str:
    detail = str(error)
    lowered = detail.lower()
    if "libnvrtc-builtins" in lowered or "nvrtc" in lowered:
        return (
            "Pyannote no pudo usar CUDA porque falta o no se puede cargar una libreria NVRTC. "
            "La app ha guardado la transcripcion bruta. Prueba CPU para la diarizacion o reinicia "
            "la aplicacion para que cargue las librerias CUDA embebidas."
        )
    if "out of memory" in lowered:
        return (
            "Pyannote se quedo sin memoria usando CUDA. La app ha guardado la transcripcion bruta. "
            "Prueba CPU o reduce el rango de audio."
        )
    return (
        "No se pudo separar hablantes. La app ha guardado la transcripcion bruta. "
        f"Detalle: {_short_detail(detail)}"
    )


def _short_detail(detail: str) -> str:
    first_line = next((line.strip() for line in detail.splitlines() if line.strip()), "")
    return first_line[:500] if first_line else "sin detalle"


def explain_diarization_error(error: Exception | None) -> str:
    detail = "" if error is None else str(error)
    if "401" in detail or "gated repo" in detail.lower() or "restricted" in detail.lower():
        return (
            "No se puede acceder al modelo de diarizacion de pyannote. Antes de procesar, "
            "pulsa 'Abrir modelo pyannote', acepta el acceso en Hugging Face y pega un Token HF "
            f"valido en la aplicacion. Modelo: {PYANNOTE_MODEL_URL}"
        )
    return (
        "No se pudo cargar el modelo de diarizacion de pyannote. Revisa el Token HF, la conexion "
        f"y el acceso al modelo. Detalle: {detail}"
    )
