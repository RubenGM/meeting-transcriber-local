from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from meeting_transcriber.audio import extract_audio_range
from meeting_transcriber.cuda_runtime import configure_cuda_runtime
from meeting_transcriber.hf_auth import apply_huggingface_token
from meeting_transcriber.types import ConversationTurn


EmbeddingExtractor = Callable[[Path, float, float], tuple[float, ...] | None]
DEFAULT_EMBEDDING_MODEL = "pyannote/embedding"


class PyannoteEmbeddingExtractor:
    def __init__(self, ffmpeg_path: Path, inference: Callable[[Path], object]) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.inference = inference

    def __call__(self, audio_path: Path, start: float, end: float) -> tuple[float, ...] | None:
        if end <= start:
            return None
        with TemporaryDirectory() as dirname:
            clip_path = Path(dirname) / "speaker_sample.wav"
            extract_audio_range(self.ffmpeg_path, audio_path, clip_path, start, end)
            return _coerce_embedding_output(self.inference(clip_path))


def load_pyannote_embedding_extractor(
    ffmpeg_path: Path,
    *,
    huggingface_token: str | None,
    device: str,
    model_id: str = DEFAULT_EMBEDDING_MODEL,
) -> PyannoteEmbeddingExtractor:
    configure_cuda_runtime()
    apply_huggingface_token(huggingface_token)
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained(model_id, token=huggingface_token)
    inference_kwargs: dict[str, object] = {"window": "whole"}
    if device == "cuda":
        try:
            import torch

            inference_kwargs["device"] = torch.device("cuda")
        except Exception:
            pass
    inference = Inference(model, **inference_kwargs)
    return PyannoteEmbeddingExtractor(ffmpeg_path, inference)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def best_embedding_match(
    embedding: tuple[float, ...],
    candidates: dict[str, tuple[tuple[float, ...], ...]],
    *,
    threshold: float,
) -> str | None:
    best_name: str | None = None
    best_score = threshold
    for name, embeddings in candidates.items():
        for candidate in embeddings:
            score = cosine_similarity(embedding, candidate)
            if score > best_score:
                best_name = name
                best_score = score
    return best_name


def extract_speaker_embeddings(
    audio_path: Path,
    turns: list[ConversationTurn],
    extractor: EmbeddingExtractor,
    *,
    min_duration_seconds: float = 2.0,
    max_samples_per_speaker: int = 3,
) -> dict[str, tuple[float, ...]]:
    samples: dict[str, list[tuple[float, ...]]] = {}
    for turn in turns:
        if turn.end - turn.start < min_duration_seconds:
            continue
        speaker_samples = samples.setdefault(turn.speaker, [])
        if len(speaker_samples) >= max_samples_per_speaker:
            continue
        embedding = extractor(audio_path, turn.start, turn.end)
        if embedding is not None:
            speaker_samples.append(embedding)
    return {
        speaker: _average_embeddings(embeddings)
        for speaker, embeddings in samples.items()
        if embeddings
    }


def match_speaker_embeddings(
    speaker_embeddings: dict[str, tuple[float, ...]],
    candidates: dict[str, tuple[tuple[float, ...], ...]],
    *,
    threshold: float,
) -> dict[str, str]:
    scored: list[tuple[float, str, str]] = []
    for speaker, embedding in speaker_embeddings.items():
        for name, candidate_embeddings in candidates.items():
            for candidate in candidate_embeddings:
                score = cosine_similarity(embedding, candidate)
                if score >= threshold:
                    scored.append((score, speaker, name))

    mapping: dict[str, str] = {}
    used_names: set[str] = set()
    for _score, speaker, name in sorted(scored, reverse=True):
        if speaker in mapping or name in used_names:
            continue
        mapping[speaker] = name
        used_names.add(name)
    return mapping


def is_cuda_embedding_error(error: Exception) -> bool:
    detail = str(error).lower()
    return (
        "cuda" in detail
        or "cublas" in detail
        or "cudnn" in detail
        or "nvrtc" in detail
        or "out of memory" in detail
    )


def _average_embeddings(embeddings: list[tuple[float, ...]]) -> tuple[float, ...]:
    if not embeddings:
        return ()
    dimensions = len(embeddings[0])
    compatible = [embedding for embedding in embeddings if len(embedding) == dimensions]
    if not compatible:
        return ()
    return tuple(
        sum(embedding[index] for embedding in compatible) / len(compatible)
        for index in range(dimensions)
    )


def _coerce_embedding_output(output: object) -> tuple[float, ...] | None:
    if isinstance(output, tuple) and len(output) == 1:
        output = output[0]
    if hasattr(output, "data"):
        output = getattr(output, "data")
    if hasattr(output, "tolist"):
        output = output.tolist()  # type: ignore[union-attr]
    flattened = _numeric_rows(output)
    if not flattened:
        return None
    return _average_embeddings(flattened)


def _numeric_rows(value: object) -> list[tuple[float, ...]]:
    if isinstance(value, (int, float)):
        return [(float(value),)]
    if not isinstance(value, (list, tuple)):
        return []
    if all(isinstance(item, (int, float)) for item in value):
        return [tuple(float(item) for item in value)]
    rows: list[tuple[float, ...]] = []
    for item in value:
        rows.extend(_numeric_rows(item))
    return rows
