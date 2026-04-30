from __future__ import annotations

import math


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
