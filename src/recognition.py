from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceIdentity:
    name: str
    embedding: np.ndarray
    images_count: int


@dataclass(frozen=True)
class MatchResult:
    name: str
    score: float


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def best_match(
    face_embedding: np.ndarray,
    references: list[ReferenceIdentity],
    min_score: float,
) -> MatchResult | None:
    normalized_embedding = l2_normalize(face_embedding)
    scores = sorted(
        (
            MatchResult(reference.name, cosine_similarity(reference.embedding, normalized_embedding))
            for reference in references
        ),
        key=lambda match: match.score,
        reverse=True,
    )

    if not scores or scores[0].score < min_score:
        return None

    return scores[0]
