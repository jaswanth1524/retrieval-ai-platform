"""Cross-encoder reranking for fused retrieval candidates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder

from api.retrieval import RetrievalError, RetrievedChunk
from api.settings import AppSettings


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid; avoids OverflowError for large-magnitude logits."""

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class RerankingError(RuntimeError):
    """Raised when cross-encoder reranking fails."""


class CrossEncoderModel(Protocol):
    """Cross-encoder model surface used for reranking."""

    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        batch_size: int = 64,
        **kwargs: Any,
    ) -> Iterable[float]: ...


@dataclass(frozen=True)
class RerankedChunk:
    """A retrieval candidate after cross-encoder reranking."""

    point_id: str
    filename: str
    page: int
    section: str
    chunk_id: str
    text: str
    retrieval_score: float
    rerank_score: float


class LocalCrossEncoderReranker:
    """Rerank candidates locally with FastEmbed's TextCrossEncoder."""

    def __init__(
        self,
        settings: AppSettings,
        model: CrossEncoderModel | None = None,
    ) -> None:
        self.settings = settings
        self.model = model or TextCrossEncoder(
            model_name=settings.reranker_model,
            lazy_load=True,
        )

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return cross-encoder scores for query-document pairs."""

        try:
            # The model is constructed with lazy_load=True, so the actual model
            # download/load happens on this first call, not at __init__ — this is
            # the boundary that must translate a load failure into RerankingError.
            scores = list(
                self.model.rerank(
                    query=query,
                    documents=documents,
                    batch_size=int(self.settings.reranker_batch_size),
                )
            )
        except Exception as exc:
            raise RerankingError(f"Reranker model failed: {exc}") from exc
        return [float(score) for score in scores]


class Reranker(Protocol):
    """Reranker surface used by the application pipeline."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


def rerank_candidates(
    query: str,
    candidates: Sequence[RetrievedChunk],
    reranker: Reranker,
    settings: AppSettings,
) -> list[RerankedChunk]:
    """Rerank the fused top-N candidates, drop low-relevance ones, keep the top-K.

    ``rerank_score`` is the raw cross-encoder logit sigmoid-normalized to [0, 1] so
    ``rerank_min_score`` is comparable across reranker models. Because the candidate
    list is sorted descending before filtering, dropping everything below the
    threshold and then slicing to ``rerank_top_k`` is equivalent to slicing first and
    filtering after — either way the result is the sorted prefix that clears the bar,
    which may be shorter than ``rerank_top_k`` or empty.
    """

    normalized_query = query.strip()
    if not normalized_query:
        raise RetrievalError("Query text is required.")
    if not candidates:
        return []

    fused_top_n = int(settings.fused_top_n)
    rerank_top_k = int(settings.rerank_top_k)
    min_score = float(settings.rerank_min_score)
    candidates_to_score = list(candidates[:fused_top_n])
    documents = [candidate.text for candidate in candidates_to_score]
    scores = reranker.score(normalized_query, documents)

    if len(scores) != len(candidates_to_score):
        raise RerankingError(
            f"Reranker score count mismatch: got {len(scores)}, "
            f"expected {len(candidates_to_score)}."
        )

    reranked = [
        RerankedChunk(
            point_id=candidate.point_id,
            filename=candidate.filename,
            page=candidate.page,
            section=candidate.section,
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            retrieval_score=candidate.score,
            rerank_score=_sigmoid(float(score)),
        )
        for candidate, score in zip(candidates_to_score, scores, strict=True)
    ]
    sorted_chunks = sorted(
        reranked,
        key=lambda chunk: (-chunk.rerank_score, -chunk.retrieval_score, chunk.chunk_id),
    )
    filtered = [chunk for chunk in sorted_chunks if chunk.rerank_score >= min_score]
    return filtered[:rerank_top_k]

