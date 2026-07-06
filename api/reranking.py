"""Cross-encoder reranking for fused retrieval candidates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder

from api.retrieval import RetrievalError, RetrievedChunk
from api.settings import AppSettings


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

        return [
            float(score)
            for score in self.model.rerank(
                query=query,
                documents=documents,
                batch_size=int(self.settings.reranker_batch_size),
            )
        ]


class Reranker(Protocol):
    """Reranker surface used by the application pipeline."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


def rerank_candidates(
    query: str,
    candidates: Sequence[RetrievedChunk],
    reranker: Reranker,
    settings: AppSettings,
) -> list[RerankedChunk]:
    """Rerank only the fused top-N candidates and return the top-K results."""

    normalized_query = query.strip()
    if not normalized_query:
        raise RetrievalError("Query text is required.")
    if not candidates:
        return []

    fused_top_n = int(settings.fused_top_n)
    rerank_top_k = int(settings.rerank_top_k)
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
            rerank_score=float(score),
        )
        for candidate, score in zip(candidates_to_score, scores, strict=True)
    ]
    return sorted(
        reranked,
        key=lambda chunk: (-chunk.rerank_score, -chunk.retrieval_score, chunk.chunk_id),
    )[:rerank_top_k]

