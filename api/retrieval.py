"""Hybrid retrieval: turns fused vector-store candidates into retrieval chunks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import models

from api.embeddings import EmbeddedText, EmbeddingError
from api.settings import AppSettings

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Raised when candidate retrieval fails."""


class RetrievalPayloadError(RetrievalError):
    """Raised when a stored point is missing required citation metadata."""


class RetrievalConfigError(RetrievalError):
    """Raised when a per-request retrieval override is invalid against server config."""


class QueryEmbeddingProvider(Protocol):
    """Embedding provider surface used by retrieval."""

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]: ...


class SearchRepository(Protocol):
    """Vector-store surface retrieval needs: fused dense+sparse candidates."""

    def ensure_ready(self, settings: AppSettings) -> None: ...

    def hybrid_search(
        self,
        settings: AppSettings,
        query_embedding: EmbeddedText,
        filenames: Sequence[str] | None = None,
    ) -> list[models.ScoredPoint]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    """A fused retrieval candidate ready for reranking."""

    point_id: str
    filename: str
    page: int
    section: str
    chunk_id: str
    text: str
    score: float
    # None for points indexed before chunk_ordinal existed — neighbor expansion
    # skips those gracefully rather than treating a missing ordinal as an error.
    chunk_ordinal: int | None = None


def retrieve_candidates(
    repository: SearchRepository,
    settings: AppSettings,
    query: str,
    embedding_provider: QueryEmbeddingProvider,
    filenames: Sequence[str] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve fused dense+sparse candidates for a user query.

    ``filenames``, when given, restricts retrieval to those documents only (both the
    dense and sparse prefetch legs are filtered before fusion, not after).
    """

    normalized_query = query.strip()
    if not normalized_query:
        raise RetrievalError("Query text is required.")

    # Validate before embedding: an embedding-tag mismatch should refuse the query
    # without paying for embedding work first.
    repository.ensure_ready(settings)
    query_embedding = embed_query(normalized_query, embedding_provider)
    points = repository.hybrid_search(settings, query_embedding, filenames)
    return points_to_chunks(points)


def points_to_chunks(points: Sequence[models.ScoredPoint]) -> list[RetrievedChunk]:
    """Convert fused points to chunks, skipping any with malformed payloads.

    One legacy/malformed point (e.g. from an older schema) must not abort an entire
    answer when dozens of other candidates are fine — only raise if literally every
    candidate is unusable.
    """

    chunks: list[RetrievedChunk] = []
    for point in points:
        try:
            chunks.append(scored_point_to_chunk(point))
        except RetrievalPayloadError:
            logger.warning("Skipping retrieval candidate with malformed payload: %s", point.id)
    if points and not chunks:
        raise RetrievalPayloadError("All retrieved points are missing required payload fields.")
    return chunks


def embed_query(query: str, embedding_provider: QueryEmbeddingProvider) -> EmbeddedText:
    """Embed a single query with the configured local embedding provider.

    Prefers the provider's ``embed_query`` (query/document embedding symmetry: dense
    query instruction + BM25 query-side encoding) when available, falling back to
    ``embed_texts`` for providers/test fakes that only implement the batch path.
    """

    query_embed = getattr(embedding_provider, "embed_query", None)
    if callable(query_embed):
        result = query_embed(query)
        if not isinstance(result, EmbeddedText):
            raise EmbeddingError("embed_query must return a single EmbeddedText.")
        return result

    embeddings = embedding_provider.embed_texts([query])
    if len(embeddings) != 1:
        raise EmbeddingError(f"Query embedding count mismatch: got {len(embeddings)}, expected 1.")
    return embeddings[0]


def scored_point_to_chunk(point: models.ScoredPoint) -> RetrievedChunk:
    """Convert a Qdrant scored point payload into a retrieval candidate."""

    payload = point.payload or {}
    return RetrievedChunk(
        point_id=str(point.id),
        filename=payload_string(payload, "filename"),
        page=payload_page(payload),
        section=payload_string(payload, "section"),
        chunk_id=payload_string(payload, "chunk_id"),
        text=payload_string(payload, "text"),
        score=float(point.score),
        chunk_ordinal=payload_optional_int(payload, "chunk_ordinal"),
    )


def payload_string(payload: dict[str, object], key: str) -> str:
    """Read a required string payload field."""

    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RetrievalPayloadError(f"Retrieved point is missing payload field '{key}'.")
    return value


def payload_page(payload: dict[str, object]) -> int:
    """Read the required one-based page payload field."""

    value = payload.get("page")
    if not isinstance(value, int) or isinstance(value, bool):
        raise RetrievalPayloadError("Retrieved point is missing payload field 'page'.")
    return value


def payload_optional_int(payload: dict[str, object], key: str) -> int | None:
    """Read an optional integer payload field, absent on pre-upgrade points."""

    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value
