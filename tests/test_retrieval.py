from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException

from api.documents import DocumentChunk
from api.embeddings import EmbeddedText
from api.ingestion import ingest_chunks
from api.repository import VectorRepository
from api.retrieval import (
    RetrievalError,
    RetrievalPayloadError,
    retrieve_candidates,
    scored_point_to_chunk,
)
from api.settings import AppSettings


class StaticEmbeddingProvider:
    def __init__(self, embeddings: list[EmbeddedText]) -> None:
        self.embeddings = embeddings
        self.seen_texts: list[str] = []

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        self.seen_texts = list(texts)
        return self.embeddings


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "retrieval_documents",
        "qdrant_dense_vector_name": "dense",
        "qdrant_sparse_vector_name": "sparse",
        "qdrant_dense_vector_size": 3,
        "embedding_model_tag": "test-embedding:v1",
        "dense_retrieval_limit": 3,
        "sparse_retrieval_limit": 3,
        "fused_top_n": 3,
        "rrf_k": 60,
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


def make_chunk(chunk_id: str, text: str, page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        filename="guide.md",
        page=page,
        section="Setup",
        chunk_id=chunk_id,
        text=text,
    )


def make_embedding(
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
) -> EmbeddedText:
    return EmbeddedText(
        dense=dense,
        sparse=models.SparseVector(indices=sparse_indices, values=sparse_values),
    )


def seed_collection(settings: AppSettings) -> QdrantClient:
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    chunks = [
        make_chunk("c1", "alpha exact", page=1),
        make_chunk("c2", "semantic only", page=2),
        make_chunk("c3", "alpha paraphrase", page=3),
    ]
    embeddings = [
        make_embedding([1.0, 0.0, 0.0], [10], [1.0]),
        make_embedding([0.0, 1.0, 0.0], [20], [1.0]),
        make_embedding([0.8, 0.1, 0.0], [10, 20], [0.8, 0.1]),
    ]
    ingest_chunks(repository, settings, chunks, StaticEmbeddingProvider(embeddings))
    return client


def test_retrieve_candidates_uses_qdrant_server_side_rrf() -> None:
    settings = make_settings()
    client = seed_collection(settings)
    repository = VectorRepository(client)
    query_provider = StaticEmbeddingProvider([make_embedding([1.0, 0.0, 0.0], [10], [1.0])])

    results = retrieve_candidates(repository, settings, "alpha", query_provider)

    assert query_provider.seen_texts == ["alpha"]
    assert [result.chunk_id for result in results] == ["c1", "c3", "c2"]
    assert results[0].filename == "guide.md"
    assert results[0].page == 1
    assert results[0].section == "Setup"
    assert results[0].text == "alpha exact"


def test_retrieve_candidates_falls_back_to_manual_rrf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings()
    client = seed_collection(settings)
    repository = VectorRepository(client)
    original_query_points = client.query_points
    hybrid_attempted = False

    def query_points(*args: Any, **kwargs: Any) -> object:
        nonlocal hybrid_attempted
        if kwargs.get("prefetch") is not None:
            hybrid_attempted = True
            raise ApiException("server-side fusion unavailable")
        return original_query_points(*args, **kwargs)

    monkeypatch.setattr(client, "query_points", query_points)
    query_provider = StaticEmbeddingProvider([make_embedding([1.0, 0.0, 0.0], [10], [1.0])])

    results = retrieve_candidates(repository, settings, "alpha", query_provider)

    assert hybrid_attempted is True
    assert [result.chunk_id for result in results] == ["c1", "c3", "c2"]
    assert results[0].score == pytest.approx(1 / 60 + 1 / 60)


def test_retrieve_candidates_rejects_empty_query() -> None:
    settings = make_settings()
    client = seed_collection(settings)
    repository = VectorRepository(client)

    with pytest.raises(RetrievalError, match="Query text"):
        retrieve_candidates(repository, settings, "   ", StaticEmbeddingProvider([]))


def test_retrieve_candidates_validates_before_embedding() -> None:
    class RefusingRepository:
        def ensure_ready(self, settings: AppSettings) -> None:
            raise RetrievalError("refused")

        def hybrid_search(
            self, settings: AppSettings, query_embedding: EmbeddedText
        ) -> list[models.ScoredPoint]:
            raise AssertionError("hybrid_search must not run when ensure_ready refuses")

    class FailingEmbeddingProvider:
        def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
            raise AssertionError("embedding must not run when ensure_ready refuses")

    settings = make_settings()

    with pytest.raises(RetrievalError, match="refused"):
        retrieve_candidates(RefusingRepository(), settings, "alpha", FailingEmbeddingProvider())


def test_scored_point_to_chunk_requires_citation_payload() -> None:
    point = models.ScoredPoint(
        id=str(uuid5(NAMESPACE_URL, "bad")),
        version=0,
        score=1.0,
        payload={"filename": "guide.md"},
    )

    with pytest.raises(RetrievalPayloadError, match="page"):
        scored_point_to_chunk(point)
