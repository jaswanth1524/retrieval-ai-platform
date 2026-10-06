from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from api.documents import DocumentChunk
from api.embeddings import EmbeddedText
from api.ingestion import ingest_chunks
from api.qdrant_schema import VectorStoreUnavailableError
from api.repository import VectorRepository
from api.retrieval import (
    RetrievalPayloadError,
    RetrievedChunk,
    points_to_chunks,
    scored_point_to_chunk,
)
from api.settings import AppSettings
from tests.factories import in_memory_qdrant, make_test_settings


def _unexpected_response(status_code: int) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status_code, reason_phrase="", content=b"", headers=httpx.Headers()
    )


class StaticEmbeddingProvider:
    def __init__(self, embeddings: list[EmbeddedText]) -> None:
        self.embeddings = embeddings
        self.seen_texts: list[str] = []

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        self.seen_texts = list(texts)
        return self.embeddings


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        **in_memory_qdrant("retrieval_documents"),
        "dense_retrieval_limit": 3,
        "sparse_retrieval_limit": 3,
        "fused_top_n": 3,
        "rrf_k": 60,
    }
    defaults.update(overrides)
    return make_test_settings(**defaults)


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


# The query "alpha": dense-closest to c1, and sharing c1's (and c3's) sparse term.
ALPHA_QUERY = make_embedding([1.0, 0.0, 0.0], [10], [1.0])


def search(
    repository: VectorRepository,
    settings: AppSettings,
    filenames: Sequence[str] | None = None,
) -> list[RetrievedChunk]:
    return points_to_chunks(repository.hybrid_search(settings, ALPHA_QUERY, filenames))


def test_hybrid_search_uses_qdrant_server_side_rrf() -> None:
    settings = make_settings()
    client = seed_collection(settings)
    repository = VectorRepository(client)

    results = search(repository, settings)

    assert [result.chunk_id for result in results] == ["c1", "c3", "c2"]
    assert results[0].filename == "guide.md"
    assert results[0].page == 1
    assert results[0].section == "Setup"
    assert results[0].text == "alpha exact"


def test_hybrid_search_falls_back_to_manual_rrf(
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
            # A 400 signals "this server/client combination doesn't support
            # prefetch+RRF" — the one case that should trigger manual fusion.
            raise _unexpected_response(400)
        return original_query_points(*args, **kwargs)

    monkeypatch.setattr(client, "query_points", query_points)
    results = search(repository, settings)

    assert hybrid_attempted is True
    assert [result.chunk_id for result in results] == ["c1", "c3", "c2"]
    assert results[0].score == pytest.approx(1 / 60 + 1 / 60)


def test_manual_fusion_fallback_is_remembered_per_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once a server rejects the hybrid query shape, later questions skip the doomed
    round trip — it failed identically on every question before."""

    settings = make_settings()
    client = seed_collection(settings)
    repository = VectorRepository(client)
    original_query_points = client.query_points
    hybrid_attempts = 0

    def query_points(*args: Any, **kwargs: Any) -> object:
        nonlocal hybrid_attempts
        if kwargs.get("prefetch") is not None:
            hybrid_attempts += 1
            raise _unexpected_response(400)
        return original_query_points(*args, **kwargs)

    monkeypatch.setattr(client, "query_points", query_points)
    first = search(repository, settings)
    second = search(VectorRepository(client), settings)

    assert hybrid_attempts == 1
    assert [r.chunk_id for r in first] == [r.chunk_id for r in second] == ["c1", "c3", "c2"]


def test_a_400_that_manual_fusion_also_fails_is_not_remembered_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 400 for another reason (a query vector of the wrong size) fails the manual
    query too; remembering it turned server-side hybrid off for the process."""

    from api.repository import _server_side_hybrid_unsupported

    settings = make_settings()
    client = seed_collection(settings)
    original_query_points = client.query_points
    broken = True

    def query_points(*args: Any, **kwargs: Any) -> object:
        if broken:
            raise _unexpected_response(400)
        return original_query_points(*args, **kwargs)

    monkeypatch.setattr(client, "query_points", query_points)
    with pytest.raises(UnexpectedResponse):
        search(VectorRepository(client), settings)

    assert client not in _server_side_hybrid_unsupported
    broken = False
    assert [r.chunk_id for r in search(VectorRepository(client), settings)] == ["c1", "c3", "c2"]


def test_hybrid_search_does_not_fall_back_on_real_query_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine server error (not an RRF-unsupported signal) must not be masked by a
    confusing second failure from the manual-fusion fallback — it should propagate.
    """

    settings = make_settings()
    client = seed_collection(settings)
    repository = VectorRepository(client)
    manual_fusion_attempted = False

    def query_points(*args: Any, **kwargs: Any) -> object:
        nonlocal manual_fusion_attempted
        if kwargs.get("prefetch") is not None:
            raise _unexpected_response(500)
        manual_fusion_attempted = True
        raise AssertionError("must not fall back to manual fusion for a real 500")

    monkeypatch.setattr(client, "query_points", query_points)
    with pytest.raises(UnexpectedResponse):
        search(repository, settings)

    assert manual_fusion_attempted is False


def test_hybrid_search_wraps_connection_failure_as_service_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings()
    client = seed_collection(settings)
    repository = VectorRepository(client)

    def query_points(*args: Any, **kwargs: Any) -> object:
        raise ResponseHandlingException(ConnectionError("connection refused"))

    monkeypatch.setattr(client, "query_points", query_points)
    with pytest.raises(VectorStoreUnavailableError, match="unreachable"):
        search(repository, settings)


def test_hybrid_search_filters_by_filenames() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    chunks = [
        DocumentChunk(filename="a.md", page=1, section="Setup", chunk_id="a1", text="alpha in a"),
        DocumentChunk(filename="b.md", page=1, section="Setup", chunk_id="b1", text="alpha in b"),
    ]
    embeddings = [
        make_embedding([1.0, 0.0, 0.0], [10], [1.0]),
        make_embedding([1.0, 0.0, 0.0], [10], [1.0]),
    ]
    ingest_chunks(repository, settings, chunks, StaticEmbeddingProvider(embeddings))
    results = search(repository, settings, filenames=["b.md"])

    assert [result.filename for result in results] == ["b.md"]


def test_scored_point_to_chunk_requires_citation_payload() -> None:
    point = models.ScoredPoint(
        id=str(uuid5(NAMESPACE_URL, "bad")),
        version=0,
        score=1.0,
        payload={"filename": "guide.md"},
    )

    with pytest.raises(RetrievalPayloadError, match="page"):
        scored_point_to_chunk(point)


def _good_point(chunk_id: str) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=str(uuid5(NAMESPACE_URL, chunk_id)),
        version=0,
        score=1.0,
        payload={
            "filename": "guide.md",
            "page": 1,
            "section": "Setup",
            "chunk_id": chunk_id,
            "text": "hi",
        },
    )


def _malformed_point() -> models.ScoredPoint:
    return models.ScoredPoint(
        id=str(uuid5(NAMESPACE_URL, "malformed")),
        version=0,
        score=1.0,
        payload={"filename": "guide.md"},
    )


def test_points_to_chunks_skips_malformed_points_when_others_are_valid() -> None:
    """One legacy/malformed point among many good candidates must not abort the
    whole query — only the malformed one is dropped."""

    points = [_good_point("c1"), _malformed_point(), _good_point("c2")]

    chunks = points_to_chunks(points)

    assert [chunk.chunk_id for chunk in chunks] == ["c1", "c2"]


def test_points_to_chunks_raises_only_when_all_points_are_malformed() -> None:
    with pytest.raises(RetrievalPayloadError, match="missing required payload"):
        points_to_chunks([_malformed_point(), _malformed_point()])


def test_points_to_chunks_returns_empty_for_no_points() -> None:
    assert points_to_chunks([]) == []
