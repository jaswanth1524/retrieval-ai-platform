"""Tests that need a real Qdrant server, not the in-process ``:memory:`` client.

The in-process client is a Python reimplementation: it has no payload indexes, and its
hybrid query and sparse-vector scoring are not the server's. These run only when
``DOCRAG_TEST_QDRANT_URL`` points at a server (CI's ``qdrant`` job starts
``qdrant/qdrant:v1.19.1``, the version docker-compose pins); otherwise they skip.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator, Sequence
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from api.documents import DocumentChunk
from api.embeddings import EmbeddedText
from api.ingestion import ingest_chunks
from api.qdrant_schema import clear_readiness_cache
from api.repository import (
    VectorRepository,
    _server_side_hybrid_unsupported,
    clear_hybrid_fallback_cache,
)
from api.settings import AppSettings

QDRANT_URL = os.environ.get("DOCRAG_TEST_QDRANT_URL", "")
# Compose runs the API with QDRANT_PREFER_GRPC=true, so every test runs over both
# transports: gRPC converts each request (the IDF modifier, set_payload's filter
# selector) through its own protobuf types.
QDRANT_GRPC_PORT = int(os.environ.get("DOCRAG_TEST_QDRANT_GRPC_PORT", "6334"))

pytestmark = pytest.mark.skipif(
    not QDRANT_URL, reason="set DOCRAG_TEST_QDRANT_URL to run against a Qdrant server"
)


class StaticEmbeddingProvider:
    def __init__(self, embeddings: list[EmbeddedText]) -> None:
        self.embeddings = embeddings

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        return self.embeddings[: len(texts)]


def embedding(dense: list[float], indices: list[int], values: list[float]) -> EmbeddedText:
    return EmbeddedText(dense=dense, sparse=models.SparseVector(indices=indices, values=values))


def chunk(chunk_id: str, text: str, filename: str = "guide.md", page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        filename=filename, page=page, section="Setup", chunk_id=chunk_id, text=text
    )


@pytest.fixture(params=[False, True], ids=["rest", "grpc"])
def server(request: pytest.FixtureRequest) -> Generator[tuple[QdrantClient, AppSettings]]:
    clear_readiness_cache()
    clear_hybrid_fallback_cache()
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=request.param, grpc_port=QDRANT_GRPC_PORT)
    defaults: dict[str, Any] = {
        "qdrant_url": QDRANT_URL,
        # Unique per test: the server outlives the test and is shared by all of them.
        "qdrant_collection": f"docrag_test_{uuid.uuid4().hex}",
        "qdrant_dense_vector_size": 3,
        "embedding_model_tag": "test-embedding:v1",
        "dense_retrieval_limit": 3,
        "sparse_retrieval_limit": 3,
        "fused_top_n": 3,
    }
    settings = AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]
    try:
        yield client, settings
    finally:
        client.delete_collection(settings.qdrant_collection)
        clear_readiness_cache()
        clear_hybrid_fallback_cache()


def seed(client: QdrantClient, settings: AppSettings) -> VectorRepository:
    repository = VectorRepository(client)
    ingest_chunks(
        repository,
        settings,
        [
            chunk("c1", "alpha exact", page=1),
            chunk("c2", "semantic only", page=2),
            chunk("c3", "alpha paraphrase", filename="notes.md", page=1),
        ],
        StaticEmbeddingProvider(
            [
                embedding([1.0, 0.0, 0.0], [10], [1.0]),
                embedding([0.0, 1.0, 0.0], [20], [1.0]),
                embedding([0.8, 0.1, 0.0], [10, 20], [0.8, 0.1]),
            ]
        ),
    )
    return repository


def test_the_server_runs_the_hybrid_query_itself(server: tuple[QdrantClient, AppSettings]) -> None:
    client, settings = server
    repository = seed(client, settings)

    points = repository.hybrid_search(settings, embedding([1.0, 0.0, 0.0], [10], [1.0]))

    # No fallback to application-side fusion was needed.
    assert client not in _server_side_hybrid_unsupported
    assert [point.payload["chunk_id"] for point in points if point.payload][:2] == ["c1", "c3"]


def test_filename_scope_and_metadata_use_the_servers_payload_index(
    server: tuple[QdrantClient, AppSettings],
) -> None:
    client, settings = server
    repository = seed(client, settings)

    scoped = repository.hybrid_search(
        settings, embedding([1.0, 0.0, 0.0], [10], [1.0]), filenames=["notes.md"]
    )
    metadata = repository.filename_metadata(settings)
    repository.delete_by_ids(settings, repository.point_ids_for_filename(settings, "guide.md"))

    assert {point.payload["filename"] for point in scoped if point.payload} == {"notes.md"}
    assert {name: meta.chunk_count for name, meta in metadata.items()} == {
        "guide.md": 2,
        "notes.md": 1,
    }
    assert sorted(repository.filename_chunk_counts(settings)) == ["notes.md"]


def test_bm25_ranks_a_rare_term_above_a_common_one_after_migration(
    server: tuple[QdrantClient, AppSettings],
) -> None:
    """A collection from before the IDF modifier: term 1 ("the") is in every document,
    term 2 ("zebra") in one. Frequency alone ranks the document with the most "the"s
    first; with IDF, the one containing the rare query term wins."""

    client, settings = server
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            settings.qdrant_dense_vector_name: models.VectorParams(
                size=3, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={settings.qdrant_sparse_vector_name: models.SparseVectorParams()},
        metadata={
            "embedding_model_tag": settings.embedding_model_tag,
            "sparse_embedding_model": settings.sparse_embedding_model,
        },
    )
    sparse_docs = {1: ([1], [3.0]), 2: ([1, 2], [1.0, 1.0]), 3: ([1], [2.0])}
    client.upsert(
        settings.qdrant_collection,
        [
            models.PointStruct(
                id=point_id,
                vector={
                    settings.qdrant_dense_vector_name: [1.0, 0.0, 0.0],
                    settings.qdrant_sparse_vector_name: models.SparseVector(
                        indices=indices, values=values
                    ),
                },
            )
            for point_id, (indices, values) in sparse_docs.items()
        ],
        wait=True,
    )
    query = models.SparseVector(indices=[1, 2], values=[1.0, 1.0])

    def sparse_ranking() -> list[int | str]:
        return [
            point.id
            for point in client.query_points(
                settings.qdrant_collection, query=query, using=settings.qdrant_sparse_vector_name
            ).points
        ]

    assert sparse_ranking()[0] == 1
    VectorRepository(client).ensure_ready(settings)
    assert sparse_ranking()[0] == 2
    # Read back as migrated on the next cold start, not re-migrated every time.
    sparse = client.get_collection(settings.qdrant_collection).config.params.sparse_vectors
    assert sparse is not None
    assert sparse[settings.qdrant_sparse_vector_name].modifier == models.Modifier.IDF
    clear_readiness_cache()
    VectorRepository(client).ensure_ready(settings)
    assert sparse_ranking()[0] == 2


def test_tags_are_set_in_place_and_scope_the_hybrid_query(
    server: tuple[QdrantClient, AppSettings],
) -> None:
    client, settings = server
    repository = seed(client, settings)

    repository.set_tags(settings, "notes.md", ["legal"])
    scoped = repository.hybrid_search(
        settings, embedding([1.0, 0.0, 0.0], [10], [1.0]), tags=["legal"]
    )

    assert {point.payload["filename"] for point in scoped if point.payload} == {"notes.md"}
    assert repository.tags_for_filename(settings, "notes.md") == ("legal",)
    assert repository.filename_metadata(settings)["notes.md"].tags == ("legal",)
    assert "tags" in (client.get_collection(settings.qdrant_collection).payload_schema or {})


def test_a_collection_deleted_out_of_band_is_recreated(
    server: tuple[QdrantClient, AppSettings],
) -> None:
    client, settings = server
    repository = seed(client, settings)
    client.delete_collection(settings.qdrant_collection)

    assert repository.filename_metadata(settings) == {}
    client.delete_collection(settings.qdrant_collection)
    assert repository.hybrid_search(settings, embedding([1.0, 0.0, 0.0], [10], [1.0])) == []
    assert client.collection_exists(settings.qdrant_collection)


def test_a_document_is_paged_by_ordinal_range(server: tuple[QdrantClient, AppSettings]) -> None:
    client, settings = server
    repository = VectorRepository(client)
    ingest_chunks(
        repository,
        settings,
        [
            DocumentChunk(
                filename="long.md", page=1, section="S", chunk_id=f"k{i}", text=f"t{i}", ordinal=i
            )
            for i in range(1, 7)
        ],
        StaticEmbeddingProvider([embedding([1.0, 0.0, 0.0], [10], [1.0])] * 6),
    )

    assert repository.chunk_count_for_filename(settings, "long.md") == 6
    assert repository.chunk_ordinal_of(settings, "long.md", "k4") == 4
    assert repository.chunk_ordinal_of(settings, "long.md", "nope") is None
    page = repository.chunks_in_ordinal_range(settings, "long.md", 3, 5)
    assert [payload["chunk_id"] for payload in page] == ["k3", "k4", "k5"]
