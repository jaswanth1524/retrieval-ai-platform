from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import pytest
from qdrant_client import QdrantClient, models

from api.documents import DocumentChunk, EmptyDocumentError
from api.embeddings import EmbeddedText, EmbeddingError
from api.ingestion import IngestionError, ingest_chunks, point_id_for_chunk
from api.qdrant_schema import CollectionSchemaError
from api.repository import VectorRepository
from api.settings import AppSettings


class FakeEmbeddingProvider:
    def __init__(self, embeddings: list[EmbeddedText]) -> None:
        self.embeddings = embeddings
        self.seen_texts: list[str] = []

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        self.seen_texts = list(texts)
        return self.embeddings


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "ingest_documents",
        "qdrant_dense_vector_name": "dense",
        "qdrant_sparse_vector_name": "sparse",
        "qdrant_dense_vector_size": 3,
        "embedding_model_tag": "test-embedding:v1",
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


def make_chunk(
    *,
    filename: str = "guide.md",
    page: int = 1,
    section: str = "Intro",
    chunk_id: str = "chunk-a",
    text: str = "alpha beta",
) -> DocumentChunk:
    return DocumentChunk(
        filename=filename,
        page=page,
        section=section,
        chunk_id=chunk_id,
        text=text,
    )


def make_embedding(value: float = 0.1) -> EmbeddedText:
    return EmbeddedText(
        dense=[value, value + 0.1, value + 0.2],
        sparse=models.SparseVector(indices=[1, 3], values=[0.7, 0.2]),
    )


def test_point_id_for_chunk_is_stable_valid_uuid() -> None:
    chunk = make_chunk()

    point_id = point_id_for_chunk(chunk)

    assert point_id == point_id_for_chunk(chunk)
    assert str(UUID(point_id)) == point_id


def test_ingest_chunks_upserts_named_vectors_and_payload() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    chunk = make_chunk(page=3, section="Setup", text="run docker compose")
    provider = FakeEmbeddingProvider([make_embedding()])

    result = ingest_chunks(repository, settings, [chunk], provider)

    assert result.points_count == 1
    assert provider.seen_texts == ["run docker compose"]
    records, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        with_payload=True,
        with_vectors=True,
    )
    assert len(records) == 1
    record = records[0]
    assert record.id == point_id_for_chunk(chunk)
    assert record.payload == {
        "filename": "guide.md",
        "page": 3,
        "section": "Setup",
        "chunk_id": "chunk-a",
        "text": "run docker compose",
    }
    assert isinstance(record.vector, dict)
    assert set(record.vector) == {"dense", "sparse"}
    assert isinstance(record.vector["sparse"], models.SparseVector)


def test_ingest_chunks_replaces_prior_points_for_same_filename() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    original = make_chunk(chunk_id="old", text="original text")
    ingest_chunks(repository, settings, [original], FakeEmbeddingProvider([make_embedding()]))

    # An edited re-upload of the same file yields a new chunk_id (new point id) — the
    # old point must end up deleted as stale, not left behind alongside the new one.
    edited = make_chunk(chunk_id="new", text="edited text")
    ingest_chunks(repository, settings, [edited], FakeEmbeddingProvider([make_embedding(0.2)]))

    records, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        with_payload=True,
        with_vectors=False,
    )
    assert len(records) == 1
    assert records[0].payload is not None
    assert records[0].payload["chunk_id"] == "new"


def test_ingest_chunks_keeps_matching_point_when_chunk_is_unchanged() -> None:
    """A chunk whose id/text is unchanged across re-ingests must NOT be treated as
    stale and deleted — only ids absent from the new upsert should be removed."""

    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    unchanged = make_chunk(chunk_id="stable", text="same text")
    ingest_chunks(repository, settings, [unchanged], FakeEmbeddingProvider([make_embedding()]))

    new_chunk = make_chunk(chunk_id="added", text="new text")
    ingest_chunks(
        repository,
        settings,
        [unchanged, new_chunk],
        FakeEmbeddingProvider([make_embedding(), make_embedding(0.2)]),
    )

    records, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        with_payload=True,
        with_vectors=False,
    )
    chunk_ids = {record.payload["chunk_id"] for record in records if record.payload}
    assert chunk_ids == {"stable", "added"}


def test_ingest_chunks_validates_before_embedding() -> None:
    class RefusingRepository:
        def ensure_ready(self, settings: AppSettings) -> None:
            raise CollectionSchemaError("refused")

        def upsert(self, settings: AppSettings, points: object) -> None:
            raise AssertionError("upsert must not run when ensure_ready refuses")

        def point_ids_for_filename(self, settings: AppSettings, filename: str) -> list[str]:
            raise AssertionError("lookup must not run when ensure_ready refuses")

        def delete_by_ids(self, settings: AppSettings, point_ids: object) -> None:
            raise AssertionError("delete must not run when ensure_ready refuses")

    class FailingEmbeddingProvider:
        def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
            raise AssertionError("embedding must not run when ensure_ready refuses")

    settings = make_settings()

    with pytest.raises(CollectionSchemaError, match="refused"):
        ingest_chunks(RefusingRepository(), settings, [make_chunk()], FailingEmbeddingProvider())


def test_ingest_chunks_rejects_empty_input() -> None:
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    provider = FakeEmbeddingProvider([])

    with pytest.raises(EmptyDocumentError):
        ingest_chunks(repository, settings, [], provider)


def test_ingest_chunks_rejects_embedding_count_mismatch() -> None:
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    provider = FakeEmbeddingProvider([])

    with pytest.raises(EmbeddingError, match="count mismatch"):
        ingest_chunks(repository, settings, [make_chunk()], provider)


def test_ingest_chunks_upsert_failure_leaves_previous_version_intact() -> None:
    """Points upsert first, stale cleanup happens after — so an upsert failure must
    leave whatever was previously indexed fully intact and queryable, not wipe it."""

    class FailsToUpsert:
        def __init__(self, inner: VectorRepository) -> None:
            self._inner = inner

        def ensure_ready(self, settings: AppSettings) -> None:
            self._inner.ensure_ready(settings)

        def point_ids_for_filename(self, settings: AppSettings, filename: str) -> list[str]:
            return self._inner.point_ids_for_filename(settings, filename)

        def upsert(self, settings: AppSettings, points: object) -> None:
            raise ConnectionError("connection refused")

        def delete_by_ids(self, settings: AppSettings, point_ids: object) -> None:
            raise AssertionError("delete must not run when upsert fails")

    settings = make_settings()
    client = QdrantClient(":memory:")
    inner = VectorRepository(client)
    original = make_chunk(chunk_id="old", text="original text")
    ingest_chunks(inner, settings, [original], FakeEmbeddingProvider([make_embedding()]))

    failing_repository = FailsToUpsert(inner)
    provider = FakeEmbeddingProvider([make_embedding(0.2)])
    edited = make_chunk(chunk_id="new", text="edited text")

    with pytest.raises(IngestionError, match="previous version remains indexed"):
        ingest_chunks(failing_repository, settings, [edited], provider)

    records, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        with_payload=True,
        with_vectors=False,
    )
    assert len(records) == 1
    assert records[0].payload is not None
    assert records[0].payload["chunk_id"] == "old"


def test_ingest_chunks_wraps_stale_cleanup_failure_distinctly() -> None:
    """A failure deleting stale points is milder than an upsert failure — the new
    content is already indexed — so it gets its own distinct error message."""

    class FailsToDeleteStale:
        def __init__(self, inner: VectorRepository) -> None:
            self._inner = inner

        def ensure_ready(self, settings: AppSettings) -> None:
            self._inner.ensure_ready(settings)

        def point_ids_for_filename(self, settings: AppSettings, filename: str) -> list[str]:
            return self._inner.point_ids_for_filename(settings, filename)

        def upsert(self, settings: AppSettings, points: object) -> None:
            self._inner.upsert(settings, points)  # type: ignore[arg-type]

        def delete_by_ids(self, settings: AppSettings, point_ids: object) -> None:
            raise ConnectionError("connection refused")

    settings = make_settings()
    client = QdrantClient(":memory:")
    inner = VectorRepository(client)
    original = make_chunk(chunk_id="old", text="original text")
    ingest_chunks(inner, settings, [original], FakeEmbeddingProvider([make_embedding()]))

    failing_repository = FailsToDeleteStale(inner)
    provider = FakeEmbeddingProvider([make_embedding(0.2)])
    edited = make_chunk(chunk_id="new", text="edited text")

    with pytest.raises(IngestionError, match="failed to remove"):
        ingest_chunks(failing_repository, settings, [edited], provider)

    records, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        with_payload=True,
        with_vectors=False,
    )
    chunk_ids = {record.payload["chunk_id"] for record in records if record.payload}
    # The new content IS indexed despite the error — only the stale "old" point
    # failed to clean up, proving this failure mode is milder than an upsert failure.
    assert chunk_ids == {"old", "new"}
