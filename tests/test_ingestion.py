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
    return AppSettings(**defaults)


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

    # An edited re-upload of the same file yields a new chunk_id (new point id).
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


def test_ingest_chunks_validates_before_embedding() -> None:
    class RefusingRepository:
        def ensure_ready(self, settings: AppSettings) -> None:
            raise CollectionSchemaError("refused")

        def upsert(self, settings: AppSettings, points: object) -> None:
            raise AssertionError("upsert must not run when ensure_ready refuses")

        def delete_by_filename(self, settings: AppSettings, filenames: object) -> None:
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


def test_ingest_chunks_wraps_upsert_failure_with_un_indexed_warning() -> None:
    """Delete-then-upsert is forced (deterministic point IDs share the filename, so
    upserting first would let a later delete wipe the new points too) — but that
    means an upsert failure after a successful delete leaves the document with zero
    indexed chunks. The caller must get a clear signal to retry, not a bare
    connection-error traceback."""

    class DeletesThenFailsToUpsert:
        def __init__(self, inner: VectorRepository) -> None:
            self._inner = inner
            self.deleted = False

        def ensure_ready(self, settings: AppSettings) -> None:
            self._inner.ensure_ready(settings)

        def delete_by_filename(self, settings: AppSettings, filenames: object) -> None:
            self.deleted = True

        def upsert(self, settings: AppSettings, points: object) -> None:
            raise ConnectionError("connection refused")

    settings = make_settings()
    repository = DeletesThenFailsToUpsert(VectorRepository(QdrantClient(":memory:")))
    provider = FakeEmbeddingProvider([make_embedding()])

    with pytest.raises(IngestionError, match="un-indexed"):
        ingest_chunks(repository, settings, [make_chunk()], provider)

    assert repository.deleted is True
