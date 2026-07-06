"""Chunk embedding and point construction for ingested documents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models

from api.documents import DocumentChunk, EmptyDocumentError
from api.embeddings import EmbeddedText, EmbeddingError
from api.settings import AppSettings


class EmbeddingProvider(Protocol):
    """Embedding provider surface used by ingestion."""

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]: ...


class WriteRepository(Protocol):
    """Vector-store surface ingestion needs: index chunks, clean up stale ones."""

    def ensure_ready(self, settings: AppSettings) -> None: ...

    def upsert(self, settings: AppSettings, points: Sequence[models.PointStruct]) -> None: ...

    def delete_by_filename(self, settings: AppSettings, filenames: Sequence[str]) -> None: ...


@dataclass(frozen=True)
class IngestResult:
    """Result returned after chunks are upserted."""

    collection_name: str
    points_count: int


def ingest_chunks(
    repository: WriteRepository,
    settings: AppSettings,
    chunks: Sequence[DocumentChunk],
    embedding_provider: EmbeddingProvider,
) -> IngestResult:
    """Embed chunks locally and index them, replacing any prior points for the file."""

    if not chunks:
        raise EmptyDocumentError("No document chunks were provided for ingestion.")

    # Validate before embedding: an embedding-tag mismatch should refuse the ingest
    # without paying for embedding work on chunks that can't be written anyway.
    repository.ensure_ready(settings)
    embeddings = embedding_provider.embed_texts([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise EmbeddingError(
            f"Embedding count mismatch: got {len(embeddings)}, expected {len(chunks)}."
        )

    points = [
        build_point(chunk, embedding, settings)
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    repository.delete_by_filename(settings, [chunk.filename for chunk in chunks])
    repository.upsert(settings, points)
    return IngestResult(
        collection_name=settings.qdrant_collection,
        points_count=len(points),
    )


def build_point(
    chunk: DocumentChunk,
    embedding: EmbeddedText,
    settings: AppSettings,
) -> models.PointStruct:
    """Build a Qdrant point with named dense and sparse vectors."""

    return models.PointStruct(
        id=point_id_for_chunk(chunk),
        vector={
            settings.qdrant_dense_vector_name: embedding.dense,
            settings.qdrant_sparse_vector_name: embedding.sparse,
        },
        payload=chunk.to_payload(),
    )


def point_id_for_chunk(chunk: DocumentChunk) -> str:
    """Return a deterministic valid Qdrant UUID for a document chunk."""

    raw = f"docrag:{chunk.filename}:{chunk.page}:{chunk.section}:{chunk.chunk_id}"
    return str(uuid5(NAMESPACE_URL, raw))
