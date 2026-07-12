"""Chunk embedding and point construction for ingested documents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models

from api.documents import DocumentChunk, EmptyDocumentError
from api.embeddings import EmbeddedText, EmbeddingError
from api.settings import AppSettings


class IngestionError(RuntimeError):
    """Raised when indexing or stale-chunk cleanup fails during ingestion.

    Points use deterministic IDs derived from filename/page/section/chunk_id (see
    ``point_id_for_chunk``), so re-ingesting an edited chunk overwrites its prior
    version in place on upsert. ``ingest_chunks`` upserts the new points *first*, then
    deletes only the point IDs that are now stale (the filename's previously indexed
    IDs minus the new ones) — so an upsert failure leaves the previous version of the
    document fully intact and queryable, rather than leaving it un-indexed. A failure
    during the stale-cleanup step is milder still: the new content is already correctly
    indexed, only a few now-orphaned old chunks remain until the next re-ingest retries
    the cleanup.
    """


class EmbeddingProvider(Protocol):
    """Embedding provider surface used by ingestion."""

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]: ...


class WriteRepository(Protocol):
    """Vector-store surface ingestion needs: index chunks, clean up stale ones."""

    def ensure_ready(self, settings: AppSettings) -> None: ...

    def upsert(self, settings: AppSettings, points: Sequence[models.PointStruct]) -> None: ...

    def point_ids_for_filename(self, settings: AppSettings, filename: str) -> list[str]: ...

    def delete_by_ids(self, settings: AppSettings, point_ids: Sequence[str]) -> None: ...


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
    on_progress: Callable[[int, int], None] | None = None,
) -> IngestResult:
    """Embed chunks locally and index them, replacing any prior points for the file.

    Chunks are embedded and upserted in batches of ``settings.ingest_batch_size`` so a
    single huge document reports incremental progress (via ``on_progress(done, total)``)
    and never holds one giant embedding call in memory. Stale-cleanup ordering is
    unchanged: every filename's currently-indexed IDs are snapshotted *before* the
    first batch is written, and only IDs absent from every new batch are deleted
    afterward — a failure partway through a multi-batch ingest still leaves whatever
    batches already succeeded correctly indexed, never wiped.
    """

    if not chunks:
        raise EmptyDocumentError("No document chunks were provided for ingestion.")

    # Validate before embedding: an embedding-tag mismatch should refuse the ingest
    # without paying for embedding work on chunks that can't be written anyway.
    repository.ensure_ready(settings)

    # Snapshot each filename's currently-indexed point IDs *before* upserting — this
    # is what "stale" gets computed against once the new points are written.
    filenames = sorted({chunk.filename for chunk in chunks})
    stale_candidate_ids: set[str] = set()
    for filename in filenames:
        stale_candidate_ids.update(repository.point_ids_for_filename(settings, filename))

    batch_size = int(settings.ingest_batch_size)
    total = len(chunks)
    new_ids: set[str] = set()
    batches_written = 0
    for start in range(0, total, batch_size):
        batch = list(chunks[start : start + batch_size])
        embeddings = embedding_provider.embed_texts([_contextual_embedding_text(c) for c in batch])
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                f"Embedding count mismatch: got {len(embeddings)}, expected {len(batch)}."
            )
        points = [
            build_point(chunk, embedding, settings)
            for chunk, embedding in zip(batch, embeddings, strict=True)
        ]
        try:
            repository.upsert(settings, points)
        except Exception as exc:
            if batches_written == 0:
                raise IngestionError(
                    f"Indexing failed while updating {', '.join(filenames)}; the previous "
                    "version remains indexed. Please retry the upload."
                ) from exc
            raise IngestionError(
                f"Indexing failed partway through updating {', '.join(filenames)} "
                f"({start} of {total} chunks already indexed alongside the previous "
                "version). Please retry the upload."
            ) from exc
        new_ids.update(str(point.id) for point in points)
        batches_written += 1
        if on_progress is not None:
            on_progress(min(start + batch_size, total), total)

    stale_ids = stale_candidate_ids - new_ids
    if stale_ids:
        try:
            repository.delete_by_ids(settings, list(stale_ids))
        except Exception as exc:
            raise IngestionError(
                f"Indexed the new version of {', '.join(filenames)}, but failed to "
                f"remove {len(stale_ids)} stale chunk(s) from the previous version. "
                "Re-ingesting again will complete the cleanup."
            ) from exc

    return IngestResult(
        collection_name=settings.qdrant_collection,
        points_count=total,
    )


def _contextual_embedding_text(chunk: DocumentChunk) -> str:
    """Prefix a chunk's embedded text with its filename/section (contextual retrieval).

    Only the *embedded* text changes — the stored payload keeps ``chunk.text`` raw
    (see ``build_point``), so citations and the generation prompt are unaffected.
    Giving both the dense and sparse (BM25) models the filename/section as context
    improves matches on queries that reference a document or heading by name, which a
    bare chunk of body text wouldn't otherwise surface.
    """

    return f"{chunk.filename} › {chunk.section}\n{chunk.text}"


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
