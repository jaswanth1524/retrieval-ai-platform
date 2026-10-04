"""Chunk embedding and point construction for ingested documents."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from threading import Lock
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models

from api.documents import DocumentChunk, EmptyDocumentError, contextual_text
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

    def tags_for_filename(self, settings: AppSettings, filename: str) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class IngestResult:
    """Result returned after chunks are upserted."""

    collection_name: str
    points_count: int


# Serializes every read-modify-write of one filename's points. The ingest executor runs
# multiple workers (see get_ingest_executor), so two concurrent ingests of the same
# filename would otherwise each snapshot the same pre-ingest point IDs and each subtract
# only their own new IDs from the stale set — the loser's freshly written points get
# deleted by the winner's cleanup. DELETE /documents/{filename} has the same shape
# (snapshot ids, then delete them), so it takes this lock too; guarding only the ingest
# path would still let a delete landing mid-ingest remove points the ingest just wrote
# and then report success. One process-wide registry, coarse per filename, is enough:
# same-filename concurrency is rare and this only serializes that case.
@dataclass
class _LockEntry:
    """One filename's lock plus a count of threads currently using it."""

    lock: Lock
    users: int


_filename_locks_guard = Lock()
_filename_locks: dict[str, _LockEntry] = {}


@contextmanager
def filename_write_lock(filename: str) -> Iterator[None]:
    """Hold the process-wide write lock for ``filename`` for the duration of the block.

    Entries are reference-counted and removed once the last user releases, so the
    registry stays proportional to *in-flight* work rather than growing by one Lock for
    every filename the process has ever seen (uploads and deletes are client-driven, so
    that set is unbounded). ``users`` is incremented under the guard *before* acquiring,
    which is what stops a concurrent releaser from evicting an entry someone is about to
    block on; the ``is entry`` identity check keeps a slow releaser from deleting a
    fresher entry that replaced its own.

    Not reentrant (a plain ``Lock``): never enter this for a filename from inside a span
    that already holds it for that same filename.
    """

    with _filename_locks_guard:
        entry = _filename_locks.get(filename)
        if entry is None:
            entry = _LockEntry(lock=Lock(), users=0)
            _filename_locks[filename] = entry
        entry.users += 1
    try:
        with entry.lock:
            yield
    finally:
        with _filename_locks_guard:
            entry.users -= 1
            if entry.users == 0 and _filename_locks.get(filename) is entry:
                del _filename_locks[filename]


class IngestSequencer:
    """Stops an older ingest of a filename from overwriting a newer one.

    Ingests of one filename are serialized by ``filename_write_lock``, but in the order
    they *reach* the lock — after parsing and embedding — not the order they were
    admitted. So a large v1 upload finishing after a small v2, or a queued re-index of
    the old original finishing after a fresh re-upload, would put the older content
    back. Every ingest takes a sequence number at admission; one that reaches the lock
    after a newer ingest of the same filename has been applied is refused.

    "Applied" (not "admitted") is what counts, so a newer upload that then fails doesn't
    also block the older one. Entries are dropped once a filename has no ingest in
    flight, so the registry tracks in-flight work only.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._next = 0
        self._in_flight: dict[str, int] = {}
        # Newest applied (sequence, was_a_delete) per filename.
        self._applied: dict[str, tuple[int, bool]] = {}

    def admit(self, filename: str) -> int:
        with self._lock:
            self._next += 1
            self._in_flight[filename] = self._in_flight.get(filename, 0) + 1
            return self._next

    def check(self, filename: str, sequence: int) -> None:
        """Raise if a newer ingest (or delete) of ``filename`` was applied. Call under
        its lock."""

        with self._lock:
            applied, was_delete = self._applied.get(filename, (0, False))
            if applied <= sequence:
                return
            if was_delete:
                raise IngestionError(
                    f"'{filename}' was deleted after this upload was queued; it was not indexed."
                )
            raise IngestionError(
                f"A newer version of '{filename}' was indexed while this one was "
                "waiting; kept the newer version."
            )

    def mark_applied(self, filename: str, sequence: int, *, deleted: bool = False) -> None:
        """Record an applied ingest — or, with ``deleted``, a delete: an upload queued
        before a delete would otherwise index the document straight back."""

        with self._lock:
            if sequence > self._applied.get(filename, (0, False))[0]:
                self._applied[filename] = (sequence, deleted)

    def release(self, filename: str) -> None:
        with self._lock:
            remaining = self._in_flight.get(filename, 0) - 1
            if remaining > 0:
                self._in_flight[filename] = remaining
                return
            self._in_flight.pop(filename, None)
            self._applied.pop(filename, None)


ingest_sequencer = IngestSequencer()


def ingest_chunks(
    repository: WriteRepository,
    settings: AppSettings,
    chunks: Sequence[DocumentChunk],
    embedding_provider: EmbeddingProvider,
    on_progress: Callable[[int, int], None] | None = None,
    on_indexed: Callable[[], None] | None = None,
    precondition: Callable[[], None] | None = None,
    carry_tags: bool = False,
    before_first_write: Callable[[], None] | None = None,
    tags: tuple[str, ...] | None = None,
) -> IngestResult:
    """Embed chunks locally and index them, replacing any prior points for the file.

    Chunks are embedded and upserted in batches of ``settings.ingest_batch_size`` so a
    single huge document reports incremental progress (via ``on_progress(done, total)``)
    and never holds one giant embedding call in memory. Stale-cleanup ordering is
    unchanged: every filename's currently-indexed IDs are snapshotted *before* the
    first batch is written, and only IDs absent from every new batch are deleted
    afterward — a failure partway through a multi-batch ingest still leaves whatever
    batches already succeeded correctly indexed, never wiped.

    ``on_indexed`` runs once the new version is fully written (before stale cleanup),
    still inside the filename lock — for side effects that must be atomic with the
    index (the stored original), so a concurrent DELETE can't interleave with them.

    ``precondition`` runs first thing under the same lock and aborts the ingest by
    raising — a re-index checks there that its document wasn't deleted while the job
    waited in the queue, since a DELETE in that window would otherwise be undone.

    ``carry_tags`` stamps each chunk with its document's current tags, read under the
    same lock ``PATCH /documents/{filename}/tags`` takes — read any earlier and a tag
    change landing while this ingest parsed or queued was overwritten by the old tags.
    Tags aren't part of the embedded text, so this never changes a vector. ``tags``,
    when given, sets the tags outright instead (a restore from a backup).

    ``before_first_write`` runs once, right before the first batch is upserted, and
    aborts the ingest by raising — the last point where nothing has changed yet (job
    cancellation).
    """

    if not chunks:
        raise EmptyDocumentError("No document chunks were provided for ingestion.")

    # Validate before embedding: an embedding-tag mismatch should refuse the ingest
    # without paying for embedding work on chunks that can't be written anyway.
    repository.ensure_ready(settings)

    # Filenames are locked in sorted order (already sorted below) so two multi-file
    # ingests that share more than one filename can never deadlock on each other.
    filenames = sorted({chunk.filename for chunk in chunks})

    with ExitStack() as locks:
        for filename in filenames:
            locks.enter_context(filename_write_lock(filename))

        if precondition is not None:
            precondition()

        if tags is not None:
            chunks = [replace(chunk, tags=tags) for chunk in chunks]
        elif carry_tags:
            current_tags = {
                filename: repository.tags_for_filename(settings, filename) for filename in filenames
            }
            chunks = [replace(chunk, tags=current_tags[chunk.filename]) for chunk in chunks]

        # Snapshot each filename's currently-indexed point IDs *before* upserting —
        # this is what "stale" gets computed against once the new points are written.
        # Holding each filename's lock for the whole snapshot-upsert-delete span is
        # what prevents a concurrent same-filename ingest from deleting these points.
        stale_candidate_ids: set[str] = set()
        for filename in filenames:
            stale_candidate_ids.update(repository.point_ids_for_filename(settings, filename))

        batch_size = int(settings.ingest_batch_size)
        total = len(chunks)
        new_ids: set[str] = set()
        batches_written = 0
        for start in range(0, total, batch_size):
            batch = list(chunks[start : start + batch_size])
            embeddings = embedding_provider.embed_texts(
                [_contextual_embedding_text(c) for c in batch]
            )
            if len(embeddings) != len(batch):
                raise EmbeddingError(
                    f"Embedding count mismatch: got {len(embeddings)}, expected {len(batch)}."
                )
            points = [
                build_point(chunk, embedding, settings)
                for chunk, embedding in zip(batch, embeddings, strict=True)
            ]
            if batches_written == 0 and before_first_write is not None:
                before_first_write()
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

        # Every batch of the new version is live from here on, so its original bytes
        # belong to it now — before stale cleanup, whose failure still leaves the new
        # version indexed (and would otherwise keep serving the previous original).
        if on_indexed is not None:
            on_indexed()

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

    return contextual_text(chunk.filename, chunk.section, chunk.text)


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
