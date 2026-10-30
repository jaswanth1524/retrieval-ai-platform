"""``GET /export``: one zip holding what DocRAG knows beyond its vectors.

- ``manifest.json``: every indexed document's metadata (chunks, pages, size, upload time,
  chunker version, tags, whether an original is kept) plus the settings that decide
  whether an index built elsewhere is compatible (embedding tag, sparse model).
- ``originals/<filename>``: the stored uploads (``RAW_DOCUMENT_DIR``), which are what a
  rebuild needs — re-indexing from them after an embedding model change, or moving to a
  new instance, without asking anyone to upload again.
- ``feedback.jsonl``: every answer rating, when feedback is on.

Qdrant's own data is backed up with a Qdrant snapshot (README "Backup and upgrade");
this complements it rather than duplicating vectors a new model would replace anyway.
"""

from __future__ import annotations

import contextvars
import json
import os
import queue
import shutil
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import asdict
from typing import BinaryIO

from api.documents import CHUNKER_VERSION, chunking_fingerprint
from api.feedback import Feedback, FeedbackCursor, FeedbackStore
from api.raw_documents import RawDocumentStore
from api.repository import DocumentMetadata, VectorRepository
from api.settings import AppSettings
from api.version import app_version
from api.workspaces import workspace_of

_CHUNK_BYTES = 64 * 1024
# Blocks between the zip writer and the response: the writer waits for a slow client
# rather than holding the archive in memory (or, before, in a tmpfs spool file counted
# against the container's memory limit).
_QUEUE_BLOCKS = 8
_PUT_POLL_SECONDS = 0.5
_FEEDBACK_PAGE = 1000
MANIFEST_VERSION = 1

_END = object()


class _ClientGone(Exception):
    """The response stopped reading (the client left): the writer stops too."""


class _QueueWriter:
    """A write-only, unseekable file for ``zipfile`` that hands out 64 KiB blocks.

    ``zipfile`` writes data descriptors after each member when it can't seek back,
    so the archive streams out as it is written.
    """

    def __init__(self, put: Callable[[object], None]) -> None:
        self._put = put
        self._buffer = bytearray()

    def write(self, data: bytes, /) -> int:
        self._buffer += data
        if len(self._buffer) >= _CHUNK_BYTES:
            self._put(bytes(self._buffer))
            self._buffer.clear()
        return len(data)

    def flush(self) -> None:
        if self._buffer:
            self._put(bytes(self._buffer))
            self._buffer.clear()

    def close(self) -> None:
        self.flush()


def _write_member(archive: zipfile.ZipFile, name: str, source: BinaryIO) -> None:
    """Copy ``source`` into ``archive`` in blocks, never whole into memory."""

    info = zipfile.ZipInfo(name, date_time=time.localtime(os.fstat(source.fileno()).st_mtime)[:6])
    # ZIP_STORED: uploads are mostly already compressed (PDF, DOCX).
    info.compress_type = zipfile.ZIP_STORED
    with archive.open(info, "w", force_zip64=True) as member:
        shutil.copyfileobj(source, member, _CHUNK_BYTES)


def build_export(
    settings: AppSettings,
    repository: VectorRepository,
    raw_store: RawDocumentStore | None,
    feedback_store: FeedbackStore | None,
) -> ExportStream:
    """The export zip as a stream of blocks, written while it is sent.

    The document listing and the first page of feedback are read here, before the
    caller starts a response, so a Qdrant or sqlite failure is still an error status
    rather than a cut-off download. Everything after that streams.
    """

    metadata = repository.filename_metadata(settings)
    stored = raw_store.stored_names() if raw_store is not None else set()
    # Known before the response starts, for its header; the manifest records what was
    # actually written (a document can be deleted meanwhile).
    missing_up_front = sum(1 for filename in metadata if filename not in stored)
    # A backup holds one workspace: its own ratings only.
    workspace = workspace_of(settings)
    first_feedback = (
        feedback_store.rows_oldest_first(None, _FEEDBACK_PAGE, workspace)
        if feedback_store is not None
        else None
    )

    def write_archive(writer: _QueueWriter) -> None:
        with zipfile.ZipFile(writer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            _write_archive(archive, settings, metadata, raw_store, feedback_store, first_feedback)
        writer.flush()

    stream = _stream(write_archive)
    stream.originals_missing = missing_up_front
    return stream


def _source_fields(meta: DocumentMetadata) -> dict[str, str | int]:
    if meta.source_url is None:
        return {}
    fields: dict[str, str | int] = {"source_url": meta.source_url}
    if meta.source_etag is not None:
        fields["source_etag"] = meta.source_etag
    if meta.source_last_modified is not None:
        fields["source_last_modified"] = meta.source_last_modified
    if meta.source_refresh_minutes is not None:
        fields["source_refresh_minutes"] = meta.source_refresh_minutes
    return fields


def _write_archive(
    archive: zipfile.ZipFile,
    settings: AppSettings,
    metadata: dict[str, DocumentMetadata],
    raw_store: RawDocumentStore | None,
    feedback_store: FeedbackStore | None,
    first_feedback: tuple[list[Feedback], FeedbackCursor | None] | None,
) -> None:
    documents = []
    for filename, meta in sorted(metadata.items()):
        # A document deleted while the export runs simply has no original here,
        # instead of failing the whole export with FileNotFoundError.
        handle = raw_store.open(filename) if raw_store is not None else None
        original = handle is not None
        if handle is not None:
            with handle:
                _write_member(archive, f"originals/{filename}", handle)
        documents.append(
            {
                "filename": filename,
                "chunk_count": meta.chunk_count,
                "page_count": meta.page_count,
                "byte_size": meta.byte_size,
                "uploaded_at": meta.uploaded_at,
                "chunker_version": meta.chunker_version,
                "chunking_fingerprints": sorted(meta.chunking_fingerprints),
                "tags": list(meta.tags),
                "original": f"originals/{filename}" if original else None,
                # Where a URL-ingested document came from, so a restored copy can
                # still be refreshed. Absent for uploads (older restores ignore it).
                **_source_fields(meta),
            }
        )
    manifest = {
        # Bumped on any change a restore (POST /import) must handle differently.
        "manifest_version": MANIFEST_VERSION,
        "docrag_version": app_version(),
        "exported_at": time.time(),
        "qdrant_collection": settings.qdrant_collection,
        "embedding_model_tag": settings.embedding_model_tag,
        "sparse_embedding_model": settings.sparse_embedding_model,
        "chunker_version": CHUNKER_VERSION,
        "chunking_fingerprint": chunking_fingerprint(settings),
        "documents": documents,
        # Documents this backup can't restore (no original kept: RAW_DOCUMENT_DIR is
        # unset, or the original was never stored). `complete` is false when there are
        # any: such a backup is not a way to rebuild the index.
        "originals_missing": [doc["filename"] for doc in documents if doc["original"] is None],
        "complete": all(doc["original"] is not None for doc in documents),
    }
    archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    if feedback_store is None or first_feedback is None:
        return
    # Oldest first, a page at a time: the whole table used to be read into one string.
    with archive.open("feedback.jsonl", "w") as member:
        page, cursor = first_feedback
        while page:
            member.write("".join(_feedback_line(row) for row in page).encode())
            if cursor is None or len(page) < _FEEDBACK_PAGE:
                break
            page, cursor = feedback_store.rows_oldest_first(
                cursor, _FEEDBACK_PAGE, workspace_of(settings)
            )


def _feedback_line(row: Feedback) -> str:
    """One feedback.jsonl row: the fields a restore reads, without the workspace (the
    rows join the workspace the backup is restored into)."""

    fields = asdict(row)
    del fields["workspace"]
    return json.dumps(fields, sort_keys=True) + "\n"


class ExportStream:
    """The export zip's blocks, in order, from a writer on its own thread.

    The queue is bounded, so the writer runs at the client's pace. ``close()`` stops it
    — from any thread — and the writer's next block raises ``_ClientGone``, so it
    unwinds and closes every file it opened. The route calls it when the response ends,
    however it ends: Starlette never closes a plain iterator when a client leaves, and
    the writer then sat on its open original until the iterator was garbage-collected.
    A failure in the writer is raised from ``__next__``, so the response aborts rather
    than ending as if the zip were complete.
    """

    # Documents with no original to put in the zip, counted before it is written.
    originals_missing = 0

    def __init__(self, write: Callable[[_QueueWriter], None]) -> None:
        self._blocks: queue.Queue[object] = queue.Queue(maxsize=_QUEUE_BLOCKS)
        self._stop = threading.Event()
        self._write = write
        self._thread = threading.Thread(
            # A copy of the request's context, so log lines keep its request id.
            target=contextvars.copy_context().run,
            args=(self._produce,),
            name="docrag-export",
            daemon=True,
        )

    def _put(self, item: object) -> None:
        while not self._stop.is_set():
            try:
                self._blocks.put(item, timeout=_PUT_POLL_SECONDS)
                return
            except queue.Full:
                continue
        raise _ClientGone

    def _produce(self) -> None:
        try:
            self._write(_QueueWriter(self._put))
            self._put(_END)
        except _ClientGone:
            pass
        except BaseException as exc:
            try:
                self._put(exc)
            except _ClientGone:
                pass

    def __iter__(self) -> ExportStream:
        return self

    def __next__(self) -> bytes:
        if not self._thread.is_alive() and self._thread.ident is None:
            self._thread.start()
        if self._stop.is_set():
            raise StopIteration
        item = self._blocks.get()
        if item is _END or item is _CLOSED:
            raise StopIteration
        if isinstance(item, BaseException):
            self._stop.set()
            raise item
        return item  # type: ignore[return-value]

    def close(self) -> None:
        """Stop the writer, and wake a reader waiting on the next block."""

        self._stop.set()
        try:
            self._blocks.put_nowait(_CLOSED)
        except queue.Full:
            pass  # a full queue: no reader is waiting on it


_CLOSED = object()


def _stream(write: Callable[[_QueueWriter], None]) -> ExportStream:
    return ExportStream(write)
