"""Index every stored original that has no document (``POST /admin/rebuild``,
``docrag rebuild``): the way back from a lost index — a wiped Qdrant volume, a
collection rebuilt for another embedding setup, a data directory moved to a new machine.

Runs as a background import (``api/imports.py``): one at a time, fed to the indexing
queue as room frees, followed and cancelled through ``/documents/imports``. Each
document comes back with the tags, upload time, source and watched folder kept beside
its original (``api/original_details.py``); an original from before those were kept is
indexed as a plain upload.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException

from api.document_names import indexing_now
from api.documents import DocumentError, normalize_tags
from api.imports import BackgroundImport, wait_for_room
from api.ingest_jobs import IngestQueue
from api.ingestion import IngestionError
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.restore import _source, _uploaded_at, _watched_from
from api.settings import AppSettings

logger = logging.getLogger(__name__)


def originals_to_rebuild(
    repository: VectorRepository, settings: AppSettings, raw_store: RawDocumentStore
) -> list[str]:
    """Stored originals with no document, indexed or on its way (read from the index
    now, not the cached listing)."""

    repository.forget_listing(settings)
    indexed = set(repository.filename_metadata(settings))
    return sorted(raw_store.stored_names() - indexed - indexing_now(settings))


def _tags(details: dict[str, object]) -> tuple[str, ...]:
    raw = details.get("tags")
    if not isinstance(raw, list):
        return ()
    try:
        return normalize_tags([tag for tag in raw if isinstance(tag, str)])
    except DocumentError:
        return ()


def rebuild(
    names: list[str],
    repository: VectorRepository,
    settings: AppSettings,
    raw_store: RawDocumentStore,
    queue: IngestQueue,
    status: BackgroundImport,
) -> None:
    """Queue each of ``names`` from its original, waiting for room in the queue."""

    def still_absent(name: str) -> None:
        if repository.content_identity(settings, name) is not None:
            raise IngestionError(f"'{name}' was indexed while the rebuild ran; left as it is.")

    for name in names:
        if status.cancel.is_set():
            return
        size = raw_store.size(name)
        if size is None:  # deleted since the list was read
            status.skipped += 1
            continue
        if not wait_for_room(queue.backlog.has_spare_room, size, status):
            return
        content = raw_store.read(name)
        if content is None:
            status.skipped += 1
            continue
        details = raw_store.details(name)
        try:
            queue.enqueue(
                filename=name,
                content=content,
                # Already stored, byte-identical: a re-save that failed would delete it.
                raw_store=None,
                uploaded_at=_uploaded_at(details.get("uploaded_at")),
                precondition=lambda name=name: still_absent(name),  # type: ignore[misc]
                tags=_tags(details),
                source=_source(details),
                watched_from=_watched_from(details, name),
                kind="rebuild",
            )
        except HTTPException as exc:
            # The queue filled between the check and the reservation (an upload got
            # in), or the workspace is being deleted.
            status.failed += 1
            status.note(f"{name}: {exc.detail}")
            if exc.status_code != 503:
                return
            continue
        status.queued += 1
