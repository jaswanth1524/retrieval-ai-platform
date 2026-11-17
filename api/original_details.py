"""What the index knows about a stored original and the file doesn't, kept beside it.

A lost index (a wiped Qdrant volume, a move to another machine) can be rebuilt from the
stored originals (``api/rebuild.py``), but a document's tags, upload time, source URL
and watched folder live only in its points' payloads. ``record_details`` writes them to
``RawDocumentStore.save_details`` whenever one changes; the keys are the ones a backup's
manifest uses for a document, so restore's validation reads them back.
"""

from __future__ import annotations

import logging

from api.raw_documents import RawDocumentStore
from api.repository import DocumentMetadata, VectorRepository
from api.settings import AppSettings

logger = logging.getLogger(__name__)


def details_of(meta: DocumentMetadata) -> dict[str, object]:
    details: dict[str, object] = {"tags": list(meta.tags)}
    for key in (
        "uploaded_at",
        "source_url",
        "source_etag",
        "source_last_modified",
        "source_refresh_minutes",
        "watched_from",
    ):
        value = getattr(meta, key)
        if value is not None:
            details[key] = value
    return details


def record_details(
    repository: VectorRepository,
    settings: AppSettings,
    raw_store: RawDocumentStore | None,
    filename: str,
) -> None:
    """Write ``filename``'s details beside its original, read from the index now. The
    caller holds its filename lock. Best effort: a failure is logged, since the change
    it follows is already made."""

    if raw_store is None or raw_store.path(filename) is None:
        return
    try:
        meta = repository.document_metadata(settings, filename)
        if meta is not None:
            raw_store.save_details(filename, details_of(meta))
    except Exception:  # noqa: BLE001 - never fails the change it records
        logger.warning("Could not keep the details of %r beside its original.", filename)


def record_missing_details(
    metadata: dict[str, DocumentMetadata], raw_store: RawDocumentStore
) -> int:
    """Write details for every indexed document whose original has none (originals kept
    by a release before details were), from the listing's ``metadata``. Once per store
    and process: after it, every change writes its own. How many were written."""

    if raw_store.details_backfilled:
        return 0
    raw_store.details_backfilled = True
    written = 0
    for name in sorted(raw_store.stored_names() - raw_store.names_with_details()):
        meta = metadata.get(name)
        if meta is None:
            continue
        try:
            raw_store.save_details(name, details_of(meta))
            written += 1
        except OSError:
            logger.warning("Could not keep the details of %r beside its original.", name)
            break
    return written
