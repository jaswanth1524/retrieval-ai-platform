"""Operations on indexed documents that go through the ingest jobs: re-index (one, or
every stale one), refresh from a URL, delete, and which failed jobs can be retried."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from fastapi import HTTPException

from api.audit import audit
from api.corpus import bump_corpus_generation
from api.document_names import (
    indexing_now,
)
from api.documents import (
    DocumentNotFoundError,
    DocumentSource,
    chunking_fingerprint,
)
from api.ingest_jobs import (
    IngestQueue,
    admit_or_refuse,
    backlog_full_error,
)
from api.ingestion import (
    IngestionError,
    filename_write_lock,
    ingest_sequencer,
)
from api.jobs import IngestJob
from api.raw_documents import RawDocumentStore
from api.repository import DocumentMetadata, VectorRepository
from api.schemas import (
    DocumentJobAcceptedResponse,
)
from api.settings import AppSettings
from api.url_fetch import FetchedDocument
from api.workspaces import document_key, workspace_of

logger = logging.getLogger(__name__)


def original_still_stored(store: RawDocumentStore, filename: str) -> Callable[[], None]:
    """A re-index precondition: fail if the document was deleted while queued.

    Checked under the filename lock DELETE also takes (and DELETE removes the original
    under it), so "original gone" means the delete landed first — re-indexing anyway
    would silently bring the deleted document back.
    """

    def check() -> None:
        if store.path(filename) is None:
            raise IngestionError(
                f"'{filename}' was deleted before its re-index ran; nothing was re-indexed."
            )

    return check


def staleness(settings: AppSettings, token_counter: object) -> Callable[[DocumentMetadata], bool]:
    """The one staleness rule the listing, bulk re-index and AUTO_REINDEX_STALE share."""

    fingerprint = chunking_fingerprint(settings)
    counter_mode = str(getattr(token_counter, "mode", "unknown"))
    return lambda meta: meta.is_stale(fingerprint, counter_mode)


def unchanged_since_fetch(
    repository: VectorRepository, settings: AppSettings, name: str, expected: DocumentMetadata
) -> Callable[[], None]:
    """A refresh precondition: the document is still the one the refresh started from.

    A fetch takes up to URL_INGEST_TIMEOUT_SECONDS before its job is even admitted, so
    a delete, an upload or a version restore can land in between — and the refresh,
    admitted after them, would bring the deleted document back or overwrite the newer
    version with the URL's. Checked under the filename lock, like every precondition.
    """

    def check() -> None:
        identity = repository.content_identity(settings, name)
        same = identity is not None and identity[1] == expected.source_url
        if same and expected.content_sha256 is not None:
            same = identity is not None and identity[0] == expected.content_sha256
        if not same:
            raise IngestionError(
                f"'{name}' was changed or deleted while its refresh was being fetched; "
                "nothing was refreshed."
            )

    return check


def enqueue_refreshed(
    name: str,
    source_url: str,
    fetched: FetchedDocument,
    queue: IngestQueue,
    raw_store: RawDocumentStore | None,
    *,
    expected: DocumentMetadata,
    repository: VectorRepository,
) -> DocumentJobAcceptedResponse:
    """Index what a refresh of ``name`` fetched (the route's and the scheduled
    refresh's): its tags carry over, as on any re-upload, unchanged bytes are left
    alone, and the server's new validators are kept for the next conditional fetch.
    ``expected`` is the document as it was before the fetch; anything else by the time
    the job writes refuses it (``unchanged_since_fetch``)."""

    settings = queue.ingest_service.settings
    return queue.enqueue(
        filename=name,
        content=fetched.content,
        raw_store=raw_store,
        source=DocumentSource(
            source_url, fetched.etag, fetched.last_modified, expected.source_refresh_minutes
        ),
        dedupe=True,
        precondition=unchanged_since_fetch(repository, settings, name, expected),
        kind="url",
    )


# Why a failed job of each kind can't be retried by the server, as the 409 says it.
_NOT_RETRYABLE = {
    "upload": "Its bytes weren't kept (an original is stored only once indexing "
    "succeeds); upload the file again.",
    "import": "Its bytes weren't kept; import the backup again.",
    "watch": "The folder watcher sends the file again when it changes.",
    "restore": "Restore that version again.",
}


def retry_refusal(job: IngestJob) -> str | None:
    """Why ``job`` can't be retried (POST /documents/jobs/{id}/retry), or None when it
    can: a failed fetch from a URL (fetched again) or a failed re-index (run again from
    the stored original) — unless the original itself is what failed to parse."""

    if job.state != "failed":
        return f"Only a failed job can be retried; this one is {job.state}."
    if job.kind == "url":
        return None if job.source_url else "It has no URL to fetch again."
    if job.kind == "reindex":
        if job.error_type == "document":
            return "The stored original itself couldn't be read; upload a corrected file instead."
        return None
    return _NOT_RETRYABLE.get(job.kind, "This job can't be retried.")


def enqueue_refetched(
    job: IngestJob,
    repository: VectorRepository,
    settings: AppSettings,
    queue: IngestQueue,
    raw_store: RawDocumentStore | None,
    fetched: FetchedDocument,
    meta: DocumentMetadata | None,
) -> DocumentJobAcceptedResponse:
    """Queue a failed URL job's document as fetched again (``fetched``, under the name it
    was stored under), with its tags; ``meta`` is the document as it was before the
    fetch. A document changed or deleted meanwhile is not replaced (as for a refresh)."""

    assert job.source_url is not None
    return queue.enqueue(
        filename=job.filename,
        content=fetched.content,
        raw_store=raw_store,
        source=DocumentSource(
            job.source_url,
            fetched.etag,
            fetched.last_modified,
            meta.source_refresh_minutes if meta is not None else None,
        ),
        tags=job.tags,
        dedupe=True,
        kind="url",
        precondition=(
            unchanged_since_fetch(repository, settings, job.filename, meta)
            if meta is not None
            else None
        ),
    )


def source_of(meta: DocumentMetadata, *, validators: bool = True) -> DocumentSource | None:
    """Where a URL-fetched document came from, carried over by a re-index.

    Without ``validators`` only the URL: a version restore puts back older bytes than
    the ones the stored ETag describes, and sending it would make the next refresh a
    304 that keeps the old version for good.
    """

    if not meta.source_url:
        return None
    if not validators:
        return DocumentSource(meta.source_url, refresh_minutes=meta.source_refresh_minutes)
    return DocumentSource(
        meta.source_url,
        meta.source_etag,
        meta.source_last_modified,
        meta.source_refresh_minutes,
    )


@dataclass(frozen=True)
class StaleReindex:
    """What one pass over the stale documents queued (``reindex_stale``)."""

    jobs: list[DocumentJobAcceptedResponse]
    # Not queued: the backlog was full (or ``limit`` reached); try again later.
    deferred: list[str]
    # Already being indexed: their job will make them current.
    in_progress: list[str]
    # Couldn't be queued for another reason (logged).
    failed: list[str]
    # Left alone because ``skip`` said so (the background pass's failed documents).
    skipped: list[str] = field(default_factory=list)


def reindex_stale(
    repository: VectorRepository,
    settings: AppSettings,
    queue: IngestQueue,
    store: RawDocumentStore,
    token_counter: object,
    *,
    limit: int | None = None,
    skip: Callable[[str], bool] | None = None,
) -> StaleReindex:
    """Queue a re-index of every stale or incomplete document with a stored original (at most
    ``limit`` jobs; the rest are deferred) — ``POST /documents/reindex`` and the
    background AUTO_REINDEX_STALE pass. A name ``skip`` answers True for is passed
    over without taking a place in ``limit``."""

    is_stale = staleness(settings, token_counter)
    # 409 now, not one failed job per document, when the index can't take them.
    repository.ensure_writable(settings)
    metadata = repository.filename_metadata(settings)
    busy = indexing_now(settings)
    jobs: list[DocumentJobAcceptedResponse] = []
    deferred: list[str] = []
    in_progress: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    for name, meta in sorted(metadata.items()):
        # Incomplete documents too: re-indexing from the original is their repair.
        if not (is_stale(meta) or meta.incomplete):
            continue
        if skip is not None and skip(name):
            skipped.append(name)
            continue
        if name in busy:
            # Still listed as stale until its job finishes: a second click would
            # parse and embed the whole document again.
            in_progress.append(name)
            continue
        if limit is not None and len(jobs) >= limit:
            deferred.append(name)
            continue
        try:
            content = read_within_backlog(
                queue, store.size(name), functools.partial(store.read, name)
            )
        except HTTPException:  # the backlog is full: the rest wait for the next pass
            deferred.append(name)
            continue
        if content is None:
            continue
        try:
            jobs.append(
                queue.enqueue(
                    filename=name,
                    content=content,
                    raw_store=None,
                    uploaded_at=meta.uploaded_at,
                    precondition=original_still_stored(store, name),
                    source=source_of(meta),
                    kind="reindex",
                )
            )
        except HTTPException as exc:
            if exc.status_code == 503:
                deferred.append(name)
            else:
                failed.append(name)
        except Exception:
            # The jobs already queued keep running; failing the request here left
            # the client without their ids.
            logger.exception("Could not queue the re-index of %s.", name)
            failed.append(name)
    return StaleReindex(jobs, deferred, in_progress, failed, skipped)


def admit_from_store(
    repository: VectorRepository,
    settings: AppSettings,
    queue: IngestQueue,
    name: str,
    size: int | None,
    read: Callable[[], bytes | None],
    missing: str,
) -> bytes:
    """The stored bytes a re-index or version restore indexes, admitted as an upload is.

    409 while the document is being indexed (a second job would parse and embed it all
    again, or race the first), 404 (``missing``) when there are no bytes, and the
    backlog asked with ``size`` before ``read`` runs: a full backlog would otherwise
    read the whole file into memory only to refuse it.
    """

    repository.ensure_writable(settings)
    if name in indexing_now(settings):
        raise HTTPException(
            status_code=409,
            detail=f"'{name}' is already being indexed; wait for that job to finish.",
        )
    content = read_within_backlog(queue, size, read)
    if content is None:
        raise DocumentNotFoundError(missing)
    return content


def read_within_backlog(
    queue: IngestQueue, size: int | None, read: Callable[[], bytes | None]
) -> bytes | None:
    """Stored bytes to index, read only once the backlog has room for ``size``: None
    when there are none (``size`` None, or gone by the read), a 503 (``HTTPException``)
    when the backlog is full — before reading, or a full backlog would read every
    remaining original into memory only to refuse it. The core of every re-index:
    the routes' (``admit_from_store``) and the stale pass's (``reindex_stale``)."""

    if size is None:
        return None
    if not queue.backlog.has_room(size):
        raise backlog_full_error()
    return read()


def require_raw_store(raw_store: RawDocumentStore | None) -> RawDocumentStore:
    """The raw store, or a 409: re-indexing reads originals only kept when it's on."""

    if raw_store is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Re-indexing needs stored originals: set RAW_DOCUMENT_DIR, then upload "
                "documents again once so their originals are kept."
            ),
        )
    return raw_store


def delete_indexed_document(
    repository: VectorRepository,
    settings: AppSettings,
    raw_store: RawDocumentStore | None,
    filename: str,
) -> int:
    """Delete a stored document's points and original; the number of points removed.

    ``DELETE /documents/{filename}`` and the folder watcher's removals (api/watch.py).
    404 when there is nothing to delete, 409 when a newer upload won the race.

    Same lock the ingest path takes: snapshot-then-delete is a read-modify-write, so
    without it a delete landing between a concurrent ingest's upsert and its
    stale-cleanup would remove the points that ingest just wrote, and the ingest would
    still report success. The raw copy (when stored) is removed under the same lock, or
    a deleted document's original bytes would accumulate forever.
    """

    # Sequenced like an ingest, and admitted before waiting for the lock, so an
    # upload already queued for this name is refused when it reaches the lock
    # instead of indexing the document straight back after the delete.
    key = document_key(settings, filename)
    sequence = admit_or_refuse(key)
    try:
        with filename_write_lock(key):
            # The lock isn't fair: an upload admitted after this delete can reach
            # it first and be indexed. Deleting then would remove the newer
            # document the user just uploaded, so the delete stands down.
            superseded = ingest_sequencer.superseded_by(key, sequence)
            if superseded == "upload":
                raise HTTPException(
                    status_code=409,
                    detail=f"A newer upload of '{filename}' was indexed after this "
                    "delete was requested; it was kept.",
                )
            if superseded == "delete":
                raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
            points_deleted = repository.delete_filename(settings, filename)
            if points_deleted:
                # Right away, not after the original below: if removing that
                # raises, cached answers must still stop citing the deleted points.
                bump_corpus_generation([filename], collection=settings.qdrant_collection)
            # Also when no points remain: an original orphaned by an earlier failed
            # ingest was otherwise undeletable, since the 404 came first.
            removed_original = _delete_original(
                raw_store, filename, points_gone=bool(points_deleted)
            )
            if not points_deleted and not removed_original:
                raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
            ingest_sequencer.mark_applied(key, sequence, deleted=True)
    finally:
        ingest_sequencer.release(key)
    audit(
        "document.delete",
        workspace=workspace_of(settings),
        filename=filename,
        points_deleted=points_deleted,
        original_removed=removed_original,
    )
    return points_deleted


def _delete_original(
    raw_store: RawDocumentStore | None, filename: str, *, points_gone: bool
) -> bool:
    """Remove the stored original; True when one was removed.

    Once the points are gone the delete has happened: a failure removing the original
    (a read-only or permission-locked RAW_DOCUMENT_DIR) is logged rather than turned
    into a 500, and a second DELETE removes the orphan through the no-points path. When
    the orphan was all there was to delete, the failure is the request's: a 500 that
    says what failed, not an unhandled ``OSError``.
    """

    if raw_store is None:
        return False
    try:
        return raw_store.delete(filename)
    except OSError as exc:
        logger.exception("Could not remove the stored original of '%s'", filename)
        if not points_gone:
            raise HTTPException(
                status_code=500,
                detail=f"The stored original of '{filename}' could not be removed; "
                "check that RAW_DOCUMENT_DIR is writable.",
            ) from exc
        return True
