"""Background ingest jobs: enqueueing against the backlog, running, and recording outcomes."""

from __future__ import annotations

import contextvars
import logging
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException

from api.audit import audit
from api.corpus import bump_corpus_generation
from api.documents import (
    DocumentError,
    DocumentNotFoundError,
    DocumentSource,
    UnsupportedDocumentError,
    chunking_fingerprint,
    normalize_filename,
)
from api.embeddings import EmbeddingError
from api.ingestion import (
    ClosedForWorkError,
    IngestionError,
    filename_write_lock,
    ingest_sequencer,
)
from api.jobs import INTERRUPTED_JOB_ERROR, IngestBacklog, IngestJob, JobStore
from api.metrics import (
    NO_ERROR_TYPE,
    ingest_chunks_total,
    ingest_job_seconds,
    ingest_jobs_total,
)
from api.pipeline import IngestService
from api.qdrant_schema import CollectionSchemaError, VectorStoreUnavailableError
from api.raw_documents import RawDocumentStore
from api.repository import DocumentMetadata, VectorRepository
from api.schemas import (
    DocumentJobAcceptedResponse,
)
from api.settings import AppSettings
from api.url_fetch import FetchedDocument
from api.workspaces import WorkspaceClosingError, document_key, filename_of, workspace_of

logger = logging.getLogger(__name__)


# Domain errors IngestService.ingest can actually raise (documents.py's DocumentError
# tree, ingestion.py, embeddings.py, qdrant_schema.py) — these carry a message that's
# safe and useful to return to a polling client. Anything else is an exception the
# ingest path wasn't built to anticipate, so its str() may carry internal detail
# (a path, a config value) that shouldn't leave the server; it gets a generic message
# instead, with the real detail already captured by the logger.warning below.
INGEST_USER_FACING_ERRORS: tuple[type[Exception], ...] = (
    DocumentError,
    IngestionError,
    EmbeddingError,
    CollectionSchemaError,
    VectorStoreUnavailableError,
)


SHUTTING_DOWN_ERROR = "The server is shutting down and could not start indexing. Retry the upload."

UNRECORDED_OUTCOME_ERROR = (
    "The server could not record how indexing ended; check the document list and "
    "re-upload if the document is missing."
)
JOB_CANCELLED_ERROR = "Cancelled before anything was indexed; the document is unchanged."


class IngestCancelledError(IngestionError):
    """The job was cancelled at one of its checkpoints, before anything was written."""


class JobHandle:
    """What cancelling a job needs to reach: its future (to drop it from the queue) and
    a flag its checkpoints read once it runs.

    Cancellation is only clean before the first batch is written: point ids are
    deterministic, so a new batch overwrites the previous version's points in place and
    there is no earlier state left to restore. ``begin_write`` closes that window
    atomically with the flag, so a cancel either lands before it or is refused.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.future: Future[None] | None = None
        self.cancel_requested = False
        self.writing = False

    def check_cancelled(self) -> None:
        with self._lock:
            if self.cancel_requested:
                raise IngestCancelledError(JOB_CANCELLED_ERROR)

    def begin_write(self) -> None:
        with self._lock:
            if self.cancel_requested:
                raise IngestCancelledError(JOB_CANCELLED_ERROR)
            self.writing = True

    def request_cancel(self) -> bool:
        """False when the job has already started writing (too late to cancel)."""

        with self._lock:
            if self.writing:
                return False
            self.cancel_requested = True
            return True


# In-flight jobs of this process, by id; finished ones are dropped. A single worker
# process is assumed throughout (locks, sequencer, slots), and this is no different.
_handles: dict[str, JobHandle] = {}
_handles_lock = threading.Lock()


CancelOutcome = Literal["cancelled", "requested", "too_late", "not_running"]


def cancel_ingest_job(job_id: str) -> CancelOutcome:
    """Cancel a queued job outright, or flag a running one for its next checkpoint."""

    with _handles_lock:
        handle = _handles.get(job_id)
    if handle is None:
        return "not_running"
    if not handle.request_cancel():
        return "too_late"
    future = handle.future
    # cancel() succeeds only for a job still waiting in the executor's queue; its done
    # callback (finish_ingest_future) records the cancellation on the job.
    if future is not None and future.cancel():
        return "cancelled"
    return "requested"


def job_error_message(exc: Exception) -> str:
    if isinstance(exc, INGEST_USER_FACING_ERRORS):
        return str(exc)
    return "Ingestion failed due to an unexpected server error."


def ingest_error_type(exc: Exception) -> str:
    """Coarse error_type label for docrag_ingest_jobs_total{outcome="failed"}."""

    if isinstance(exc, DocumentError):
        return "document"
    if isinstance(exc, IngestionError):
        return "ingestion"
    if isinstance(exc, EmbeddingError):
        return "embedding"
    if isinstance(exc, CollectionSchemaError):
        return "collection_schema"
    if isinstance(exc, VectorStoreUnavailableError):
        return "vector_store_unavailable"
    return "unexpected"


def run_ingest_job(
    job_store: JobStore,
    job_id: str,
    ingest_service: IngestService,
    filename: str,
    content: bytes,
    raw_store: RawDocumentStore | None = None,
    uploaded_at: float | None = None,
    precondition: Callable[[], None] | None = None,
    sequence: int | None = None,
    handle: JobHandle | None = None,
    tags: tuple[str, ...] | None = None,
    source: DocumentSource | None = None,
    dedupe: bool = False,
) -> None:
    """Background-executor entry point: parse/chunk/embed/index and update job status.

    Runs on ``get_ingest_executor()``'s worker threads, outside any request's
    lifetime — exceptions are caught and recorded on the job rather than raised,
    since there is no HTTP response left to attach them to. The original bytes (when
    ``raw_store`` is set) are saved only after indexing succeeds and under the same
    filename lock: saving them at upload time left orphans for uploads that then
    failed, and overwrote the original of a still-indexed older version.
    """

    # The sequencer's key: the filename, workspace-qualified outside the default one.
    key = document_key(ingest_service.settings, filename)

    def on_progress(done: int, total: int) -> None:
        job_store.update(job_id, state="embedding", chunks_done=done, chunks_total=total)

    def check_sequence() -> None:
        if handle is not None:
            handle.check_cancelled()
        # The re-index precondition first: its "deleted before its re-index ran" names
        # the cause more exactly than the sequencer's generic refusal.
        if precondition is not None:
            precondition()
        if sequence is not None:
            ingest_sequencer.check(key, sequence)

    def save_original() -> None:
        if sequence is not None:
            ingest_sequencer.mark_applied(key, sequence)
        if raw_store is None:
            return
        try:
            raw_store.save(filename, content)
        except OSError:
            # The index is already updated; failing the job would report a document
            # as not indexed when it is. The original is an opt-in extra — but a
            # previous version's original must not outlive the index it described, or
            # /original would serve bytes that no longer match what is indexed. No
            # original (404) beats a wrong one.
            logger.exception("Indexed %r but could not store its original bytes.", filename)
            try:
                raw_store.delete(filename)
            except OSError:
                logger.exception(
                    "Could not remove the outdated original of %r either; "
                    "GET /documents/%s/original may serve the previous version.",
                    filename,
                    filename,
                )

    started = time.monotonic()
    try:
        if handle is not None:
            handle.check_cancelled()
        job_store.update(job_id, state="parsing")
        outcome = ingest_service.ingest(
            filename,
            content,
            on_progress,
            save_original,
            uploaded_at=uploaded_at,
            precondition=check_sequence,
            before_first_write=handle.begin_write if handle is not None else None,
            tags=tags,
            source=source,
            dedupe=dedupe,
        )
    except IngestCancelledError as exc:
        # Asked for, not a fault: one line, no traceback.
        logger.info("Ingest job %s for %r cancelled.", job_id, filename)
        ingest_job_seconds.labels(outcome="failed").observe(time.monotonic() - started)
        ingest_jobs_total.labels(outcome="failed", error_type="cancelled").inc()
        job_store.update(
            job_id, state="failed", error=job_error_message(exc), error_type="cancelled"
        )
        return
    except Exception as exc:
        logger.warning("Ingest job %s for %r failed: %s", job_id, filename, exc, exc_info=True)
        ingest_job_seconds.labels(outcome="failed").observe(time.monotonic() - started)
        error_type = ingest_error_type(exc)
        ingest_jobs_total.labels(outcome="failed", error_type=error_type).inc()
        job_store.update(
            job_id, state="failed", error=job_error_message(exc), error_type=error_type
        )
        return

    ingest_job_seconds.labels(outcome="done").observe(time.monotonic() - started)
    ingest_jobs_total.labels(outcome="done", error_type=NO_ERROR_TYPE).inc()
    if not outcome.unchanged:  # nothing was indexed
        ingest_chunks_total.inc(outcome.chunks_ingested)
    job_store.update(
        job_id,
        state="done",
        chunks_done=outcome.chunks_ingested,
        chunks_total=outcome.chunks_ingested,
        result={
            "filename": outcome.filename,
            "sections_parsed": outcome.sections_parsed,
            "chunks_ingested": outcome.chunks_ingested,
            "collection_name": outcome.collection_name,
            "unchanged": outcome.unchanged,
        },
    )


BACKLOG_FULL_ERROR = (
    "Too many documents are already waiting to be indexed. Retry once some of them finish."
)


def backlog_full_error() -> HTTPException:
    """The 503 an upload gets while the ingest backlog has no room."""

    return HTTPException(status_code=503, detail=BACKLOG_FULL_ERROR, headers={"Retry-After": "15"})


def admit_or_refuse(key: str) -> int:
    """``ingest_sequencer.admit(key)``, or a 409 while its workspace is being deleted."""

    try:
        return ingest_sequencer.admit(key)
    except ClosedForWorkError as exc:
        raise WorkspaceClosingError(exc.group) from None


@dataclass(frozen=True)
class IngestQueue:
    """Where an upload, URL fetch, re-index, version restore or import is queued: the
    job store, the ingest executor, the backlog and the request's ingest service."""

    job_store: JobStore
    executor: ThreadPoolExecutor
    backlog: IngestBacklog
    ingest_service: IngestService

    def enqueue(
        self,
        *,
        filename: str,
        content: bytes,
        raw_store: RawDocumentStore | None,
        uploaded_at: float | None = None,
        precondition: Callable[[], None] | None = None,
        tags: tuple[str, ...] | None = None,
        source: DocumentSource | None = None,
        dedupe: bool = False,
        kind: str = "upload",
    ) -> DocumentJobAcceptedResponse:
        """``enqueue_ingest`` with this queue's collaborators."""

        return enqueue_ingest(
            job_store=self.job_store,
            executor=self.executor,
            backlog=self.backlog,
            ingest_service=self.ingest_service,
            filename=filename,
            content=content,
            raw_store=raw_store,
            uploaded_at=uploaded_at,
            precondition=precondition,
            tags=tags,
            source=source,
            dedupe=dedupe,
            kind=kind,
        )


def enqueue_ingest(
    *,
    job_store: JobStore,
    executor: ThreadPoolExecutor,
    backlog: IngestBacklog,
    ingest_service: IngestService,
    filename: str,
    content: bytes,
    raw_store: RawDocumentStore | None,
    uploaded_at: float | None = None,
    precondition: Callable[[], None] | None = None,
    tags: tuple[str, ...] | None = None,
    source: DocumentSource | None = None,
    dedupe: bool = False,
    kind: str = "upload",
) -> DocumentJobAcceptedResponse:
    """Admit ``content`` against the backlog, create its job, and start it. Blocking.

    ``kind`` (``jobs.JOB_KINDS``) says what started it, recorded on the job with the
    source URL and tags, for a retry.

    Shared by upload and re-index so both get the same backlog 503, the same
    shutting-down handling, and the same done-callback that frees the bytes.
    """

    size = len(content)
    if not backlog.try_reserve(size):
        raise backlog_full_error()
    key = document_key(ingest_service.settings, filename)
    try:
        sequence = admit_or_refuse(key)
    except HTTPException:
        backlog.release(size)
        raise

    def release() -> None:
        backlog.release(size)
        ingest_sequencer.release(key)

    try:
        job = job_store.create(
            filename,
            workspace_of(ingest_service.settings),
            kind=kind,
            source_url=source.url if source is not None else None,
            tags=tags,
        )
    except BaseException:
        release()
        raise
    handle = JobHandle()
    with _handles_lock:
        _handles[job.id] = handle

    def forget() -> None:
        with _handles_lock:
            _handles.pop(job.id, None)

    try:
        # Ingestion runs on a dedicated executor (not FastAPI's request threadpool) so
        # it survives independently of this request/response cycle — the client can
        # disconnect and poll the job later without interrupting the work.
        # In the request's context: the job's log lines carry the upload's request id.
        future = executor.submit(
            contextvars.copy_context().run,
            run_ingest_job,
            job_store,
            job.id,
            ingest_service,
            filename,
            content,
            raw_store,
            uploaded_at,
            precondition,
            sequence,
            handle,
            tags,
            source,
            dedupe,
        )
    except RuntimeError as exc:
        # The executor is shut down (the process is stopping). The job already exists,
        # so fail it rather than leave it "queued" forever for a poller.
        release()
        forget()
        job_store.update(job.id, state="failed", error=SHUTTING_DOWN_ERROR)
        raise HTTPException(
            status_code=503, detail=SHUTTING_DOWN_ERROR, headers={"Retry-After": "15"}
        ) from exc
    except BaseException:
        release()
        forget()
        job_store.update(job.id, state="failed", error="Could not start indexing.")
        raise
    handle.future = future

    def finished(done: Future[None]) -> None:
        forget()
        finish_ingest_future(done, release, job_store, job.id, cancelled=handle.cancel_requested)

    future.add_done_callback(finished)
    return DocumentJobAcceptedResponse(job_id=job.id, filename=filename, state="queued")


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


def stored_filename(filename: str) -> str:
    """``filename`` as a path parameter naming a stored document, or a 404.

    Every stored name went through ``normalize_filename`` at upload, so a name it would
    change (padding spaces, a backslash path, control characters) was never stored. The
    raw-document store normalizes on its own, so passing such a name through let
    ``DELETE /documents/%20a.pdf`` remove ``a.pdf``'s original while holding a lock on,
    and deleting the points of, a different name.
    """

    try:
        name = normalize_filename(filename)
    except UnsupportedDocumentError:
        name = ""
    if name != filename:
        raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
    return name


class FilenameConflictError(DocumentError):
    """Another document's name differs from this one only by case or Unicode form."""


def _folded(filename: str) -> str:
    return unicodedata.normalize("NFC", filename).casefold()


def refuse_near_duplicate_name(filename: str, existing: Iterable[str]) -> None:
    """Refuse a name that only case or Unicode normalization tells apart from another.

    Only matters when originals are stored: on a case-insensitive or normalizing disk
    (a macOS host, a Windows bind mount) ``Report.pdf`` and ``report.pdf`` are one file,
    so the second upload replaced the first document's original — and deleting either
    removed both. The index itself tells them apart, so this is checked only with a raw
    store, and re-uploading the exact same name is always allowed.
    """

    folded = _folded(filename)
    for other in existing:
        if other != filename and _folded(other) == folded:
            raise FilenameConflictError(
                f"'{filename}' differs from the existing document '{other}' only by "
                "letter case or accents, and their stored originals would overwrite each "
                f"other. Rename the file, or delete '{other}' first."
            )


def indexing_now(settings: AppSettings) -> set[str]:
    """Filenames in ``settings``' workspace with an upload, re-index or delete admitted
    and not yet finished."""

    in_flight = (filename_of(settings, key) for key in ingest_sequencer.in_flight())
    return {name for name in in_flight if name}


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


def reindex_stale(
    repository: VectorRepository,
    settings: AppSettings,
    queue: IngestQueue,
    store: RawDocumentStore,
    token_counter: object,
    *,
    limit: int | None = None,
) -> StaleReindex:
    """Queue a re-index of every stale or incomplete document with a stored original (at most
    ``limit`` jobs; the rest are deferred) — ``POST /documents/reindex`` and the
    background AUTO_REINDEX_STALE pass."""

    is_stale = staleness(settings, token_counter)
    # 409 now, not one failed job per document, when the index can't take them.
    repository.ensure_writable(settings)
    metadata = repository.filename_metadata(settings)
    busy = indexing_now(settings)
    jobs: list[DocumentJobAcceptedResponse] = []
    deferred: list[str] = []
    in_progress: list[str] = []
    failed: list[str] = []
    for name, meta in sorted(metadata.items()):
        # Incomplete documents too: re-indexing from the original is their repair.
        if not (is_stale(meta) or meta.incomplete):
            continue
        if name in busy:
            # Still listed as stale until its job finishes: a second click would
            # parse and embed the whole document again.
            in_progress.append(name)
            continue
        size = store.size(name)
        if size is None:
            continue
        # Checked before reading: once the backlog is full, reading every remaining
        # original into memory only to defer it cost the whole corpus in RAM.
        if (limit is not None and len(jobs) >= limit) or not queue.backlog.has_room(size):
            deferred.append(name)
            continue
        content = store.read(name)
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
    return StaleReindex(jobs, deferred, in_progress, failed)


def known_filenames(repository: VectorRepository, settings: AppSettings) -> list[str]:
    """Every indexed filename in ``settings``' workspace plus those with an upload or
    delete in flight: what a new name is checked against by
    ``refuse_near_duplicate_name`` (uploads, URL ingest, import and the folder watcher)."""

    return [*repository.filename_metadata(settings), *indexing_now(settings)]


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


def finish_ingest_future(
    future: Future[None],
    release: Callable[[], None],
    job_store: JobStore,
    job_id: str,
    *,
    cancelled: bool = False,
) -> None:
    """Release the job's backlog slot, and surface anything that escaped the job.

    ``run_ingest_job`` records failures on the job, but the job store itself can fail
    while doing so (sqlite disk full); nothing observed the future, so that left the
    job stuck in a non-terminal state with nothing in the log.

    A cancelled future is a job shutdown dropped from the queue before it started; it
    is failed here so a client still polling this process doesn't see "queued" forever.
    """

    release()
    if future.cancelled():
        # Cancelled by DELETE /documents/jobs/{id}, or dropped by shutdown.
        error = JOB_CANCELLED_ERROR if cancelled else INTERRUPTED_JOB_ERROR
        try:
            job_store.update(job_id, state="failed", error=error)
        except Exception:
            logger.exception("Could not mark cancelled ingest job %s as failed.", job_id)
        return
    exc = future.exception()
    if exc is not None:
        logger.error("Ingest job %s crashed outside its own error handling.", job_id, exc_info=exc)
        # Mostly the job store failing to record the outcome, so the indexing may have
        # succeeded; a terminal state the poller can act on still beats "parsing"
        # forever. The ingest metrics were already counted by run_ingest_job.
        try:
            job_store.update(job_id, state="failed", error=UNRECORDED_OUTCOME_ERROR)
        except Exception:
            logger.exception("Could not mark ingest job %s as failed.", job_id)


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
    into a 500, and a second DELETE removes the orphan through the no-points path.
    """

    if raw_store is None:
        return False
    try:
        return raw_store.delete(filename)
    except OSError:
        if not points_gone:
            raise
        logger.exception("Could not remove the stored original of '%s'", filename)
        return True
