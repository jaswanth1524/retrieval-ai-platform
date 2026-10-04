"""Background ingest jobs: enqueueing against the backlog, running, and recording outcomes."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Literal

from fastapi import HTTPException

from api.documents import (
    DocumentError,
    DocumentNotFoundError,
    UnsupportedDocumentError,
    normalize_filename,
)
from api.embeddings import EmbeddingError
from api.ingestion import IngestionError, ingest_sequencer
from api.jobs import INTERRUPTED_JOB_ERROR, IngestBacklog, JobStore
from api.metrics import (
    NO_ERROR_TYPE,
    ingest_chunks_total,
    ingest_job_seconds,
    ingest_jobs_total,
)
from api.pipeline import IngestService
from api.qdrant_schema import CollectionSchemaError, VectorStoreUnavailableError
from api.raw_documents import RawDocumentStore
from api.schemas import (
    DocumentJobAcceptedResponse,
)

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
) -> None:
    """Background-executor entry point: parse/chunk/embed/index and update job status.

    Runs on ``get_ingest_executor()``'s worker threads, outside any request's
    lifetime — exceptions are caught and recorded on the job rather than raised,
    since there is no HTTP response left to attach them to. The original bytes (when
    ``raw_store`` is set) are saved only after indexing succeeds and under the same
    filename lock: saving them at upload time left orphans for uploads that then
    failed, and overwrote the original of a still-indexed older version.
    """

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
            ingest_sequencer.check(filename, sequence)

    def save_original() -> None:
        if sequence is not None:
            ingest_sequencer.mark_applied(filename, sequence)
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
        )
    except Exception as exc:
        logger.warning("Ingest job %s for %r failed: %s", job_id, filename, exc, exc_info=True)
        ingest_job_seconds.labels(outcome="failed").observe(time.monotonic() - started)
        ingest_jobs_total.labels(outcome="failed", error_type=ingest_error_type(exc)).inc()
        job_store.update(job_id, state="failed", error=job_error_message(exc))
        return

    ingest_job_seconds.labels(outcome="done").observe(time.monotonic() - started)
    ingest_jobs_total.labels(outcome="done", error_type=NO_ERROR_TYPE).inc()
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
        },
    )


BACKLOG_FULL_ERROR = (
    "Too many documents are already waiting to be indexed. Retry once some of them finish."
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
) -> DocumentJobAcceptedResponse:
    """Admit ``content`` against the backlog, create its job, and start it. Blocking.

    Shared by upload and re-index so both get the same backlog 503, the same
    shutting-down handling, and the same done-callback that frees the bytes.
    """

    size = len(content)
    if not backlog.try_reserve(size):
        raise HTTPException(
            status_code=503, detail=BACKLOG_FULL_ERROR, headers={"Retry-After": "15"}
        )
    sequence = ingest_sequencer.admit(filename)

    def release() -> None:
        backlog.release(size)
        ingest_sequencer.release(filename)

    try:
        job = job_store.create(filename)
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
        future = executor.submit(
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
