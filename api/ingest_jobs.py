"""Background ingest jobs: enqueueing against the backlog, running, and recording outcomes."""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException

from api.documents import (
    DocumentError,
    DocumentSource,
)
from api.embeddings import EmbeddingError
from api.ingestion import (
    ClosedForWorkError,
    IngestionError,
    ingest_sequencer,
)
from api.jobs import INTERRUPTED_JOB_ERROR, IngestBacklog, JobStore
from api.metrics import (
    NO_ERROR_TYPE,
    ingest_chunks_total,
    ingest_job_failures_total,
    ingest_job_seconds,
    ingest_jobs_total,
)
from api.parse_isolation import (
    ParseEnvironmentError,
    ParseInterruptedError,
    ParseMemoryError,
)
from api.pipeline import IngestService
from api.qdrant_schema import CollectionSchemaError, VectorStoreUnavailableError
from api.raw_documents import RawDocumentStore
from api.schemas import (
    DocumentJobAcceptedResponse,
)
from api.workspaces import WorkspaceClosingError, document_key, workspace_of

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

    if isinstance(exc, ParseMemoryError):
        return "parse_memory"
    if isinstance(exc, ParseEnvironmentError):
        return "parse_failed"
    if isinstance(exc, ParseInterruptedError):
        return "parse_timeout"
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


@dataclass(frozen=True)
class IngestRequest:
    """One document to index, as an upload, URL fetch, re-index, restore, import or the
    watcher hands it over.

    ``raw_store`` keeps the original once indexed (None: not kept, or not replaced — a
    re-index must not rewrite the only copy). ``precondition`` refuses the job by
    raising, before the parse and again under the filename lock. ``tags`` set the
    document's tags outright (None carries the stored ones over). ``dedupe`` leaves
    identical bytes alone. ``kind`` (``jobs.JOB_KINDS``) says what started it,
    recorded on the job with the source URL and tags, for a retry.
    """

    filename: str
    content: bytes
    raw_store: RawDocumentStore | None = None
    uploaded_at: float | None = None
    precondition: Callable[[], None] | None = None
    tags: tuple[str, ...] | None = None
    source: DocumentSource | None = None
    dedupe: bool = False
    kind: str = "upload"
    # The watched folder that owns the document (a restore carries it; None keeps the
    # stored one, or the ingest service's own when it is the watcher's).
    watched_from: str | None = None


def run_ingest_job(
    job_store: JobStore,
    job_id: str,
    ingest_service: IngestService,
    request: IngestRequest,
    sequence: int | None = None,
    handle: JobHandle | None = None,
) -> None:
    """Background-executor entry point: parse/chunk/embed/index and update job status.

    Runs on ``get_ingest_executor()``'s worker threads, outside any request's
    lifetime — exceptions are caught and recorded on the job rather than raised,
    since there is no HTTP response left to attach them to. The original bytes (when
    ``raw_store`` is set) are saved only after indexing succeeds and under the same
    filename lock: saving them at upload time left orphans for uploads that then
    failed, and overwrote the original of a still-indexed older version.
    """

    filename, content, raw_store = request.filename, request.content, request.raw_store

    # The sequencer's key: the filename, workspace-qualified outside the default one.
    key = document_key(ingest_service.settings, filename)

    def on_progress(done: int, total: int) -> None:
        job_store.update(job_id, state="embedding", chunks_done=done, chunks_total=total)

    def check_sequence() -> None:
        if handle is not None:
            handle.check_cancelled()
        # The re-index precondition first: its "deleted before its re-index ran" names
        # the cause more exactly than the sequencer's generic refusal.
        if request.precondition is not None:
            request.precondition()
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
            # original (404) beats a wrong one. Its earlier versions stay: they are
            # still right, and a full disk shouldn't cost the document its history.
            logger.exception("Indexed %r but could not store its original bytes.", filename)
            try:
                raw_store.remove_current(filename)
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
            uploaded_at=request.uploaded_at,
            precondition=check_sequence,
            before_first_write=handle.begin_write if handle is not None else None,
            tags=request.tags,
            source=request.source,
            dedupe=request.dedupe,
            watched_from=request.watched_from,
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
        ingest_job_failures_total.labels(kind=request.kind).inc()
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
            "chunks_reused": outcome.chunks_reused,
            "chunks_removed": outcome.chunks_removed,
            "duplicate_of": outcome.duplicate_of,
            "warnings": list(outcome.warnings),
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
        watched_from: str | None = None,
    ) -> DocumentJobAcceptedResponse:
        """``enqueue_ingest`` with this queue's collaborators (``IngestRequest``'s fields
        as keywords)."""

        return enqueue_ingest(
            job_store=self.job_store,
            executor=self.executor,
            backlog=self.backlog,
            ingest_service=self.ingest_service,
            request=IngestRequest(
                filename=filename,
                content=content,
                raw_store=raw_store,
                uploaded_at=uploaded_at,
                precondition=precondition,
                tags=tags,
                source=source,
                dedupe=dedupe,
                kind=kind,
                watched_from=watched_from,
            ),
        )


def enqueue_ingest(
    *,
    job_store: JobStore,
    executor: ThreadPoolExecutor,
    backlog: IngestBacklog,
    ingest_service: IngestService,
    request: IngestRequest,
) -> DocumentJobAcceptedResponse:
    """Admit ``request``'s content against the backlog, create its job, and start it.
    Blocking.

    Shared by every way a document is indexed, so all get the same backlog 503, the
    same shutting-down handling, and the same done-callback that frees the bytes.
    """

    filename = request.filename
    size = len(request.content)
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
            kind=request.kind,
            source_url=request.source.url if request.source is not None else None,
            tags=request.tags,
            refresh_minutes=(
                request.source.refresh_minutes if request.source is not None else None
            ),
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
            request,
            sequence,
            handle,
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
