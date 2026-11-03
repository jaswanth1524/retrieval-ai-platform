"""Helpers the document routers share: job status responses and the URL fetch admission."""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from api.document_ops import retry_refusal
from api.ingest_jobs import (
    JOB_CANCELLED_ERROR,
    IngestQueue,
    backlog_full_error,
)
from api.jobs import IngestJob, JobNotFoundError, JobStore
from api.schemas import (
    DocumentIngestResponse,
    DocumentJobStatusResponse,
)
from api.url_fetch import (
    BacklogFullError,
    FetchedDocument,
    fetch_slots,
)

logger = logging.getLogger("api.routes.documents")


# POST /documents/from-url downloads at most URL_FETCH_MAX_CONCURRENT documents at once,
# shared with the scheduled refresh.
_url_fetch_slots = fetch_slots


def _job_status(job_store: JobStore, job_id: str, workspace: str) -> DocumentJobStatusResponse:
    job = job_store.get(job_id)
    # Another workspace's job is "not found" here, the same as an unknown id: its
    # status carries the filename and error text.
    if job is None or job.workspace != workspace:
        raise JobNotFoundError(f"No ingestion job found with id '{job_id}'.")
    return _job_response(job)


def _job_response(job: IngestJob) -> DocumentJobStatusResponse:
    result = DocumentIngestResponse.model_validate(job.result) if job.result is not None else None
    return DocumentJobStatusResponse(
        job_id=job.id,
        filename=job.filename,
        state=job.state,
        chunks_total=job.chunks_total,
        chunks_done=job.chunks_done,
        error=job.error,
        result=result,
        cancelled=job.error == JOB_CANCELLED_ERROR,
        created_at=job.created_at_wall,
        unchanged=bool(result is not None and result.unchanged),
        kind=job.kind,
        retryable=job.state == "failed" and retry_refusal(job) is None,
    )


async def _fetch_for_ingest(
    queue: IngestQueue, fetch: Callable[[], FetchedDocument]
) -> FetchedDocument:
    """Run a URL fetch for indexing (add from URL, refresh), admitted like an upload.

    A full backlog refuses before the download, as an upload is refused before its
    body is read; ``fetch_document`` asks again with the size the server declares,
    before reading the body, and ``enqueue`` makes the real reservation. Each fetch
    holds a request thread for up to URL_INGEST_TIMEOUT_SECONDS and up to
    MAX_UPLOAD_BYTES of memory: a few at a time, the rest told to retry.
    """

    if not queue.backlog.has_room(1):
        raise backlog_full_error()
    if not _url_fetch_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Too many URL fetches are running. Try again in a moment.",
            headers={"Retry-After": "5"},
        )
    try:
        return await run_in_threadpool(fetch)
    except BacklogFullError:
        raise backlog_full_error() from None
    finally:
        _url_fetch_slots.release()
