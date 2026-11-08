"""Ingest job routes: list, poll, retry and cancel jobs."""

from __future__ import annotations

import functools
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from api.audit import audit
from api.dependencies import (
    require_api_key,
    require_full_key,
)
from api.document_names import indexing_now
from api.document_ops import (
    enqueue_refetched,
    refuse_if_replaced,
    reindex_from_original,
    retry_refusal,
)
from api.ingest_jobs import (
    IngestQueue,
    cancel_ingest_job,
)
from api.jobs import IngestJob, JobNotFoundError
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.routes.deps import (
    IngestJobStoreDep,
    IngestQueueDep,
    RawDocumentStoreDep,
    SettingsDep,
    VectorRepositoryDep,
)
from api.routes.documents_common import (
    _fetch_for_ingest,
    _job_response,
    _job_status,
)
from api.schemas import (
    DocumentJobAcceptedResponse,
    DocumentJobListResponse,
    DocumentJobStatusResponse,
)
from api.settings import AppSettings
from api.url_fetch import (
    fetch_document,
)
from api.workspaces import workspace_of

logger = logging.getLogger("api.routes.documents")

router = APIRouter()


# GET /documents/jobs?state=recent|failed returns at most this many.
JOB_LIST_MAX_LIMIT = 500


# Guarded like its sibling document routes: the response carries the filename (and
# on failure the error text), and the frontend polls this URL once a second during
# every upload, so the job id lands in browser history and any intermediate proxy
# log — places a party who never held the key can read it from.
@router.get(
    "/documents/jobs",
    response_model=DocumentJobListResponse,
    dependencies=[Depends(require_api_key)],
)
def list_document_jobs(
    job_store: IngestJobStoreDep,
    settings: SettingsDep,
    state: Annotated[Literal["active", "recent", "failed"], Query()] = "active",
    limit: Annotated[int, Query(ge=1, le=JOB_LIST_MAX_LIMIT)] = 50,
) -> DocumentJobListResponse:
    """Queued and running jobs (``active``, oldest first), or finished ones still
    retained (``recent``, or only ``failed``, newest first) — a watcher's or an
    import's failures, which no client polled."""

    workspace = workspace_of(settings)
    if state == "active":
        jobs = job_store.list_active(workspace)
    else:
        jobs = job_store.list_finished(workspace, failed_only=state == "failed", limit=limit)
    return DocumentJobListResponse(jobs=[_job_response(job) for job in jobs])


@router.get(
    "/documents/jobs/{job_id}",
    response_model=DocumentJobStatusResponse,
    dependencies=[Depends(require_api_key)],
)
def get_document_job(
    job_id: str, job_store: IngestJobStoreDep, settings: SettingsDep
) -> DocumentJobStatusResponse:
    return _job_status(job_store, job_id, workspace_of(settings))


# Runs a failed job again as it was asked for: a URL fetched again, a re-index run
# again from the stored original. A failed upload's bytes were never kept (an
# original is stored only once indexing succeeds), so it needs the file sent again.
@router.post(
    "/documents/jobs/{job_id}/retry",
    response_model=DocumentJobAcceptedResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={
        404: {"description": "No such job (or another workspace's)."},
        409: {
            "description": "The job hasn't failed, can't be retried by the server (an "
            "upload: send the file again), or its document is being indexed."
        },
    },
)
async def retry_document_job(
    job_id: str,
    job_store: IngestJobStoreDep,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
) -> DocumentJobAcceptedResponse:
    workspace = workspace_of(settings)
    # Off the event loop: the sqlite job store blocks (async for the URL fetch below).
    job = await run_in_threadpool(job_store.get, job_id)
    if job is None or job.workspace != workspace:
        raise JobNotFoundError(f"No ingestion job found with id '{job_id}'.")
    refusal = retry_refusal(job)
    if refusal is not None:
        raise HTTPException(status_code=409, detail=refusal)
    if job.filename in await run_in_threadpool(indexing_now, settings):
        raise HTTPException(
            status_code=409,
            detail=f"'{job.filename}' is being indexed; wait for that job to finish.",
        )
    if job.kind == "reindex":
        accepted = await run_in_threadpool(
            reindex_from_original, job.filename, repository, settings, queue, raw_store
        )
    else:
        accepted = await _fetch_again(job, repository, settings, queue, raw_store)
    audit(
        "job.retry",
        workspace=workspace,
        job_id=job_id,
        filename=job.filename,
        new_job_id=accepted.job_id,
    )
    return accepted


async def _fetch_again(
    job: IngestJob,
    repository: VectorRepository,
    settings: AppSettings,
    queue: IngestQueue,
    raw_store: RawDocumentStore | None,
) -> DocumentJobAcceptedResponse:
    """A failed URL job's fetch, run again under the name it was stored under: the
    whole document (no validators — the failure may have been after a 200)."""

    if not settings.url_ingest_enabled:
        raise HTTPException(status_code=409, detail="URL ingest is not enabled.")
    assert job.source_url is not None
    await run_in_threadpool(repository.ensure_writable, settings)
    meta = await run_in_threadpool(repository.document_metadata, settings, job.filename)
    refuse_if_replaced(job, meta)
    fetched = await _fetch_for_ingest(
        queue,
        functools.partial(
            fetch_document,
            job.source_url,
            settings,
            filename=job.filename,
            has_room=queue.backlog.has_room,
        ),
    )
    return await run_in_threadpool(
        enqueue_refetched, job, repository, settings, queue, raw_store, fetched, meta
    )


# Stops a queued job outright, or a running one at its next checkpoint — but only
# before it writes anything: point ids are deterministic, so once a batch is
# written the previous version is partly overwritten and can't be restored (409;
# delete the document afterwards instead). Changes the corpus, so full key only.
@router.delete(
    "/documents/jobs/{job_id}",
    response_model=DocumentJobStatusResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
)
def cancel_document_job(
    job_id: str, job_store: IngestJobStoreDep, settings: SettingsDep
) -> DocumentJobStatusResponse:
    workspace = workspace_of(settings)
    # Checked before cancelling: another workspace's job is a 404, left running.
    _job_status(job_store, job_id, workspace)
    outcome = cancel_ingest_job(job_id)
    if outcome == "not_running":
        # A finished job can't be stopped.
        status = _job_status(job_store, job_id, workspace)
        raise HTTPException(status_code=409, detail=f"Job '{job_id}' has already {status.state}.")
    if outcome == "too_late":
        raise HTTPException(
            status_code=409,
            detail=(
                "This job is already writing to the index and can't be cancelled "
                "cleanly. Delete the document once it finishes instead."
            ),
        )
    status = _job_status(job_store, job_id, workspace)
    audit("job.cancel", workspace=workspace, job_id=job_id, filename=status.filename)
    return status
