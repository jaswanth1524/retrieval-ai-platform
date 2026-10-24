"""Document routes: list, upload, re-index, delete, tags, jobs, content, original, export."""

from __future__ import annotations

import functools
import logging
import mimetypes
import os
import threading
import time
from collections.abc import Callable, Iterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from api.audit import audit
from api.corpus import bump_corpus_generation
from api.dependencies import (
    require_api_key,
    require_full_key,
)
from api.documents import (
    DocumentNotFoundError,
    DocumentSource,
    normalize_tags,
    validate_upload_filename,
)
from api.export import build_export
from api.ingest_jobs import (
    JOB_CANCELLED_ERROR,
    FilenameConflictError,
    IngestQueue,
    backlog_full_error,
    cancel_ingest_job,
    delete_indexed_document,
    indexing_now,
    known_filenames,
    original_still_stored,
    refuse_near_duplicate_name,
    reindex_stale,
    require_raw_store,
    source_of,
    staleness,
    stored_filename,
)
from api.ingestion import filename_write_lock
from api.jobs import IngestJob, JobNotFoundError, JobStore
from api.metrics import corpus_documents
from api.restore import BackupImportError, restore_backup
from api.routes.deps import (
    FeedbackStoreDep,
    IngestJobStoreDep,
    IngestQueueDep,
    RawDocumentStoreDep,
    SettingsDep,
    TokenCounterDep,
    VectorRepositoryDep,
)
from api.routes.responses import content_chunk
from api.schemas import (
    DocumentContentResponse,
    DocumentDeleteResponse,
    DocumentIngestResponse,
    DocumentJobAcceptedResponse,
    DocumentJobListResponse,
    DocumentJobStatusResponse,
    DocumentListResponse,
    DocumentRefreshResponse,
    DocumentReindexAllResponse,
    DocumentTagsRequest,
    DocumentTagsResponse,
    DocumentVersionResponse,
    DocumentVersionsResponse,
    ImportResponse,
    UrlIngestRequest,
)
from api.streaming import ClosingStreamingResponse
from api.upload import read_upload_within_limit
from api.url_fetch import BacklogFullError, FetchedDocument, fetch_document, loggable_url
from api.workspaces import document_key, workspace_of

logger = logging.getLogger(__name__)

_ORIGINAL_CHUNK_BYTES = 64 * 1024
# Held while an export streams; a second one meanwhile is a 409.
_export_running = threading.Lock()
# A page of /content is at most 2 * this + 1 chunks.
CONTENT_MAX_RADIUS = 100
# Far past any real document; a 400-digit `start` used to reach Qdrant's filter and 500.
CONTENT_MAX_ORDINAL = 1_000_000
# /content with no paging parameters returns at most this many chunks.
CONTENT_UNPAGED_MAX_CHUNKS = 2000
# POST /documents/from-url downloads at most this many documents at once.
URL_FETCH_MAX_CONCURRENT = 4
_url_fetch_slots = threading.BoundedSemaphore(URL_FETCH_MAX_CONCURRENT)


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


# The document and export routes; registered on the app by ``register``.
router = APIRouter()


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    dependencies=[Depends(require_api_key)],
)
def list_documents(
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    raw_store: RawDocumentStoreDep,
    token_counter: TokenCounterDep,
) -> DocumentListResponse:
    metadata = repository.filename_metadata(settings)
    corpus_documents.set(len(metadata))
    is_stale = staleness(settings, token_counter)
    stored_originals = raw_store.stored_names() if raw_store is not None else set()
    return DocumentListResponse(
        filenames=sorted(metadata),
        chunk_counts={name: meta.chunk_count for name, meta in metadata.items()},
        page_counts={name: meta.page_count for name, meta in metadata.items()},
        byte_sizes={
            name: meta.byte_size for name, meta in metadata.items() if meta.byte_size is not None
        },
        uploaded_ats={
            name: meta.uploaded_at
            for name, meta in metadata.items()
            if meta.uploaded_at is not None
        },
        stale_filenames=sorted(name for name, meta in metadata.items() if is_stale(meta)),
        reindexable_filenames=sorted(set(metadata) & stored_originals),
        tags={name: list(meta.tags) for name, meta in metadata.items() if meta.tags},
        # Without the query string or fragment, which can carry a signed token: the
        # listing is readable with the read key. Refresh uses the stored full URL.
        source_urls={
            name: loggable_url(meta.source_url)
            for name, meta in metadata.items()
            if meta.source_url
        },
    )


@router.post(
    "/documents",
    response_model=DocumentJobAcceptedResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={
        409: {
            "description": "With stored originals, the name differs from an existing "
            "document's only by letter case or accents; or the collection was built "
            "for another embedding setup and still holds documents."
        }
    },
)
async def upload_document(
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
    file: Annotated[UploadFile, File()],
) -> DocumentJobAcceptedResponse:
    # Rejected here, not in the background job: an unusable name or type used to
    # be accepted with a 202 and only fail once the client polled the job.
    filename = validate_upload_filename(file.filename or "")
    # 409 now, as /reindex and /import do, rather than a 202 for a job that can
    # only fail on a collection built for another embedding setup.
    await run_in_threadpool(repository.ensure_writable, settings)
    if raw_store is not None:
        known = await run_in_threadpool(known_filenames, repository, settings)
        try:
            refuse_near_duplicate_name(filename, known)
        except FilenameConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    # The multipart parser has already spooled the part and knows its size: a full
    # backlog refuses it before it is read into memory, not after (enqueue_ingest
    # still makes the real reservation).
    if file.size is not None and not queue.backlog.has_room(file.size):
        raise backlog_full_error()
    content = await read_upload_within_limit(file, int(settings.max_upload_bytes))
    # job_store.create is a sqlite INSERT under JOB_STORE_BACKEND=sqlite; off the
    # event loop so it can't stall other in-flight requests.
    return await run_in_threadpool(
        queue.enqueue,
        filename=filename,
        content=content,
        raw_store=raw_store,
        # The same bytes again (a re-upload, a file sent twice): left as indexed.
        dedupe=True,
    )


# The server downloads the document (URL_INGEST_ENABLED, off by default) and it
# takes the upload's path from there: same name checks, backlog, job and polling.
# api/url_fetch.py guards against fetching private addresses (SSRF).
@router.post(
    "/documents/from-url",
    response_model=DocumentJobAcceptedResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={
        404: {"description": "URL ingest is off (URL_INGEST_ENABLED)."},
        409: {
            "description": "The remote server named the document after one already "
            "indexed (pass `filename` to replace it); or, with stored originals, the "
            "name differs from an existing document's only by letter case or accents."
        },
    },
)
async def upload_document_from_url(
    request: UrlIngestRequest,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
) -> DocumentJobAcceptedResponse:
    if not settings.url_ingest_enabled:
        raise HTTPException(status_code=404, detail="URL ingest is not enabled.")
    await run_in_threadpool(repository.ensure_writable, settings)
    fetched = await _fetch_for_ingest(
        queue,
        functools.partial(
            fetch_document,
            request.url,
            settings,
            filename=request.filename,
            has_room=queue.backlog.has_room,
        ),
    )
    if raw_store is not None or fetched.remote_named:
        known = await run_in_threadpool(known_filenames, repository, settings)
        # A name the remote server picked (Content-Disposition, a redirect) must not
        # replace a document already here: re-adding the same URL keeps its own
        # name and replaces it as an upload would; a page calling itself
        # "handbook.pdf" doesn't.
        if fetched.remote_named and fetched.filename in known:
            raise HTTPException(
                status_code=409,
                detail=f"The server at that URL names the document "
                f"'{fetched.filename}', and a document with that name is already "
                "indexed; it was not replaced. Delete or rename that document first, "
                "or give this one a name (`filename` in the API, --name on the CLI).",
            )
        if raw_store is not None:
            try:
                refuse_near_duplicate_name(fetched.filename, known)
            except FilenameConflictError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
    logger.info(
        "Fetched %s (%d bytes) for indexing.",
        loggable_url(fetched.final_url),
        len(fetched.content),
    )
    return await run_in_threadpool(
        queue.enqueue,
        filename=fetched.filename,
        content=fetched.content,
        raw_store=raw_store,
        # The URL asked for (not where a redirect went): what a refresh fetches again.
        source=DocumentSource(request.url.strip(), fetched.etag, fetched.last_modified),
        dedupe=True,
    )


# Earlier versions of a document's original, kept by re-uploads when
# RAW_DOCUMENT_VERSIONS is set; 409 without stored originals, like /reindex.
@router.get(
    "/documents/{filename}/versions",
    response_model=DocumentVersionsResponse,
    dependencies=[Depends(require_api_key)],
)
def list_document_versions(
    filename: str, raw_store: RawDocumentStoreDep
) -> DocumentVersionsResponse:
    store = require_raw_store(raw_store)
    name = stored_filename(filename)
    return DocumentVersionsResponse(
        filename=name,
        versions=[
            DocumentVersionResponse(version=v.version, saved_at=v.saved_at, size=v.size)
            for v in store.versions(name)
        ],
    )


# Puts an earlier version back: it is indexed as an upload of those bytes would be,
# and the version replaced is kept in its turn.
@router.post(
    "/documents/{filename}/versions/{version}/restore",
    response_model=DocumentJobAcceptedResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
)
def restore_document_version(
    filename: str,
    version: str,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
) -> DocumentJobAcceptedResponse:
    store = require_raw_store(raw_store)
    name = stored_filename(filename)
    repository.ensure_writable(settings)
    content = store.read_version(name, version)
    if content is None:
        raise DocumentNotFoundError(f"No version '{version}' of '{name}'.")
    accepted = queue.enqueue(
        filename=name,
        content=content,
        raw_store=store,
    )
    audit(
        "document.version_restore",
        workspace=workspace_of(settings),
        filename=name,
        version=version,
        job_id=accepted.job_id,
    )
    return accepted


# Fetches a URL-ingested document again from where it came from, conditionally: a
# server that says nothing changed (304) costs no download and no job.
@router.post(
    "/documents/{filename}/refresh",
    response_model=DocumentJobAcceptedResponse | DocumentRefreshResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={
        200: {
            "model": DocumentRefreshResponse,
            "description": "Unchanged at the source (304); nothing was fetched.",
        },
        404: {"description": "No such document, it wasn't fetched by URL, or URL ingest is off."},
    },
)
async def refresh_document_from_url(
    filename: str,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
) -> DocumentJobAcceptedResponse | Response:
    if not settings.url_ingest_enabled:
        raise HTTPException(status_code=404, detail="URL ingest is not enabled.")
    name = stored_filename(filename)
    meta = await run_in_threadpool(repository.document_metadata, settings, name)
    if meta is None:
        raise DocumentNotFoundError(f"No indexed document named '{name}'.")
    if not meta.source_url:
        raise HTTPException(
            status_code=404, detail=f"'{name}' was uploaded, not fetched from a URL."
        )
    await run_in_threadpool(repository.ensure_writable, settings)
    fetched = await _fetch_for_ingest(
        queue,
        functools.partial(
            fetch_document,
            meta.source_url,
            settings,
            # Its own name, whatever the server calls it now.
            filename=name,
            has_room=queue.backlog.has_room,
            etag=meta.source_etag,
            last_modified=meta.source_last_modified,
        ),
    )
    if fetched.not_modified:
        return Response(
            content=DocumentRefreshResponse(filename=name).model_dump_json(),
            status_code=200,
            media_type="application/json",
        )
    accepted = await run_in_threadpool(
        queue.enqueue,
        filename=name,
        content=fetched.content,
        raw_store=raw_store,
        source=DocumentSource(meta.source_url, fetched.etag, fetched.last_modified),
        # Its tags carry over (as on any re-upload); unchanged bytes are left alone.
        dedupe=True,
    )
    audit(
        "document.refresh", workspace=workspace_of(settings), filename=name, job_id=accepted.job_id
    )
    return accepted


# Re-chunks and re-embeds a document from its stored original (RAW_DOCUMENT_DIR),
# so a chunking/embedding change applies without the user re-uploading. Same job
# flow as an upload: 202 + a job id to poll.
@router.post(
    "/documents/reindex",
    response_model=DocumentReindexAllResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
)
def reindex_stale_documents(
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
    token_counter: TokenCounterDep,
) -> DocumentReindexAllResponse:
    store = require_raw_store(raw_store)
    outcome = reindex_stale(repository, settings, queue, store, token_counter)
    audit(
        "document.reindex_stale",
        workspace=workspace_of(settings),
        queued=len(outcome.jobs),
        deferred=len(outcome.deferred),
        in_progress=len(outcome.in_progress),
        failed=len(outcome.failed),
    )
    return DocumentReindexAllResponse(
        jobs=outcome.jobs,
        deferred=outcome.deferred,
        in_progress=outcome.in_progress,
        failed=outcome.failed,
    )


@router.post(
    "/documents/{filename}/reindex",
    response_model=DocumentJobAcceptedResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={
        409: {
            "description": "The document is already being indexed, originals aren't "
            "kept (RAW_DOCUMENT_DIR), or the index was built for another embedding setup."
        }
    },
)
def reindex_document(
    filename: str,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
) -> DocumentJobAcceptedResponse:
    store = require_raw_store(raw_store)
    name = stored_filename(filename)
    repository.ensure_writable(settings)
    if name in indexing_now(settings):
        raise HTTPException(
            status_code=409,
            detail=f"'{name}' is already being indexed; wait for that job to finish.",
        )
    # Sized before reading, as the bulk path does: a full backlog would otherwise
    # read the whole original into memory only to refuse it.
    size = store.size(name)
    if size is None:
        raise DocumentNotFoundError(f"No stored original for '{name}' to re-index.")
    if not queue.backlog.has_room(size):
        raise backlog_full_error()
    content = store.read(name)
    if content is None:
        raise DocumentNotFoundError(f"No stored original for '{name}' to re-index.")
    meta = repository.document_metadata(settings, name)
    accepted = queue.enqueue(
        filename=name,
        content=content,
        # The original is already stored, byte-identical. Passing the store would
        # re-save it, and a failed save deletes it — losing the only copy.
        raw_store=None,
        uploaded_at=meta.uploaded_at if meta is not None else None,
        precondition=original_still_stored(store, name),
        # A re-index is the same document: where it came from carries over.
        source=source_of(meta) if meta is not None else None,
    )
    audit(
        "document.reindex", workspace=workspace_of(settings), filename=name, job_id=accepted.job_id
    )
    return accepted


@router.delete(
    "/documents/{filename}",
    response_model=DocumentDeleteResponse,
    dependencies=[Depends(require_full_key)],
    responses={
        409: {
            "description": "A newer upload of the document was indexed after this "
            "delete was requested; it was kept."
        }
    },
)
def delete_document(
    filename: str,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    raw_store: RawDocumentStoreDep,
) -> DocumentDeleteResponse:
    # Filenames are always basenames (normalize_filename strips any path
    # component at upload time), so a plain path segment is sufficient — a
    # filename containing "/" simply can't match this route, which is fine
    # since one can never have been uploaded.
    #
    # Same lock the ingest path takes: snapshot-then-delete is a read-modify-write,
    # so without it a delete landing between a concurrent ingest's upsert and its
    # stale-cleanup would remove the points that ingest just wrote, and the ingest
    # would still report success. The raw copy (when stored) is removed under the
    # same lock, or a deleted document's original bytes would accumulate forever.
    filename = stored_filename(filename)
    deleted = delete_indexed_document(repository, settings, raw_store, filename)
    return DocumentDeleteResponse(filename=filename, points_deleted=deleted)


# Payload-only: no re-embedding, so it is instant even for a large document.
@router.patch(
    "/documents/{filename}/tags",
    response_model=DocumentTagsResponse,
    dependencies=[Depends(require_full_key)],
)
def set_document_tags(
    filename: str,
    request: DocumentTagsRequest,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
) -> DocumentTagsResponse:
    filename = stored_filename(filename)
    tags = normalize_tags(request.tags)
    # Under the filename lock ingest also reads tags under (ingest_chunks'
    # carry_tags), so a re-upload in flight either sees this change or runs after
    # it and carries it over — it never writes the tags from before it.
    with filename_write_lock(document_key(settings, filename)):
        # A count, not a scroll of every point id just to see whether there is one.
        if repository.chunk_count_for_filename(settings, filename) == 0:
            raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
        repository.set_tags(settings, filename, tags)
    bump_corpus_generation([filename], collection=settings.qdrant_collection)
    audit("document.tags", workspace=workspace_of(settings), filename=filename, tags=list(tags))
    return DocumentTagsResponse(filename=filename, tags=list(tags))


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
    state: Annotated[Literal["active"], Query()] = "active",
) -> DocumentJobListResponse:
    # Only "active" for now: finished jobs are what each client polled itself.
    return DocumentJobListResponse(
        jobs=[_job_response(job) for job in job_store.list_active(workspace_of(settings))]
    )


@router.get(
    "/documents/jobs/{job_id}",
    response_model=DocumentJobStatusResponse,
    dependencies=[Depends(require_api_key)],
)
def get_document_job(
    job_id: str, job_store: IngestJobStoreDep, settings: SettingsDep
) -> DocumentJobStatusResponse:
    return _job_status(job_store, job_id, workspace_of(settings))


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


# Guarded alongside GET /documents: this returns every chunk's raw text, so leaving
# it open would let an unauthenticated caller list the corpus and then dump it in
# full — the key would protect Q&A while the documents themselves stayed readable.
@router.get(
    "/documents/{filename}/content",
    response_model=DocumentContentResponse,
    dependencies=[Depends(require_api_key)],
)
def document_content(
    filename: str,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    around: Annotated[str | None, Query(max_length=200)] = None,
    radius: Annotated[int, Query(ge=0, le=CONTENT_MAX_RADIUS)] = 30,
    start: Annotated[int | None, Query(ge=1, le=CONTENT_MAX_ORDINAL)] = None,
    end: Annotated[int | None, Query(ge=1, le=CONTENT_MAX_ORDINAL)] = None,
) -> DocumentContentResponse:
    # No parameters: the whole document, as before. `around` (a chunk id) returns
    # `radius` chunks either side of it; `start`/`end` a range of ordinals. A 50 MB
    # CSV used to ship every chunk so the viewer could show the few around a citation.
    filename = stored_filename(filename)
    total = repository.chunk_count_for_filename(settings, filename)
    if total == 0:
        raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
    if around is None and start is None and end is None:
        # The whole document, up to CONTENT_UNPAGED_MAX_CHUNKS: unbounded, one
        # request could pull a 100k-chunk document into memory. total_chunks still
        # reports the real size, so a client can tell the list was cut short.
        if total <= CONTENT_UNPAGED_MAX_CHUNKS:
            payloads = repository.chunks_for_filename(settings, filename)
        else:
            payloads = (
                repository.chunks_in_ordinal_range(
                    settings, filename, 1, CONTENT_UNPAGED_MAX_CHUNKS
                )
                or repository.chunks_for_filename(settings, filename)[:CONTENT_UNPAGED_MAX_CHUNKS]
            )  # a legacy document with no ordinals to range over
        return DocumentContentResponse(
            filename=filename,
            chunks=[content_chunk(payload) for payload in payloads],
            total_chunks=total,
        )
    target_found: bool | None = None
    if around is not None:
        ordinal = repository.chunk_ordinal_of(settings, filename, around)
        target_found = ordinal is not None
        # A chunk a re-index replaced: show the start of the document instead.
        center = ordinal if ordinal is not None else 1
        first, last = max(1, center - radius), center + radius
    else:
        first = start if start is not None else 1
        last = end if end is not None else first + 2 * CONTENT_MAX_RADIUS
        if last < first:
            raise HTTPException(status_code=422, detail="`end` must not be before `start`.")
        last = min(last, first + 2 * CONTENT_MAX_RADIUS)
    payloads = repository.chunks_in_ordinal_range(settings, filename, first, last)
    return DocumentContentResponse(
        filename=filename,
        chunks=[content_chunk(payload) for payload in payloads],
        total_chunks=total,
        target_found=target_found,
    )


# Guarded at least as strictly as GET /documents/{filename}/content above: raw
# original bytes are strictly more sensitive than derived chunk text. Returns 404
# both when raw_document_dir is unset (feature off) and when it's set but this
# particular filename predates the feature or was never stored — the client sees
# the same "not found" either way, not a distinct "feature unavailable" state.
@router.get("/documents/{filename}/original", dependencies=[Depends(require_api_key)])
def document_original(filename: str, raw_store: RawDocumentStoreDep) -> Response:
    filename = stored_filename(filename)
    path = raw_store.path(filename) if raw_store is not None else None
    if path is None:
        raise DocumentNotFoundError(f"No stored original for '{filename}'.")
    # Open before building the response. FileResponse re-stats and opens the path
    # only when the response is sent, so a DELETE landing in between turned this
    # 404 into a 500; an already-open handle keeps streaming the bytes it opened
    # even if the file is unlinked meanwhile.
    try:
        handle = path.open("rb")
    except FileNotFoundError:
        raise DocumentNotFoundError(f"No stored original for '{filename}'.") from None
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    # Headers from FileResponse (given the stat, it does no I/O of its own): it
    # RFC 5987-encodes the name — a hand-built latin-1 header 500'd on any
    # non-Latin-1 filename and broke on one containing a quote.
    template = FileResponse(
        path,
        media_type=media_type,
        filename=filename,
        stat_result=os.fstat(handle.fileno()),
        headers={"X-Content-Type-Options": "nosniff"},
    )
    headers = {
        name: value
        for name, value in template.headers.items()
        # Ranges need seeking the streamed body can't do; its own type is set below.
        if name not in ("accept-ranges", "content-type")
    }

    def chunks() -> Iterator[bytes]:
        with handle:
            while block := handle.read(_ORIGINAL_CHUNK_BYTES):
                yield block

    # Closing: a client that leaves mid-download lets go of the file at once.
    return ClosingStreamingResponse(chunks(), media_type=media_type, headers=headers)


# Everything beyond the vectors, as one zip: manifest, stored originals, feedback.
# See api/export.py. Operator data, so the full key only.
@router.get("/export", dependencies=[Depends(require_full_key)])
def export_corpus(
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    raw_store: RawDocumentStoreDep,
    feedback_store: FeedbackStoreDep,
) -> ClosingStreamingResponse:
    # One at a time: each export reads every stored original on its own thread,
    # and nothing else bounds how many a client could start.
    if not _export_running.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="An export is already running. Try again when it ends."
        )
    try:
        # Streamed as it is written: nothing is spooled to /tmp or held in memory.
        archive = build_export(settings, repository, raw_store, feedback_store)
    except BaseException:
        _export_running.release()
        raise
    released = threading.Event()

    def close() -> None:
        # However the response ends, a client that leaves included: the writer
        # thread stops and lets go of the original it was reading.
        archive.close()
        if not released.is_set():
            released.set()
            _export_running.release()

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return ClosingStreamingResponse(
        archive,
        close=close,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="docrag-export-{stamp}.zip"',
            "X-Content-Type-Options": "nosniff",
            # Documents the backup holds no original for, and so can't restore.
            "X-DocRAG-Originals-Missing": str(archive.originals_missing),
        },
    )


# The other half of /export: re-ingests each stored original with its tags and
# upload time, and adds back the feedback (api/restore.py). Documents already
# indexed here are skipped, so importing the same backup twice is harmless and a
# deferred one can be finished by importing again. Full key only.
@router.post(
    "/import",
    response_model=ImportResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
)
def import_backup(
    file: Annotated[UploadFile, File()],
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
    feedback_store: FeedbackStoreDep,
) -> ImportResponse:
    # Read once for the whole backup, then kept up to date with what it queues:
    # read per document, each job finishing mid-import moved the corpus generation
    # and the next document paid for a fresh listing.
    known: set[str] | None = None

    def enqueue(
        filename: str, content: bytes, uploaded_at: float | None, tags: tuple[str, ...]
    ) -> DocumentJobAcceptedResponse:
        nonlocal known
        if raw_store is not None:
            if known is None:
                known = set(known_filenames(repository, settings))
            # A DocumentError: listed under `rejected`, the rest still restored.
            refuse_near_duplicate_name(filename, known)
            known.add(filename)
        return queue.enqueue(
            filename=filename,
            content=content,
            raw_store=raw_store,
            uploaded_at=uploaded_at,
            tags=tags,
        )

    # The last step of recovering from an embedding change (export, delete all,
    # import): an emptied collection is rebuilt here; a non-empty one is a 409.
    repository.ensure_writable(settings)
    try:
        outcome = restore_backup(
            file.file,
            existing_filenames=set(repository.filename_metadata(settings)),
            max_original_bytes=int(settings.max_upload_bytes),
            has_room=queue.backlog.has_room,
            enqueue=enqueue,
            feedback_store=feedback_store,
            workspace=workspace_of(settings),
        )
    except BackupImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit(
        "backup.import",
        workspace=workspace_of(settings),
        queued=len(outcome.jobs),
        deferred=len(outcome.deferred),
        skipped=len(outcome.skipped),
        rejected=len(outcome.rejected),
        feedback_imported=outcome.feedback_imported,
    )
    return ImportResponse(
        jobs=outcome.jobs,
        deferred=outcome.deferred,
        skipped=outcome.skipped,
        missing_originals=outcome.missing_originals,
        rejected=outcome.rejected,
        feedback_imported=outcome.feedback_imported,
        feedback_skipped=outcome.feedback_skipped,
    )


def register(app: FastAPI) -> None:
    """Register the document and export routes."""

    app.include_router(router)
