"""Document write routes: upload, preview, add from URL, refresh, re-index, version restore,
delete, tags and refresh settings."""

from __future__ import annotations

import functools
import logging
import threading
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from api.audit import audit
from api.corpus import bump_corpus_generation
from api.dependencies import (
    require_full_key,
)
from api.document_names import (
    FilenameConflictError,
    indexing_now,
    known_filenames,
    refuse_near_duplicate_name,
    stored_filename,
)
from api.document_ops import (
    admit_from_store,
    delete_indexed_document,
    enqueue_refreshed,
    original_still_stored,
    reindex_stale,
    require_raw_store,
    source_of,
)
from api.documents import (
    DocumentNotFoundError,
    DocumentSource,
    chunk_sections,
    normalize_tags,
    validate_upload_filename,
)
from api.ingest_jobs import backlog_full_error
from api.ingestion import filename_write_lock
from api.parse_isolation import parse_with_limit
from api.routes.deps import (
    IngestQueueDep,
    RawDocumentStoreDep,
    SettingsDep,
    TokenCounterDep,
    VectorRepositoryDep,
)
from api.routes.documents_common import (
    _fetch_for_ingest,
)
from api.schemas import (
    DocumentDeleteResponse,
    DocumentJobAcceptedResponse,
    DocumentPreviewResponse,
    DocumentRefreshResponse,
    DocumentRefreshSettingsRequest,
    DocumentRefreshSettingsResponse,
    DocumentReindexAllResponse,
    DocumentTagsRequest,
    DocumentTagsResponse,
    PreviewChunkResponse,
    UrlIngestRequest,
)
from api.upload import read_upload_within_limit
from api.url_fetch import (
    fetch_document,
    loggable_url,
)
from api.workspaces import document_key, workspace_of

logger = logging.getLogger("api.routes.documents")

router = APIRouter()


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
    tags: Annotated[
        list[str] | None,
        Form(
            description="Tags to give the document (replacing any it has). Giving tags "
            "re-indexes the same bytes rather than leaving them as indexed.",
        ),
    ] = None,
) -> DocumentJobAcceptedResponse:
    # Rejected here, not in the background job: an unusable name or type used to
    # be accepted with a 202 and only fail once the client polled the job.
    filename = validate_upload_filename(file.filename or "")
    given_tags = normalize_tags(tags) if tags else None
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
    accepted = await run_in_threadpool(
        queue.enqueue,
        filename=filename,
        content=content,
        raw_store=raw_store,
        tags=given_tags,
        # The same bytes again (a re-upload, a file sent twice): left as indexed —
        # unless tags were given, which the ingest has to write.
        dedupe=True,
    )
    audit(
        "document.upload",
        workspace=workspace_of(settings),
        filename=filename,
        size=len(content),
        job_id=accepted.job_id,
        **({"tags": list(given_tags)} if given_tags else {}),
    )
    return accepted


# At most this many previews parse at once (each holds a parse process and the file).
PREVIEW_MAX_CONCURRENT = 2


PREVIEW_FIRST_CHUNKS = 20


PREVIEW_SECTION_LABELS = 100


_preview_slots = threading.BoundedSemaphore(PREVIEW_MAX_CONCURRENT)


# How a file would be indexed — its sections, passages and the first of them — without
# indexing anything: to check a parser's take on a scanned PDF or a spreadsheet before
# uploading. The parse runs as an upload's does (in a child process, PARSE_TIMEOUT_
# SECONDS); nothing is embedded, stored or queued.
@router.post(
    "/documents/preview",
    response_model=DocumentPreviewResponse,
    dependencies=[Depends(require_full_key)],
    responses={
        400: {"description": "An unsupported, unreadable or empty file."},
        429: {"description": "Too many previews are running; try again in a moment."},
    },
)
async def preview_document(
    settings: SettingsDep,
    token_counter: TokenCounterDep,
    file: Annotated[UploadFile, File()],
) -> DocumentPreviewResponse:
    filename = validate_upload_filename(file.filename or "")
    if not _preview_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Too many previews are running. Try again in a moment.",
            headers={"Retry-After": "5"},
        )
    try:
        content = await read_upload_within_limit(file, int(settings.max_upload_bytes))
        sections = await run_in_threadpool(parse_with_limit, filename, content, settings)
        chunks = await run_in_threadpool(chunk_sections, sections, settings, token_counter)
    finally:
        _preview_slots.release()
    labels = list(dict.fromkeys(section.section for section in sections))
    return DocumentPreviewResponse(
        filename=filename,
        section_count=len(sections),
        chunk_count=len(chunks),
        page_count=max((chunk.page_end or chunk.page for chunk in chunks), default=1),
        section_labels=labels[:PREVIEW_SECTION_LABELS],
        first_chunks=[
            PreviewChunkResponse(
                ordinal=chunk.ordinal,
                page=chunk.page,
                page_end=chunk.page_end,
                section=chunk.section,
                text=chunk.text,
            )
            for chunk in chunks[:PREVIEW_FIRST_CHUNKS]
        ],
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
    given_tags = normalize_tags(request.tags) if request.tags else None
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
    accepted = await run_in_threadpool(
        queue.enqueue,
        filename=fetched.filename,
        content=fetched.content,
        raw_store=raw_store,
        # The URL asked for (not where a redirect went): what a refresh fetches again.
        source=DocumentSource(
            request.url.strip(), fetched.etag, fetched.last_modified, request.refresh_minutes
        ),
        tags=given_tags,
        dedupe=True,
        kind="url",
    )
    audit(
        "document.upload_from_url",
        workspace=workspace_of(settings),
        filename=fetched.filename,
        url=loggable_url(request.url.strip()),
        size=len(fetched.content),
        job_id=accepted.job_id,
        **({"tags": list(given_tags)} if given_tags else {}),
    )
    return accepted


# Puts an earlier version back: it is indexed as an upload of those bytes would be,
# and the version replaced is kept in its turn.
@router.post(
    "/documents/{filename}/versions/{version}/restore",
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
    # The same admission as a re-index: never alongside a job for the same document,
    # and sized against the backlog before the bytes are read.
    content = admit_from_store(
        repository,
        settings,
        queue,
        name,
        store.version_size(name, version),
        lambda: store.read_version(name, version),
        f"No version '{version}' of '{name}'.",
    )
    meta = repository.document_metadata(settings, name)
    accepted = queue.enqueue(
        filename=name,
        content=content,
        raw_store=store,
        # An earlier version of a fetched document still came from that URL; a refresh
        # brings the current one back — a full fetch, since the stored validators
        # describe the version being replaced.
        source=source_of(meta, validators=False) if meta is not None else None,
        kind="restore",
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
        409: {"description": "The document is being indexed."},
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
    if name in await run_in_threadpool(indexing_now, settings):
        # A second click would fetch again and queue a job that can only fail as
        # "changed while it was being fetched" once the first one lands.
        raise HTTPException(
            status_code=409, detail=f"'{name}' is being indexed; refresh it once that finishes."
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
        functools.partial(
            enqueue_refreshed,
            name,
            meta.source_url,
            fetched,
            queue,
            raw_store,
            expected=meta,
            repository=repository,
        )
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
    content = admit_from_store(
        repository,
        settings,
        queue,
        name,
        store.size(name),
        lambda: store.read(name),
        f"No stored original for '{name}' to re-index.",
    )
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
        kind="reindex",
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


# A fetched document's own refresh interval; payload-only, like tags.
@router.patch(
    "/documents/{filename}/refresh-settings",
    response_model=DocumentRefreshSettingsResponse,
    dependencies=[Depends(require_full_key)],
    responses={404: {"description": "No such document, or it wasn't fetched from a URL."}},
)
def set_document_refresh_settings(
    filename: str,
    request: DocumentRefreshSettingsRequest,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
) -> DocumentRefreshSettingsResponse:
    filename = stored_filename(filename)
    # Under the filename lock: a refresh or re-upload in flight writes the setting it
    # read under the same lock, so it can't put back the one from before this.
    with filename_write_lock(document_key(settings, filename)):
        meta = repository.document_metadata(settings, filename)
        if meta is None:
            raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
        if not meta.source_url:
            raise HTTPException(
                status_code=404, detail=f"'{filename}' was uploaded, not fetched from a URL."
            )
        repository.set_source_details(
            settings,
            filename,
            meta.source_etag,
            meta.source_last_modified,
            request.refresh_minutes,
        )
    bump_corpus_generation([filename], collection=settings.qdrant_collection)
    audit(
        "document.refresh_settings",
        workspace=workspace_of(settings),
        filename=filename,
        refresh_minutes=request.refresh_minutes,
    )
    return DocumentRefreshSettingsResponse(
        filename=filename, refresh_minutes=request.refresh_minutes
    )
