"""Document routes: list, upload, re-index, delete, tags, jobs, content, original, export."""

from __future__ import annotations

import logging
import mimetypes
import os
import time
from collections.abc import Callable, Iterator
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse

from api.chunking import TokenCounter
from api.corpus import bump_corpus_generation
from api.dependencies import (
    require_api_key,
    require_full_key,
)
from api.documents import (
    DocumentNotFoundError,
    chunking_fingerprint,
    normalize_tags,
    validate_upload_filename,
)
from api.export import build_export, iter_file
from api.ingest_jobs import (
    JOB_CANCELLED_ERROR,
    FilenameConflictError,
    cancel_ingest_job,
    enqueue_ingest,
    original_still_stored,
    refuse_near_duplicate_name,
    require_raw_store,
    stored_filename,
)
from api.ingestion import filename_write_lock, ingest_sequencer
from api.jobs import JobNotFoundError, JobStore
from api.raw_documents import RawDocumentStore
from api.repository import DocumentMetadata, VectorRepository
from api.restore import BackupImportError, restore_backup
from api.routes.deps import (
    FeedbackStoreDep,
    IngestBacklogDep,
    IngestExecutorDep,
    IngestJobStoreDep,
    IngestServiceDep,
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
    DocumentJobStatusResponse,
    DocumentListResponse,
    DocumentReindexAllResponse,
    DocumentTagsRequest,
    DocumentTagsResponse,
    ImportResponse,
)
from api.settings import AppSettings
from api.upload import read_upload_within_limit

logger = logging.getLogger(__name__)

_ORIGINAL_CHUNK_BYTES = 64 * 1024
# A page of /content is at most 2 * this + 1 chunks.
CONTENT_MAX_RADIUS = 100
# /content with no paging parameters returns at most this many chunks.
CONTENT_UNPAGED_MAX_CHUNKS = 2000


def _job_status(job_store: JobStore, job_id: str) -> DocumentJobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise JobNotFoundError(f"No ingestion job found with id '{job_id}'.")
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
    )


def register(app: FastAPI) -> None:
    """Register the document and export routes."""

    @app.get(
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
        is_stale = _staleness(settings, token_counter)
        return DocumentListResponse(
            filenames=sorted(metadata),
            chunk_counts={name: meta.chunk_count for name, meta in metadata.items()},
            page_counts={name: meta.page_count for name, meta in metadata.items()},
            byte_sizes={
                name: meta.byte_size
                for name, meta in metadata.items()
                if meta.byte_size is not None
            },
            uploaded_ats={
                name: meta.uploaded_at
                for name, meta in metadata.items()
                if meta.uploaded_at is not None
            },
            stale_filenames=sorted(name for name, meta in metadata.items() if is_stale(meta)),
            reindexable_filenames=sorted(
                name
                for name in metadata
                if raw_store is not None and raw_store.path(name) is not None
            ),
            tags={name: list(meta.tags) for name, meta in metadata.items() if meta.tags},
        )

    @app.post(
        "/documents",
        response_model=DocumentJobAcceptedResponse,
        status_code=202,
        dependencies=[Depends(require_full_key)],
        responses={
            409: {
                "description": "With stored originals, the name differs from an existing "
                "document's only by letter case or accents."
            }
        },
    )
    async def upload_document(
        ingest_service: IngestServiceDep,
        repository: VectorRepositoryDep,
        settings: SettingsDep,
        job_store: IngestJobStoreDep,
        executor: IngestExecutorDep,
        backlog: IngestBacklogDep,
        raw_store: RawDocumentStoreDep,
        file: Annotated[UploadFile, File()],
    ) -> DocumentJobAcceptedResponse:
        # Rejected here, not in the background job: an unusable name or type used to
        # be accepted with a 202 and only fail once the client polled the job.
        filename = validate_upload_filename(file.filename or "")
        if raw_store is not None:
            known = await run_in_threadpool(_known_filenames, repository, settings)
            try:
                refuse_near_duplicate_name(filename, known)
            except FilenameConflictError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        content = await read_upload_within_limit(file, int(settings.max_upload_bytes))
        # job_store.create is a sqlite INSERT under JOB_STORE_BACKEND=sqlite; off the
        # event loop so it can't stall other in-flight requests.
        return await run_in_threadpool(
            enqueue_ingest,
            job_store=job_store,
            executor=executor,
            backlog=backlog,
            ingest_service=ingest_service,
            filename=filename,
            content=content,
            raw_store=raw_store,
        )

    # Re-chunks and re-embeds a document from its stored original (RAW_DOCUMENT_DIR),
    # so a chunking/embedding change applies without the user re-uploading. Same job
    # flow as an upload: 202 + a job id to poll.
    @app.post(
        "/documents/reindex",
        response_model=DocumentReindexAllResponse,
        status_code=202,
        dependencies=[Depends(require_full_key)],
    )
    def reindex_stale_documents(
        repository: VectorRepositoryDep,
        ingest_service: IngestServiceDep,
        settings: SettingsDep,
        job_store: IngestJobStoreDep,
        executor: IngestExecutorDep,
        backlog: IngestBacklogDep,
        raw_store: RawDocumentStoreDep,
        token_counter: TokenCounterDep,
    ) -> DocumentReindexAllResponse:
        store = require_raw_store(raw_store)
        is_stale = _staleness(settings, token_counter)
        # 409 now, not one failed job per document, when the index can't take them.
        repository.ensure_writable(settings)
        metadata = repository.filename_metadata(settings)
        jobs: list[DocumentJobAcceptedResponse] = []
        deferred: list[str] = []
        for name, meta in sorted(metadata.items()):
            if not is_stale(meta):
                continue
            size = store.size(name)
            if size is None:
                continue
            # Checked before reading: once the backlog is full, reading every remaining
            # original into memory only to defer it cost the whole corpus in RAM.
            if not backlog.has_room(size):
                deferred.append(name)
                continue
            content = store.read(name)
            if content is None:
                continue
            try:
                jobs.append(
                    enqueue_ingest(
                        job_store=job_store,
                        executor=executor,
                        backlog=backlog,
                        ingest_service=ingest_service,
                        filename=name,
                        content=content,
                        raw_store=None,
                        uploaded_at=meta.uploaded_at,
                        precondition=original_still_stored(store, name),
                    )
                )
            except HTTPException as exc:
                if exc.status_code != 503:
                    raise
                deferred.append(name)
        return DocumentReindexAllResponse(jobs=jobs, deferred=deferred)

    @app.post(
        "/documents/{filename}/reindex",
        response_model=DocumentJobAcceptedResponse,
        status_code=202,
        dependencies=[Depends(require_full_key)],
    )
    def reindex_document(
        filename: str,
        repository: VectorRepositoryDep,
        ingest_service: IngestServiceDep,
        settings: SettingsDep,
        job_store: IngestJobStoreDep,
        executor: IngestExecutorDep,
        backlog: IngestBacklogDep,
        raw_store: RawDocumentStoreDep,
    ) -> DocumentJobAcceptedResponse:
        store = require_raw_store(raw_store)
        name = stored_filename(filename)
        repository.ensure_writable(settings)
        content = store.read(name)
        if content is None:
            raise DocumentNotFoundError(f"No stored original for '{name}' to re-index.")
        return enqueue_ingest(
            job_store=job_store,
            executor=executor,
            backlog=backlog,
            ingest_service=ingest_service,
            filename=name,
            content=content,
            # The original is already stored, byte-identical. Passing the store would
            # re-save it, and a failed save deletes it — losing the only copy.
            raw_store=None,
            uploaded_at=repository.uploaded_at_for_filename(settings, name),
            precondition=original_still_stored(store, name),
        )

    @app.delete(
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
        # Sequenced like an ingest, and admitted before waiting for the lock, so an
        # upload already queued for this name is refused when it reaches the lock
        # instead of indexing the document straight back after the delete.
        sequence = ingest_sequencer.admit(filename)
        try:
            with filename_write_lock(filename):
                # The lock isn't fair: an upload admitted after this delete can reach
                # it first and be indexed. Deleting then would remove the newer
                # document the user just uploaded, so the delete stands down.
                superseded = ingest_sequencer.superseded_by(filename, sequence)
                if superseded == "upload":
                    raise HTTPException(
                        status_code=409,
                        detail=f"A newer upload of '{filename}' was indexed after this "
                        "delete was requested; it was kept.",
                    )
                if superseded == "delete":
                    raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
                point_ids = repository.point_ids_for_filename(settings, filename)
                if point_ids:
                    repository.delete_by_ids(settings, point_ids)
                    # Right away, not after the original below: if removing that
                    # raises, cached answers must still stop citing the deleted points.
                    bump_corpus_generation()
                    ingest_sequencer.mark_applied(filename, sequence, deleted=True)
                # Also when no points remain: an original orphaned by an earlier failed
                # ingest was otherwise undeletable, since the 404 came first.
                removed_original = _delete_original(
                    raw_store, filename, points_gone=bool(point_ids)
                )
                if not point_ids and not removed_original:
                    raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
                ingest_sequencer.mark_applied(filename, sequence, deleted=True)
        finally:
            ingest_sequencer.release(filename)
        return DocumentDeleteResponse(filename=filename, points_deleted=len(point_ids))

    # Payload-only: no re-embedding, so it is instant even for a large document.
    @app.patch(
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
        with filename_write_lock(filename):
            # A count, not a scroll of every point id just to see whether there is one.
            if repository.chunk_count_for_filename(settings, filename) == 0:
                raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
            repository.set_tags(settings, filename, tags)
        bump_corpus_generation()
        return DocumentTagsResponse(filename=filename, tags=list(tags))

    # Guarded like its sibling document routes: the response carries the filename (and
    # on failure the error text), and the frontend polls this URL once a second during
    # every upload, so the job id lands in browser history and any intermediate proxy
    # log — places a party who never held the key can read it from.
    @app.get(
        "/documents/jobs/{job_id}",
        response_model=DocumentJobStatusResponse,
        dependencies=[Depends(require_api_key)],
    )
    def get_document_job(job_id: str, job_store: IngestJobStoreDep) -> DocumentJobStatusResponse:
        return _job_status(job_store, job_id)

    # Stops a queued job outright, or a running one at its next checkpoint — but only
    # before it writes anything: point ids are deterministic, so once a batch is
    # written the previous version is partly overwritten and can't be restored (409;
    # delete the document afterwards instead). Changes the corpus, so full key only.
    @app.delete(
        "/documents/jobs/{job_id}",
        response_model=DocumentJobStatusResponse,
        status_code=202,
        dependencies=[Depends(require_full_key)],
    )
    def cancel_document_job(job_id: str, job_store: IngestJobStoreDep) -> DocumentJobStatusResponse:
        outcome = cancel_ingest_job(job_id)
        if outcome == "not_running":
            # Unknown is a 404 (raised by _job_status); a finished job can't be stopped.
            status = _job_status(job_store, job_id)
            raise HTTPException(
                status_code=409, detail=f"Job '{job_id}' has already {status.state}."
            )
        if outcome == "too_late":
            raise HTTPException(
                status_code=409,
                detail=(
                    "This job is already writing to the index and can't be cancelled "
                    "cleanly. Delete the document once it finishes instead."
                ),
            )
        return _job_status(job_store, job_id)

    # Guarded alongside GET /documents: this returns every chunk's raw text, so leaving
    # it open would let an unauthenticated caller list the corpus and then dump it in
    # full — the key would protect Q&A while the documents themselves stayed readable.
    @app.get(
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
        start: Annotated[int | None, Query(ge=1)] = None,
        end: Annotated[int | None, Query(ge=1)] = None,
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
                    or repository.chunks_for_filename(settings, filename)[
                        :CONTENT_UNPAGED_MAX_CHUNKS
                    ]
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
    @app.get("/documents/{filename}/original", dependencies=[Depends(require_api_key)])
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

        return StreamingResponse(chunks(), media_type=media_type, headers=headers)

    # Everything beyond the vectors, as one zip: manifest, stored originals, feedback.
    # See api/export.py. Operator data, so the full key only.
    @app.get("/export", dependencies=[Depends(require_full_key)])
    def export_corpus(
        repository: VectorRepositoryDep,
        settings: SettingsDep,
        raw_store: RawDocumentStoreDep,
        feedback_store: FeedbackStoreDep,
    ) -> StreamingResponse:
        archive = build_export(settings, repository, raw_store, feedback_store)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        return StreamingResponse(
            iter_file(archive),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="docrag-export-{stamp}.zip"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    # The other half of /export: re-ingests each stored original with its tags and
    # upload time, and adds back the feedback (api/restore.py). Documents already
    # indexed here are skipped, so importing the same backup twice is harmless and a
    # deferred one can be finished by importing again. Full key only.
    @app.post(
        "/import",
        response_model=ImportResponse,
        status_code=202,
        dependencies=[Depends(require_full_key)],
    )
    def import_backup(
        file: Annotated[UploadFile, File()],
        repository: VectorRepositoryDep,
        ingest_service: IngestServiceDep,
        settings: SettingsDep,
        job_store: IngestJobStoreDep,
        executor: IngestExecutorDep,
        backlog: IngestBacklogDep,
        raw_store: RawDocumentStoreDep,
        feedback_store: FeedbackStoreDep,
    ) -> ImportResponse:
        def enqueue(
            filename: str, content: bytes, uploaded_at: float | None, tags: tuple[str, ...]
        ) -> DocumentJobAcceptedResponse:
            if raw_store is not None:
                # A DocumentError: listed under `rejected`, the rest still restored.
                refuse_near_duplicate_name(filename, _known_filenames(repository, settings))
            return enqueue_ingest(
                job_store=job_store,
                executor=executor,
                backlog=backlog,
                ingest_service=ingest_service,
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
                has_room=backlog.has_room,
                enqueue=enqueue,
                feedback_store=feedback_store,
            )
        except BackupImportError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ImportResponse(
            jobs=outcome.jobs,
            deferred=outcome.deferred,
            skipped=outcome.skipped,
            missing_originals=outcome.missing_originals,
            rejected=outcome.rejected,
            feedback_imported=outcome.feedback_imported,
            feedback_skipped=outcome.feedback_skipped,
        )


def _staleness(
    settings: AppSettings, token_counter: TokenCounter
) -> Callable[[DocumentMetadata], bool]:
    """The one staleness rule the listing and bulk re-index share."""

    fingerprint = chunking_fingerprint(settings)
    counter_mode = str(getattr(token_counter, "mode", "unknown"))
    return lambda meta: meta.is_stale(fingerprint, counter_mode)


def _known_filenames(repository: VectorRepository, settings: AppSettings) -> list[str]:
    """Every indexed filename plus those with an upload or delete in flight."""

    return [*repository.filename_metadata(settings), *ingest_sequencer.in_flight()]


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
