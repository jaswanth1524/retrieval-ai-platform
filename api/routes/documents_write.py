"""Document write routes: upload, preview, add from URL, refresh, re-index, version restore,
delete, tags and refresh settings."""

from __future__ import annotations

import dataclasses
import functools
import io
import logging
import lzma
import tempfile
import threading
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path, PurePosixPath
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
    free_or_from,
    held_by_another_source,
    reindex_from_original,
    reindex_stale,
    require_raw_store,
    source_of,
)
from api.documents import (
    SUPPORTED_EXTENSIONS,
    DocumentError,
    DocumentNotFoundError,
    DocumentSource,
    chunk_sections,
    normalize_document_name,
    normalize_tags,
    validate_document_name,
    validate_upload_filename,
)
from api.imports import (
    BackgroundImport,
    remember,
    spool_directory,
    spool_upload,
    wait_for_room,
)
from api.imports import start as start_import
from api.ingest_jobs import IngestQueue, backlog_full_error
from api.ingestion import filename_write_lock, measured_embed_rate
from api.original_details import record_details
from api.parse_isolation import parse_with_limit
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.routes.deps import (
    IngestQueueDep,
    RawDocumentStoreDep,
    SettingsDep,
    TokenCounterDep,
    VectorRepositoryDep,
)
from api.routes.documents_common import (
    _fetch_for_ingest,
    _url_fetch_slots,
)
from api.schemas import (
    REQUEST_FILENAME_MAX_CHARS,
    ArchiveSkipResponse,
    ArchiveUploadResponse,
    ArchiveUrlIngestRequest,
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
    SitemapIngestRequest,
    SitemapIngestResponse,
    SitemapPageResponse,
    UrlIngestRequest,
)
from api.settings import AppSettings
from api.sitemap import KnownPage, PageImport, read_sitemap, start_page_import
from api.storage import require_free_space
from api.upload import read_upload_within_limit
from api.url_fetch import (
    FetchedDocument,
    fetch_document,
    loggable_url,
)
from api.workspaces import document_key, workspace_of
from api.zip_safety import too_many_entries

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
    name: Annotated[
        str | None,
        Form(
            max_length=REQUEST_FILENAME_MAX_CHARS,
            description="The name to store the document under, when it isn't the "
            "file's own: it may carry folders (reports/2024/q1.pdf). Without it the "
            "file's base name is used, as always.",
        ),
    ] = None,
) -> DocumentJobAcceptedResponse:
    # Rejected here, not in the background job: an unusable name or type used to
    # be accepted with a 202 and only fail once the client polled the job.
    filename = (
        validate_document_name(name) if name else validate_upload_filename(file.filename or "")
    )
    given_tags = normalize_tags(tags) if tags else None
    # 409 now, as /reindex and /import do, rather than a 202 for a job that can
    # only fail on a collection built for another embedding setup.
    await run_in_threadpool(repository.ensure_writable, settings)
    require_free_space(settings, settings.raw_document_dir)
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


def _names_and_sources(
    repository: VectorRepository, settings: AppSettings
) -> tuple[dict[str, str | None], dict[str, KnownPage]]:
    """Every name in use, with the URL its document came from (None for an upload and
    for a name with an upload in flight); and each fetched document by its URL."""

    names: dict[str, str | None] = dict.fromkeys(indexing_now(settings))
    pages: dict[str, KnownPage] = {}
    for name, meta in repository.filename_metadata(settings).items():
        names[name] = meta.source_url
        if meta.source_url:
            pages[meta.source_url] = KnownPage(
                meta.uploaded_at, meta.source_etag, meta.source_last_modified
            )
    return names, pages


def page_state(before: KnownPage | None, listed: float | None) -> str:
    """What a sitemap import would do with a page (its dry run): ``new``, ``unchanged``
    (skipped, as ``run_page_import`` skips it) or ``check`` (fetched conditionally)."""

    if before is None:
        return "new"
    if before.uploaded_at is not None and listed is not None and listed <= before.uploaded_at:
        return "unchanged"
    return "check"


def checked_folder(folder: str | None) -> str | None:
    """A ``folder`` to keep fetched documents under, checked as a document name's
    folders are (400 otherwise); None for none."""

    if folder is None or not folder.strip("/ "):
        return None
    cleaned = folder.strip().strip("/")
    try:
        normalize_document_name(f"{cleaned}/x.txt")
    except DocumentError as exc:
        raise HTTPException(status_code=400, detail=f"Not a usable folder: {exc}") from exc
    return cleaned


def in_folder(fetched: FetchedDocument, folder: str | None) -> FetchedDocument:
    """``fetched`` named under ``folder`` (unchanged without one)."""

    if folder is None:
        return fetched
    return dataclasses.replace(
        fetched, filename=normalize_document_name(f"{folder}/{fetched.filename}")
    )


# A site's pages from its sitemap (or sitemap index, RSS or Atom feed): read now, its
# pages fetched and queued one by one in the background (api/sitemap.py). The full key
# and URL_INGEST_ENABLED, as /documents/from-url.
@router.post(
    "/documents/from-sitemap",
    response_model=SitemapIngestResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={
        400: {"description": "Not a sitemap or feed, or it could not be fetched."},
        404: {"description": "URL ingest is not enabled (URL_INGEST_ENABLED)."},
        429: {"description": "Too many URL fetches are running."},
    },
)
async def add_from_sitemap(
    request: SitemapIngestRequest,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
) -> SitemapIngestResponse:
    if not settings.url_ingest_enabled:
        raise HTTPException(status_code=404, detail="URL ingest is not enabled.")
    folder = checked_folder(request.folder)
    given_tags = normalize_tags(request.tags) if request.tags else None
    await run_in_threadpool(repository.ensure_writable, settings)
    require_free_space(settings, settings.raw_document_dir)
    limit = min(
        request.max_pages or int(settings.sitemap_max_pages), int(settings.sitemap_max_pages)
    )

    def fetch_listing(url: str) -> FetchedDocument:
        # A name only so the fetch takes any content type (XML isn't one DocRAG indexes);
        # the listing is read here, never indexed.
        return fetch_document(url, settings, filename="sitemap.txt")

    if not _url_fetch_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Too many URL fetches are running. Try again in a moment.",
            headers={"Retry-After": "5"},
        )
    try:
        found = await run_in_threadpool(
            read_sitemap, request.url, fetch_listing, same_host=request.same_host, limit=limit
        )
    finally:
        _url_fetch_slots.release()
    known, indexed = await run_in_threadpool(_names_and_sources, repository, settings)
    if request.dry_run:
        return SitemapIngestResponse(
            urls=found.urls,
            left_out=found.left_out,
            child_sitemaps=len(found.child_sitemaps),
            pages=[
                SitemapPageResponse(
                    url=url, state=page_state(indexed.get(url), found.lastmods.get(url))
                )
                for url in found.urls
            ],
        )

    def fetch_page(
        url: str, has_room: Callable[[int], bool], etag: str | None, modified: str | None
    ) -> FetchedDocument:
        return in_folder(
            fetch_document(url, settings, has_room=has_room, etag=etag, last_modified=modified),
            folder,
        )

    status = start_page_import(
        found.urls,
        PageImport(
            queue=queue,
            raw_store=raw_store,
            fetch=fetch_page,
            tags=given_tags,
            refresh_minutes=request.refresh_minutes,
            known=known,
            indexed=indexed,
            lastmods=found.lastmods,
            still_ours=functools.partial(free_or_from, repository, settings),
        ),
        url=request.url,
    )
    logger.info(
        "Adding %d page(s) from the sitemap %s.", len(found.urls), loggable_url(request.url)
    )
    return SitemapIngestResponse(
        urls=found.urls,
        left_out=found.left_out,
        child_sitemaps=len(found.child_sitemaps),
        import_id=status.id,
    )


# A zip of documents, each indexed as its own (as if uploaded one by one, with the same
# tags): the folder a team keeps, in one request. Members are named by their base name;
# a name met twice, an unsupported type, a hidden file or one over MAX_UPLOAD_BYTES is
# skipped and said so. Nothing in the zip is ever extracted to disk.
@router.post(
    "/documents/archive",
    response_model=ArchiveUploadResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={400: {"description": "Not a zip, or one past MAX_ARCHIVE_MEMBERS."}},
)
def upload_archive(
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
    file: Annotated[UploadFile, File()],
    tags: Annotated[list[str] | None, Form(description="Tags for every document.")] = None,
    background: Annotated[
        bool,
        Form(
            description="Import in the background: every document is queued as room "
            "frees (none deferred); follow it with GET /documents/imports/{import_id}."
        ),
    ] = False,
    keep_paths: Annotated[
        bool,
        Form(
            description="Name each document by its path in the zip (reports/2024/q1.pdf) "
            "instead of its base name, so same-named files in different folders all "
            "come in."
        ),
    ] = False,
) -> ArchiveUploadResponse:
    given_tags = normalize_tags(tags) if tags else None
    repository.ensure_writable(settings)
    require_free_space(settings, settings.raw_document_dir)
    # Before the zip's directory is read into memory: a few hundred documents, with
    # room for the entries skipped silently (folders, __MACOSX, dotfiles).
    if too_many_entries(file.file, int(settings.max_archive_members) * _ENTRIES_PER_DOCUMENT):
        raise HTTPException(
            status_code=400,
            detail=f"The zip lists far more entries than {settings.max_archive_members} "
            "documents could need (MAX_ARCHIVE_MEMBERS); split it.",
        )
    if background:
        return _archive_in_background(
            repository, settings, queue, raw_store, file, given_tags, keep_paths
        )
    try:
        archive = zipfile.ZipFile(file.file)
    except (zipfile.BadZipFile, OSError) as exc:
        raise HTTPException(status_code=400, detail="That file isn't a zip archive.") from exc
    with archive:
        members = _archive_members(archive, settings)
        jobs, skipped, deferred = _queue_archive_members(
            archive,
            members,
            repository,
            settings,
            queue,
            raw_store,
            given_tags,
            keep_paths=keep_paths,
        )
    _audit_archive(
        settings, file.filename or "", given_tags, len(jobs), len(skipped), len(deferred)
    )
    return ArchiveUploadResponse(jobs=jobs, skipped=skipped, deferred=deferred)


def _audit_archive(
    settings: AppSettings,
    name: str,
    tags: tuple[str, ...] | None,
    queued: int,
    skipped: int,
    deferred: int,
    **extra: object,
) -> None:
    audit(
        "document.upload",
        workspace=workspace_of(settings),
        source="archive",
        archive=name,
        queued=queued,
        skipped=skipped,
        deferred=deferred,
        **({"tags": list(tags)} if tags else {}),
        **extra,
    )


# Zip entries allowed per document of MAX_ARCHIVE_MEMBERS (a Finder zip has two per
# file, plus folders).
_ENTRIES_PER_DOCUMENT = 20


def _archive_members(archive: zipfile.ZipFile, settings: AppSettings) -> list[zipfile.ZipInfo]:
    """The zip's file members; 400 past MAX_ARCHIVE_MEMBERS."""

    members = [info for info in archive.infolist() if not info.is_dir()]
    # Counted without what is skipped silently (a Finder zip carries a __MACOSX entry
    # for every file): the cap is on documents.
    counted = [info for info in members if not _silently_skipped(info)]
    if len(counted) > int(settings.max_archive_members):
        raise HTTPException(
            status_code=400,
            detail=f"The zip has more than {settings.max_archive_members} files "
            "(MAX_ARCHIVE_MEMBERS); split it.",
        )
    return members


def _queue_archive_members(
    archive: zipfile.ZipFile,
    members: list[zipfile.ZipInfo],
    repository: VectorRepository,
    settings: AppSettings,
    queue: IngestQueue,
    raw_store: RawDocumentStore | None,
    given_tags: tuple[str, ...] | None,
    status: BackgroundImport | None = None,
    *,
    keep_paths: bool = False,
) -> tuple[list[DocumentJobAcceptedResponse], list[ArchiveSkipResponse], list[str]]:
    """Queue ``members``: (jobs, skipped, deferred). With ``status`` (a background
    import) a full queue is waited out instead of deferring the rest, the import's
    counts follow along, and a cancel stops it between documents."""

    limit = int(settings.max_upload_bytes)
    budget = int(settings.max_import_bytes)
    jobs: list[DocumentJobAcceptedResponse] = []
    skipped: list[ArchiveSkipResponse] = []
    deferred: list[str] = []

    def skip(name: str, reason: str) -> None:
        skipped.append(ArchiveSkipResponse(name=name, reason=reason))
        if status is not None:
            status.skipped += 1
            status.note(f"{name}: {reason}")

    known = set(known_filenames(repository, settings)) if raw_store is not None else set()
    seen: set[str] = set()
    for info in members:
        if status is not None and status.cancel.is_set():
            break
        member_path = info.filename.replace("\\", "/").lstrip("/")
        # Its base name, or with keep_paths its whole path in the zip.
        base = member_path if keep_paths else PurePosixPath(member_path).name
        reason = _archive_skip_reason(info, base, seen, limit)
        if reason is None:
            try:
                name = (
                    validate_document_name(base) if keep_paths else validate_upload_filename(base)
                )
            except DocumentError as exc:
                reason = str(exc)
            else:
                try:
                    if raw_store is not None:
                        refuse_near_duplicate_name(name, known)
                except FilenameConflictError as exc:
                    reason = str(exc)
        if reason is not None:
            if reason:  # hidden files and folders' metadata are skipped silently
                skip(info.filename, reason)
            continue
        seen.add(base.casefold())
        if status is not None:
            if not wait_for_room(queue.backlog.has_spare_room, info.file_size, status):
                break
        elif deferred or not queue.backlog.has_room(info.file_size):
            deferred.append(name)
            continue
        try:
            with archive.open(info) as handle:
                content = handle.read(limit + 1)
        except (
            zipfile.BadZipFile,
            zlib.error,
            lzma.LZMAError,
            EOFError,
            OSError,  # a damaged bzip2 member
            RuntimeError,
            NotImplementedError,
        ):
            skip(info.filename, "It is damaged.")
            continue
        # The header's size isn't trusted: what it expands to is counted — only for
        # a member that is kept (one past the limit is skipped, not budgeted).
        if len(content) <= limit:
            budget -= len(content)
        if len(content) > limit or budget < 0:
            skip(info.filename, "Too large (MAX_UPLOAD_BYTES, or the zip past MAX_IMPORT_BYTES).")
            continue
        while True:
            try:
                accepted = queue.enqueue(
                    filename=name,
                    content=content,
                    raw_store=raw_store,
                    tags=given_tags,
                    dedupe=True,
                )
            except HTTPException as exc:
                if exc.status_code == 503 and status is not None:
                    # Room went to an upload meanwhile: wait for it again.
                    if wait_for_room(queue.backlog.has_spare_room, len(content), status):
                        continue
                    accepted = None
                elif exc.status_code == 503:
                    deferred.append(name)
                    accepted = None
                else:
                    skip(info.filename, str(exc.detail))
                    accepted = None
            break
        if accepted is None:
            continue
        known.add(name)
        jobs.append(accepted)
        if status is not None:
            status.queued += 1
    return jobs, skipped, deferred


def _archive_in_background(
    repository: VectorRepository,
    settings: AppSettings,
    queue: IngestQueue,
    raw_store: RawDocumentStore | None,
    file: UploadFile,
    given_tags: tuple[str, ...] | None,
    keep_paths: bool = False,
) -> ArchiveUploadResponse:
    """Spool the zip, check it, and queue its documents on the import thread."""

    directory = spool_directory(settings.import_spool_dir)
    require_free_space(settings, directory or tempfile.gettempdir())
    spool = spool_upload(file.file, "docrag-import-", directory)
    name = file.filename or "archive.zip"
    record = None
    if directory is not None:
        try:
            record = remember(
                directory,
                {
                    "kind": "archive",
                    "name": name,
                    "workspace": workspace_of(settings),
                    "tags": list(given_tags) if given_tags is not None else None,
                    "keep_paths": keep_paths,
                },
                spool,
            )
        except BaseException:
            # A record that can't be written (a full disk) left the spooled zip behind,
            # with nothing to resume or remove it, as /import already handled.
            spool.unlink(missing_ok=True)
            raise
    started = start_archive_import(
        spool,
        name,
        repository,
        settings,
        queue,
        raw_store,
        given_tags,
        record=record,
        keep_paths=keep_paths,
    )
    return ArchiveUploadResponse(jobs=[], skipped=[], deferred=[], import_id=started.id)


# A zip of documents at a URL: fetched within the URL-ingest guards (public addresses,
# MAX_IMPORT_BYTES, the deadline), then imported in the background as an uploaded one
# sent with background=true is. Follow it with GET /documents/imports/{import_id}.
_archive_fetches = threading.BoundedSemaphore(1)


@router.post(
    "/documents/archive-from-url",
    response_model=ArchiveUploadResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={
        400: {"description": "The URL couldn't be fetched, or isn't a zip."},
        404: {"description": "URL ingest is off (URL_INGEST_ENABLED)."},
    },
)
async def upload_archive_from_url(
    request: ArchiveUrlIngestRequest,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
) -> ArchiveUploadResponse:
    if not settings.url_ingest_enabled:
        raise HTTPException(status_code=404, detail="URL ingest is not enabled.")
    given_tags = normalize_tags(request.tags) if request.tags else None
    await run_in_threadpool(repository.ensure_writable, settings)
    directory = spool_directory(settings.import_spool_dir)
    require_free_space(settings, directory or tempfile.gettempdir())
    # One archive download at a time: each is held whole in memory (up to
    # MAX_IMPORT_BYTES, twice while it is copied out), and four at once came near the
    # container's memory limit.
    if not _archive_fetches.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Another archive is being downloaded; try again when it has finished.",
            headers={"Retry-After": "30"},
        )
    try:
        fetched = await _fetch_for_ingest(
            queue,
            functools.partial(
                fetch_document, request.url, settings, has_room=lambda size: True, archive=True
            ),
            # Spooled and imported in the background, waiting for room as it goes: a
            # full indexing queue is no reason to refuse the download.
            needs_room=False,
        )
    finally:
        _archive_fetches.release()
    if too_many_entries(
        io.BytesIO(fetched.content), int(settings.max_archive_members) * _ENTRIES_PER_DOCUMENT
    ):
        raise HTTPException(
            status_code=400,
            detail=f"The zip lists far more entries than {settings.max_archive_members} "
            "documents could need (MAX_ARCHIVE_MEMBERS); split it.",
        )
    spool = await run_in_threadpool(
        spool_upload, io.BytesIO(fetched.content), "docrag-import-", directory
    )
    record = None
    if directory is not None:
        record = remember(
            directory,
            {
                "kind": "archive",
                "name": fetched.filename,
                "workspace": workspace_of(settings),
                "tags": list(given_tags) if given_tags is not None else None,
                "keep_paths": request.keep_paths,
            },
            spool,
        )
    started = await run_in_threadpool(
        functools.partial(
            start_archive_import,
            spool,
            fetched.filename,
            repository,
            settings,
            queue,
            raw_store,
            given_tags,
            record=record,
            keep_paths=request.keep_paths,
        )
    )
    return ArchiveUploadResponse(jobs=[], skipped=[], deferred=[], import_id=started.id)


def start_archive_import(
    spool: Path,
    name: str,
    repository: VectorRepository,
    settings: AppSettings,
    queue: IngestQueue,
    raw_store: RawDocumentStore | None,
    given_tags: tuple[str, ...] | None,
    *,
    record: Path | None = None,
    resumed: bool = False,
    keep_paths: bool = False,
) -> BackgroundImport:
    """Queue a spooled zip's documents on the import thread (a request's, or one a
    restart cut off: ``record`` is what lets the next start run it again). A file that
    isn't a zip is a 400, with the spool removed."""

    def cleanup() -> None:
        spool.unlink(missing_ok=True)
        if record is not None:
            record.unlink(missing_ok=True)

    try:
        with zipfile.ZipFile(spool) as archive:
            _archive_members(archive, settings)
    except (zipfile.BadZipFile, OSError) as exc:
        cleanup()
        raise HTTPException(status_code=400, detail="That file isn't a zip archive.") from exc
    except HTTPException:
        cleanup()
        raise

    def run(status: BackgroundImport) -> None:
        with zipfile.ZipFile(spool) as archive:
            members = _archive_members(archive, settings)
            jobs, skipped, _ = _queue_archive_members(
                archive,
                members,
                repository,
                settings,
                queue,
                raw_store,
                given_tags,
                status,
                keep_paths=keep_paths,
            )
        _audit_archive(settings, name, given_tags, len(jobs), len(skipped), 0, import_id=status.id)

    return start_import(
        "archive",
        name,
        workspace_of(settings),
        str(settings.qdrant_collection),
        run,
        cleanup=cleanup,
        resumable=record is not None,
        resumed=resumed,
    )


def _silently_skipped(info: zipfile.ZipInfo) -> bool:
    """Hidden files and macOS folder metadata: not documents, skipped without a word."""

    parts = PurePosixPath(info.filename.replace("\\", "/")).parts
    base = parts[-1] if parts else ""
    return not base or base.startswith(".") or "__MACOSX" in parts


def _archive_skip_reason(
    info: zipfile.ZipInfo, base: str, seen: set[str], limit: int
) -> str | None:
    """Why a zip member isn't indexed ("" for one skipped without a word: hidden files,
    macOS folder metadata), or None to index it."""

    if _silently_skipped(info):
        return ""
    if PurePosixPath(base).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return "Not a supported document type."
    if base.casefold() in seen:
        return "Another file in the zip has the same name."
    if info.file_size > limit:
        return "Larger than MAX_UPLOAD_BYTES."
    return None


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
    rate = measured_embed_rate()
    limit = int(settings.max_chunks_per_document)
    return DocumentPreviewResponse(
        filename=filename,
        section_count=len(sections),
        chunk_count=len(chunks),
        estimated_seconds=round(len(chunks) / rate, 1) if rate else None,
        over_chunk_limit=bool(limit) and len(chunks) > limit,
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
    folder = checked_folder(request.folder)
    given_tags = normalize_tags(request.tags) if request.tags else None
    await run_in_threadpool(repository.ensure_writable, settings)
    require_free_space(settings, settings.raw_document_dir)
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
    fetched = in_folder(fetched, folder)
    url = request.url.strip()
    # Naming the document (`filename`) says which one to replace, as it always has;
    # a name taken from the URL alone doesn't.
    replacing = request.replace or bool((request.filename or "").strip())
    if not replacing and await run_in_threadpool(
        held_by_another_source, repository, settings, fetched.filename, url
    ):
        raise HTTPException(
            status_code=409,
            detail=f"A document named '{fetched.filename}' is already indexed from "
            "another source; it was not replaced. Give this one a name (`filename` in "
            "the API, --name on the CLI), or replace it (`replace`, --replace).",
        )
    if raw_store is not None or fetched.remote_named:
        known = await run_in_threadpool(known_filenames, repository, settings)
        # A name the remote server picked (Content-Disposition, a redirect) must not
        # replace a document already here. One indexed from another source was refused
        # above; what is left is a name being indexed right now, whose source isn't
        # known yet. The same URL added again keeps its name and is replaced as an
        # upload would be, and `replace` says to go ahead either way.
        indexed = await run_in_threadpool(repository.content_identity, settings, fetched.filename)
        if (
            fetched.remote_named
            and not request.replace
            and indexed is None
            and fetched.filename in known
        ):
            raise HTTPException(
                status_code=409,
                detail=f"The server at that URL names the document "
                f"'{fetched.filename}', and a document with that name is being "
                "indexed; it was not replaced. Wait for it, give this one a name "
                "(`filename` in the API, --name on the CLI), or replace it (`replace`, "
                "--replace).",
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
        precondition=(
            None if replacing else free_or_from(repository, settings, fetched.filename, url)
        ),
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
    "/documents/{filename:path}/versions/{version}/restore",
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
    "/documents/{filename:path}/refresh",
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
        no_original=len(outcome.no_original),
    )
    return DocumentReindexAllResponse(
        jobs=outcome.jobs,
        deferred=outcome.deferred,
        in_progress=outcome.in_progress,
        failed=outcome.failed,
        no_original=outcome.no_original,
    )


@router.post(
    "/documents/{filename:path}/reindex",
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
    return reindex_from_original(stored_filename(filename), repository, settings, queue, raw_store)


@router.delete(
    "/documents/{filename:path}",
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
    # A stored name may carry folders (reports/2024/q1.pdf: ``{filename:path}``);
    # stored_filename 404s any name normalize_document_name would change.
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
    "/documents/{filename:path}/tags",
    response_model=DocumentTagsResponse,
    dependencies=[Depends(require_full_key)],
)
def set_document_tags(
    filename: str,
    request: DocumentTagsRequest,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    raw_store: RawDocumentStoreDep,
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
        record_details(repository, settings, raw_store, filename)
    bump_corpus_generation([filename], collection=settings.qdrant_collection)
    audit("document.tags", workspace=workspace_of(settings), filename=filename, tags=list(tags))
    return DocumentTagsResponse(filename=filename, tags=list(tags))


# A fetched document's own refresh interval; payload-only, like tags.
@router.patch(
    "/documents/{filename:path}/refresh-settings",
    response_model=DocumentRefreshSettingsResponse,
    dependencies=[Depends(require_full_key)],
    responses={404: {"description": "No such document, or it wasn't fetched from a URL."}},
)
def set_document_refresh_settings(
    filename: str,
    request: DocumentRefreshSettingsRequest,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    raw_store: RawDocumentStoreDep,
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
        # Only the listing shows it: no answer depends on the refresh interval.
        repository.refresh_listed(settings, [filename])
        record_details(repository, settings, raw_store, filename)
    audit(
        "document.refresh_settings",
        workspace=workspace_of(settings),
        filename=filename,
        refresh_minutes=request.refresh_minutes,
    )
    return DocumentRefreshSettingsResponse(
        filename=filename, refresh_minutes=request.refresh_minutes
    )
