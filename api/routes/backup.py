"""Backup routes: export the corpus as a zip, and import one."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from api.audit import audit
from api.audit_log import AuditLog
from api.chunking import TokenCounter
from api.consistency import LostIndexError, consistency_report, repair
from api.dependencies import (
    get_audit_log,
    get_token_counter,
    require_full_key,
)
from api.doctor import diagnose
from api.document_names import (
    known_filenames,
    refuse_near_duplicate_name,
)
from api.documents import (
    DocumentSource,
    chunk_sections,
)
from api.export import build_export
from api.feedback import FeedbackStore
from api.imports import (
    BackgroundImport,
    adopt_upload,
    remember,
    spool_directory,
    spool_upload,
    wait_for_room,
)
from api.imports import start as start_import
from api.ingest_jobs import IngestQueue
from api.ingestion import IngestionError
from api.notify import TEST_EVENT, NotifyError, current_notifier
from api.parse_isolation import parse_with_limit
from api.raw_documents import RawDocumentStore
from api.rebuild import originals_to_rebuild, rebuild
from api.repository import VectorRepository
from api.restore import (
    MAX_MEMBERS,
    BackupCheck,
    BackupImportError,
    DrillReport,
    drill_backup,
    restore_backup,
    verify_backup,
)
from api.routes.deps import (
    FeedbackStoreDep,
    IngestQueueDep,
    RawDocumentStoreDep,
    SettingsDep,
    TokenCounterDep,
    VectorRepositoryDep,
)
from api.scheduled_backup import (
    BACKUP_NAME_RE,
    ScheduledBackup,
    backups_in,
    check_path,
    load_check,
    save_check,
)
from api.schemas import (
    AuditEntryResponse,
    AuditResponse,
    BackupCheckResponse,
    BackupDeleteResponse,
    BackupDrillResponse,
    BackupFileResponse,
    BackupListResponse,
    BackupRunResponse,
    ConsistencyRepairRequest,
    ConsistencyRepairResponse,
    ConsistencyResponse,
    DoctorFindingResponse,
    DoctorResponse,
    DocumentJobAcceptedResponse,
    DrillDocumentResponse,
    ImportResponse,
    NotifyTestResponse,
    RebuildResponse,
)
from api.settings import AppSettings
from api.storage import require_free_space
from api.streaming import ClosingStreamingResponse
from api.workspaces import workspace_of
from api.zip_safety import too_many_entries

logger = logging.getLogger("api.routes.documents")

router = APIRouter()


# Held while an export streams; a second one meanwhile is a 409.
_export_running = threading.Lock()


# Everything beyond the vectors, as one zip: manifest, stored originals, feedback.
# See api/export.py. Operator data, so the full key only.
@router.get("/export", dependencies=[Depends(require_full_key)])
def export_corpus(
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    raw_store: RawDocumentStoreDep,
    feedback_store: FeedbackStoreDep,
    include_versions: Annotated[
        bool,
        Query(
            description="Also include each document's kept earlier originals "
            "(RAW_DOCUMENT_VERSIONS), which an import puts back."
        ),
    ] = False,
) -> ClosingStreamingResponse:
    # One at a time: each export reads every stored original on its own thread,
    # and nothing else bounds how many a client could start.
    if not _export_running.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="An export is already running. Try again when it ends."
        )
    try:
        # Streamed as it is written: nothing is spooled to /tmp or held in memory.
        archive = build_export(
            settings, repository, raw_store, feedback_store, include_versions=include_versions
        )
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

    # A full read of the corpus: every original and the feedback leave the server.
    audit(
        "corpus.export",
        workspace=workspace_of(settings),
        originals_missing=archive.originals_missing,
    )
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
    background: Annotated[
        bool,
        Form(
            description="Restore in the background: every document is queued as room "
            "frees (none deferred); follow it with GET /documents/imports/{import_id}."
        ),
    ] = False,
) -> ImportResponse:
    # Before anything is copied or its directory read into memory.
    if too_many_entries(file.file, MAX_MEMBERS):
        raise HTTPException(
            status_code=400, detail=f"The archive has more than {MAX_MEMBERS} entries."
        )
    require_free_space(settings, settings.raw_document_dir)
    if background:
        directory = spool_directory(settings.import_spool_dir)
        require_free_space(settings, directory or tempfile.gettempdir())
        name = file.filename or "backup.zip"
        record = None
        # With IMPORT_SPOOL_DIR, copied onto the data volume (and remembered, to resume
        # after a restart). Without it, the file the web server already received is
        # kept as it is: copying it into the same temp directory needed room twice.
        spool: Path | IO[bytes]
        if directory is not None:
            spool = spool_upload(file.file, "docrag-restore-", directory)
            try:
                record = remember(
                    directory,
                    {"kind": "restore", "name": name, "workspace": workspace_of(settings)},
                    spool,
                )
            except BaseException:
                spool.unlink(missing_ok=True)
                raise
        else:
            spool = adopt_upload(file)

        def cleanup() -> None:
            if isinstance(spool, Path):
                spool.unlink(missing_ok=True)
            else:
                spool.close()
            if record is not None:
                record.unlink(missing_ok=True)

        return _restore_in_background(
            spool,
            name,
            repository,
            settings,
            queue,
            raw_store,
            feedback_store,
            cleanup=cleanup,
            resumable=record is not None,
        )
    return _restore(file.file, repository, settings, queue, raw_store, feedback_store)


@contextlib.contextmanager
def _opened(source: Path | IO[bytes]) -> Iterator[IO[bytes]]:
    """A spooled file opened, or an upload taken over read from its start (left open:
    the import's cleanup closes it, and so removes it)."""

    if isinstance(source, Path):
        with source.open("rb") as handle:
            yield handle
        return
    source.seek(0)
    yield source


def _restore_in_background(
    path: Path | IO[bytes],
    name: str,
    repository: VectorRepository,
    settings: AppSettings,
    queue: IngestQueue,
    raw_store: RawDocumentStore | None,
    feedback_store: FeedbackStore | None,
    *,
    cleanup: Callable[[], None] = lambda: None,
    source: str | None = None,
    resumable: bool = False,
    resumed: bool = False,
    only: frozenset[str] | None = None,
) -> ImportResponse:
    """Check the backup's manifest now (a bad one is a 400 here, not a failed import),
    then restore it on the import thread, waiting for room instead of deferring."""

    try:
        with zipfile.ZipFile(path) as archive:
            if "manifest.json" not in archive.namelist():
                raise BackupImportError("The archive has no manifest.json; it isn't a backup.")
    except (zipfile.BadZipFile, OSError, BackupImportError) as exc:
        cleanup()
        detail = (
            str(exc) if isinstance(exc, BackupImportError) else "That file isn't a zip archive."
        )
        raise HTTPException(status_code=400, detail=detail) from exc
    try:
        repository.ensure_writable(settings)
    except BaseException:
        cleanup()  # a 409 (index built for another embedding setup): nothing will run
        raise

    def run(status: BackgroundImport) -> None:
        with _opened(path) as handle:
            outcome = _restore(
                handle,
                repository,
                settings,
                queue,
                raw_store,
                feedback_store,
                source=source,
                status=status,
                only=only,
            )
        status.skipped += len(outcome.skipped) + len(outcome.missing_originals)
        # Cancelled: a document it was waiting to queue was left out, not refused.
        left_out = [] if status.cancel.is_set() else outcome.deferred
        status.failed += len(outcome.rejected) + len(left_out)
        for filename in outcome.rejected:
            # "b.txt (not in this backup)" says why already.
            status.note(
                filename
                if filename.endswith(")")
                else f"{filename}: refused by restore's rules (or damaged)"
            )
        for filename in outcome.missing_originals:
            status.note(f"{filename}: the backup holds no original of it")

    started = start_import(
        "restore",
        name,
        workspace_of(settings),
        str(settings.qdrant_collection),
        run,
        cleanup=cleanup,
        resumable=resumable,
        resumed=resumed,
    )
    return ImportResponse(
        jobs=[],
        deferred=[],
        skipped=[],
        missing_originals=[],
        rejected=[],
        feedback_imported=0,
        feedback_skipped=0,
        import_id=started.id,
    )


def _still_absent(repository: VectorRepository, settings: AppSettings, name: str) -> None:
    if repository.content_identity(settings, name) is not None:
        raise IngestionError(
            f"'{name}' was indexed while the restore ran; the newer document was kept."
        )


def _restore(
    upload: IO[bytes],
    repository: VectorRepository,
    settings: AppSettings,
    queue: IngestQueue,
    raw_store: RawDocumentStore | None,
    feedback_store: FeedbackStore | None,
    *,
    source: str | None = None,
    status: BackgroundImport | None = None,
    only: frozenset[str] | None = None,
    dry_run: bool = False,
) -> ImportResponse:
    """Restore one backup into ``settings``' workspace: an upload (``/import``) or one
    of BACKUP_DIR's (``source`` names it, for the audit). With ``status`` (a background
    import) a full queue is waited out instead of deferring, and its count follows.
    ``dry_run``: what it would restore, skip and reject, with nothing queued, stored or
    changed (feedback included)."""

    # Read once for the whole backup, then kept up to date with what it queues:
    # read per document, each job finishing mid-import moved the corpus generation
    # and the next document paid for a fresh listing.
    known: set[str] | None = None

    def enqueue(
        filename: str,
        content: bytes,
        uploaded_at: float | None,
        tags: tuple[str, ...],
        source: DocumentSource | None,
        watched_from: str | None,
    ) -> DocumentJobAcceptedResponse:
        nonlocal known
        if raw_store is not None:
            if known is None:
                known = set(known_filenames(repository, settings))
            # A DocumentError: listed under `rejected`, the rest still restored.
            refuse_near_duplicate_name(filename, known)
            known.add(filename)
        # After the name checks: a dry run rejects what the restore would.
        if dry_run:
            return DocumentJobAcceptedResponse(job_id="", filename=filename, state="queued")
        accepted = queue.enqueue(
            filename=filename,
            content=content,
            raw_store=raw_store,
            uploaded_at=uploaded_at,
            # A restore never overwrites: a document uploaded while it waited for room
            # (a background restore can wait hours) is newer than the backup's copy.
            precondition=lambda name=filename: _still_absent(repository, settings, name),  # type: ignore[misc]
            tags=tags,
            source=source,
            watched_from=watched_from,
            kind="import",
        )
        if status is not None:
            status.queued += 1
        return accepted

    def has_room(size: int) -> bool:
        if dry_run:
            return True
        if status is None:
            return queue.backlog.has_room(size)
        return wait_for_room(queue.backlog.has_spare_room, size, status)

    # The last step of recovering from an embedding change (export, delete all,
    # import): an emptied collection is rebuilt here; a non-empty one is a 409.
    if not dry_run:
        repository.ensure_writable(settings)
    try:
        # A dry run reads every original through, as a verify does: one at a time.
        with _one_verify() if dry_run else contextlib.nullcontext():
            outcome = restore_backup(
                upload,
                # Indexed, or being indexed: importing the same backup again while the
                # first import's jobs run must not queue each document a second time.
                existing_filenames=set(known_filenames(repository, settings)),
                max_original_bytes=int(settings.max_upload_bytes),
                has_room=has_room,
                enqueue=enqueue,
                feedback_store=None if dry_run else feedback_store,
                workspace=workspace_of(settings),
                keep_version=raw_store.add_version
                if raw_store is not None and not dry_run
                else None,
                # One of BACKUP_DIR's own (``source`` names it): no upload caps.
                # A backup this server wrote (BACKUP_DIR) is trusted past the upload caps; one
                # inside an uploaded migration bundle is an upload like any other.
                stored=source is not None and not source.startswith("bundle:"),
                stop=status.cancel.is_set if status is not None else None,
                only=only,
                # In a migration bundle: a whole server's worth of entries may be in it.
                max_members=(
                    int(settings.max_migrate_members)
                    if source is not None and source.startswith("bundle:")
                    else None
                ),
            )
    except BackupImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if dry_run:
        return ImportResponse(
            jobs=[],
            deferred=outcome.deferred,
            skipped=outcome.skipped,
            missing_originals=outcome.missing_originals,
            rejected=outcome.rejected,
            feedback_imported=0,
            feedback_skipped=0,
            would_restore=[job.filename for job in outcome.jobs],
            dry_run=True,
        )
    audit(
        "backup.import",
        workspace=workspace_of(settings),
        queued=len(outcome.jobs),
        deferred=len(outcome.deferred),
        skipped=len(outcome.skipped),
        rejected=len(outcome.rejected),
        feedback_imported=outcome.feedback_imported,
        # Which of BACKUP_DIR's backups it came from (an upload names none).
        **({"backup": source} if source else {}),
    )
    return ImportResponse(
        jobs=outcome.jobs,
        deferred=outcome.deferred,
        skipped=outcome.skipped,
        missing_originals=outcome.missing_originals,
        rejected=outcome.rejected,
        feedback_imported=outcome.feedback_imported,
        feedback_skipped=outcome.feedback_skipped,
        versions_restored=outcome.versions_restored,
    )


def _backup_dir(settings: AppSettings) -> Path:
    if not settings.backup_dir:
        raise HTTPException(status_code=404, detail="Scheduled backups are off (BACKUP_DIR).")
    return Path(settings.backup_dir) / (workspace_of(settings) or "default")


def _stored_backup(settings: AppSettings, name: str) -> Path:
    """One of this workspace's backups by its file name; 404 for anything else (no
    path in the name, only names the loop writes)."""

    directory = _backup_dir(settings)
    if not BACKUP_NAME_RE.fullmatch(name) or not (directory / name).is_file():
        raise HTTPException(status_code=404, detail=f"No backup named '{name}'.")
    return directory / name


def _check_response(check: BackupCheck) -> BackupCheckResponse:
    fields = dataclasses.asdict(check)
    del fields["drill_ok"]
    return BackupCheckResponse(ok=check.ok, **fields)


# One stored backup, downloaded (to keep it somewhere else) or deleted. Operator data:
# the full key (request_guard covers every /backups path).
@router.get(
    "/backups/{name}",
    response_class=FileResponse,
    dependencies=[Depends(require_full_key)],
    responses={
        200: {"content": {"application/zip": {}}},
        404: {"description": "No such backup, or BACKUP_DIR is unset."},
    },
)
def download_stored_backup(name: str, settings: SettingsDep) -> FileResponse:
    path = _stored_backup(settings, name)
    audit("backup.download", workspace=workspace_of(settings), backup=name)
    return FileResponse(path, media_type="application/zip", filename=name)


@router.delete(
    "/backups/{name}",
    response_model=BackupDeleteResponse,
    dependencies=[Depends(require_full_key)],
    responses={404: {"description": "No such backup, or BACKUP_DIR is unset."}},
)
def delete_stored_backup(
    name: str, request: Request, settings: SettingsDep
) -> BackupDeleteResponse:
    path = _stored_backup(settings, name)
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("Could not delete the backup %s.", path, exc_info=True)
        raise HTTPException(
            status_code=500, detail="The backup could not be deleted; see the server log."
        ) from exc
    check_path(path).unlink(missing_ok=True)
    loop: ScheduledBackup | None = getattr(request.app.state, "backup", None)
    if loop is not None:
        loop.checks.pop(str(path), None)
    audit("backup.delete", workspace=workspace_of(settings), backup=name)
    return BackupDeleteResponse(name=name)


# BACKUP_DIR's backups of this workspace (api/scheduled_backup.py), newest first, with
# whether each was read back cleanly when written. Operator data: the full key.
@router.get(
    "/backups",
    response_model=BackupListResponse,
    dependencies=[Depends(require_full_key)],
    responses={404: {"description": "Scheduled backups are off (BACKUP_DIR)."}},
)
def list_stored_backups(request: Request, settings: SettingsDep) -> BackupListResponse:
    directory = _backup_dir(settings)
    loop: ScheduledBackup | None = getattr(request.app.state, "backup", None)
    files: list[BackupFileResponse] = []
    for path in reversed(backups_in(directory)):
        try:
            stat = path.stat()
        except OSError:
            continue
        check = loop.check_for(path) if loop is not None else load_check(path)
        files.append(
            BackupFileResponse(
                name=path.name,
                size=stat.st_size,
                created_at=stat.st_mtime,
                verified=check.ok if check is not None else None,
                problems=list(check.problems) if check is not None else [],
            )
        )
    return BackupListResponse(
        backups=files,
        interval_hours=float(settings.backup_interval_hours),
        keep=int(settings.backup_keep),
    )


# A backup of this workspace now, whatever the newest one's age: written on a pass the
# background loop starts at once (202; GET /backups lists it once written).
@router.post(
    "/backups/run",
    response_model=BackupRunResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={404: {"description": "Scheduled backups are off (BACKUP_DIR)."}},
)
def run_backup_now(request: Request, settings: SettingsDep) -> BackupRunResponse:
    require_free_space(settings, _backup_dir(settings))
    loop: ScheduledBackup | None = getattr(request.app.state, "backup", None)
    if loop is None:
        raise HTTPException(status_code=404, detail="Scheduled backups are not running.")
    loop.request_now(settings)
    # Asked for, not yet written: the backup's own corpus.export is audited when it is.
    audit("corpus.export_requested", workspace=workspace_of(settings))
    return BackupRunResponse(workspace=workspace_of(settings) or "default")


# One verify at a time: each reads (and decompresses) a whole archive.
_verify_running = threading.Lock()


@contextlib.contextmanager
def _one_verify() -> Iterator[None]:
    if not _verify_running.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="A backup is already being checked; try again shortly."
        )
    try:
        yield
    finally:
        _verify_running.release()


# A backup read through without restoring anything: the manifest, every member's
# checksum, every listed original and version present. An upload (any backup, from
# anywhere), or one of BACKUP_DIR's below.
@router.post(
    "/backups/verify",
    response_model=BackupCheckResponse,
    dependencies=[Depends(require_full_key)],
)
def verify_uploaded_backup(
    file: Annotated[UploadFile, File()], settings: SettingsDep
) -> BackupCheckResponse:
    with _one_verify():
        return _check_response(verify_backup(file.file, int(settings.max_upload_bytes)))


@router.post(
    "/backups/{name}/verify",
    response_model=BackupCheckResponse,
    dependencies=[Depends(require_full_key)],
    responses={404: {"description": "No such backup, or BACKUP_DIR is unset."}},
)
def verify_stored_backup(name: str, request: Request, settings: SettingsDep) -> BackupCheckResponse:
    path = _stored_backup(settings, name)
    with _one_verify(), path.open("rb") as handle:
        check = verify_backup(handle, int(settings.max_upload_bytes), stored=True)
    # Listed with this result from now on, after a restart too.
    loop: ScheduledBackup | None = getattr(request.app.state, "backup", None)
    if loop is not None:
        loop.record_check(path, check)
    else:
        save_check(path, check)
    return _check_response(check)


# How long one drill request reads and parses originals before it stops and reports.
DRILL_BUDGET_SECONDS = 600.0


def drill_file(
    path: Path,
    settings: AppSettings,
    counter: TokenCounter,
    sample: int | None,
    *,
    originals_only: bool = False,
) -> DrillReport:
    """``path``'s originals (``sample`` of them; None = all) parsed and chunked with
    this server's parsers and settings, nothing indexed (POST /backups/{name}/drill,
    and BACKUP_DRILL_SAMPLE after each scheduled backup)."""

    def parse(filename: str, content: bytes) -> int:
        sections = parse_with_limit(filename, content, settings)
        return len(chunk_sections(sections, settings, counter))

    with path.open("rb") as handle:
        return drill_backup(
            handle,
            parse,
            sample=sample,
            max_original_bytes=int(settings.max_upload_bytes),
            budget_seconds=DRILL_BUDGET_SECONDS,
            originals_only=originals_only,
        )


# A restore rehearsed on one of this workspace's backups: a sample of its originals
# (``sample``; 0 = all) read, parsed and chunked with this server's parsers and
# settings — nothing indexed or stored. Verify proves the zip is whole; this proves its
# originals still make documents. One at a time (it shares verify's guard).
@router.post(
    "/backups/{name}/drill",
    response_model=BackupDrillResponse,
    dependencies=[Depends(require_full_key)],
    responses={404: {"description": "No such backup, or BACKUP_DIR is unset."}},
)
def drill_stored_backup(
    name: str,
    settings: SettingsDep,
    counter: Annotated[TokenCounter, Depends(get_token_counter)],
    sample: Annotated[int, Query(ge=0, le=10_000, description="Documents to read; 0 = all.")] = 20,
) -> BackupDrillResponse:
    path = _stored_backup(settings, name)
    with _one_verify():
        report = drill_file(path, settings, counter, sample or None)
    failed = sum(not d.ok for d in report.documents)
    audit("backup.drill", workspace=workspace_of(settings), name=name, failed=failed)
    return BackupDrillResponse(
        name=name,
        listed=report.listed,
        checked=len(report.documents),
        failed=failed,
        documents=[DrillDocumentResponse(**dataclasses.asdict(d)) for d in report.documents],
        problems=report.problems,
        ok=not failed and not report.problems,
    )


# Restore one of this workspace's backups from BACKUP_DIR, as /import would the same
# file uploaded — without sending it over the network first.
@router.post(
    "/backups/{name}/restore",
    response_model=ImportResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={404: {"description": "No such backup, or BACKUP_DIR is unset."}},
)
def restore_stored_backup(
    name: str,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    queue: IngestQueueDep,
    raw_store: RawDocumentStoreDep,
    feedback_store: FeedbackStoreDep,
    background: bool = False,
    only: Annotated[
        list[str] | None,
        Query(
            max_length=500,
            description="Restore just these documents (repeatable); feedback isn't restored then.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        Query(description="List what it would restore, skip and reject; change nothing."),
    ] = False,
) -> ImportResponse:
    path = _stored_backup(settings, name)
    chosen = frozenset(only) if only else None
    if dry_run:
        with path.open("rb") as handle:
            return _restore(
                handle,
                repository,
                settings,
                queue,
                raw_store,
                feedback_store,
                source=name,
                only=chosen,
                dry_run=True,
            )
    require_free_space(settings, settings.raw_document_dir)
    if background:
        directory = spool_directory(settings.import_spool_dir)
        record = None
        if directory is not None:
            # No copy: the backup stays where it is; the record says which one it was.
            record = remember(
                directory,
                {
                    "kind": "restore",
                    "name": name,
                    "workspace": workspace_of(settings),
                    "stored": name,
                    **({"only": sorted(chosen)} if chosen else {}),
                },
            )
        kept = record
        return _restore_in_background(
            path,
            name,
            repository,
            settings,
            queue,
            raw_store,
            feedback_store,
            source=name,
            cleanup=(lambda: kept.unlink(missing_ok=True)) if kept is not None else lambda: None,
            resumable=record is not None,
            only=chosen,
        )
    with path.open("rb") as handle:
        return _restore(
            handle, repository, settings, queue, raw_store, feedback_store, source=name, only=chosen
        )


# The audit events kept in AUDIT_LOG_PATH (api/audit_log.py): this workspace's, or every
# workspace's with all_workspaces. 404 when the store is off.
@router.get(
    "/admin/audit",
    response_model=AuditResponse,
    dependencies=[Depends(require_full_key)],
    responses={404: {"description": "AUDIT_LOG_PATH is not set."}},
)
def list_audit(
    settings: SettingsDep,
    audit_log: Annotated[AuditLog | None, Depends(get_audit_log)],
    event: Annotated[
        str | None,
        Query(max_length=64, description='An event name, or a prefix ending in "." (document.).'),
    ] = None,
    since: Annotated[
        float | None, Query(ge=0, description="Only events since (epoch seconds).")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    all_workspaces: bool = False,
) -> AuditResponse:
    if audit_log is None:
        raise HTTPException(status_code=404, detail="The audit log is off (AUDIT_LOG_PATH).")
    entries = audit_log.recent(
        None if all_workspaces else (workspace_of(settings) or "default"),
        event=event,
        since=since,
        limit=limit,
    )
    return AuditResponse(
        entries=[AuditEntryResponse(**dataclasses.asdict(entry)) for entry in entries]
    )


# Risky configurations of this server (api/doctor.py). A report only.
@router.get(
    "/admin/doctor",
    response_model=DoctorResponse,
    dependencies=[Depends(require_full_key)],
)
def run_doctor(repository: VectorRepositoryDep, settings: SettingsDep) -> DoctorResponse:
    try:
        counts = {
            name: meta.chunk_count for name, meta in repository.filename_metadata(settings).items()
        }
    except Exception:  # noqa: BLE001 - the index's checks are left out, the rest still run
        logger.warning("Doctor: could not read the document listing.", exc_info=True)
        counts = {}
    return DoctorResponse(
        findings=[
            DoctorFindingResponse(**dataclasses.asdict(finding))
            for finding in diagnose(
                settings, chunk_counts=counts, index=repository.index_size(settings)
            )
        ]
    )


# How the index and the stored originals agree (api/consistency.py). A report only.
@router.get(
    "/admin/consistency",
    response_model=ConsistencyResponse,
    dependencies=[Depends(require_full_key)],
)
def check_consistency(
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    raw_store: RawDocumentStoreDep,
    token_counter: TokenCounterDep,
) -> ConsistencyResponse:
    report = consistency_report(repository, settings, raw_store, token_counter)
    return ConsistencyResponse(problems=report.problems, **dataclasses.asdict(report))


# What the report finds, put right — only the remedies asked for, each checked again
# under the document's lock: orphaned originals and version folders deleted, incomplete
# documents re-indexed from their originals. Audited.
@router.post(
    "/admin/consistency/repair",
    response_model=ConsistencyRepairResponse,
    dependencies=[Depends(require_full_key)],
)
def repair_consistency(
    request: ConsistencyRepairRequest,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    raw_store: RawDocumentStoreDep,
    queue: IngestQueueDep,
    token_counter: TokenCounterDep,
) -> ConsistencyRepairResponse:
    try:
        outcome = repair(
            repository,
            settings,
            raw_store,
            queue,
            token_counter,
            delete_orphan_originals=request.delete_orphan_originals,
            delete_orphan_versions=request.delete_orphan_versions,
            reindex_incomplete=request.reindex_incomplete,
            delete_despite_lost_index=request.delete_despite_lost_index,
        )
    except LostIndexError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    audit(
        "admin.repair",
        workspace=workspace_of(settings),
        originals_deleted=outcome.originals_deleted,
        versions_deleted=outcome.versions_deleted,
        reindexed=[job.filename for job in outcome.reindex_jobs],
    )
    return ConsistencyRepairResponse(**dataclasses.asdict(outcome))


# The way back from a lost index (api/rebuild.py): every stored original with no
# document is indexed again, in the background, with the tags and source kept beside it.
@router.post(
    "/admin/rebuild",
    response_model=RebuildResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={409: {"description": "Originals aren't kept, or the index can't be written."}},
)
def rebuild_from_originals(
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    raw_store: RawDocumentStoreDep,
    queue: IngestQueueDep,
) -> RebuildResponse:
    if raw_store is None:
        raise HTTPException(
            status_code=409,
            detail="A rebuild reads stored originals, and none are kept (RAW_DOCUMENT_DIR).",
        )
    # An emptied collection built for another embedding setup is recreated here; one
    # that still holds documents is a 409.
    repository.ensure_writable(settings)
    names = originals_to_rebuild(repository, settings, raw_store)
    if not names:
        return RebuildResponse(originals=0)
    store = raw_store
    started = start_import(
        "rebuild",
        f"{len(names)} originals",
        workspace_of(settings),
        str(settings.qdrant_collection),
        lambda status: rebuild(names, repository, settings, store, queue, status),
    )
    audit("admin.rebuild", workspace=workspace_of(settings), originals=len(names))
    return RebuildResponse(import_id=started.id, originals=len(names))


# One notification now, to check NOTIFY_URL end to end (sent on the request, so the
# answer says whether the webhook took it).
@router.post(
    "/admin/notify-test",
    response_model=NotifyTestResponse,
    dependencies=[Depends(require_full_key)],
    responses={404: {"description": "Notifications are off (NOTIFY_URL)."}},
)
def send_test_notification() -> NotifyTestResponse:
    notifier = current_notifier()
    if notifier is None:
        raise HTTPException(status_code=404, detail="Notifications are off (NOTIFY_URL).")
    try:
        notifier.send(TEST_EVENT, "This is a test: notifications from this server reach you.")
    except NotifyError as exc:
        return NotifyTestResponse(sent=False, error=str(exc))
    audit("admin.notify_test", workspace="")
    return NotifyTestResponse(sent=True)
