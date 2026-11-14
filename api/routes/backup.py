"""Backup routes: export the corpus as a zip, and import one."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import shutil
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
from api.consistency import consistency_report, repair
from api.dependencies import (
    get_audit_log,
    require_full_key,
)
from api.doctor import diagnose
from api.document_names import (
    known_filenames,
    refuse_near_duplicate_name,
)
from api.documents import (
    DocumentSource,
)
from api.export import build_export
from api.feedback import FeedbackStore
from api.imports import BackgroundImport, wait_for_room
from api.imports import start as start_import
from api.ingest_jobs import IngestQueue
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.restore import BackupCheck, BackupImportError, restore_backup, verify_backup
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
    BackupFileResponse,
    BackupListResponse,
    BackupRunResponse,
    ConsistencyRepairRequest,
    ConsistencyRepairResponse,
    ConsistencyResponse,
    DoctorFindingResponse,
    DoctorResponse,
    DocumentJobAcceptedResponse,
    ImportResponse,
)
from api.settings import AppSettings
from api.streaming import ClosingStreamingResponse
from api.workspaces import workspace_of

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
    if background:
        spool = _spool_upload(file)
        return _restore_in_background(
            spool,
            file.filename or "backup.zip",
            repository,
            settings,
            queue,
            raw_store,
            feedback_store,
            cleanup=lambda: spool.unlink(missing_ok=True),
        )
    return _restore(file.file, repository, settings, queue, raw_store, feedback_store)


def _spool_upload(file: UploadFile) -> Path:
    """The upload copied to a temporary file the import thread reads after the request."""

    handle = tempfile.NamedTemporaryFile(prefix="docrag-restore-", suffix=".zip", delete=False)
    with handle:
        shutil.copyfileobj(file.file, handle, length=1 << 20)
    return Path(handle.name)


def _restore_in_background(
    path: Path,
    name: str,
    repository: VectorRepository,
    settings: AppSettings,
    queue: IngestQueue,
    raw_store: RawDocumentStore | None,
    feedback_store: FeedbackStore | None,
    *,
    cleanup: Callable[[], None] = lambda: None,
    source: str | None = None,
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
    repository.ensure_writable(settings)

    def run(status: BackgroundImport) -> None:
        with path.open("rb") as handle:
            outcome = _restore(
                handle,
                repository,
                settings,
                queue,
                raw_store,
                feedback_store,
                source=source,
                status=status,
            )
        status.skipped += len(outcome.skipped) + len(outcome.missing_originals)
        status.failed += len(outcome.rejected) + len(outcome.deferred)
        for filename in outcome.rejected:
            status.note(f"{filename}: refused by restore's rules (or damaged)")
        for filename in outcome.missing_originals:
            status.note(f"{filename}: the backup holds no original of it")

    started = start_import(
        "restore",
        name,
        workspace_of(settings),
        str(settings.qdrant_collection),
        run,
        cleanup=cleanup,
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
) -> ImportResponse:
    """Restore one backup into ``settings``' workspace: an upload (``/import``) or one
    of BACKUP_DIR's (``source`` names it, for the audit). With ``status`` (a background
    import) a full queue is waited out instead of deferring, and its count follows."""

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
        accepted = queue.enqueue(
            filename=filename,
            content=content,
            raw_store=raw_store,
            uploaded_at=uploaded_at,
            tags=tags,
            source=source,
            watched_from=watched_from,
            kind="import",
        )
        if status is not None:
            status.queued += 1
        return accepted

    def has_room(size: int) -> bool:
        if status is None:
            return queue.backlog.has_room(size)
        return wait_for_room(queue.backlog.has_room, size, status)

    # The last step of recovering from an embedding change (export, delete all,
    # import): an emptied collection is rebuilt here; a non-empty one is a 409.
    repository.ensure_writable(settings)
    try:
        outcome = restore_backup(
            upload,
            # Indexed, or being indexed: importing the same backup again while the
            # first import's jobs run must not queue each document a second time.
            existing_filenames=set(known_filenames(repository, settings)),
            max_original_bytes=int(settings.max_upload_bytes),
            has_room=has_room,
            enqueue=enqueue,
            feedback_store=feedback_store,
            workspace=workspace_of(settings),
            keep_version=raw_store.add_version if raw_store is not None else None,
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
    return BackupCheckResponse(ok=check.ok, **dataclasses.asdict(check))


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
    _backup_dir(settings)
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
        check = verify_backup(handle, int(settings.max_upload_bytes))
    # Listed with this result from now on, after a restart too.
    loop: ScheduledBackup | None = getattr(request.app.state, "backup", None)
    if loop is not None:
        loop.record_check(path, check)
    else:
        save_check(path, check)
    return _check_response(check)


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
) -> ImportResponse:
    path = _stored_backup(settings, name)
    if background:
        return _restore_in_background(
            path, name, repository, settings, queue, raw_store, feedback_store, source=name
        )
    with path.open("rb") as handle:
        return _restore(handle, repository, settings, queue, raw_store, feedback_store, source=name)


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
            for finding in diagnose(settings, chunk_counts=counts)
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
    outcome = repair(
        repository,
        settings,
        raw_store,
        queue,
        token_counter,
        delete_orphan_originals=request.delete_orphan_originals,
        delete_orphan_versions=request.delete_orphan_versions,
        reindex_incomplete=request.reindex_incomplete,
    )
    audit(
        "admin.repair",
        workspace=workspace_of(settings),
        originals_deleted=outcome.originals_deleted,
        versions_deleted=outcome.versions_deleted,
        reindexed=[job.filename for job in outcome.reindex_jobs],
    )
    return ConsistencyRepairResponse(**dataclasses.asdict(outcome))
