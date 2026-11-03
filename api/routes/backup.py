"""Backup routes: export the corpus as a zip, and import one."""

from __future__ import annotations

import logging
import threading
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from api.audit import audit
from api.dependencies import (
    require_full_key,
)
from api.document_names import (
    known_filenames,
    refuse_near_duplicate_name,
)
from api.documents import (
    DocumentSource,
)
from api.export import build_export
from api.restore import BackupImportError, restore_backup
from api.routes.deps import (
    FeedbackStoreDep,
    IngestQueueDep,
    RawDocumentStoreDep,
    SettingsDep,
    VectorRepositoryDep,
)
from api.schemas import (
    DocumentJobAcceptedResponse,
    ImportResponse,
)
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
) -> ImportResponse:
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
            source=source,
            kind="import",
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
