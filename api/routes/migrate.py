"""The whole server as one bundle, and back (api/migrate.py). Full key; every workspace,
whatever ``X-Workspace`` says."""

from __future__ import annotations

import logging
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient
from starlette.background import BackgroundTask

from api.audit import audit
from api.chunking import TokenCounter
from api.corpus import bump_corpus_generation
from api.dependencies import (
    get_app_settings,
    get_embedding_provider,
    get_workspace_settings_store,
    raw_store_for,
    require_full_key,
)
from api.embeddings import LocalEmbeddingProvider
from api.imports import BackgroundImport, spool_directory, spool_upload
from api.imports import start as start_import
from api.ingest_jobs import IngestQueue
from api.migrate import (
    MigrateError,
    QdrantSnapshots,
    backup_of,
    can_use_snapshot,
    pack,
    read_manifest,
    recover_workspace,
    store_originals,
)
from api.pipeline import IngestService
from api.repository import VectorRepository
from api.routes import backup as backup_routes
from api.routes.deps import (
    FeedbackStoreDep,
    IngestBacklogDep,
    IngestExecutorDep,
    IngestJobStoreDep,
    QdrantClientDep,
    TokenCounterDep,
    VectorRepositoryDep,
)
from api.schemas import MigrateImportResponse
from api.settings import AppSettings
from api.storage import require_free_space
from api.workspace_settings import WORKSPACE_SETTING_FIELDS, WorkspaceSettingsStore
from api.workspace_target import WorkspaceTarget
from api.workspaces import (
    MAX_WORKSPACES,
    WorkspaceError,
    list_workspaces,
    settings_for,
    validate_name,
)

logger = logging.getLogger("api.routes.documents")

router = APIRouter()

BaseSettingsDep = Annotated[AppSettings, Depends(get_app_settings)]
EmbeddingProviderDep = Annotated[LocalEmbeddingProvider, Depends(get_embedding_provider)]
WorkspaceSettingsStoreDep = Annotated[WorkspaceSettingsStore, Depends(get_workspace_settings_store)]


class _Server:
    """What packing or unpacking needs to reach any workspace of this server."""

    def __init__(
        self,
        settings: AppSettings,
        client: QdrantClient,
        repository: VectorRepository,
        embedding: LocalEmbeddingProvider,
        token_counter: TokenCounter,
        job_store: IngestJobStoreDep,
        executor: IngestExecutorDep,
        backlog: IngestBacklogDep,
    ) -> None:
        self.settings = settings
        self.client = client
        self.repository = repository
        self._embedding = embedding
        self._token_counter = token_counter
        self._queue_parts = (job_store, executor, backlog)

    def target(self, name: str) -> WorkspaceTarget:
        """``name``'s workspace ("" or "default": the default one)."""

        settings = (
            settings_for(self.settings, name) if name and name != "default" else self.settings
        )
        service = IngestService(self.repository, self._embedding, settings, self._token_counter)
        return WorkspaceTarget(
            settings, raw_store_for(settings), IngestQueue(*self._queue_parts, service)
        )

    def targets(self) -> list[WorkspaceTarget]:
        names = [""]
        if self.settings.workspaces_enabled:
            names += list_workspaces(self.client, self.settings.qdrant_collection)
        return [self.target(name) for name in names]

    def snapshots(self) -> QdrantSnapshots:
        return QdrantSnapshots(self.client, self.settings)


def get_server(
    settings: BaseSettingsDep,
    client: QdrantClientDep,
    repository: VectorRepositoryDep,
    embedding: EmbeddingProviderDep,
    token_counter: TokenCounterDep,
    job_store: IngestJobStoreDep,
    executor: IngestExecutorDep,
    backlog: IngestBacklogDep,
) -> _Server:
    return _Server(
        settings, client, repository, embedding, token_counter, job_store, executor, backlog
    )


ServerDep = Annotated[_Server, Depends(get_server)]


def _scratch(settings: AppSettings) -> Path | None:
    """Where the bundle is assembled and read: IMPORT_SPOOL_DIR when set (in Docker
    /tmp is memory), else the system's temp directory (None)."""

    return spool_directory(settings.import_spool_dir)


# Every workspace's documents, versions, tags, sources, ratings and settings, with a
# Qdrant snapshot of each index: one file to move the server with. Written to disk
# first (a snapshot's size isn't known before it is made), then sent.
@router.get(
    "/admin/migrate/export",
    dependencies=[Depends(require_full_key)],
    responses={409: {"description": "Another export or backup is being written."}},
)
def migrate_export(
    server: ServerDep,
    feedback_store: FeedbackStoreDep,
    own_settings: WorkspaceSettingsStoreDep,
) -> FileResponse:
    settings = server.settings
    directory = _scratch(settings)
    require_free_space(settings, directory or tempfile.gettempdir())
    if not backup_routes._export_running.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="Another export or backup is being written; try again."
        )
    handle = tempfile.NamedTemporaryFile(
        prefix="docrag-migrate-", suffix=".zip", delete=False, dir=directory
    )
    handle.close()
    path = Path(handle.name)
    try:
        manifest = pack(
            settings,
            server.repository,
            server.targets(),
            feedback_store,
            own_settings.get,
            server.snapshots(),
            path,
        )
    except MigrateError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        backup_routes._export_running.release()
    audit(
        "admin.migrate_export",
        workspace="",
        workspaces=[entry["name"] for entry in manifest["workspaces"]],
        snapshots=sum(1 for entry in manifest["workspaces"] if entry["snapshot"]),
    )
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"docrag-migrate-{stamp}.zip",
        background=BackgroundTask(path.unlink, missing_ok=True),
    )


# A bundle put back, in the background (follow it with GET /documents/imports/{id}, in
# the default workspace). A workspace whose snapshot fits this server gets its index
# back as it was; any other is indexed from its originals, as a restore would.
@router.post(
    "/admin/migrate/import",
    response_model=MigrateImportResponse,
    status_code=202,
    dependencies=[Depends(require_full_key)],
    responses={400: {"description": "Not a migration bundle."}},
)
def migrate_import(
    file: Annotated[UploadFile, File()],
    server: ServerDep,
    feedback_store: FeedbackStoreDep,
    own_settings: WorkspaceSettingsStoreDep,
) -> MigrateImportResponse:
    settings = server.settings
    directory = _scratch(settings)
    require_free_space(settings, directory or tempfile.gettempdir())
    spool = spool_upload(file.file, "docrag-migrate-", directory)
    try:
        with zipfile.ZipFile(spool) as bundle:
            manifest = read_manifest(bundle)
    except (zipfile.BadZipFile, OSError) as exc:
        spool.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="That file isn't a zip archive.") from exc
    except MigrateError as exc:
        spool.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    name = file.filename or "bundle.zip"

    def run(status: BackgroundImport) -> None:
        with zipfile.ZipFile(spool) as bundle:
            for entry in manifest["workspaces"]:
                if status.cancel.is_set():
                    return
                if isinstance(entry, dict):
                    _unpack_workspace(
                        bundle, manifest, entry, server, feedback_store, own_settings, status
                    )

    started = start_import(
        "migrate",
        name,
        "",
        str(settings.qdrant_collection),
        run,
        cleanup=lambda: spool.unlink(missing_ok=True),
    )
    audit("admin.migrate_import", workspace="", bundle=name, import_id=started.id)
    return MigrateImportResponse(
        import_id=started.id,
        workspaces=[str(e.get("name")) for e in manifest["workspaces"] if isinstance(e, dict)],
    )


def _unpack_workspace(
    bundle: zipfile.ZipFile,
    manifest: dict[str, object],
    entry: dict[str, object],
    server: _Server,
    feedback_store: FeedbackStoreDep,
    own_settings: WorkspaceSettingsStore,
    status: BackgroundImport,
) -> None:
    """One workspace of the bundle, by its snapshot when that fits here, else from its
    originals. A workspace that can't be unpacked is noted and the rest go on."""

    label = str(entry.get("name") or "")
    base = server.settings
    try:
        name = "" if label == "default" else validate_name(label, new=True)
    except WorkspaceError as exc:
        status.failed += 1
        status.note(f"{label!r}: not a workspace name ({exc})")
        return
    if name and not base.workspaces_enabled:
        status.failed += 1
        status.note(f"{label}: workspaces are off here (WORKSPACES_ENABLED); left out")
        return
    if name and name not in list_workspaces(server.client, base.qdrant_collection):
        if len(list_workspaces(server.client, base.qdrant_collection)) >= MAX_WORKSPACES:
            status.failed += 1
            status.note(f"{label}: at most {MAX_WORKSPACES} workspaces can exist; left out")
            return
    target = server.target(name)
    assert target.queue is not None
    backup: Path | None = None
    try:
        server.repository.ensure_writable(target.settings, create_workspace=bool(name))
        values = entry.get("settings")
        if isinstance(values, dict) and values:
            kept = {k: v for k, v in values.items() if k in WORKSPACE_SETTING_FIELDS}
            own_settings.put(name, kept)
        collection = str(target.settings.qdrant_collection)
        points = server.client.count(collection_name=collection, exact=True).count
        backup = backup_of(bundle, entry, _scratch(base))
        reason = (
            can_use_snapshot(manifest, target.settings, points)
            if entry.get("snapshot")
            else "the bundle holds no snapshot of it"
        )
        if reason is None:
            try:
                recover_workspace(
                    bundle, entry, target, server.client, server.snapshots(), _scratch(base)
                )
            except MigrateError as exc:
                reason = str(exc)
        if reason is None:
            stored = store_originals(
                backup, target.raw_store, int(base.max_upload_bytes), feedback_store, name
            )
            bump_corpus_generation(collection=collection)
            documents = len(server.repository.filename_metadata(target.settings))
            status.skipped += documents
            status.note(
                f"{label}: index recovered from its snapshot, nothing embedded again "
                f"({documents} documents, {stored} originals stored)"
            )
            return
        status.note(f"{label}: indexed from its originals ({reason})")
        with backup.open("rb") as handle:
            backup_routes._restore(
                handle,
                server.repository,
                target.settings,
                target.queue,
                target.raw_store,
                feedback_store,
                source=f"bundle:{label}",
                status=status,
            )
    except HTTPException as exc:
        status.failed += 1
        status.note(f"{label}: {exc.detail}")
    except Exception as exc:  # noqa: BLE001 - this workspace's failure; the rest go on
        logger.exception("Could not unpack workspace %r from the bundle.", label)
        status.failed += 1
        status.note(f"{label}: could not be unpacked ({type(exc).__name__})")
    finally:
        if backup is not None:
            backup.unlink(missing_ok=True)
