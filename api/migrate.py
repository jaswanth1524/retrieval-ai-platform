"""The whole server in one file, to move it to another machine (``docrag migrate
pack|unpack``, ``GET /admin/migrate/export``, ``POST /admin/migrate/import``).

A backup (``api/export.py``) holds one workspace's documents and has every one of them
embedded again on restore. A bundle holds every workspace — its backup (originals,
earlier versions, tags, sources, ratings), its own settings, and a Qdrant snapshot of
its collection — so a server with the same embedding setup gets its index back as it
was, without embedding anything. Where a snapshot can't be used (another embedding
setup, a collection that already holds documents, a Qdrant that can't make one), the
workspace's documents are indexed from their originals instead, as a restore does.

Layout: ``manifest.json``, then per workspace ``workspaces/<name>/backup.zip`` and
``workspaces/<name>/snapshot`` (both stored, not compressed again).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import httpx
from qdrant_client import QdrantClient

from api.export import build_export
from api.feedback import FeedbackStore
from api.qdrant_schema import collection_metadata, forget_collection
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository, forget_metadata
from api.restore import _MANIFEST_MAX_BYTES, BackupImportError, restore_backup
from api.schemas import DocumentJobAcceptedResponse
from api.settings import AppSettings
from api.version import app_version
from api.workspace_target import WorkspaceTarget
from api.workspaces import workspace_of

logger = logging.getLogger(__name__)

MIGRATE_VERSION = 1
_COPY_BLOCK = 1 << 20
_SNAPSHOT_TIMEOUT = httpx.Timeout(30.0, read=3600.0, write=3600.0)


class MigrateError(Exception):
    """The bundle can't be written or read; the message is safe to show."""


def embedding_setup(settings: AppSettings) -> dict[str, str]:
    """What must be equal on both servers for a snapshot's vectors to be usable."""

    return {**collection_metadata(settings), "dense_size": str(settings.qdrant_dense_vector_size)}


class QdrantSnapshots:
    """Snapshots of a collection over Qdrant's REST API: the client library makes and
    lists them, but can't download one or recover from an upload."""

    def __init__(self, client: QdrantClient, settings: AppSettings) -> None:
        self._client = client
        self._base = settings.qdrant_url.rstrip("/")
        self._headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}

    def write(self, collection: str, out: IO[bytes]) -> bool:
        """Snapshot ``collection`` into ``out``; False when this Qdrant can't (the
        embedded one tests use, or a server that refuses)."""

        try:
            made = self._client.create_snapshot(collection_name=collection, wait=True)
        except Exception as exc:  # noqa: BLE001 - no snapshot: the bundle works without
            logger.info(
                "No snapshot of %s (%s): the bundle holds its originals only.", collection, exc
            )
            return False
        if made is None:
            return False
        url = f"{self._base}/collections/{collection}/snapshots/{made.name}"
        try:
            with httpx.stream(
                "GET", url, headers=self._headers, timeout=_SNAPSHOT_TIMEOUT
            ) as response:
                response.raise_for_status()
                for block in response.iter_bytes(_COPY_BLOCK):
                    out.write(block)
        except httpx.HTTPError as exc:
            raise MigrateError(f"Could not download the snapshot of {collection}.") from exc
        finally:
            try:
                self._client.delete_snapshot(collection_name=collection, snapshot_name=made.name)
            except Exception:  # noqa: BLE001 - left on Qdrant's disk: logged, not fatal
                logger.warning("Could not delete the snapshot %s of %s.", made.name, collection)
        return True

    def recover(self, collection: str, snapshot: Path) -> None:
        """Replace ``collection`` with the snapshot's contents."""

        url = f"{self._base}/collections/{collection}/snapshots/upload"
        try:
            with snapshot.open("rb") as handle:
                response = httpx.post(
                    url,
                    params={"priority": "snapshot", "wait": "true"},
                    files={"snapshot": ("snapshot", handle, "application/octet-stream")},
                    headers=self._headers,
                    timeout=_SNAPSHOT_TIMEOUT,
                )
            response.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            raise MigrateError(f"Qdrant could not recover {collection} from its snapshot.") from exc


def pack(
    settings: AppSettings,
    repository: VectorRepository,
    targets: Sequence[WorkspaceTarget],
    feedback_store: FeedbackStore | None,
    own_settings: Callable[[str], dict[str, Any]],
    snapshots: QdrantSnapshots | None,
    out: Path,
) -> dict[str, Any]:
    """Write the bundle to ``out``; its manifest."""

    workspaces: list[dict[str, Any]] = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for target in targets:
            name = workspace_of(target.settings) or "default"
            prefix = f"workspaces/{name}"
            stream = build_export(
                target.settings,
                repository,
                target.raw_store,
                feedback_store,
                include_versions=True,
            )
            try:
                with bundle.open(f"{prefix}/backup.zip", "w", force_zip64=True) as member:
                    for block in stream:
                        member.write(block)
            finally:
                stream.close()
            has_snapshot = False
            if snapshots is not None:
                with bundle.open(f"{prefix}/snapshot", "w", force_zip64=True) as member:
                    has_snapshot = snapshots.write(str(target.settings.qdrant_collection), member)
            workspaces.append(
                {
                    "name": name,
                    "backup": f"{prefix}/backup.zip",
                    "snapshot": f"{prefix}/snapshot" if has_snapshot else None,
                    "settings": own_settings(workspace_of(target.settings)),
                    "documents": len(repository.filename_metadata(target.settings)),
                }
            )
        manifest = {
            "migrate_version": MIGRATE_VERSION,
            "app_version": app_version(),
            "created_at": time.time(),
            "embedding": embedding_setup(settings),
            "workspaces": workspaces,
        }
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2))
    return manifest


def read_manifest(bundle: zipfile.ZipFile) -> dict[str, Any]:
    try:
        info = bundle.getinfo("manifest.json")
    except KeyError:
        raise MigrateError("The file has no manifest.json; is it a migration bundle?") from None
    if info.file_size > _MANIFEST_MAX_BYTES:
        raise MigrateError("The bundle's manifest.json is too large to be one.")
    try:
        manifest = json.loads(bundle.read(info))
    except (ValueError, RecursionError, zipfile.BadZipFile, OSError):
        raise MigrateError("The bundle's manifest.json can't be read.") from None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("workspaces"), list):
        raise MigrateError("The bundle's manifest.json is not a migration manifest.")
    if manifest.get("migrate_version") != MIGRATE_VERSION:
        raise MigrateError(
            f"Unsupported bundle version {manifest.get('migrate_version')!r}; this server "
            f"reads version {MIGRATE_VERSION}."
        )
    return manifest


@dataclass
class WorkspaceUnpacked:
    name: str
    # "snapshot" (the index came back as it was), "originals" (indexed again from them),
    # or "skipped" (see note).
    how: str
    documents: int = 0
    note: str = ""


@dataclass
class Unpacked:
    workspaces: list[WorkspaceUnpacked] = field(default_factory=list)


def _extract(bundle: zipfile.ZipFile, member: str, directory: Path | None) -> Path:
    """One member copied to a temporary file (a nested zip, or a snapshot to upload)."""

    handle = tempfile.NamedTemporaryFile(prefix="docrag-migrate-", delete=False, dir=directory)
    path = Path(handle.name)
    try:
        with handle, bundle.open(member) as source:
            shutil.copyfileobj(source, handle, length=_COPY_BLOCK)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def store_originals(
    backup: Path,
    raw_store: RawDocumentStore | None,
    max_bytes: int,
    feedback_store: FeedbackStore | None,
    workspace: str,
) -> int:
    """Put a backup's originals and versions straight into ``raw_store`` (when
    originals are kept here) and its ratings into ``feedback_store``, indexing nothing:
    the documents are already in the recovered index. How many originals."""

    stored = 0

    def keep(
        filename: str,
        content: bytes,
        uploaded_at: float | None,
        tags: tuple[str, ...],
        source: object,
        watched_from: str | None,
    ) -> DocumentJobAcceptedResponse:
        nonlocal stored
        if raw_store is not None:
            raw_store.save(filename, content)
            stored += 1
        return DocumentJobAcceptedResponse(job_id="", filename=filename, state="queued")

    with backup.open("rb") as handle:
        restore_backup(
            handle,
            existing_filenames=set(),
            max_original_bytes=max_bytes,
            has_room=lambda size: True,
            enqueue=keep,
            feedback_store=feedback_store,
            workspace=workspace,
            keep_version=raw_store.add_version if raw_store is not None else None,
            stored=True,
        )
    if raw_store is not None:
        # Their details (tags, sources) are in the recovered index: written beside the
        # originals at the next listing.
        raw_store.details_backfilled = False
    return stored


def can_use_snapshot(
    manifest: dict[str, Any], settings: AppSettings, points_here: int
) -> str | None:
    """Why a workspace's snapshot can't be recovered here, or None when it can."""

    if manifest.get("embedding") != embedding_setup(settings):
        return "the embedding setup here differs from the bundle's"
    if points_here:
        return "the workspace already holds documents here"
    return None


def recover_workspace(
    bundle: zipfile.ZipFile,
    entry: dict[str, Any],
    target: WorkspaceTarget,
    client: QdrantClient,
    snapshots: QdrantSnapshots,
    spool_directory: Path | None,
) -> None:
    """Recover ``target``'s collection from the bundle's snapshot of it (and forget
    what was cached about the collection before)."""

    snapshot = _extract(bundle, str(entry["snapshot"]), spool_directory)
    try:
        snapshots.recover(str(target.settings.qdrant_collection), snapshot)
    finally:
        snapshot.unlink(missing_ok=True)
    forget_collection(client, target.settings)
    forget_metadata(client, str(target.settings.qdrant_collection))


def backup_of(bundle: zipfile.ZipFile, entry: dict[str, Any], spool_directory: Path | None) -> Path:
    member = entry.get("backup")
    if not isinstance(member, str) or member not in bundle.namelist():
        raise BackupImportError("the bundle holds no backup of it")
    return _extract(bundle, member, spool_directory)
