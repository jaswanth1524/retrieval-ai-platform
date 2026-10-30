"""Workspaces: separate corpora on one server (WORKSPACES_ENABLED, off by default).

A workspace is its own Qdrant collection, ``<QDRANT_COLLECTION>__ws_<name>``, with its own
stored originals under ``<RAW_DOCUMENT_DIR>/workspaces/<name>``. A request picks one with
the ``X-Workspace`` header; without it (or with an empty one) it uses the default
workspace — the plain ``QDRANT_COLLECTION`` — exactly as before workspaces existed.

The per-request ``AppSettings`` copy (``settings_for``) is the whole mechanism: the
repository, pipeline, ingest service and raw store all read the collection and directory
from the settings they are given. What is keyed by filename alone in-process — the
filename write lock and the ingest sequencer — uses ``document_key`` so the same name in
two workspaces never blocks or overrides the other.

Not access control: any key that can use the API can use every workspace.
"""

from __future__ import annotations

import os
import re
import threading
from weakref import WeakKeyDictionary

from fastapi import HTTPException
from qdrant_client import QdrantClient

from api.settings import AppSettings

WORKSPACE_HEADER = "X-Workspace"
_SEPARATOR = "__ws_"
# Lowercase, digits and hyphens: valid in a Qdrant collection name, a directory name and a
# header value alike, and never confusable with the separator.
_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$")
MAX_WORKSPACES = 50


class WorkspaceClosingError(HTTPException):
    """409: the workspace is being deleted, so nothing is admitted into it now
    (``ingest_jobs.admit_or_refuse``). Its own type so a caller (restore) can tell it
    from any other refusal."""

    def __init__(self, workspace: str) -> None:
        super().__init__(
            status_code=409,
            detail=f"Workspace '{workspace}' is being deleted; nothing can be added to or "
            "removed from it now.",
        )


class WorkspaceError(ValueError):
    """An unusable workspace name, or workspaces are off (a 400)."""


# What clients (the CLI, MCP, the UI's picker) call the default workspace; a workspace
# created under it could never be reached from them, and its metric and audit labels
# would read as the default one's.
RESERVED_NAMES = frozenset({"default"})


def validate_name(name: str, *, new: bool = False) -> str:
    """``name`` stripped, or a ``WorkspaceError``. With ``new`` (creating one), the
    reserved names are refused too; one that already exists can still be used."""

    name = name.strip()
    if not _NAME_RE.match(name):
        raise WorkspaceError(
            "A workspace name is 1-32 lowercase letters, digits or hyphens, starting and "
            "ending with a letter or digit."
        )
    if new and name in RESERVED_NAMES:
        raise WorkspaceError(f"'{name}' is reserved for the default workspace.")
    return name


def collection_for(base_collection: str, name: str) -> str:
    return f"{base_collection}{_SEPARATOR}{name}" if name else base_collection


def workspace_of(settings: AppSettings) -> str:
    """The workspace these settings point at ("" for the default)."""

    _, separator, name = settings.qdrant_collection.partition(_SEPARATOR)
    return name if separator else ""


def settings_for(settings: AppSettings, name: str) -> AppSettings:
    """``settings`` pointed at workspace ``name`` ("" returns them unchanged)."""

    if not name:
        return settings
    update: dict[str, object] = {
        "qdrant_collection": collection_for(settings.qdrant_collection, name)
    }
    if settings.raw_document_dir:
        update["raw_document_dir"] = os.path.join(settings.raw_document_dir, "workspaces", name)
    return settings.model_copy(update=update)


def document_key(settings: AppSettings, filename: str) -> str:
    """The in-process key for ``filename`` in these settings' workspace: the filename
    itself in the default workspace (unchanged), workspace-qualified in any other."""

    workspace = workspace_of(settings)
    return f"{workspace}\x1f{filename}" if workspace else filename


def filename_of(settings: AppSettings, key: str) -> str | None:
    """The filename ``key`` names when it belongs to these settings' workspace, else None."""

    workspace, separator, filename = key.rpartition("\x1f")
    if not separator:
        workspace, filename = "", key
    return filename if workspace == workspace_of(settings) else None


def list_workspaces(client: QdrantClient, base_collection: str) -> list[str]:
    """The workspaces that exist (have a collection), sorted; the default is not listed."""

    prefix = f"{base_collection}{_SEPARATOR}"
    names = [
        collection.name[len(prefix) :]
        for collection in client.get_collections().collections
        if collection.name.startswith(prefix)
    ]
    return sorted(name for name in names if _NAME_RE.match(name))


# Workspace collections this process has seen exist, per client. Every request naming a
# workspace checks it with Qdrant; while Qdrant is briefly unreachable, one seen before
# counts as existing, so a request that needs nothing from Qdrant (a job poll, every
# second during an upload) still works instead of failing with a 503.
_seen_workspaces: WeakKeyDictionary[QdrantClient, set[str]] = WeakKeyDictionary()
_seen_lock = threading.Lock()


def workspace_exists(client: QdrantClient, base_collection: str, name: str) -> bool:
    collection = collection_for(base_collection, name)
    try:
        exists = bool(client.collection_exists(collection))
    except Exception:
        with _seen_lock:
            if collection in _seen_workspaces.get(client, set()):
                return True
        raise
    with _seen_lock:
        seen = _seen_workspaces.setdefault(client, set())
        if exists:
            seen.add(collection)
        else:
            seen.discard(collection)
    return exists


def forget_workspace(client: QdrantClient, collection: str) -> None:
    """A deleted workspace no longer counts as existing when Qdrant can't be asked."""

    with _seen_lock:
        _seen_workspaces.get(client, set()).discard(collection)
