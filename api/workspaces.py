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

from qdrant_client import QdrantClient

from api.settings import AppSettings

WORKSPACE_HEADER = "X-Workspace"
_SEPARATOR = "__ws_"
# Lowercase, digits and hyphens: valid in a Qdrant collection name, a directory name and a
# header value alike, and never confusable with the separator.
_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$")
MAX_WORKSPACES = 50


class WorkspaceError(ValueError):
    """An unusable workspace name, or workspaces are off (a 400)."""


def validate_name(name: str) -> str:
    name = name.strip()
    if not _NAME_RE.match(name):
        raise WorkspaceError(
            "A workspace name is 1-32 lowercase letters, digits or hyphens, starting and "
            "ending with a letter or digit."
        )
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


def workspace_exists(client: QdrantClient, base_collection: str, name: str) -> bool:
    return bool(client.collection_exists(collection_for(base_collection, name)))
