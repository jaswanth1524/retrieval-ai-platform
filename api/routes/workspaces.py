"""Workspace routes: list, create and delete (an empty) workspace (api/workspaces.py)."""

from __future__ import annotations

import functools
import os
import shutil
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from qdrant_client import QdrantClient

from api.corpus import bump_corpus_generation
from api.dependencies import (
    clear_workspace_raw_stores,
    get_app_settings,
    require_api_key,
    require_full_key,
)
from api.ingestion import ingest_sequencer
from api.qdrant_schema import forget_collection
from api.raw_documents import VERSIONS_DIR, is_temp_file_name
from api.repository import forget_metadata
from api.routes.deps import QdrantClientDep, VectorRepositoryDep
from api.schemas import WorkspaceCreateRequest, WorkspaceListResponse, WorkspaceResponse
from api.settings import AppSettings
from api.workspaces import (
    MAX_WORKSPACES,
    WorkspaceError,
    filename_of,
    forget_workspace,
    list_workspaces,
    settings_for,
    validate_name,
    workspace_exists,
)

# The app's own settings, never a workspace's: these routes manage workspaces by name.
BaseSettingsDep = Annotated[AppSettings, Depends(get_app_settings)]


def _require_enabled(settings: AppSettings) -> None:
    if not settings.workspaces_enabled:
        raise HTTPException(status_code=404, detail="Workspaces are not enabled on this server.")


def _name(raw: str) -> str:
    try:
        return validate_name(raw)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _delete_if_empty(client: QdrantClient, target: AppSettings, name: str) -> None:
    """Delete ``target``'s workspace when it holds no points and no originals, else 409."""

    # Only an empty one: deleting a corpus is one document at a time (or a backup,
    # then delete each), never a single call that drops everything in it.
    points = client.count(collection_name=target.qdrant_collection, exact=True).count
    originals = target.raw_document_dir
    entries = os.listdir(originals) if originals and os.path.isdir(originals) else []
    # Earlier versions belong to documents: with none left, they are history of
    # deleted ones, and an empty workspace stayed undeletable over them.
    entries = [entry for entry in entries if entry != VERSIONS_DIR]
    # A save a crash interrupted leaves an `.upload-*` temp file behind; it is not a
    # document, and must not keep an empty workspace undeletable.
    temp_files = [entry for entry in entries if is_temp_file_name(entry)]
    if points or len(temp_files) < len(entries):
        raise HTTPException(
            status_code=409,
            detail=f"Workspace '{name}' still holds documents; delete them first.",
        )
    # The directory first, so a failure there (permissions, a file that appeared
    # meanwhile) is a 409 with the workspace still whole, not a 500 after its
    # collection was already gone.
    if originals and os.path.isdir(originals):
        try:
            for entry in temp_files:
                os.unlink(os.path.join(originals, entry))
            shutil.rmtree(os.path.join(originals, VERSIONS_DIR), ignore_errors=True)
            os.rmdir(originals)
        except OSError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Workspace '{name}': its stored originals could not be removed "
                f"({exc.strerror or exc}).",
            ) from exc
    client.delete_collection(target.qdrant_collection)
    forget_collection(client, target)
    forget_metadata(client, target.qdrant_collection)
    forget_workspace(client, target.qdrant_collection)
    # Answers cached for it are for a corpus that is gone (a workspace created again
    # under this name starts at a generation it never had).
    bump_corpus_generation(collection=target.qdrant_collection)
    # The cached store for its directory: dropped, so a workspace created again
    # under this name starts from a fresh one (save also remakes the directory).
    clear_workspace_raw_stores()


def register(app: FastAPI) -> None:
    """Register the workspace routes."""

    @app.get(
        "/workspaces",
        response_model=WorkspaceListResponse,
        dependencies=[Depends(require_api_key)],
    )
    def workspaces(settings: BaseSettingsDep, client: QdrantClientDep) -> WorkspaceListResponse:
        # Answers when the feature is off too (enabled: false), so a client can ask.
        if not settings.workspaces_enabled:
            return WorkspaceListResponse(enabled=False, workspaces=[])
        return WorkspaceListResponse(
            enabled=True, workspaces=list_workspaces(client, settings.qdrant_collection)
        )

    @app.post(
        "/workspaces",
        response_model=WorkspaceResponse,
        status_code=201,
        dependencies=[Depends(require_full_key)],
    )
    async def create_workspace(
        request: WorkspaceCreateRequest,
        settings: BaseSettingsDep,
        client: QdrantClientDep,
        repository: VectorRepositoryDep,
    ) -> WorkspaceResponse:
        _require_enabled(settings)
        name = _name(request.name)
        existing = await run_in_threadpool(list_workspaces, client, settings.qdrant_collection)
        if name in existing:
            raise HTTPException(status_code=409, detail=f"Workspace '{name}' already exists.")
        if len(existing) >= MAX_WORKSPACES:
            raise HTTPException(
                status_code=409, detail=f"At most {MAX_WORKSPACES} workspaces can exist."
            )
        await run_in_threadpool(
            functools.partial(
                repository.ensure_writable, settings_for(settings, name), create_workspace=True
            )
        )
        return WorkspaceResponse(name=name)

    @app.delete(
        "/workspaces/{name}",
        response_model=WorkspaceResponse,
        dependencies=[Depends(require_full_key)],
        responses={409: {"description": "The workspace still holds documents or originals."}},
    )
    def delete_workspace(
        name: str, settings: BaseSettingsDep, client: QdrantClientDep
    ) -> WorkspaceResponse:
        _require_enabled(settings)
        name = _name(name)
        if not workspace_exists(client, settings.qdrant_collection, name):
            raise HTTPException(status_code=404, detail=f"No workspace named '{name}'.")
        # The watched folder's workspace: its watcher would fail every pass once the
        # collection is gone (it is never recreated implicitly), until a restart.
        if settings.watch_dir and settings.watch_workspace.strip() == name:
            raise HTTPException(
                status_code=409,
                detail=f"Workspace '{name}' is WATCH_WORKSPACE, where the watched folder "
                "is indexed; change that setting before deleting it.",
            )
        target = settings_for(settings, name)
        # Not while anything for it is queued or running: a job would recreate the
        # collection on its first write (the repository recreates a missing one), or
        # fail, after the workspace was reported deleted. Keys are workspace-qualified
        # (api/workspaces.document_key), so only this workspace's work counts. The check
        # also closes the workspace to new work until the delete is done, in one step,
        # so an upload can't be admitted between the check and the delete.
        group = name
        if not ingest_sequencer.close_if_idle(
            group, lambda key: filename_of(target, key) is not None
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Workspace '{name}' has documents being indexed or deleted; try "
                "again when they finish.",
            )
        try:
            _delete_if_empty(client, target, name)
        finally:
            ingest_sequencer.reopen(group)
        return WorkspaceResponse(name=name)
