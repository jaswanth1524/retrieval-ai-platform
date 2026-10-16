"""Workspace routes: list, create and delete (an empty) workspace (api/workspaces.py)."""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_app_settings, require_api_key, require_full_key
from api.qdrant_schema import forget_collection
from api.repository import clear_metadata_cache
from api.routes.deps import QdrantClientDep, VectorRepositoryDep
from api.schemas import WorkspaceCreateRequest, WorkspaceListResponse, WorkspaceResponse
from api.settings import AppSettings
from api.workspaces import (
    MAX_WORKSPACES,
    WorkspaceError,
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
        await run_in_threadpool(repository.ensure_writable, settings_for(settings, name))
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
        target = settings_for(settings, name)
        # Only an empty one: deleting a corpus is one document at a time (or a backup,
        # then delete each), never a single call that drops everything in it.
        points = client.count(collection_name=target.qdrant_collection, exact=True).count
        originals = target.raw_document_dir
        if points or (originals and os.path.isdir(originals) and os.listdir(originals)):
            raise HTTPException(
                status_code=409,
                detail=f"Workspace '{name}' still holds documents; delete them first.",
            )
        client.delete_collection(target.qdrant_collection)
        forget_collection(client, target)
        clear_metadata_cache()
        if originals and os.path.isdir(originals):
            os.rmdir(originals)
        return WorkspaceResponse(name=name)
