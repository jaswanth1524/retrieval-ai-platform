"""Workspace routes: list, create and delete (an empty) workspace (api/workspaces.py)."""

from __future__ import annotations

import functools
import logging
import os
import shutil
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from qdrant_client import QdrantClient

from api.answer_cache import AnswerCache
from api.audit import audit
from api.corpus import bump_corpus_generation
from api.dependencies import (
    clear_workspace_raw_stores,
    get_answer_cache,
    get_app_settings,
    get_feedback_store,
    get_ingest_job_store,
    get_query_log_store,
    get_trace_store,
    get_workspace_settings_store,
    require_api_key,
    require_full_key,
)
from api.feedback import FeedbackStore
from api.ingestion import ingest_sequencer
from api.jobs import JobStore
from api.metrics import forget_workspace_metrics
from api.qdrant_schema import forget_collection
from api.query_log import QueryLogStore
from api.raw_documents import VERSIONS_DIR, is_temp_file_name
from api.repository import forget_metadata
from api.routes.deps import QdrantClientDep, VectorRepositoryDep
from api.schemas import (
    WorkspaceCreateRequest,
    WorkspaceListResponse,
    WorkspaceResponse,
    WorkspaceSettingsRequest,
    WorkspaceSettingsResponse,
)
from api.settings import AppSettings
from api.sitemap import importing_into
from api.tracing import TraceStore
from api.watch import watched_workspaces
from api.workspace_settings import WORKSPACE_SETTING_FIELDS, WorkspaceSettingsStore
from api.workspaces import (
    MAX_WORKSPACES,
    WorkspaceError,
    filename_of,
    forget_workspace,
    list_workspaces,
    settings_for,
    validate_name,
    workspace_exists,
    workspace_of,
)

logger = logging.getLogger(__name__)

# The app's own settings, never a workspace's: these routes manage workspaces by name.
BaseSettingsDep = Annotated[AppSettings, Depends(get_app_settings)]


def _require_enabled(settings: AppSettings) -> None:
    if not settings.workspaces_enabled:
        raise HTTPException(status_code=404, detail="Workspaces are not enabled on this server.")


def _name(raw: str, *, new: bool = False) -> str:
    try:
        return validate_name(raw, new=new)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


WorkspaceSettingsStoreDep = Annotated[WorkspaceSettingsStore, Depends(get_workspace_settings_store)]


def _settings_target(settings: AppSettings, client: QdrantClient, raw: str) -> AppSettings:
    """The settings of the workspace ``raw`` names ("default": the default one)."""

    if raw == "default":
        return settings
    _require_enabled(settings)
    name = _name(raw)
    if not workspace_exists(client, settings.qdrant_collection, name):
        raise HTTPException(status_code=404, detail=f"No workspace named '{name}'.")
    return settings_for(settings, name)


def _settings_response(
    target: AppSettings, store: WorkspaceSettingsStore
) -> WorkspaceSettingsResponse:
    own = store.get(workspace_of(target))
    return WorkspaceSettingsResponse(
        workspace=workspace_of(target) or "default",
        settings=WorkspaceSettingsRequest(**own),
        effective=WorkspaceSettingsRequest(
            **{field: own.get(field, getattr(target, field)) for field in WORKSPACE_SETTING_FIELDS}
        ),
    )


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


def _forget_rows(
    name: str,
    feedback: FeedbackStore | None,
    query_log: QueryLogStore | None,
    traces: TraceStore,
    jobs: JobStore,
    workspace_settings: WorkspaceSettingsStore | None = None,
) -> dict[str, int]:
    """Remove a deleted workspace's rows from every store keyed by workspace, so one
    created again under its name starts empty (its ratings, asked questions, traces and
    job history would otherwise come back, and its next backup would carry the
    feedback). Best effort: the workspace is already gone, so a failure is logged."""

    removed: dict[str, int] = {}
    stores: list[tuple[str, Callable[[str], int] | None]] = [
        ("feedback", feedback.delete_workspace if feedback is not None else None),
        ("queries", query_log.delete_workspace if query_log is not None else None),
        ("traces", traces.delete_workspace),
        ("jobs", jobs.delete_finished),
        (
            "settings",
            workspace_settings.delete_workspace if workspace_settings is not None else None,
        ),
    ]
    for label, delete in stores:
        if delete is None:
            continue
        try:
            removed[label] = delete(name)
        except Exception:  # noqa: BLE001 - one store's failure must not hide the others
            logger.exception("Could not remove workspace %r's %s.", name, label)
    return removed


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
        name = _name(request.name, new=True)
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
        audit("workspace.create", workspace=name)
        return WorkspaceResponse(name=name)

    # A workspace's own relevance settings ("default" for the default workspace): read
    # with any key, changed with the full key. A change empties its cached answers.
    @app.get(
        "/workspaces/{name}/settings",
        response_model=WorkspaceSettingsResponse,
        dependencies=[Depends(require_api_key)],
    )
    def workspace_settings(
        name: str,
        settings: BaseSettingsDep,
        client: QdrantClientDep,
        store: WorkspaceSettingsStoreDep,
    ) -> WorkspaceSettingsResponse:
        return _settings_response(_settings_target(settings, client, name), store)

    @app.put(
        "/workspaces/{name}/settings",
        response_model=WorkspaceSettingsResponse,
        dependencies=[Depends(require_full_key)],
    )
    def put_workspace_settings(
        name: str,
        request: WorkspaceSettingsRequest,
        settings: BaseSettingsDep,
        client: QdrantClientDep,
        store: WorkspaceSettingsStoreDep,
        cache: Annotated[AnswerCache | None, Depends(get_answer_cache)],
    ) -> WorkspaceSettingsResponse:
        target = _settings_target(settings, client, name)
        values = request.model_dump(exclude_none=True)
        store.put(workspace_of(target), values)
        # Answers cached under the old settings would be served as if given under these.
        if cache is not None:
            cache.drop_collection(target.qdrant_collection)
        bump_corpus_generation(collection=target.qdrant_collection)
        audit("workspace.settings", workspace=workspace_of(target), **values)
        return _settings_response(target, store)

    @app.delete(
        "/workspaces/{name}",
        response_model=WorkspaceResponse,
        dependencies=[Depends(require_full_key)],
        responses={409: {"description": "The workspace still holds documents or originals."}},
    )
    def delete_workspace(
        name: str,
        settings: BaseSettingsDep,
        client: QdrantClientDep,
        feedback: Annotated[FeedbackStore | None, Depends(get_feedback_store)],
        query_log: Annotated[QueryLogStore | None, Depends(get_query_log_store)],
        traces: Annotated[TraceStore, Depends(get_trace_store)],
        jobs: Annotated[JobStore, Depends(get_ingest_job_store)],
        workspace_settings: WorkspaceSettingsStoreDep,
    ) -> WorkspaceResponse:
        _require_enabled(settings)
        name = _name(name)
        if not workspace_exists(client, settings.qdrant_collection, name):
            raise HTTPException(status_code=404, detail=f"No workspace named '{name}'.")
        # The watched folder's workspace: its watcher would fail every pass once the
        # collection is gone (it is never recreated implicitly), until a restart.
        if name in watched_workspaces(settings):
            raise HTTPException(
                status_code=409,
                detail=f"Workspace '{name}' is WATCH_WORKSPACE, where the watched folder "
                "is indexed; change that setting before deleting it.",
            )
        target = settings_for(settings, name)
        if importing_into(target.qdrant_collection):
            raise HTTPException(
                status_code=409,
                detail=f"Workspace '{name}' has a sitemap import still adding pages; try "
                "again when it finishes.",
            )
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
        forget_workspace_metrics(name)
        removed = _forget_rows(name, feedback, query_log, traces, jobs, workspace_settings)
        audit("workspace.delete", workspace=name, **{f"{k}_removed": v for k, v in removed.items()})
        return WorkspaceResponse(name=name)
