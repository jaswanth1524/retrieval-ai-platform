"""Watched-folder routes: what the watcher last saw, and a pass on demand (api/watch.py)."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import Depends, FastAPI, HTTPException, Request

from api.audit import audit
from api.dependencies import require_api_key, require_full_key
from api.schemas import WatchFolderStatusResponse, WatchStatusResponse
from api.watch import FolderWatcher


def _watcher(request: Request) -> FolderWatcher:
    watcher: FolderWatcher | None = getattr(request.app.state, "watcher", None)
    if watcher is None:
        raise HTTPException(status_code=404, detail="No folder is watched (WATCH_DIR is unset).")
    return watcher


def _watchers(request: Request) -> list[FolderWatcher]:
    watchers: list[FolderWatcher] | None = getattr(request.app.state, "watchers", None)
    return watchers or [_watcher(request)]


def _status(request: Request) -> WatchStatusResponse:
    """The first folder's status (the fields every client reads), and each folder's."""

    first = _watcher(request)
    folders = [
        WatchFolderStatusResponse.model_validate(
            {
                **asdict(watcher.status()),
                "workspace": getattr(watcher, "workspace", "") or "default",
            }
        )
        for watcher in _watchers(request)
    ]
    return WatchStatusResponse.model_validate({**asdict(first.status()), "folders": folders})


def register(app: FastAPI) -> None:
    """Register the watched-folder routes."""

    # The folder's path, file names and errors: the read key may see them (as it sees
    # the document listing), but only the full key may start a pass.
    @app.get(
        "/watch",
        response_model=WatchStatusResponse,
        dependencies=[Depends(require_api_key)],
    )
    def watch_status(request: Request) -> WatchStatusResponse:
        return _status(request)

    @app.post("/watch/scan", status_code=202, dependencies=[Depends(require_full_key)])
    def watch_scan_now(request: Request) -> WatchStatusResponse:
        watchers = _watchers(request)
        for watcher in watchers:
            watcher.scan_now()
        # Each folder's workspace (WATCH_DIRS feeds several), not WATCH_WORKSPACE alone.
        for workspace in dict.fromkeys(getattr(w, "workspace", "") for w in watchers):
            audit("watch.scan", workspace=workspace)
        # The status as of the previous pass: the new one runs on the watchers' threads.
        return _status(request)
