"""Watched-folder routes: what the watcher last saw, and a pass on demand (api/watch.py)."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import Depends, FastAPI, HTTPException, Request

from api.dependencies import require_api_key, require_full_key
from api.schemas import WatchStatusResponse
from api.watch import FolderWatcher


def _watcher(request: Request) -> FolderWatcher:
    watcher: FolderWatcher | None = getattr(request.app.state, "watcher", None)
    if watcher is None:
        raise HTTPException(status_code=404, detail="No folder is watched (WATCH_DIR is unset).")
    return watcher


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
        return WatchStatusResponse.model_validate(asdict(_watcher(request).status()))

    @app.post("/watch/scan", status_code=202, dependencies=[Depends(require_full_key)])
    def watch_scan_now(request: Request) -> WatchStatusResponse:
        watcher = _watcher(request)
        watcher.scan_now()
        # The status as of the previous pass: the new one runs on the watcher's thread.
        return WatchStatusResponse.model_validate(asdict(watcher.status()))
