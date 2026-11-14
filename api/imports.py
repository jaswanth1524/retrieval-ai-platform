"""Zips and backups imported in the background (``background=true`` on
``POST /documents/archive``, ``POST /import`` and ``POST /backups/{name}/restore``).

A zip or a backup holds more documents than the indexing queue does (50 jobs at once by
default), so a request imported only the first batch and listed the rest as ``deferred``
for the caller to send again. In the background, the upload is spooled to a temporary
file and its documents are fed to the queue as room frees — waiting instead of
deferring — one import at a time. ``GET /documents/imports[/{id}]`` follows one and
``DELETE`` stops it (documents already queued stay queued). Kept in memory, the newest
``MAX_IMPORTS_KEPT`` of them: a restart forgets them (and stops a running one).
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_IMPORTS_KEPT = 50
# How often an import waiting for room in the indexing queue looks again.
ROOM_WAIT_SECONDS = 2.0
# Problems kept per import.
_MAX_PROBLEMS = 50


@dataclass
class BackgroundImport:
    id: str
    # "archive" (a zip of documents) or "restore" (a backup).
    kind: str
    # The zip's or backup's name.
    name: str
    workspace: str
    collection: str
    started_at: float
    queued: int = 0
    skipped: int = 0
    failed: int = 0
    finished_at: float | None = None
    problems: list[str] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def state(self) -> str:
        if self.finished_at is None:
            return "cancelling" if self.cancel.is_set() else "running"
        return "cancelled" if self.cancel.is_set() else "done"

    def note(self, problem: str) -> None:
        if len(self.problems) < _MAX_PROBLEMS:
            self.problems.append(problem[:300])


_lock = threading.Lock()
_imports: OrderedDict[str, BackgroundImport] = OrderedDict()
_pool: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docrag-import")
        return _pool


def start(
    kind: str,
    name: str,
    workspace: str,
    collection: str,
    run: Callable[[BackgroundImport], None],
    cleanup: Callable[[], None] = lambda: None,
) -> BackgroundImport:
    """Run ``run(status)`` on the import thread; ``cleanup`` (the spooled file's removal)
    runs after it however it ends, or at once when it can't start."""

    status = BackgroundImport(
        id=str(uuid.uuid4()),
        kind=kind,
        name=name,
        workspace=workspace,
        collection=collection,
        started_at=time.time(),
    )
    with _lock:
        _imports[status.id] = status
        while len(_imports) > MAX_IMPORTS_KEPT:
            oldest = next((k for k, v in _imports.items() if v.finished_at is not None), None)
            if oldest is None:
                break
            del _imports[oldest]

    def task() -> None:
        try:
            run(status)
        except Exception as exc:  # noqa: BLE001 - reported on the import, not raised
            logger.exception("Background %s import %s failed.", kind, name)
            status.note(f"The import stopped: {type(exc).__name__}: {exc}")
        finally:
            status.finished_at = time.time()
            cleanup()

    # The request's context (its id, the audit actor) goes with the work.
    context = contextvars.copy_context()
    try:
        _executor().submit(context.run, task)
    except RuntimeError:  # shutting down
        status.cancel.set()
        status.finished_at = time.time()
        cleanup()
    return status


def wait_for_room(
    has_room: Callable[[int], bool],
    size: int,
    status: BackgroundImport,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Wait until the indexing queue has room for ``size`` bytes; False when the import
    was cancelled meanwhile."""

    while not has_room(size):
        if status.cancel.is_set():
            return False
        sleep(ROOM_WAIT_SECONDS)
    return not status.cancel.is_set()


def status_of(import_id: str, workspace: str) -> BackgroundImport | None:
    """An import of ``workspace``'s (another workspace's reads as unknown)."""

    with _lock:
        found = _imports.get(import_id)
    return found if found is not None and found.workspace == workspace else None


def imports_of(workspace: str) -> list[BackgroundImport]:
    """``workspace``'s remembered imports, newest first."""

    with _lock:
        return [s for s in reversed(_imports.values()) if s.workspace == workspace]


def importing_into(collection: str) -> bool:
    """An import into ``collection`` is still running (its workspace can't be deleted)."""

    with _lock:
        return any(s.collection == collection and s.finished_at is None for s in _imports.values())


def shutdown() -> None:
    """Stop every running import (the lifespan's exit): they stop between documents."""

    global _pool
    with _lock:
        for status in _imports.values():
            if status.finished_at is None:
                status.cancel.set()
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
