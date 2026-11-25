"""Zips and backups imported in the background (``background=true`` on
``POST /documents/archive``, ``POST /import`` and ``POST /backups/{name}/restore``).

A zip or a backup holds more documents than the indexing queue does (50 jobs at once by
default), so a request imported only the first batch and listed the rest as ``deferred``
for the caller to send again. In the background, the upload is spooled to a temporary
file and its documents are fed to the queue as room frees — waiting instead of
deferring — one import at a time. ``GET /documents/imports[/{id}]`` follows one and
``DELETE`` stops it (documents already queued stay queued). Kept in memory, the newest
``MAX_IMPORTS_KEPT`` of them: a restart forgets them (and stops a running one).

With ``IMPORT_SPOOL_DIR`` set, the spooled file is kept there with a small record of
what it is (``remember``), and an import a restart cut off runs again at startup
(``main.resume_imports``): documents it had already queued are indexed or skipped as
already there, so running it twice is harmless.
"""

from __future__ import annotations

import contextvars
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from fastapi import UploadFile

from api.notify import notify

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
    # It ended on an error (a backup that isn't one, an index it can't write to):
    # nothing, or not everything, was queued.
    errored: bool = False
    # Started again at startup, after a restart cut the first run off.
    resumed: bool = False
    # Stopped by the server shutting down, not by anyone asking.
    interrupted: bool = False
    # Its turn on the import thread has come (one runs at a time).
    began: bool = False

    @property
    def state(self) -> str:
        if self.finished_at is None:
            if self.cancel.is_set():
                return "cancelling"
            # Behind another import on the one import thread: not running yet.
            return "running" if self.began else "waiting"
        if self.errored:
            return "failed"
        return "cancelled" if self.cancel.is_set() else "done"

    def note(self, problem: str) -> None:
        if len(self.problems) < _MAX_PROBLEMS:
            self.problems.append(problem[:300])


_lock = threading.Lock()
_imports: OrderedDict[str, BackgroundImport] = OrderedDict()
_pool: ThreadPoolExecutor | None = None
# Imports submitted and not yet started, with their cleanup.
_waiting: list[tuple[BackgroundImport, Callable[[], None]]] = []


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
    *,
    resumable: bool = False,
    resumed: bool = False,
) -> BackgroundImport:
    """Run ``run(status)`` on the import thread; ``cleanup`` (the spooled file's removal)
    runs after it however it ends, or at once when it can't start. A ``resumable`` one
    the server's shutdown stops keeps its spooled file, for the next start."""

    status = BackgroundImport(
        id=str(uuid.uuid4()),
        kind=kind,
        name=name,
        workspace=workspace,
        collection=collection,
        started_at=time.time(),
        resumed=resumed,
    )
    with _lock:
        _imports[status.id] = status
        while len(_imports) > MAX_IMPORTS_KEPT:
            oldest = next((k for k, v in _imports.items() if v.finished_at is not None), None)
            if oldest is None:
                break
            del _imports[oldest]

    def task() -> None:
        with _lock:
            if entry in _waiting:
                _waiting.remove(entry)
            else:
                return  # shut down before its turn (its cleanup ran there)
        status.began = True
        try:
            # Cancelled while it waited its turn: nothing to run.
            if not status.cancel.is_set():
                run(status)
        except Exception as exc:  # noqa: BLE001 - reported on the import, not raised
            logger.exception("Background %s import %s failed.", kind, name)
            # An HTTPException carries the message a request would have answered with.
            reason = getattr(exc, "detail", None) or f"{type(exc).__name__}: {exc}"
            status.errored = True
            status.note(f"The import stopped: {reason}")
        finally:
            # Cleaned up first: a finished import's spooled file is gone.
            try:
                if not (resumable and status.interrupted):
                    cleanup()
            finally:
                status.finished_at = time.time()
                _report(status)

    # The request's context (its id, the audit actor) goes with the work.
    context = contextvars.copy_context()
    entry = (status, (lambda: None) if resumable else cleanup)
    with _lock:
        _waiting.append(entry)
    try:
        _executor().submit(context.run, task)
    except RuntimeError:  # shutting down
        with _lock:
            if entry in _waiting:
                _waiting.remove(entry)
        status.cancel.set()
        status.finished_at = time.time()
        cleanup()
    return status


def _report(status: BackgroundImport) -> None:
    """Tell NOTIFY_URL how a background import ended: ``import_failed`` when it stopped
    on an error or some documents failed, else ``import_finished`` — they run for
    minutes to hours, and polling was the only way to find out. A shutdown's stop is
    not an ending (the import resumes at the next start)."""

    if status.interrupted:
        return
    where = f" in workspace '{status.workspace}'" if status.workspace else ""
    summary = f"{status.queued} queued, {status.skipped} skipped, {status.failed} failed" + (
        f"; {status.problems[0]}" if status.problems else ""
    )
    try:
        if status.errored or status.failed:
            notify(
                "import_failed",
                f"The {status.kind} import '{status.name}'{where} {status.state}: {summary}",
            )
        else:
            notify(
                "import_finished",
                f"The {status.kind} import '{status.name}'{where} {status.state}: {summary}",
            )
    except Exception:  # noqa: BLE001 - a notification never fails the import it reports
        logger.warning("Could not report the end of import %s.", status.id, exc_info=True)


def spool_directory(setting: str) -> Path | None:
    """IMPORT_SPOOL_DIR as a directory that exists, or None when it isn't set."""

    if not setting:
        return None
    directory = Path(setting)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


_RECORD_SUFFIX = ".import.json"
# A spooled file with no record is a copy that never finished (or whose import did):
# removed once it is this old.
STALE_SPOOL_SECONDS = 3600.0


def remember(directory: Path, record: dict[str, object], spool: Path | None = None) -> Path:
    """Write what a restart needs to run this import again: ``record`` (its kind, name,
    workspace, arguments) beside ``spool`` — or on its own, for an import that reads a
    file kept elsewhere (a stored backup). The record's path."""

    path = (
        spool.with_name(spool.name + _RECORD_SUFFIX)
        if spool is not None
        else directory / f"{uuid.uuid4().hex}{_RECORD_SUFFIX}"
    )
    stored = {**record, "spool": spool.name if spool is not None else None}
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(stored), encoding="utf-8")
    os.replace(partial, path)
    return path


def leftover(directory: Path) -> list[tuple[Path, dict[str, object], Path | None]]:
    """The imports a restart cut off: each record's path, its contents and its spooled
    file (None when it has none). A record whose spooled file is gone, or that can't
    be read, is removed; so are spooled files with no record, once stale."""

    found: list[tuple[Path, dict[str, object], Path | None]] = []
    recorded: set[str] = set()
    for path in sorted(directory.glob(f"*{_RECORD_SUFFIX}")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        if not isinstance(record, dict):
            path.unlink(missing_ok=True)
            continue
        spool_name = record.get("spool")
        spool = directory / Path(str(spool_name)).name if spool_name else None
        if spool is not None and not spool.is_file():
            path.unlink(missing_ok=True)
            continue
        if spool is not None:
            recorded.add(spool.name)
        found.append((path, record, spool))
    cutoff = time.time() - STALE_SPOOL_SECONDS
    for stray in directory.glob("docrag-*.zip"):
        try:
            if stray.name not in recorded and stray.stat().st_mtime < cutoff:
                stray.unlink()
        except OSError:
            logger.warning("Could not remove the stale spooled import %s.", stray)
    return found


def spool_upload(source: IO[bytes], prefix: str, directory: Path | None = None) -> Path:
    """``source`` copied to a temporary file the import thread reads after the request
    (in ``directory``: IMPORT_SPOOL_DIR, else the system's); a copy that fails (a full
    disk) leaves no partial file behind."""

    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".zip", delete=False, dir=directory)
    path = Path(handle.name)
    try:
        with handle:
            shutil.copyfileobj(source, handle, length=1 << 20)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def adopt_upload(upload: UploadFile) -> IO[bytes]:
    """The file an upload was received into, taken over for the import thread (which
    closes it, and so removes it): the request's own cleanup closes a stand-in. Saves a
    second copy of a large upload in the same temp directory."""

    held = upload.file
    upload.file = io.BytesIO()
    held.seek(0)
    return held


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
                status.interrupted = True
                status.cancel.set()
        pool, _pool = _pool, None
        waiting, _waiting[:] = list(_waiting), []
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    # An import still queued behind the one worker never ran: it ends here, and its
    # spooled file goes with it.
    for status, cleanup in waiting:
        if status.finished_at is None:
            status.finished_at = time.time()
            cleanup()
