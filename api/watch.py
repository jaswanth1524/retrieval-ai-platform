"""Index a folder as it changes (WATCH_DIR, off by default).

A poller, not a filesystem-event listener: it works the same on a Docker bind mount, a
network share and macOS, where inotify-style events are unreliable or absent, and needs
no extra dependency. Every ``WATCH_INTERVAL_SECONDS`` it lists the folder (recursively,
supported types only, dotfiles skipped) and:

- indexes a file the index doesn't have, or one modified since its document was
  uploaded — through the same backlog, job and sequencing as an upload;
- with ``WATCH_DELETE_MISSING``, deletes the document of a file that has disappeared —
  only documents this watcher has itself seen in the folder, never an upload from
  elsewhere that merely shares nothing with it.

Documents are stored by file name, as uploads are: two files with the same name in
different subfolders are one document, so the second is skipped with a warning.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from api.documents import (
    SUPPORTED_EXTENSIONS,
    DocumentError,
    DocumentNotFoundError,
    validate_upload_filename,
)
from api.settings import AppSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileState:
    path: Path
    size: int
    mtime: float


# What the watcher needs from the app, as plain callables, so it is testable without one.
ListIndexed = Callable[[], dict[str, float | None]]  # filename -> uploaded_at
Enqueue = Callable[[str, bytes], None]  # raises HTTPException(503) when the backlog is full
Delete = Callable[[str], None]


def scan_folder(root: Path, max_bytes: int) -> tuple[dict[str, FileState], list[str]]:
    """Supported files under ``root`` by stored name, and the problems found (duplicate
    names, unusable names, files over the upload limit), as log-ready strings."""

    found: dict[str, FileState] = {}
    problems: list[str] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = sorted(d for d in subdirectories if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            path = Path(directory) / name
            try:
                stat = path.stat()
                stored = validate_upload_filename(name)
            except (OSError, DocumentError) as exc:
                problems.append(f"{path}: {exc}")
                continue
            if stat.st_size > max_bytes:
                problems.append(f"{path}: larger than MAX_UPLOAD_BYTES ({max_bytes})")
                continue
            if stored in found:
                problems.append(f"{path}: another file is already indexed as '{stored}'")
                continue
            found[stored] = FileState(path, stat.st_size, stat.st_mtime)
    return found, problems


class FolderWatcher:
    """Keeps one workspace's index in step with one folder (the settings it is given
    point at that workspace)."""

    def __init__(
        self,
        settings: AppSettings,
        list_indexed: ListIndexed,
        enqueue: Enqueue,
        delete: Delete,
    ) -> None:
        self._root = Path(settings.watch_dir)
        self._interval = float(settings.watch_interval_seconds)
        self._delete_missing = bool(settings.watch_delete_missing)
        self._max_bytes = int(settings.max_upload_bytes)
        self._list_indexed = list_indexed
        self._enqueue = enqueue
        self._delete = delete
        # What was last sent for each name (size, mtime): changed files are re-sent once,
        # not on every pass while their job runs.
        self._sent: dict[str, tuple[int, float]] = {}
        # Names this watcher has seen in the folder: the only ones it may delete.
        self._seen: set[str] = set()
        self._reported: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def scan_once(self) -> dict[str, list[str]]:
        """One pass; what it did, by kind (for the log and the tests)."""

        done: dict[str, list[str]] = {"indexed": [], "deleted": [], "deferred": []}
        if not self._root.is_dir():
            logger.warning("WATCH_DIR %s is not a directory; nothing to watch.", self._root)
            return done
        files, problems = scan_folder(self._root, self._max_bytes)
        for problem in problems:
            if problem not in self._reported:  # once, not every interval
                self._reported.add(problem)
                logger.warning("Watched folder: skipping %s", problem)
        indexed = self._list_indexed()
        for name, state in files.items():
            self._seen.add(name)
            signature = (state.size, state.mtime)
            uploaded_at = indexed.get(name)
            current = name in indexed and (uploaded_at is None or uploaded_at >= state.mtime)
            if current or self._sent.get(name) == signature:
                continue
            try:
                content = state.path.read_bytes()
                self._enqueue(name, content)
            except HTTPException as exc:
                if exc.status_code == 503:  # the backlog is full: next pass
                    done["deferred"].append(name)
                    continue
                logger.warning("Watched folder: %s was refused: %s", state.path, exc.detail)
            except (OSError, DocumentError) as exc:
                logger.warning("Watched folder: could not index %s: %s", state.path, exc)
            self._sent[name] = signature
            done["indexed"].append(name)
        if self._delete_missing:
            for name in sorted(self._seen - files.keys()):
                if name in indexed:
                    try:
                        self._delete(name)
                        done["deleted"].append(name)
                    except (DocumentNotFoundError, HTTPException) as exc:
                        logger.warning("Watched folder: could not delete %s: %s", name, exc)
                self._seen.discard(name)
                self._sent.pop(name, None)
        if any(done.values()):
            logger.info(
                "Watched folder: %d sent for indexing, %d deleted, %d waiting for room.",
                len(done["indexed"]),
                len(done["deleted"]),
                len(done["deferred"]),
            )
        return done

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception:  # noqa: BLE001 - the watcher must outlive one bad pass
                logger.exception("Watched folder: a scan failed; trying again next interval.")
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="docrag-watch", daemon=True)
        self._thread.start()
        logger.info("Watching %s every %gs.", self._root, self._interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
