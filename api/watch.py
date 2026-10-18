"""Index a folder as it changes (WATCH_DIR, off by default).

A poller, not a filesystem-event listener: it works the same on a Docker bind mount, a
network share and macOS, where inotify-style events are unreliable or absent, and needs
no extra dependency. Every ``WATCH_INTERVAL_SECONDS`` it lists the folder (recursively,
supported types only, dotfiles skipped) and:

- indexes a file the index doesn't have, or one whose size or modification time no
  longer matches what was indexed — through the same backlog, job and sequencing as an
  upload — once it has stayed unchanged for one interval (a file still being copied in
  is not read half-written);
- with ``WATCH_DELETE_MISSING``, deletes the document of a file that has disappeared —
  only documents this watcher has itself seen in the folder, never an upload from
  elsewhere that merely shares nothing with it, and never in a pass that could not read
  the whole folder (a subfolder it may not list, a network mount that dropped).

Only regular files are read: a symbolic link is skipped (it could point anywhere on the
server, ``/dev/zero`` included), and so is a FIFO, socket or device.

Documents are stored by file name, as uploads are: two files with the same name in
different subfolders are one document, so the second is skipped with a warning.
"""

from __future__ import annotations

import logging
import os
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

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
# (name, content, mtime): the document is stamped with the file's mtime as its
# uploaded_at, which is how the next pass knows it is current. Raises HTTPException(503)
# when the backlog is full.
Enqueue = Callable[[str, bytes, float], None]
Delete = Callable[[str], None]


class UnreadableFileError(DocumentError):
    """A watched file that is not a regular file, or grew past the upload limit."""


class FolderScan(NamedTuple):
    """One listing of the folder: supported files by stored name, the problems found (as
    log-ready strings), and whether part of the folder could not be read at all."""

    files: dict[str, FileState]
    problems: list[str]
    incomplete: bool


def scan_folder(root: Path, max_bytes: int) -> FolderScan:
    """Supported regular files under ``root`` by stored name, and the problems found
    (duplicate names, unusable names, links and special files, files over the upload
    limit).

    ``incomplete`` is set when a directory could not be listed or a file could not be
    looked at: the files in it are then missing from ``files`` without having gone
    anywhere, so nothing may be deleted on the strength of this scan.
    """

    found: dict[str, FileState] = {}
    problems: list[str] = []
    incomplete = False

    def unreadable_directory(exc: OSError) -> None:
        nonlocal incomplete
        incomplete = True
        problems.append(f"{exc.filename or root}: cannot be listed ({exc.strerror or exc})")

    # followlinks stays False: a linked directory is not descended into, for the same
    # reason linked files are skipped below.
    for directory, subdirectories, filenames in os.walk(root, onerror=unreadable_directory):
        subdirectories[:] = sorted(d for d in subdirectories if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            path = Path(directory) / name
            try:
                info = path.lstat()  # the link itself, never what it points at
                stored = validate_upload_filename(name)
            except FileNotFoundError:
                continue  # removed since the directory was listed: it is gone
            except OSError as exc:
                incomplete = True
                problems.append(f"{path}: {exc}")
                continue
            except DocumentError as exc:
                problems.append(f"{path}: {exc}")
                continue
            if stat.S_ISLNK(info.st_mode):
                problems.append(f"{path}: a symbolic link; only regular files are indexed")
                continue
            if not stat.S_ISREG(info.st_mode):
                problems.append(f"{path}: not a regular file")
                continue
            if info.st_size > max_bytes:
                problems.append(f"{path}: larger than MAX_UPLOAD_BYTES ({max_bytes})")
                continue
            if stored in found:
                problems.append(f"{path}: another file is already indexed as '{stored}'")
                continue
            found[stored] = FileState(path, info.st_size, info.st_mtime)
    return FolderScan(found, problems, incomplete)


def read_regular_file(path: Path, max_bytes: int) -> bytes:
    """``path``'s bytes, refusing anything but a regular file of at most ``max_bytes``.

    The listing's lstat is not enough on its own: the file can be swapped for a link or
    a FIFO between it and the read. So the open refuses to follow a link (O_NOFOLLOW),
    cannot block on a FIFO (O_NONBLOCK), and what was actually opened is checked again
    before a byte is read; the read itself stops one byte past the limit, so a file that
    grew (or never ends) is never read whole.
    """

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise UnreadableFileError(f"{path} is not a regular file.")
        content = handle.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise UnreadableFileError(f"{path} is larger than MAX_UPLOAD_BYTES ({max_bytes}).")
    return content


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
        # Each file's (size, mtime) in the previous pass: a file is read only once it
        # is the same in two passes in a row, one interval apart.
        self._previous: dict[str, tuple[int, float]] = {}
        # Names this watcher has seen in the folder: the only ones it may delete.
        self._seen: set[str] = set()
        self._reported: set[str] = set()
        # Why the last pass held deletions back, if it did: said once when it starts,
        # not every interval for as long as a mount stays dropped.
        self._standing_down: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _is_current(self, indexed: dict[str, float | None], name: str, mtime: float) -> bool:
        """Whether the indexed document is this exact version of the file.

        The watcher stamps each document's ``uploaded_at`` with the file's mtime when it
        sends it, so equality — not "uploaded after it was modified" — is the test. The
        old ``uploaded_at >= mtime`` missed a file replaced by an older copy (``cp -p``,
        ``rsync -t`` and restores keep the source's mtime), and was stamped when the job
        ran, not when the file was read.

        So every watched document indexed before this (stamped with the time its job
        ran), every document with no stamp at all, and an upload from the UI that shares
        a name with a file in the folder, are each re-indexed once from the folder copy,
        after which they carry the file's mtime and are current.
        """

        return name in indexed and indexed[name] == mtime

    def scan_once(self) -> dict[str, list[str]]:
        """One pass; what it did, by kind (for the log and the tests)."""

        done: dict[str, list[str]] = {"indexed": [], "deleted": [], "deferred": []}
        if not self._root.is_dir():
            logger.warning("WATCH_DIR %s is not a directory; nothing to watch.", self._root)
            return done
        scan = scan_folder(self._root, self._max_bytes)
        files = scan.files
        for problem in scan.problems:
            if problem not in self._reported:  # once, not every interval
                self._reported.add(problem)
                logger.warning("Watched folder: skipping %s", problem)
        indexed = self._list_indexed()
        previous, self._previous = self._previous, {}
        for name, state in files.items():
            self._seen.add(name)
            signature = (state.size, state.mtime)
            self._previous[name] = signature
            if self._is_current(indexed, name, state.mtime) or self._sent.get(name) == signature:
                continue
            if previous.get(name) != signature:
                # New, or changed since the last pass: it may still be being written.
                # Read it next pass if it has stayed the same.
                continue
            try:
                content = read_regular_file(state.path, self._max_bytes)
                self._enqueue(name, content, state.mtime)
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
            self._delete_removed(files, indexed, scan.incomplete, done)
        if any(done.values()):
            logger.info(
                "Watched folder: %d sent for indexing, %d deleted, %d waiting for room.",
                len(done["indexed"]),
                len(done["deleted"]),
                len(done["deferred"]),
            )
        return done

    def _delete_removed(
        self,
        files: dict[str, FileState],
        indexed: dict[str, float | None],
        incomplete: bool,
        done: dict[str, list[str]],
    ) -> None:
        """Delete the documents of watched files that are gone — unless this pass can't
        tell "gone" from "couldn't see it". ``_seen`` is left alone when it stands down,
        so the next complete pass still deletes what really went."""

        missing = sorted(self._seen - files.keys())
        if not missing:
            self._standing_down = None
            return
        if incomplete:
            self._stand_down(
                "incomplete",
                "Watched folder: part of %s could not be read; deleting nothing until it can.",
                self._root,
            )
            return
        if not files:
            # Every watched file gone at once is far more often a network share or bind
            # mount that dropped (leaving an empty mount point) than a real deletion, and
            # acting on it would empty the index. Stays this way while the folder is
            # empty, not just for one pass: a dropped mount stays dropped.
            self._stand_down(
                "empty",
                "Watched folder: %s lists no files at all; not deleting the %d documents "
                "seen there (until it lists files again). Delete them through the API if "
                "that was intended.",
                self._root,
                len(missing),
            )
            return
        self._standing_down = None
        for name in missing:
            if name in indexed:
                try:
                    self._delete(name)
                    done["deleted"].append(name)
                except (DocumentNotFoundError, HTTPException) as exc:
                    logger.warning("Watched folder: could not delete %s: %s", name, exc)
            self._seen.discard(name)
            self._sent.pop(name, None)

    def _stand_down(self, reason: str, message: str, *args: object) -> None:
        if self._standing_down != reason:
            self._standing_down = reason
            logger.warning(message, *args)

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
