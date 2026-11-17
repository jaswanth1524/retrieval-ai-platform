"""Optional storage of original uploaded document bytes.

Off by default (``AppSettings.raw_document_dir`` unset) — every prior release
discards the uploaded bytes once ingestion is queued. Enabling this adds a side
channel: ``GET /documents/{filename}/original`` downloads the exact bytes that were
uploaded, and ``POST /documents/{filename}/reindex`` re-chunks a document from them.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from api.documents import (
    UnsupportedDocumentError,
    normalize_document_name,
    normalize_filename,
)

logger = logging.getLogger(__name__)

_STALE_TEMP_FILE_SECONDS = 3600.0
# What tempfile.mkstemp(prefix=".upload-") produces: 8 characters from
# [a-z0-9_] and no suffix — which no accepted upload name can have.
_TEMP_FILE_NAME = re.compile(r"\.upload-[a-z0-9_]{8}")


# Earlier versions of each original (RAW_DOCUMENT_VERSIONS), in a subdirectory per
# document: ``.versions/<name>/<nanoseconds>``. A directory, so the listing of stored
# originals (files only) never shows it, and a name no upload can have (no extension).
VERSIONS_DIR = ".versions"
# What the index knows about each original and the file itself doesn't (its tags, upload
# time, source URL, watched folder): ``.meta/<name>`` (JSON), written with the original and
# when one of them changes, so originals alone can rebuild a lost index
# (``POST /admin/rebuild``). A directory for the same reason as ``.versions``.
DETAILS_DIR = ".meta"
# Directories of the store's own, never a document's.
OWN_DIRECTORIES = (VERSIONS_DIR, DETAILS_DIR)
_DETAILS_MAX_BYTES = 64 * 1024
_VERSION_ID = re.compile(r"\d{1,20}")


@dataclass(frozen=True)
class StoredVersion:
    """One earlier version of a document's original."""

    version: str
    saved_at: float
    size: int


def is_temp_file_name(name: str) -> bool:
    """Whether ``name`` is one of ``save``'s own temp files, never a stored original."""

    return _TEMP_FILE_NAME.fullmatch(name) is not None


class RawDocumentStore:
    """Filesystem-backed store for original uploaded document bytes.

    Keyed by the same normalized (basename-only) filename ingestion itself uses
    (``api.documents.normalize_filename``), applied here too rather than trusted from
    the caller — defense in depth against a client-supplied filename containing a
    path component.
    """

    def __init__(self, directory: str, keep_versions: int = 0) -> None:
        self._directory = Path(directory)
        # How many earlier versions of each original a save keeps (0: none).
        self._keep_versions = keep_versions
        # Set once the details of originals stored before details were kept have been
        # written (``original_details.record_missing_details``).
        self.details_backfilled = False
        self._directory.mkdir(parents=True, exist_ok=True)
        self._sweep_stale_temp_files()

    def _sweep_stale_temp_files(self) -> None:
        """Remove ``.upload-*`` temp files a crash or kill left mid-save.

        ``save`` removes its temp file on any error it sees, but not when the process
        dies in between; those accumulated forever. An hour old is never a save still
        running in another process sharing the directory.
        """

        cutoff = time.time() - _STALE_TEMP_FILE_SECONDS
        for temp in self._directory.glob(".upload-*"):
            # Only mkstemp's own names: a stored original may be called ".upload-x.txt"
            # (a leading dot is a valid name), and it has a suffix a temp file never has.
            if not _TEMP_FILE_NAME.fullmatch(temp.name):
                continue
            try:
                if temp.is_file() and temp.stat().st_mtime < cutoff:
                    temp.unlink()
            except OSError:
                logger.warning("Could not remove stale temp file %s", temp, exc_info=True)

    def save(self, filename: str, content: bytes) -> None:
        """Write atomically: a crash mid-write leaves the previous original intact."""

        target = self._path_for(filename)
        # Made again if it went: deleting a workspace removes its directory, and a store
        # built for it before (cached, or the folder watcher's) must still work if a
        # workspace of that name is created again.
        self._directory.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=self._directory, prefix=".upload-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            if self._keep_versions > 0 and _is_file(target):
                self._keep_as_version(target, filename, content)
            # A name with folders ("reports/2024/q1.pdf") is kept under them.
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_path, target)
        except BaseException:
            Path(temp_path).unlink(missing_ok=True)
            raise

    def _versions_dir(self, filename: str) -> Path:
        return self._directory / VERSIONS_DIR / _store_name(filename)

    def _keep_as_version(self, current: Path, filename: str, incoming: bytes) -> None:
        """Move the current original into its versions (unless it is the same bytes),
        then drop all but the newest ``keep_versions``. A failure here is logged: the
        new original is saved either way."""

        try:
            if current.stat().st_size == len(incoming) and current.read_bytes() == incoming:
                return
            directory = self._versions_dir(filename)
            directory.mkdir(parents=True, exist_ok=True)
            kept = directory / str(time.time_ns())
            try:
                os.link(current, kept)  # the same bytes, no copy
            except OSError:
                shutil.copy2(current, kept)
            for old in self._version_paths(filename)[self._keep_versions :]:
                old.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not keep the previous version of %s", filename, exc_info=True)

    def _version_paths(self, filename: str) -> list[Path]:
        """A document's version files, newest first."""

        directory = self._versions_dir(filename)
        try:
            paths = [
                p for p in directory.iterdir() if _VERSION_ID.fullmatch(p.name) and p.is_file()
            ]
        except OSError:
            return []
        return sorted(paths, key=lambda path: int(path.name), reverse=True)

    def versions(self, filename: str) -> list[StoredVersion]:
        """The earlier versions kept for ``filename``, newest first."""

        result: list[StoredVersion] = []
        for path in self._version_paths(filename):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            result.append(StoredVersion(path.name, int(path.name) / 1e9, size))
        return result

    def version_size(self, filename: str, version: str) -> int | None:
        """One earlier version's size in bytes, or None when there is no such version."""

        if not _VERSION_ID.fullmatch(version):
            return None
        try:
            path = self._versions_dir(filename) / version
            return path.stat().st_size if path.is_file() else None
        except OSError:
            return None

    def read_version(self, filename: str, version: str) -> bytes | None:
        """One earlier version's bytes, or None when there is no such version."""

        if not _VERSION_ID.fullmatch(version):
            return None
        path = self._versions_dir(filename) / version
        try:
            return path.read_bytes() if path.is_file() else None
        except OSError:
            return None

    def open_version(self, filename: str, version: str) -> BinaryIO | None:
        """An open handle on one earlier version, or None when there is no such version
        (for a backup: read while it is written, never whole into memory)."""

        if not _VERSION_ID.fullmatch(version):
            return None
        try:
            return (self._versions_dir(filename) / version).open("rb")
        except OSError:
            return None

    def add_version(self, filename: str, version: str, content: bytes) -> bool:
        """Put an earlier version back (a restored backup's), under its own id, keeping
        only the newest ``keep_versions``; whether it was kept. Nothing is kept when
        versions aren't (RAW_DOCUMENT_VERSIONS 0) or the id isn't one this store makes."""

        if self._keep_versions <= 0 or not _VERSION_ID.fullmatch(version):
            return False
        directory = self._versions_dir(filename)
        directory.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".restore-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            os.replace(temp_path, directory / version)
        except BaseException:
            Path(temp_path).unlink(missing_ok=True)
            raise
        for old in self._version_paths(filename)[self._keep_versions :]:
            old.unlink(missing_ok=True)
        return (directory / version).is_file()

    def _details_path(self, filename: str) -> Path:
        # Laid out as the originals are (``.meta/<name>``), so a name that is refused
        # as a file-and-folder clash there can't clash here either.
        return self._directory / DETAILS_DIR / _store_name(filename)

    def save_details(self, filename: str, details: dict[str, object]) -> None:
        """Keep ``details`` beside ``filename``'s original (atomically, replacing the
        ones kept before)."""

        target = self._details_path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=target.parent, prefix=".upload-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(details, handle, ensure_ascii=False, sort_keys=True)
            os.replace(temp_path, target)
        except BaseException:
            Path(temp_path).unlink(missing_ok=True)
            raise

    def details(self, filename: str) -> dict[str, object]:
        """What ``save_details`` kept for ``filename``; ``{}`` when nothing was (an
        original from before details were kept) or it can't be read."""

        path = self._details_path(filename)
        try:
            if path.stat().st_size > _DETAILS_MAX_BYTES:
                return {}
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def names_with_details(self) -> set[str]:
        """Names whose details are kept (one walk of their directory)."""

        return _files_under(self._directory / DETAILS_DIR)

    def read(self, filename: str) -> bytes | None:
        """The stored original's bytes, or None when none is stored — including one
        deleted (or made unreadable) since the caller checked for it."""

        path = self.path(filename)
        if path is None:
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def path(self, filename: str) -> Path | None:
        """Path of the stored original, or None when none is stored."""

        path = self._path_for(filename)
        return path if _is_file(path) else None

    def stored_names(self) -> set[str]:
        """Names of every stored original, from one walk of the directory (one read of
        it, when no document is kept in a folder).

        For the document listing, which used to stat each document's path in turn.
        Exact names: with originals stored, uploads that differ only by case or
        Unicode form are refused (``document_names.refuse_near_duplicate_name``), so a
        document's original is stored under its own name.
        """

        return _files_under(self._directory, skip_top=_NOT_DOCUMENT_FOLDERS)

    def versioned_names(self) -> set[str]:
        """Names with a folder of kept earlier versions (one walk of them). A document's
        folder holds only version files, so every folder without subfolders is one
        (empty ones too: what is left of a document's history)."""

        root = self._directory / VERSIONS_DIR
        names: set[str] = set()
        for directory, subdirectories, _ in os.walk(root):
            if directory != str(root) and not subdirectories:
                names.add(Path(directory).relative_to(root).as_posix())
        return names

    def size(self, filename: str) -> int | None:
        """Size in bytes of the stored original, or None when none is stored."""

        path = self.path(filename)
        if path is None:
            return None
        try:
            return path.stat().st_size
        except OSError:
            return None

    def identity(self, filename: str) -> tuple[int, int] | None:
        """(size, mtime in ns) of the stored original, or None when none is stored: what
        tells a replaced original from the one a job already failed on."""

        path = self.path(filename)
        if path is None:
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_size, stat.st_mtime_ns

    def open(self, filename: str) -> BinaryIO | None:
        """An open handle on the stored original, or None when none is stored.

        Opening (rather than returning a path to open later) closes the window in
        which a concurrent DELETE turns "it exists" into FileNotFoundError: an open
        handle keeps reading the bytes it opened even after the file is unlinked.
        """

        path = self.path(filename)
        if path is None:
            return None
        try:
            return path.open("rb")
        except OSError:
            return None

    def delete(self, filename: str) -> bool:
        """Remove the stored original; True when there was one to remove."""

        path = self._path_for(filename)
        # Its earlier versions go with it: they are versions of a document that's gone.
        versions, details = self._versions_dir(filename), self._details_path(filename)
        shutil.rmtree(versions, ignore_errors=True)
        details.unlink(missing_ok=True)
        self._prune_empty_folders(versions.parent, self._directory / VERSIONS_DIR)
        self._prune_empty_folders(details.parent, self._directory / DETAILS_DIR)
        if not _is_file(path):
            return False
        path.unlink(missing_ok=True)
        self._prune_empty_folders(path.parent, self._directory)
        return True

    @staticmethod
    def _prune_empty_folders(folder: Path, stop: Path) -> None:
        """Remove ``folder`` and its parents while they are empty, never ``stop``
        itself: the folders of a deleted document's name don't outlive it."""

        while folder != stop and stop in folder.parents:
            try:
                folder.rmdir()
            except OSError:
                return  # not empty (or already gone)
            folder = folder.parent

    def remove_current(self, filename: str) -> bool:
        """Remove only the current original, keeping its earlier versions: what a failed
        save leaves behind must not describe the newly indexed bytes, but the history
        before them is still right. True when there was one to remove."""

        path = self._path_for(filename)
        if not _is_file(path):
            return False
        path.unlink(missing_ok=True)
        return True

    def _path_for(self, filename: str) -> Path:
        return self._directory / _store_name(filename)


# Top-level folders of the store that hold no documents of its own: its bookkeeping,
# and the other workspaces' stores (``<raw>/workspaces/<name>``). No document name can
# start with one of them (``documents.RESERVED_FIRST_FOLDERS``, hidden folders).
_NOT_DOCUMENT_FOLDERS = frozenset({*OWN_DIRECTORIES, "workspaces"})


def _files_under(root: Path, skip_top: frozenset[str] = frozenset()) -> set[str]:
    """The path of every file under ``root``, relative to it with ``/`` between parts;
    ``save``'s temp files and the top-level folders in ``skip_top`` left out. A folder
    that can't be read is logged and counts as empty."""

    names: set[str] = set()
    pending = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if _TEMP_FILE_NAME.fullmatch(entry.name):
                        continue
                    try:
                        if entry.is_file():
                            names.add(prefix + entry.name)
                        elif entry.is_dir(follow_symlinks=False) and not (
                            not prefix and entry.name in skip_top
                        ):
                            pending.append((Path(entry.path), f"{prefix}{entry.name}/"))
                    except OSError:
                        continue
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning("Cannot list stored originals in %s", directory, exc_info=True)
    return names


def _store_name(filename: str) -> str:
    """``filename`` as stored: a valid name (folders included) as it is; anything else
    by its base name, as before names could have folders — so a name with ``..`` in it
    can only ever reach a file directly in the store (defense in depth: the routes
    refuse such a name before it gets here)."""

    try:
        return normalize_document_name(filename)
    except UnsupportedDocumentError:
        return normalize_filename(filename)


def _is_file(path: Path) -> bool:
    """``path.is_file()``, but a name the OS can't even look up is simply absent.

    On Python 3.12 ``is_file`` raises for ENAMETOOLONG (3.13 returns False), so a
    document indexed under a name longer than the filesystem allows made every listing
    that checks for its original fail.
    """

    try:
        return path.is_file()
    except OSError as exc:
        logger.warning("Cannot look up stored original %s: %s", path.name[:80], exc)
        return False
