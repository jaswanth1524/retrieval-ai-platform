"""Optional storage of original uploaded document bytes.

Off by default (``AppSettings.raw_document_dir`` unset) — every prior release
discards the uploaded bytes once ingestion is queued. Enabling this adds a side
channel: ``GET /documents/{filename}/original`` downloads the exact bytes that were
uploaded, and ``POST /documents/{filename}/reindex`` re-chunks a document from them.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from api.documents import normalize_filename

logger = logging.getLogger(__name__)

_STALE_TEMP_FILE_SECONDS = 3600.0
# What tempfile.mkstemp(prefix=".upload-") produces: 8 characters from
# [a-z0-9_] and no suffix — which no accepted upload name can have.
_TEMP_FILE_NAME = re.compile(r"\.upload-[a-z0-9_]{8}")


# Earlier versions of each original (RAW_DOCUMENT_VERSIONS), in a subdirectory per
# document: ``.versions/<name>/<nanoseconds>``. A directory, so the listing of stored
# originals (files only) never shows it, and a name no upload can have (no extension).
VERSIONS_DIR = ".versions"
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
                self._keep_as_version(target, content)
            os.replace(temp_path, target)
        except BaseException:
            Path(temp_path).unlink(missing_ok=True)
            raise

    def _versions_dir(self, filename: str) -> Path:
        return self._directory / VERSIONS_DIR / normalize_filename(filename)

    def _keep_as_version(self, current: Path, incoming: bytes) -> None:
        """Move the current original into its versions (unless it is the same bytes),
        then drop all but the newest ``keep_versions``. A failure here is logged: the
        new original is saved either way."""

        try:
            if current.stat().st_size == len(incoming) and current.read_bytes() == incoming:
                return
            directory = self._versions_dir(current.name)
            directory.mkdir(parents=True, exist_ok=True)
            kept = directory / str(time.time_ns())
            try:
                os.link(current, kept)  # the same bytes, no copy
            except OSError:
                shutil.copy2(current, kept)
            for old in self._version_paths(current.name)[self._keep_versions :]:
                old.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not keep the previous version of %s", current.name, exc_info=True)

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

    def read_version(self, filename: str, version: str) -> bytes | None:
        """One earlier version's bytes, or None when there is no such version."""

        if not _VERSION_ID.fullmatch(version):
            return None
        path = self._versions_dir(filename) / version
        try:
            return path.read_bytes() if path.is_file() else None
        except OSError:
            return None

    def read(self, filename: str) -> bytes | None:
        path = self.path(filename)
        return path.read_bytes() if path is not None else None

    def path(self, filename: str) -> Path | None:
        """Path of the stored original, or None when none is stored."""

        path = self._path_for(filename)
        return path if _is_file(path) else None

    def stored_names(self) -> set[str]:
        """Names of every stored original, from one directory read.

        For the document listing, which used to stat each document's path in turn.
        Exact names: with originals stored, uploads that differ only by case or
        Unicode form are refused (``ingest_jobs.refuse_near_duplicate_name``), so a
        document's original is stored under its own name.
        """

        names: set[str] = set()
        try:
            with os.scandir(self._directory) as entries:
                for entry in entries:
                    if _TEMP_FILE_NAME.fullmatch(entry.name):
                        continue
                    try:
                        if entry.is_file():
                            names.add(entry.name)
                    except OSError:
                        continue
        except OSError:
            logger.warning("Cannot list stored originals in %s", self._directory, exc_info=True)
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
        shutil.rmtree(self._versions_dir(filename), ignore_errors=True)
        if not _is_file(path):
            return False
        path.unlink(missing_ok=True)
        return True

    def _path_for(self, filename: str) -> Path:
        return self._directory / normalize_filename(filename)


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
