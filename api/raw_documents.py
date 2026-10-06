"""Optional storage of original uploaded document bytes.

Off by default (``AppSettings.raw_document_dir`` unset) — every prior release
discards the uploaded bytes once ingestion is queued. Enabling this adds a side
channel: ``GET /documents/{filename}/original`` downloads the exact bytes that were
uploaded, and ``POST /documents/{filename}/reindex`` re-chunks a document from them.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import BinaryIO

from api.documents import normalize_filename

logger = logging.getLogger(__name__)

_STALE_TEMP_FILE_SECONDS = 3600.0


class RawDocumentStore:
    """Filesystem-backed store for original uploaded document bytes.

    Keyed by the same normalized (basename-only) filename ingestion itself uses
    (``api.documents.normalize_filename``), applied here too rather than trusted from
    the caller — defense in depth against a client-supplied filename containing a
    path component.
    """

    def __init__(self, directory: str) -> None:
        self._directory = Path(directory)
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
            try:
                if temp.is_file() and temp.stat().st_mtime < cutoff:
                    temp.unlink()
            except OSError:
                logger.warning("Could not remove stale temp file %s", temp, exc_info=True)

    def save(self, filename: str, content: bytes) -> None:
        """Write atomically: a crash mid-write leaves the previous original intact."""

        target = self._path_for(filename)
        fd, temp_path = tempfile.mkstemp(dir=self._directory, prefix=".upload-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            os.replace(temp_path, target)
        except BaseException:
            Path(temp_path).unlink(missing_ok=True)
            raise

    def read(self, filename: str) -> bytes | None:
        path = self.path(filename)
        return path.read_bytes() if path is not None else None

    def path(self, filename: str) -> Path | None:
        """Path of the stored original, or None when none is stored."""

        path = self._path_for(filename)
        return path if _is_file(path) else None

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
