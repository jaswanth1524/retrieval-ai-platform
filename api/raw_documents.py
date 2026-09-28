"""Optional storage of original uploaded document bytes.

Off by default (``AppSettings.raw_document_dir`` unset) — every prior release
discards the uploaded bytes once ingestion is queued. Enabling this adds a side
channel: ``GET /documents/{filename}/original`` downloads the exact bytes that were
uploaded. Re-indexing from these originals (so a future chunking/embedding change
could apply without asking users to re-upload) is not implemented — this module only
stores and serves the bytes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from api.documents import normalize_filename


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
        return path if path.is_file() else None

    def delete(self, filename: str) -> bool:
        """Remove the stored original; True when there was one to remove."""

        path = self._path_for(filename)
        if not path.is_file():
            return False
        path.unlink(missing_ok=True)
        return True

    def _path_for(self, filename: str) -> Path:
        return self._directory / normalize_filename(filename)
