"""Optional storage of original uploaded document bytes.

Off by default (``AppSettings.raw_document_dir`` unset) — every prior release
discards the uploaded bytes once ingestion is queued. Enabling this adds a side
channel: ``GET /documents/{filename}/original`` downloads the exact bytes that were
uploaded. Re-indexing from these originals (so a future chunking/embedding change
could apply without asking users to re-upload) is not implemented — this module only
stores and serves the bytes.
"""

from __future__ import annotations

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
        self._path_for(filename).write_bytes(content)

    def read(self, filename: str) -> bytes | None:
        path = self._path_for(filename)
        if not path.is_file():
            return None
        return path.read_bytes()

    def delete(self, filename: str) -> None:
        self._path_for(filename).unlink(missing_ok=True)

    def _path_for(self, filename: str) -> Path:
        return self._directory / normalize_filename(filename)
