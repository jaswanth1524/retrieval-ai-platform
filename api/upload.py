"""Bounded reading of uploaded files so a single request cannot exhaust memory."""

from __future__ import annotations

from fastapi import UploadFile

_READ_CHUNK_BYTES = 1024 * 1024


class UploadTooLargeError(RuntimeError):
    """Raised when an uploaded file exceeds the configured size limit."""


async def read_upload_within_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, refusing it before buffering past ``max_bytes``.

    When the multipart parser recorded the part's size, an oversized upload is refused
    without reading it and a fitting one is read in a single call: collecting chunks
    and joining them held every upload in memory twice. The size is still checked
    against what was read, and a size that turns out short (or none at all) falls back
    to reading in bounded chunks, stopping the moment the total passes the limit.
    """

    chunks: list[bytes] = []
    total = 0
    size = file.size
    if isinstance(size, int) and size >= 0:
        if size > max_bytes:
            raise _too_large(max_bytes)
        data = await file.read(size + 1)
        if len(data) <= size:
            return data
        chunks.append(data)
        total = len(data)
    while True:
        if total > max_bytes:
            raise _too_large(max_bytes)
        chunk = await file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        chunks.append(chunk)
    if total > max_bytes:
        raise _too_large(max_bytes)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _too_large(max_bytes: int) -> UploadTooLargeError:
    return UploadTooLargeError(f"Upload exceeds the maximum allowed size of {max_bytes} bytes.")
