"""Bounded reading of uploaded files so a single request cannot exhaust memory."""

from __future__ import annotations

from fastapi import UploadFile

_READ_CHUNK_BYTES = 1024 * 1024


class UploadTooLargeError(RuntimeError):
    """Raised when an uploaded file exceeds the configured size limit."""


async def read_upload_within_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload incrementally, aborting before buffering past ``max_bytes``.

    ``UploadFile`` in this Starlette version exposes no reliable pre-read size, so the
    only safe cap is to read in bounded chunks and stop the moment the running total
    would exceed the limit — never buffering the full oversized body into memory.
    """

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError(
                f"Upload exceeds the maximum allowed size of {max_bytes} bytes."
            )
        chunks.append(chunk)
    return b"".join(chunks)
