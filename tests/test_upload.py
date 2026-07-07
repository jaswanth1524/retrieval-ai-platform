from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from api.upload import UploadTooLargeError, read_upload_within_limit


def make_upload(content: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(content), size=len(content), headers=Headers({}))


@pytest.mark.asyncio
async def test_read_upload_within_limit_returns_full_content_under_limit() -> None:
    upload = make_upload(b"hello world")

    content = await read_upload_within_limit(upload, max_bytes=1024)

    assert content == b"hello world"


@pytest.mark.asyncio
async def test_read_upload_within_limit_rejects_content_over_limit() -> None:
    upload = make_upload(b"x" * 100)

    with pytest.raises(UploadTooLargeError, match="exceeds"):
        await read_upload_within_limit(upload, max_bytes=10)


@pytest.mark.asyncio
async def test_read_upload_within_limit_accepts_content_exactly_at_limit() -> None:
    upload = make_upload(b"x" * 10)

    content = await read_upload_within_limit(upload, max_bytes=10)

    assert content == b"x" * 10
