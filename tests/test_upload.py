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


class CountingFile(BytesIO):
    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self.reads: list[int] = []

    def read(self, size: int | None = -1) -> bytes:
        data = super().read(size)
        self.reads.append(len(data))
        return data


@pytest.mark.asyncio
async def test_a_known_size_is_read_in_one_call_and_refused_unread_when_too_big() -> None:
    fitting = CountingFile(b"x" * (3 * 1024 * 1024))
    content = await read_upload_within_limit(
        UploadFile(file=fitting, size=3 * 1024 * 1024, headers=Headers({})), max_bytes=10**8
    )
    assert content == b"x" * (3 * 1024 * 1024)
    assert fitting.reads == [3 * 1024 * 1024]

    oversized = CountingFile(b"x" * 100)
    with pytest.raises(UploadTooLargeError):
        await read_upload_within_limit(
            UploadFile(file=oversized, size=100, headers=Headers({})), max_bytes=10
        )
    assert oversized.reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [None, 4])
async def test_a_missing_or_short_size_still_reads_everything_within_the_limit(
    declared: int | None,
) -> None:
    upload = UploadFile(file=BytesIO(b"y" * 3000), size=declared, headers=Headers({}))
    assert await read_upload_within_limit(upload, max_bytes=5000) == b"y" * 3000

    upload = UploadFile(file=BytesIO(b"y" * 3000), size=declared, headers=Headers({}))
    with pytest.raises(UploadTooLargeError):
        await read_upload_within_limit(upload, max_bytes=2999)
