"""ClosingStreamingResponse: the streamed iterator is closed however the response ends."""

from __future__ import annotations

import asyncio
import gc
import threading
from collections.abc import Generator, Iterator

import pytest
from starlette.types import Message

from api.streaming import ClosingStreamingResponse


def _scope(spec_version: str) -> dict[str, object]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "scheme": "http",
        "server": ("test", 80),
        "client": ("client", 1),
    }


def _endless(closed: threading.Event) -> Generator[str]:
    try:
        count = 0
        while True:
            count += 1
            yield f"data: {count}\n\n"
    finally:
        closed.set()


async def _serve_until_disconnect(response: ClosingStreamingResponse, spec_version: str) -> None:
    """Send the response until three body frames went out, then the client leaves:
    spec 2.3 says so through receive(), 2.4 by send() raising, as uvicorn does."""

    bodies = 0
    gone = asyncio.Event()

    async def receive() -> Message:
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        nonlocal bodies
        if gone.is_set() and spec_version == "2.4":
            raise OSError("connection reset")
        if message["type"] == "http.response.body":
            bodies += 1
            if bodies >= 3:
                gone.set()

    try:
        await response(_scope(spec_version), receive, send)
    except Exception:
        pass  # 2.4: Starlette raises ClientDisconnect


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_a_client_that_leaves_closes_the_iterator_without_waiting_for_gc(
    spec_version: str,
) -> None:
    closed = threading.Event()
    gc.disable()  # a plain StreamingResponse passed only once a collection ran
    try:
        asyncio.run(
            _serve_until_disconnect(ClosingStreamingResponse(_endless(closed)), spec_version)
        )
        assert closed.is_set()
    finally:
        gc.enable()


def test_a_finished_response_closes_the_iterator_too() -> None:
    closed = threading.Event()
    calls: list[str] = []

    def three() -> Iterator[bytes]:
        try:
            yield from (b"a", b"b", b"c")
        finally:
            closed.set()

    async def receive() -> Message:
        await asyncio.Event().wait()
        raise AssertionError  # never reached

    async def send(message: Message) -> None:
        calls.append(message["type"])

    asyncio.run(ClosingStreamingResponse(three())(_scope("2.4"), receive, send))

    assert closed.is_set()
    assert calls[-1] == "http.response.body"


def test_an_explicit_close_callback_replaces_the_iterators_own() -> None:
    closed = threading.Event()
    called = threading.Event()

    async def receive() -> Message:
        await asyncio.Event().wait()
        raise AssertionError

    async def send(message: Message) -> None:
        return None

    response = ClosingStreamingResponse(iter([b"x"]), close=called.set)
    asyncio.run(response(_scope("2.4"), receive, send))

    assert called.is_set()
    assert not closed.is_set()
