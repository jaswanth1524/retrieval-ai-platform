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


def test_a_quiet_stream_hears_the_client_leave_at_once_under_asgi_2_4() -> None:
    """Nothing is sent while a question waits for the model, so a failed send can't tell
    Starlette the client left: the response listens for it and stops the work."""

    stop = threading.Event()
    left_at: list[float] = []

    def quiet() -> Iterator[str]:
        yield "data: started\n\n"
        stop.wait(10)  # the pipeline waiting; ends when stopped

    async def scenario() -> None:
        loop = asyncio.get_running_loop()

        async def receive() -> Message:
            await asyncio.sleep(0.2)
            left_at.append(loop.time())
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            return None

        response = ClosingStreamingResponse(quiet(), on_disconnect=stop.set)
        started = loop.time()
        await response(_scope("2.4"), receive, send)
        assert loop.time() - started < 3

    asyncio.run(scenario())
    assert stop.is_set() and left_at


def test_the_client_leaving_stops_the_work_at_once_under_asgi_2_3_too() -> None:
    """Starlette hears the disconnect here, but its cleanup waits for the step in flight
    (blocked until the next keepalive): the stop must not wait with it."""

    stop = threading.Event()
    stopped_after: list[float] = []

    def quiet() -> Iterator[str]:
        yield "data: started\n\n"
        started = __import__("time").monotonic()
        stop.wait(10)
        stopped_after.append(__import__("time").monotonic() - started)

    async def scenario() -> None:
        async def receive() -> Message:
            await asyncio.sleep(0.2)
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            return None

        await ClosingStreamingResponse(quiet(), on_disconnect=stop.set)(
            _scope("2.3"), receive, send
        )

    asyncio.run(scenario())
    assert stop.is_set() and stopped_after and stopped_after[0] < 3
