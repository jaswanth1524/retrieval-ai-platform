"""A streamed response that always closes what it streams."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterator, Mapping

from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse
from starlette.types import Message, Receive, Scope, Send


class ClosingStreamingResponse(StreamingResponse):
    """A ``StreamingResponse`` over a sync iterator that closes it however the response
    ends: finished, failed, or the client gone.

    Starlette never closes a sync iterator itself. On a disconnect it cancels the send
    loop and drops the iterator, so a generator's ``finally`` (stop the pipeline
    thread, release a question slot, close a file) waited for garbage collection —
    measured at ~30 s in the container for an export's open original. ``close`` runs
    after the response has fully ended, when no worker thread is inside the iterator
    any more (Starlette's threadpool step isn't abandoned on cancellation), so it can't
    race a ``next()``.
    """

    def __init__(
        self,
        content: Iterator[str] | Iterator[bytes],
        *,
        close: Callable[[], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: BackgroundTask | None = None,
    ) -> None:
        super().__init__(
            content,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
            background=background,
        )
        self._close = close if close is not None else getattr(content, "close", None)
        self._on_disconnect = on_disconnect

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # ``on_disconnect`` runs the moment the client is known to have left. Starlette
        # does cancel the send loop then, but its cleanup waits for the threadpool step
        # in flight, and on a quiet stream (a question waiting for the local model) that
        # step is blocked until the next keepalive, up to 15 s: the question kept its
        # slot, and the pipeline its place in the model's queue, that long after the
        # reader had gone. Before ASGI 2.4 Starlette listens itself (through this
        # wrapped ``receive``); from 2.4 it only notices a failed send, so listen here.
        watcher = None
        if self._on_disconnect is not None:
            receive = _noting_disconnect(receive, self._on_disconnect)
            if _spec_version(scope) >= (2, 4):
                watcher = asyncio.create_task(_until_disconnect(receive))
        try:
            await super().__call__(scope, receive, send)
        finally:
            if watcher is not None:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
            if self._close is not None:
                self._close()


def _spec_version(scope: Scope) -> tuple[int, ...]:
    return tuple(map(int, scope.get("asgi", {}).get("spec_version", "2.0").split(".")))


def _noting_disconnect(receive: Receive, on_disconnect: Callable[[], None]) -> Receive:
    async def wrapped() -> Message:
        message = await receive()
        if message["type"] == "http.disconnect":
            on_disconnect()
        return message

    return wrapped


async def _until_disconnect(receive: Receive) -> None:
    # The request body was read before the response started: what comes now is the
    # client leaving.
    while (await receive())["type"] != "http.disconnect":
        pass
