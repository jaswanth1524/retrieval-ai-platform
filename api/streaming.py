"""A streamed response that always closes what it streams."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send


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

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self._close is not None:
                self._close()
