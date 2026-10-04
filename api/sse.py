"""Server-Sent Events plumbing: keepalive heartbeat and question-slot hand-off."""

from __future__ import annotations

import contextvars
import json
import queue
import threading
from collections.abc import Callable, Iterator

from api.admission import Release

# Comment frames sent while the pipeline is quiet (retrieval and a cold model can take
# a minute before the first token): a reverse proxy's read timeout (nginx: 60s) would
# otherwise cut the stream, and the frontend's idle timer resets on any bytes.
SSE_KEEPALIVE_SECONDS = 15.0


SSE_KEEPALIVE_FRAME = ": ping\n\n"


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


_END_OF_STREAM = object()


def with_keepalive[T](
    source: Iterator[T],
    interval: float,
    *,
    stop: threading.Event | None = None,
    on_start: Callable[[], None] | None = None,
    on_finish: Callable[[], None] | None = None,
) -> Iterator[T | None]:
    """Yield ``source``'s items, and ``None`` whenever ``interval`` passes without one.

    ``source`` runs on its own thread so a blocking step inside it (retrieval, the
    first token) can't stop the heartbeat. When the consumer stops early (the client
    disconnected), ``stop`` is set and the producer stops at its next item and closes
    ``source``, so an abandoned answer doesn't keep generating. ``on_start`` runs just
    before the thread starts and ``on_finish`` once it has closed ``source`` — the
    point where its work has really ended, which can be well after the consumer left.
    The thread runs in a copy of the caller's context, so context variables (a
    request id) carry over.
    """

    items: queue.Queue[object] = queue.Queue()
    stop = stop if stop is not None else threading.Event()

    def produce() -> None:
        try:
            for item in source:
                if stop.is_set():
                    break
                items.put(item)
        except BaseException as exc:
            items.put(exc)
        finally:
            try:
                close = getattr(source, "close", None)
                if close is not None:
                    close()
            finally:
                if on_finish is not None:
                    on_finish()
            items.put(_END_OF_STREAM)

    context = contextvars.copy_context()
    thread = threading.Thread(target=context.run, args=(produce,), name="docrag-sse", daemon=True)
    if on_start is not None:
        on_start()
    try:
        thread.start()
    except BaseException:
        if on_finish is not None:
            on_finish()
        raise
    try:
        while True:
            try:
                item = items.get(timeout=interval)
            except queue.Empty:
                yield None
                continue
            if item is _END_OF_STREAM:
                return
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]
    finally:
        stop.set()


class StreamLease:
    """Hands a question slot's release to whoever ends up owning the work.

    Before the pipeline thread starts, the response owns the slot (released by its
    background task, which also covers a client that leaves before the first frame).
    Once the thread starts, only the thread releases it, when its work has actually
    stopped: releasing on disconnect let a client that reconnects in a loop keep
    unlimited retrievals and reranks running past MAX_CONCURRENT_QUESTIONS.
    """

    def __init__(self, release: Release) -> None:
        self._release = release
        self._lock = threading.Lock()
        self._handed_off = False

    def hand_off(self) -> None:
        with self._lock:
            self._handed_off = True

    def release(self) -> None:
        self._release()

    def release_if_not_handed_off(self) -> None:
        with self._lock:
            if self._handed_off:
                return
        self._release()


def sse_event(payload: dict[str, object]) -> str:
    """Format one Server-Sent Events ``data:`` frame."""

    return f"data: {json.dumps(payload)}\n\n"
