"""Caps how many questions run at once, answering 429 past the cap instead of queueing.

Every question holds a request-threadpool thread for its whole 20-80 s run, and the
reranker scores one question at a time anyway. Unbounded, a burst of questions took
every thread, so even job polling waited behind them. Past the cap a client gets a 429
with Retry-After immediately — a clear "busy, try again" instead of a minute of silence.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import BoundedSemaphore, Lock

Release = Callable[[], None]


class QuestionSlots:
    """A non-blocking counting semaphore handing out idempotent release callbacks."""

    def __init__(self, limit: int) -> None:
        # 0 (MAX_CONCURRENT_QUESTIONS=0) means no cap.
        self._semaphore = BoundedSemaphore(limit) if limit > 0 else None

    def try_acquire(self) -> Release | None:
        """A release callback when a slot was free, else None. Never blocks."""

        semaphore = self._semaphore
        if semaphore is None:
            return _noop
        if not semaphore.acquire(blocking=False):
            return None
        released = False
        guard = Lock()

        # Idempotent: a streamed answer releases both when its generator finishes and
        # from the response's background task, since a client that disconnects before
        # the first frame never runs the generator at all.
        def release() -> None:
            nonlocal released
            with guard:
                if released:
                    return
                released = True
            semaphore.release()

        return release


def _noop() -> None:
    return None
