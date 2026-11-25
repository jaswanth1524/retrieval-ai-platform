"""Caps how many questions run at once, answering 429 past the cap instead of queueing.

Every question holds a request-threadpool thread for its whole 20-80 s run, and the
reranker scores one question at a time anyway. Unbounded, a burst of questions took
every thread, so even job polling waited behind them. Past the cap a client gets a 429
with Retry-After immediately — a clear "busy, try again" instead of a minute of silence.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import BoundedSemaphore, Lock
from typing import Literal

from api.metrics import question_slots_in_use

Release = Callable[[], None]


SlotKind = Literal["question", "summary"]

# Requests that may wait, slot-less, for an identical question being answered: this
# many per question slot. Each waits on a request thread (the shared threadpool, or an
# SSE thread), so without a cap a burst of one question behind a slow local model took
# every thread the other routes need.
FOLLOWERS_PER_SLOT = 4


def _idempotent(release: Callable[[], None]) -> Release:
    released = False
    guard = Lock()

    def once() -> None:
        nonlocal released
        with guard:
            if released:
                return
            released = True
        release()

    return once


class QuestionSlots:
    """A non-blocking counting semaphore handing out idempotent release callbacks.

    Summaries also take one of ``summary_limit`` slots of their own first, so they can
    never hold every slot: questions keep the rest.
    """

    def __init__(self, limit: int, summary_limit: int = 0) -> None:
        # 0 (MAX_CONCURRENT_QUESTIONS=0 / MAX_CONCURRENT_SUMMARIES=0) means no cap.
        self._semaphore = BoundedSemaphore(limit) if limit > 0 else None
        self._summaries = BoundedSemaphore(summary_limit) if summary_limit > 0 else None
        self._followers = BoundedSemaphore(limit * FOLLOWERS_PER_SLOT) if limit > 0 else None

    def try_follow(self) -> Release | None:
        """A release callback for one request waiting on an identical run (no question
        slot), when the cap on those leaves room, else None. Never blocks."""

        followers = self._followers
        if followers is None:
            return _idempotent(lambda: None)
        if not followers.acquire(blocking=False):
            return None
        return _idempotent(followers.release)

    def try_acquire(self, kind: SlotKind = "question") -> Release | None:
        """A release callback when a slot was free, else None. Never blocks."""

        summaries = self._summaries if kind == "summary" else None
        if summaries is not None and not summaries.acquire(blocking=False):
            return None
        semaphore = self._semaphore
        if semaphore is not None and not semaphore.acquire(blocking=False):
            if summaries is not None:
                summaries.release()
            return None
        question_slots_in_use.inc()
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
            question_slots_in_use.dec()
            if semaphore is not None:
                semaphore.release()
            if summaries is not None:
                summaries.release()

        return release
