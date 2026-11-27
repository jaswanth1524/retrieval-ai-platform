"""Caps how many questions run at once, answering 429 past the cap instead of queueing.

Every question holds a request-threadpool thread for its whole 20-80 s run, and the
reranker scores one question at a time anyway. Unbounded, a burst of questions took
every thread, so even job polling waited behind them. Past the cap a client gets a 429
with Retry-After immediately — a clear "busy, try again" instead of a minute of silence.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import BoundedSemaphore, Lock
from typing import Literal

from api.metrics import question_slots_in_use

Release = Callable[[], None]


SlotKind = Literal["question", "summary", "batch"]

# Requests that may wait, slot-less, for an identical question being answered: this
# many per question slot. Each waits on a request thread (the shared threadpool, or an
# SSE thread), so without a cap a burst of one question behind a slow local model took
# every thread the other routes need.
FOLLOWERS_PER_SLOT = 4

# With a single slot, how long after refusing someone batches leave it free: past the
# Retry-After (5 s) the refusal sent. Taking it back between questions left a gap of a
# second, which a client retrying when told to rarely hit.
BATCH_STAND_ASIDE_SECONDS = 7.0


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

    def __init__(
        self, limit: int, summary_limit: int = 0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        # 0 (MAX_CONCURRENT_QUESTIONS=0 / MAX_CONCURRENT_SUMMARIES=0) means no cap.
        self._semaphore = BoundedSemaphore(limit) if limit > 0 else None
        self._summaries = BoundedSemaphore(summary_limit) if summary_limit > 0 else None
        self._followers = BoundedSemaphore(limit * FOLLOWERS_PER_SLOT) if limit > 0 else None
        # Background batches together hold at most all slots but one, so someone asking
        # is never refused for as long as a batch runs.
        self._batches = BoundedSemaphore(limit - 1) if limit > 1 else None
        # With one slot there is none to spare: a batch stands aside for whoever was
        # just refused it.
        self._single = limit == 1
        self._clock = clock
        self._refused_at: float | None = None

    def _someone_waiting(self) -> bool:
        refused = self._refused_at
        return refused is not None and self._clock() - refused < BATCH_STAND_ASIDE_SECONDS

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

        if kind == "batch" and self._someone_waiting():
            return None
        # The kind's own cap, taken first.
        own = self._summaries if kind == "summary" else self._batches if kind == "batch" else None
        if own is not None and not own.acquire(blocking=False):
            return None
        semaphore = self._semaphore
        if semaphore is not None and not semaphore.acquire(blocking=False):
            if own is not None:
                own.release()
            if self._single and kind != "batch":
                self._refused_at = self._clock()
            return None
        if kind != "batch":
            self._refused_at = None  # whoever was waiting is in
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
            if own is not None:
                own.release()

        return release
