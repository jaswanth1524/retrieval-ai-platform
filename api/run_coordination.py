"""How an answer run waits its turn: the local model's generation slot, joining an
identical question already being answered, and the answer cache around a run. Moved
out of ``api/pipeline.py``, which re-exports these names.
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from typing import cast

from api.answer_cache import (
    AnswerCache,
    CachedAnswer,
    CacheKey,
)
from api.answer_events import DoneEvent, StreamEvent
from api.llm import (
    GenerationError,
    GenerationSlot,
    was_degraded,
)
from api.metrics import answer_inflight_joins_total
from api.settings import AppSettings


class QuestionAbandoned(Exception):
    """The client left mid-question; stop before the next expensive stage."""


class NoSlotForRun(Exception):
    """A request admitted without a question slot, to replay an identical question's
    answer, found none to replay (that run failed or kept no answer): it may not run
    the question itself — the server is full (429)."""


def follow_only(
    cache: AnswerCache,
    key: CacheKey,
    event: threading.Event,
    should_stop: Callable[[], bool] | None,
    take_slot: Callable[[], bool] | None = None,
) -> Generator[StreamEvent, None, CachedAnswer | threading.Event | None]:
    """Wait for the identical run ``event`` belongs to and return its answer: None when
    the reader left. When that run left none (it failed, was abandoned, or its answer
    isn't kept), the run claimed by this request (the event the caller leads and must
    ``finish``) if ``take_slot`` got it a question slot of its own — the caller then
    runs the question itself — else ``NoSlotForRun``: before pass 21 every follower of
    a failed leader got a 429 though slots were free.

    The run is claimed before the slot is taken, so of the followers of a failed
    leader one leads and the rest follow it without a slot: each used to take a slot
    of its own first and then wait on the new leader holding it."""

    yield {"type": "stage", "stage": "joining_identical_question"}
    while True:
        while not event.wait(_JOIN_POLL_SECONDS):
            if should_stop is not None and should_stop():
                return None
        hit = cache.get(key)
        if hit is not None:
            answer_inflight_joins_total.labels(outcome="replayed").inc()
            return hit
        _raise_shared_failure(event)
        # The leader's slot is freed just after it releases its followers: for a moment,
        # lead the question (a claim and a slot) or follow whoever does.
        deadline = time.monotonic() + _RELEAD_SETTLE_SECONDS
        while True:
            if take_slot is None:
                claim, leading = cache.inflight.running(key), False
            else:
                claim, leading = cache.inflight.claim(key)
            if leading:
                assert claim is not None and take_slot is not None
                hit = cache.get(key)
                if hit is not None:
                    cache.inflight.finish(key, claim)
                    answer_inflight_joins_total.labels(outcome="replayed").inc()
                    return hit
                if take_slot():
                    answer_inflight_joins_total.labels(outcome="ran").inc()
                    return claim
                # No slot free yet: let another follower (or a new request) lead.
                cache.inflight.finish(key, claim)
            elif claim is not None and claim is not event:
                event = claim
                break
            if time.monotonic() >= deadline:
                answer_inflight_joins_total.labels(outcome="ran").inc()
                raise NoSlotForRun(
                    "The server is answering as many questions as it allows; try again."
                )
            if should_stop is not None and should_stop():
                return None
            time.sleep(_JOIN_POLL_SECONDS / 10)


def _raise_shared_failure(event: threading.Event) -> None:
    """The run ``event`` belongs to failed in a way the same question would fail again
    (bad credentials, an unknown model, a model that timed out on this prompt): its
    followers fail with it. Each used to run the question again in turn, so with a
    provider that took its timeout to fail, the last of N identical questions waited N
    timeouts for the same error."""

    failure = getattr(event, "failure", None)
    if isinstance(failure, GenerationError) and not failure.worth_retrying:
        answer_inflight_joins_total.labels(outcome="failed").inc()
        # A copy: the leader's own exception may still be on its way up another thread.
        raise copy.copy(failure)


# How long a follower whose leader left no answer looks for a slot or a new leader.
_RELEAD_SETTLE_SECONDS = 1.0


def wait_for_model(
    settings: AppSettings, should_stop: Callable[[], bool] | None
) -> Generator[StreamEvent, None, GenerationSlot]:
    """Take this thread's place at the local model (OLLAMA_MAX_PARALLEL), held.

    Yields a ``waiting_for_model`` frame first when another request has it, so the
    reader sees why nothing moves, and raises ``QuestionAbandoned`` when the client
    leaves while waiting. The caller releases the returned slot; generator calls made
    while it is held take no second place (the slot is re-entrant per thread). With no
    limit (hosted providers, 0) it returns at once and holds nothing.
    """

    slot = GenerationSlot(settings)
    if not slot.try_acquire():
        yield {"type": "stage", "stage": "waiting_for_model"}
        if not slot.acquire(should_stop):
            raise QuestionAbandoned
    return slot


# How often a follower waiting for an identical run checks whether its reader left.
_JOIN_POLL_SECONDS = 0.5


def lead_or_join(
    cache: AnswerCache, key: CacheKey, should_stop: Callable[[], bool] | None
) -> Generator[StreamEvent, None, threading.Event | CachedAnswer | None]:
    """Lead the run of ``key`` (returns the event the caller must ``finish``), or wait
    for the identical one already running and return its cached answer.

    A follower announces itself (``joining_identical_question``) and waits for as long
    as the leader runs — a leader may legitimately wait minutes for a local model, and a
    follower that gave up would only queue behind it. None: the reader left while
    waiting. A leader that leaves no answer (failed, abandoned) makes the follower lead
    the question itself.
    """

    announced = False
    while True:
        event, leading = cache.inflight.claim(key)
        if leading:
            # The previous leader may have finished between our lookup and the claim.
            hit = cache.get(key)
            if hit is None:
                return event
            cache.inflight.finish(key, event)
            return hit
        if not announced:
            announced = True
            yield {"type": "stage", "stage": "joining_identical_question"}
        while not event.wait(_JOIN_POLL_SECONDS):
            if should_stop is not None and should_stop():
                return None
        hit = cache.get(key)
        if hit is not None:
            answer_inflight_joins_total.labels(outcome="replayed").inc()
            return hit
        _raise_shared_failure(event)
        answer_inflight_joins_total.labels(outcome="ran").inc()


def cached_run(
    cache: AnswerCache | None,
    key: CacheKey | None,
    *,
    use_cache: bool,
    should_stop: Callable[[], bool] | None,
    replay: Callable[[CachedAnswer], Iterator[StreamEvent]],
    run: Callable[[], Generator[StreamEvent]],
    keep: Callable[[DoneEvent], bool] = lambda done: True,
    on_hit: Callable[[CachedAnswer], None] | None = None,
    claimed: threading.Event | None = None,
) -> Iterator[StreamEvent]:
    """A run through the answer cache, for answers and summaries alike: a hit replayed
    (``replay``), or the identical run already going joined (``lead_or_join``), or
    ``run`` led — its ``done`` stored (unless the answer is degraded or ``keep`` says
    no) before the frame goes out, and its follow-ups added once they come.

    Stored at ``done``, not after the loop: follow-ups are generated after it, and a
    client that leaves then (or as ``done`` arrives) closes this generator at the yield,
    so a store after the loop never ran. Followers are released right after the store;
    they never wait for the follow-ups. ``use_cache=False`` (Regenerate) skips the
    lookup but still stores the result. ``on_hit`` is told of a hit (or a join) before
    it is replayed: it has no trace of its own, so the query log is told here.
    ``claimed`` (a run the caller already leads) skips the lookup and the join.
    """

    # ``claimed``: the request already leads this run (it claimed it as it was admitted).
    lead: threading.Event | None = claimed
    if claimed is None and key is not None and use_cache and cache is not None:
        hit = cache.get(key)
        if hit is None:
            joined = yield from lead_or_join(cache, key, should_stop)
            if joined is None:
                return
            if isinstance(joined, CachedAnswer):
                hit = joined
            else:
                lead = joined
        if hit is not None:
            if on_hit is not None:
                on_hit(hit)
            yield from replay(hit)
            return
    done: DoneEvent | None = None
    events = run()

    failure: BaseException | None = None

    def finish_lead() -> None:
        if lead is not None and key is not None and cache is not None:
            cache.inflight.finish(key, lead, failure)

    stored: CachedAnswer | None = None

    def store(followups: Sequence[str]) -> None:
        nonlocal stored
        if key is None or done is None or cache is None or was_degraded() or not keep(done):
            return
        stored = CachedAnswer.from_done(done, followups)
        cache.put(key, stored)

    try:
        for event in events:
            if event["type"] == "done":
                done = event
                store(())
                finish_lead()
            elif event["type"] == "followups":
                store(event["questions"])
            yield event
        # Stored at ``done``, before the follow-ups: when they failed (no frame comes
        # then), the answer is degraded after all and isn't kept, as on the sync path.
        if stored is not None and key is not None and cache is not None and was_degraded():
            cache.discard(key, stored)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        # Closing this generator (the client left) must close the inner one now: its
        # `finally` records the "closed early" trace. Left to garbage collection, it ran
        # whenever the inner generator was freed — never, while anything (a tracer, a
        # traceback) still held it.
        events.close()
        finish_lead()


def drain[T](events: Generator[StreamEvent, None, T]) -> T:
    """Run a stage-yielding generator to its return value (the sync route has no
    stream to send its frames to)."""

    while True:
        try:
            next(events)
        except StopIteration as stop:
            return cast(T, stop.value)
