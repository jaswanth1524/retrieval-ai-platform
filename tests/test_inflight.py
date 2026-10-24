"""Identical questions asked together share one run (api/answer_cache.InflightRuns,
pipeline.lead_or_join): the first leads, the others wait and replay its answer."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

import pytest
from qdrant_client import QdrantClient

from api.answer_cache import AnswerCache
from api.generation import ChatMessage
from api.ingest_service import IngestService
from api.pipeline import GroundedAnswer, RagPipeline
from api.repository import VectorRepository
from api.settings import AppSettings
from tests.test_pipeline import (
    FakeReranker,
    ListTraceStore,
    StaticEmbeddingProvider,
    make_embedding,
    make_settings,
)


class GatedGenerator:
    """Answers only once ``gate`` is set; counts calls; the first ``failures`` raise."""

    def __init__(self, failures: int = 0) -> None:
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.calls = 0
        self.failures = failures
        self._lock = threading.Lock()

    def _turn(self) -> int:
        with self._lock:
            self.calls += 1
            call = self.calls
        self.entered.set()
        assert self.gate.wait(10)
        if call <= self.failures:
            raise RuntimeError("generation boom")
        return call

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self._turn()
        return "Alpha is documented [1]."

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings):
        self._turn()
        yield "Alpha is documented [1]."


def _pipeline(generator: GatedGenerator, **overrides: Any) -> tuple[RagPipeline, ListTraceStore]:
    settings = make_settings(**overrides)
    repository = VectorRepository(QdrantClient(":memory:"))
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    traces = ListTraceStore()
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
        trace_store=traces,
        answer_cache=AnswerCache(16, 600),
    )
    return pipeline, traces


def _in_thread(target: Any) -> tuple[threading.Thread, list[Any]]:
    results: list[Any] = []

    def run() -> None:
        try:
            results.append(target())
        except Exception as exc:  # noqa: BLE001 - asserted on by the test
            results.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, results


def _wait_for_follower(pipeline: RagPipeline) -> None:
    """Until the second caller is waiting on the leader (it has nothing to signal)."""

    time.sleep(0.2)


def test_an_identical_question_waits_for_the_running_one_and_replays_it() -> None:
    generator = GatedGenerator()
    # A leader slower than the request timeout is still joined: the wait has no timeout.
    pipeline, traces = _pipeline(generator, llm_request_timeout_seconds=1)

    leader, leader_result = _in_thread(lambda: pipeline.answer("alpha"))
    assert generator.entered.wait(5)
    follower, follower_result = _in_thread(lambda: pipeline.answer("  ALPHA "))
    _wait_for_follower(pipeline)
    time.sleep(1.2)
    generator.gate.set()
    leader.join(10)
    follower.join(10)

    (first,) = leader_result
    (second,) = follower_result
    assert isinstance(first, GroundedAnswer) and isinstance(second, GroundedAnswer)
    assert generator.calls == 1
    assert not first.cached and second.cached
    assert second.trace_id == first.trace_id
    assert second.answer == first.answer
    assert len(traces.traces) == 1


def test_a_follower_runs_the_question_itself_when_the_leader_fails() -> None:
    generator = GatedGenerator(failures=1)
    pipeline, _ = _pipeline(generator)

    leader, leader_result = _in_thread(lambda: pipeline.answer("alpha"))
    assert generator.entered.wait(5)
    follower, follower_result = _in_thread(lambda: pipeline.answer("alpha"))
    _wait_for_follower(pipeline)
    generator.gate.set()
    leader.join(10)
    follower.join(10)

    assert isinstance(leader_result[0], Exception)
    (second,) = follower_result
    assert isinstance(second, GroundedAnswer) and not second.cached
    assert generator.calls == 2


def test_a_streamed_follower_announces_itself_and_stops_when_its_reader_leaves() -> None:
    generator = GatedGenerator()
    pipeline, _ = _pipeline(generator)
    leader, _ = _in_thread(lambda: list(pipeline.answer_stream("alpha")))
    assert generator.entered.wait(5)

    gone = threading.Event()
    follower = pipeline.answer_stream("alpha", should_stop=gone.is_set)
    assert next(follower) == {"type": "stage", "stage": "joining_identical_question"}
    gone.set()
    with pytest.raises(StopIteration):
        next(follower)

    generator.gate.set()
    leader.join(10)
    assert generator.calls == 1


def test_a_streamed_follower_replays_once_the_leader_is_done() -> None:
    generator = GatedGenerator()
    pipeline, _ = _pipeline(generator)
    leader, leader_events = _in_thread(lambda: list(pipeline.answer_stream("alpha")))
    assert generator.entered.wait(5)
    follower, follower_events = _in_thread(lambda: list(pipeline.answer_stream("alpha")))
    _wait_for_follower(pipeline)
    generator.gate.set()
    leader.join(10)
    follower.join(10)

    (events,) = follower_events
    assert events[0] == {"type": "stage", "stage": "joining_identical_question"}
    done = next(event for event in events if event["type"] == "done")
    assert done["cached"] is True
    leader_done = next(event for event in leader_events[0] if event["type"] == "done")
    assert done["trace_id"] == leader_done["trace_id"]
    assert generator.calls == 1


@pytest.mark.parametrize(
    "second",
    [
        pytest.param({"use_cache": False}, id="regenerate"),
        pytest.param(
            {"history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]},
            id="history",
        ),
    ],
)
def test_regenerate_and_follow_ups_never_join(second: dict[str, Any]) -> None:
    generator = GatedGenerator()
    pipeline, _ = _pipeline(generator)

    leader, _ = _in_thread(lambda: pipeline.answer("alpha"))
    assert generator.entered.wait(5)
    other, other_result = _in_thread(lambda: pipeline.answer("alpha", **second))
    _wait_for_follower(pipeline)
    generator.gate.set()
    leader.join(10)
    other.join(10)

    assert isinstance(other_result[0], GroundedAnswer)
    assert not other_result[0].cached
    # Both answered (the follow-up's condense call is one more).
    assert generator.calls >= 2


def test_nothing_is_left_registered_after_a_run() -> None:
    generator = GatedGenerator()
    generator.gate.set()
    pipeline, _ = _pipeline(generator)
    pipeline.answer("alpha")
    list(pipeline.answer_stream("beta"))
    assert len(pipeline._answer_cache.inflight) == 0  # type: ignore[union-attr]  # noqa: SLF001
