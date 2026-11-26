"""Pass 22: answer-path fixes."""

from __future__ import annotations

from typing import Any

from api.dependencies import clear_dependency_caches
from tests.routes_http.support import upload_and_wait
from tests.test_pass17_answers import _sse_events
from tests.test_pass21_features import _DOC, _client

# One passage whose fact sits well past the 200-character citation excerpt.
_LONG = (
    b"Leave\n"
    + b"This handbook explains how the company organises its working year and its offices. " * 4
    + b"Staff take annual leave of 28 days after asking their manager in writing."
)


def test_the_self_check_reads_the_whole_cited_passage_not_its_excerpt() -> None:
    answer = "Staff take annual leave of 28 days after asking their manager [1]."
    with _client(answer, chunk_size_tokens=400) as client:
        assert upload_and_wait(client, "leave.txt", _LONG)["state"] == "done"
        sync = client.post("/questions", json={"question": "leave?"}).json()
        assert len(sync["sources"]) == 1 and "28 days" not in sync["sources"][0]["text"]
        assert [c["status"] for c in sync["verification"]] == ["supported"]
        assert "28 days" in sync["verification"][0]["quote"]
        events = _sse_events(
            client.post("/questions/stream", json={"question": "leave?", "use_cache": False}).text
        )
        done: dict[str, Any] = next(e for e in events if e["type"] == "done")
        assert [c["status"] for c in done["verification"]] == ["supported"]
        hit = client.post("/questions", json={"question": "leave?"}).json()
        assert hit["cached"] is True and hit["verification"] == sync["verification"]
    clear_dependency_caches()


def _saved(client: Any, answer: str) -> dict[str, Any]:
    asked = client.post("/questions", json={"question": "leave?"}).json()
    sources = [
        {k: s[k] for k in ("source_number", "filename", "chunk_id")} for s in asked["sources"]
    ]
    return {"answer": answer, "sources": sources}


def test_a_recheck_searches_a_changed_document_once_per_sentence(monkeypatch: Any) -> None:
    from api.pipeline_core import PipelineCore

    calls: list[str] = []
    original = PipelineCore.retrieve

    def counting(self: Any, question: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(question)
        return original(self, question, *args, **kwargs)

    answer = "Staff take annual leave of 28 days after asking their manager [1]."
    with _client(answer, chunk_size_tokens=200) as client:
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        body = _saved(client, answer)
        # The same changed passage cited under many numbers: one search, not one each.
        source = body["sources"][0]
        body["sources"] = [{**source, "source_number": n} for n in range(1, 11)]
        assert upload_and_wait(client, "leave.txt", _DOC.replace(b"28", b"35"))["state"] == "done"
        monkeypatch.setattr(PipelineCore, "retrieve", counting)
        changed = client.post("/questions/recheck", json=body).json()
        assert [s["state"] for s in changed["sources"]] == ["changed"] * 10
        assert changed["sentences"][0]["status"] == "numbers_differ"
        assert len(calls) == 1
    clear_dependency_caches()


def test_a_recheck_says_when_it_left_sentences_unchecked() -> None:
    answer = "Staff take annual leave of 28 days after asking their manager [1]."
    with _client(answer, chunk_size_tokens=200) as client:
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        body = _saved(client, " ".join([answer] * 25))
        result = client.post("/questions/recheck", json=body).json()
        assert len(result["sentences"]) == 20 and result["more"] is True
        assert result["still_holds"] is False
        short = client.post("/questions/recheck", json=_saved(client, answer)).json()
        assert short["more"] is False and short["still_holds"] is True
    clear_dependency_caches()


def test_followers_of_a_failed_leader_take_one_slot_between_them() -> None:
    import threading

    from api.answer_cache import AnswerCache, CachedAnswer
    from api.run_coordination import drain, follow_only

    cache = AnswerCache(16, 600)
    key = ("q", ())
    first, leading = cache.inflight.claim(key)
    assert leading
    taken: list[int] = []
    lock = threading.Lock()

    def take_slot() -> bool:
        with lock:
            taken.append(1)
        return True

    results: list[Any] = []

    def follow() -> None:
        results.append(drain(follow_only(cache, key, first, None, take_slot)))

    threads = [threading.Thread(target=follow) for _ in range(4)]
    for thread in threads:
        thread.start()
    cache.inflight.finish(key, first)  # the leader failed: no answer stored
    # Whoever leads now answers; the others follow it, without a slot of their own.
    for _ in range(100):
        new = cache.inflight.running(key)
        if new is not None and new is not first:
            break
        threading.Event().wait(0.02)
    assert new is not None
    threading.Event().wait(0.3)
    answer = CachedAnswer.from_done({"answer": "Alpha [1].", "sources": [], "trace_id": None})
    led = next(r for r in results if isinstance(r, threading.Event))
    cache.put(key, answer)
    cache.inflight.finish(key, led)
    for thread in threads:
        thread.join(5)
    assert len(taken) == 1
    assert sum(isinstance(r, threading.Event) for r in results) == 1
    assert sum(isinstance(r, CachedAnswer) for r in results) == 3


def test_an_empty_continuation_leaves_the_answer_cut_off() -> None:
    from api.continuation import AutoContinuing
    from api.llm import reset_finish_reason, stopped_at_token_limit
    from tests.factories import make_test_settings

    class _CutThenNothing:
        calls = 0

        def complete(self, messages: Any, settings: Any) -> str:
            type(self).calls += 1
            if type(self).calls == 1:
                reset_finish_reason("length")
                return "Staff take annual leave of"
            reset_finish_reason("stop")
            return "   "

        def stream(self, messages: Any, settings: Any) -> Any:
            yield ""

    settings = make_test_settings()
    text = AutoContinuing(_CutThenNothing(), 2).complete(
        [{"role": "user", "content": "leave?"}], settings
    )
    assert text == "Staff take annual leave of"
    assert stopped_at_token_limit()
    reset_finish_reason()


def test_followers_fail_with_a_failure_asking_again_would_repeat() -> None:
    import threading

    import pytest

    from api.answer_cache import AnswerCache
    from api.llm import GenerationError
    from api.run_coordination import drain, follow_only, lead_or_join

    cache = AnswerCache(16, 600)
    key = ("q", ())
    event, _ = cache.inflight.claim(key)
    bad_key = GenerationError(
        "Generation provider request failed: credentials.", worth_retrying=False
    )
    cache.inflight.finish(key, event, bad_key)
    with pytest.raises(GenerationError, match="credentials") as raised:
        drain(follow_only(cache, key, event, None, lambda: True))
    assert raised.value is not bad_key  # a copy: the leader's is still on its way up

    # A failure worth retrying (a rate limit, a blip): the follower leads it again.
    event, _ = cache.inflight.claim(key)
    cache.inflight.finish(key, event, GenerationError("rate limited", worth_retrying=True))
    led = drain(follow_only(cache, key, event, None, lambda: True))
    assert isinstance(led, threading.Event)
    cache.inflight.finish(key, led)

    # The sync path's join (lead_or_join) shares it too.
    event, _ = cache.inflight.claim(key)
    joiner_error: list[BaseException] = []

    def join() -> None:
        try:
            drain(lead_or_join(cache, key, None))
        except GenerationError as exc:
            joiner_error.append(exc)

    joiner = threading.Thread(target=join)
    joiner.start()
    threading.Event().wait(0.1)
    cache.inflight.finish(key, event, bad_key)
    joiner.join(5)
    assert joiner_error and "credentials" in str(joiner_error[0])


def test_identical_questions_against_a_rejecting_provider_call_it_once(
    monkeypatch: Any,
) -> None:
    import threading
    import time
    from collections.abc import Sequence

    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    import api.routes.questions as question_routes
    from api.admission import QuestionSlots
    from api.dependencies import get_generator, get_question_slots
    from api.llm import GenerationError
    from tests.routes_http.support import hermetic_app
    from tests.routes_http.support import make_settings as route_settings

    monkeypatch.setattr(question_routes, "IDENTICAL_RUN_SETTLE_SECONDS", 0.0)

    class _Rejects:
        calls = 0
        lock = threading.Lock()

        def complete(self, messages: Sequence[Any], settings: Any) -> str:
            with type(self).lock:
                type(self).calls += 1
            time.sleep(0.5)
            raise GenerationError(
                "Generation provider request failed: credentials.", worth_retrying=False
            )

        def stream(self, messages: Sequence[Any], settings: Any) -> Any:
            yield self.complete(messages, settings)

    clear_dependency_caches()
    app = hermetic_app(
        route_settings(followups_enabled=False, citation_retry_enabled=False),
        QdrantClient(":memory:"),
    )
    slots = QuestionSlots(1)
    app.dependency_overrides[get_question_slots] = lambda: slots
    app.dependency_overrides[get_generator] = lambda: _Rejects()
    with TestClient(app, raise_server_exceptions=False) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        codes: list[int] = []

        def ask() -> None:
            codes.append(client.post("/questions", json={"question": "alpha?"}).status_code)

        threads = [threading.Thread(target=ask) for _ in range(4)]
        for thread in threads:
            thread.start()
            time.sleep(0.05)
        for thread in threads:
            thread.join()
        assert sorted(codes) == [502] * 4
        assert _Rejects.calls == 1
        assert slots.try_acquire() is not None
    clear_dependency_caches()
