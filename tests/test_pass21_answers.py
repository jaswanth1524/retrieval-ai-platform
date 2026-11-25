"""Pass 21: answer-path fixes."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

import pytest
from fastapi import HTTPException

from api.admission import QuestionSlots
from api.settings import AppSettings


def test_followers_of_a_leader_that_failed_run_the_question_instead_of_a_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    import api.routes.questions as question_routes
    from api.dependencies import clear_dependency_caches, get_generator, get_question_slots
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings

    monkeypatch.setattr(question_routes, "IDENTICAL_RUN_SETTLE_SECONDS", 0.0)

    class _FailsFirst:
        calls = 0
        lock = threading.Lock()

        def complete(self, messages: Sequence[Any], settings: AppSettings) -> str:
            with type(self).lock:
                type(self).calls += 1
                first = type(self).calls == 1
            time.sleep(0.5)
            if first:
                raise RuntimeError("provider fell over")
            return "Alpha is documented [1]."

        def stream(self, messages: Sequence[Any], settings: AppSettings) -> Any:
            yield self.complete(messages, settings)

    clear_dependency_caches()
    app = hermetic_app(route_settings(followups_enabled=False), QdrantClient(":memory:"))
    slots = QuestionSlots(1)
    app.dependency_overrides[get_question_slots] = lambda: slots
    app.dependency_overrides[get_generator] = lambda: _FailsFirst()
    with TestClient(app, raise_server_exceptions=False) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        codes: list[int] = []

        def ask() -> None:
            codes.append(client.post("/questions", json={"question": "alpha?"}).status_code)

        threads = [threading.Thread(target=ask) for _ in range(4)]
        for thread in threads:
            thread.start()
            time.sleep(0.05)  # the first leads
        for thread in threads:
            thread.join()
        assert 429 not in codes
        assert codes.count(200) == 3
        # One retry run, not one per follower.
        assert _FailsFirst.calls == 2
    clear_dependency_caches()


def test_requests_waiting_on_an_identical_run_are_capped() -> None:
    from api.admission import FOLLOWERS_PER_SLOT
    from api.routes.questions import question_slot_unless_cached

    slots = QuestionSlots(1)
    held = slots.try_acquire()
    assert held is not None
    event = threading.Event()
    admitted = [
        question_slot_unless_cached(slots, lambda: None, lambda: event)
        for _ in range(FOLLOWERS_PER_SLOT)
    ]
    assert all(a.follow is event for a in admitted)
    with pytest.raises(HTTPException) as refused:
        question_slot_unless_cached(slots, lambda: None, lambda: event)
    assert refused.value.status_code == 429
    admitted[0].release()
    admitted[0].release()  # idempotent
    assert question_slot_unless_cached(slots, lambda: None, lambda: event).follow is event
    held()


def test_a_follower_releases_a_slot_it_took_late() -> None:
    from api.routes.questions import question_slot_unless_cached

    slots = QuestionSlots(1)
    held = slots.try_acquire()
    assert held is not None
    event = threading.Event()
    admitted = question_slot_unless_cached(slots, lambda: None, lambda: event)
    assert admitted.take_slot is not None and not admitted.take_slot()
    held()
    assert admitted.take_slot()
    assert slots.try_acquire() is None
    admitted.release()
    again = slots.try_acquire()
    assert again is not None
    again()


def test_a_full_server_doesnt_wait_for_a_run_a_question_cant_join() -> None:
    from api.routes.questions import question_slot_unless_cached

    slots = QuestionSlots(1)
    held = slots.try_acquire()
    assert held is not None
    began = time.monotonic()
    with pytest.raises(HTTPException) as refused:
        question_slot_unless_cached(slots, lambda: None, None)
    assert refused.value.status_code == 429
    assert time.monotonic() - began < 0.5
    held()


def test_a_run_claimed_after_its_answer_was_stored_replays_it_and_frees_the_slot() -> None:
    from api.answer_cache import CachedAnswer
    from api.routes.questions import question_slot_unless_cached

    slots = QuestionSlots(1)
    answer = CachedAnswer.__new__(CachedAnswer)
    finished: list[bool] = []
    event = threading.Event()
    admitted = question_slot_unless_cached(
        slots,
        lambda: answer,
        lambda: None,
        lambda: (event, True, lambda: finished.append(True)),
    )
    assert admitted.cached is answer and finished == [True]
    free = slots.try_acquire()
    assert free is not None
    free()


def test_a_failing_claim_frees_the_slot() -> None:
    from api.routes.questions import question_slot_unless_cached

    slots = QuestionSlots(1)

    def claim() -> Any:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        question_slot_unless_cached(slots, lambda: None, lambda: None, claim)
    free = slots.try_acquire()
    assert free is not None
    free()


def test_continuing_with_a_provider_since_refused_is_a_400_before_the_stream() -> None:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches, get_app_settings, get_generator
    from api.llm import reset_finish_reason
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings
    from tests.test_pass17_answers import _sse_events

    class _Cut:
        def complete(self, messages: Sequence[Any], settings: AppSettings) -> str:
            return "unused"

        def stream(self, messages: Sequence[Any], settings: AppSettings) -> Any:
            yield "Alpha is documented [1] and"
            reset_finish_reason("length")

    clear_dependency_caches()
    settings = route_settings(citation_retry_enabled=False)
    app = hermetic_app(settings, QdrantClient(":memory:"))
    app.dependency_overrides[get_generator] = lambda: _Cut()
    with TestClient(app) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        first = _sse_events(client.post("/questions/stream", json={"question": "alpha?"}).text)
        done = next(e for e in first if e["type"] == "done")
        assert done["truncated"] is True
        # The operator has since moved to another provider and allows no other.
        app.dependency_overrides[get_app_settings] = lambda: settings.model_copy(
            update={"llm_provider": "openai_compatible", "allowed_request_providers": []}
        )
        refused = client.post("/questions/continue", json={"trace_id": done["trace_id"]})
        assert refused.status_code == 400
        assert "not allowed" in refused.json()["detail"]
    clear_dependency_caches()


def test_a_streamed_no_context_answer_reports_weak_matches_like_the_sync_route() -> None:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings
    from tests.test_pass17_answers import _sse_events

    clear_dependency_caches()
    app = hermetic_app(route_settings(rerank_min_score=0.99), QdrantClient(":memory:"))
    with TestClient(app) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        events = _sse_events(client.post("/questions/stream", json={"question": "zzz?"}).text)
        done = next(e for e in events if e["type"] == "done")
        sync = client.post("/questions", json={"question": "zzz?", "use_cache": False}).json()
        assert done["answer_status"] == sync["answer_status"]
        assert "weak_matches" in done and done["weak_matches"] == sync["weak_matches"]
    clear_dependency_caches()


def _passage(text: str) -> Any:
    from api.reranking import RerankedChunk

    return RerankedChunk("p1", "leave.txt", 1, "Leave", "c1", text, 0.5, 0.9)


def test_verify_finds_a_number_in_the_matched_passages_neighbours() -> None:
    from api.verify import check_sentence

    passage = _passage("Staff take annual leave days after the first year of service.")
    sentence = "Staff take 42 annual leave days after the first year."
    alone = check_sentence(sentence, lambda _: [passage])
    assert alone.status == "numbers_differ" and alone.missing_numbers == ("42",)
    widened = check_sentence(
        sentence, lambda _: [passage], lambda chunk: chunk.text + " The allowance is 42 days."
    )
    assert widened.status == "supported"


def test_verify_searches_a_repeated_sentence_once() -> None:
    from api.verify import verify_text

    searched: list[str] = []

    def search(sentence: str) -> list[Any]:
        searched.append(sentence)
        return []

    checks, _ = verify_text(
        "Staff get leave every year. Staff get leave every year. Pay is monthly here.",
        search,
    )
    assert len(checks) == 3
    assert searched == ["Staff get leave every year.", "Pay is monthly here."]


def test_verify_with_a_scope_naming_no_document_says_so_and_searches_nothing() -> None:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings

    clear_dependency_caches()
    app = hermetic_app(route_settings(), QdrantClient(":memory:"))
    with TestClient(app) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        text = {"text": "Alpha beta gamma delta are documented here."}
        typo = client.post("/questions/verify", json={**text, "filenames": ["gide.txt"]}).json()
        assert typo == {"sentences": [], "more": False, "empty_scope": True}
        scoped = client.post("/questions/verify", json={**text, "filenames": ["guide.txt"]})
        assert scoped.json()["empty_scope"] is False and scoped.json()["sentences"]
    clear_dependency_caches()


def test_the_sync_route_continues_a_cut_off_answer_like_the_stream() -> None:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches, get_generator
    from api.llm import reset_finish_reason
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings

    class _CutOnce:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Sequence[Any], settings: AppSettings) -> str:
            self.calls += 1
            if self.calls == 1:
                reset_finish_reason("length")
                return "Alpha is documented [1] and"
            reset_finish_reason("stop")
            return " beta follows it [1]."

        def stream(self, messages: Sequence[Any], settings: AppSettings) -> Any:
            yield self.complete(messages, settings)

    generator = _CutOnce()
    clear_dependency_caches()
    app = hermetic_app(
        route_settings(citation_retry_enabled=False, auto_continue_max=1, followups_enabled=False),
        QdrantClient(":memory:"),
    )
    app.dependency_overrides[get_generator] = lambda: generator
    with TestClient(app) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        answer = client.post("/questions", json={"question": "alpha?"}).json()
        assert answer["answer"] == "Alpha is documented [1] and beta follows it [1]."
        assert answer["truncated"] is False
        assert generator.calls == 2
    clear_dependency_caches()


def test_an_answer_finished_before_a_later_step_failed_is_not_offered_to_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.answer_stream import StreamingAnswers
    from api.dependencies import clear_dependency_caches, get_generator
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings
    from tests.test_pass17_answers import _sse_events

    class _Done:
        def complete(self, messages: Sequence[Any], settings: AppSettings) -> str:
            return "Alpha is documented [1]."

        def stream(self, messages: Sequence[Any], settings: AppSettings) -> Any:
            yield "Alpha is documented [1]."

    def broken(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the trace store fell over")
        yield  # pragma: no cover - a generator

    clear_dependency_caches()
    app = hermetic_app(route_settings(followups_enabled=False), QdrantClient(":memory:"))
    app.dependency_overrides[get_generator] = lambda: _Done()
    with TestClient(app) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        monkeypatch.setattr(StreamingAnswers, "_stream_finish", broken)
        events = _sse_events(client.post("/questions/stream", json={"question": "alpha?"}).text)
        trace_id = next(e for e in events if e["type"] == "sources")["trace_id"]
        assert any(e["type"] == "error" for e in events)
        refused = client.post("/questions/continue", json={"trace_id": trace_id})
        assert refused.status_code == 409
    clear_dependency_caches()


def test_a_continuation_repeating_the_word_cut_with_its_comma_is_joined_once() -> None:
    from api.continuation import drop_restart, join_continuation

    before = "Laptops are replaced every four years [1]. Furthermore,"
    after = "Furthermore, software must only come from the portal [1]."
    joined = join_continuation(before, drop_restart(before, after))
    assert joined == (
        "Laptops are replaced every four years [1]. "
        "Furthermore, software must only come from the portal [1]."
    )
    # A short word is still kept: "… of the" then "the policy".
    assert join_continuation(
        "It is part of the", drop_restart("It is part of the", "the policy.")
    ) == ("It is part of the the policy.")
