"""Pass 20: answer-path fixes."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import QdrantClient

from api.pipeline import IngestService
from api.repository import VectorRepository
from api.reranking import RerankedChunk
from api.settings import AppSettings
from tests.test_pipeline import StaticEmbeddingProvider, make_embedding, make_settings

_DOC = (
    b"Leave\nStaff take annual leave by asking their manager in writing first. "
    b"Requests go through the portal and are answered within a week by the team. "
    b"The allowance is 42 days for every member of staff after the first year."
)


def test_a_continued_answers_passages_get_back_the_neighbour_text_the_prompt_had() -> None:
    from api.answer_stream import _with_neighbour_text
    from api.citations import unsupported_numbers

    settings = make_settings(chunk_size_tokens=20, chunk_overlap_tokens=0)
    repository = VectorRepository(QdrantClient(":memory:"))

    class _Each(StaticEmbeddingProvider):
        def embed_texts(self, texts: Sequence[str]) -> list[Any]:
            return [make_embedding(1.0) for _ in texts]

    IngestService(repository, _Each([]), settings).ingest("leave.txt", _DOC)
    payloads = repository.chunks_for_filename(settings, "leave.txt")
    assert len(payloads) >= 3
    holder = next(i for i, p in enumerate(payloads) if "42" in str(p["text"]))
    middle = payloads[holder - 1]  # the passage just before the one stating the figure
    chunk = RerankedChunk(
        point_id=str(middle["point_id"]) if "point_id" in middle else "p1",
        filename="leave.txt",
        page=1,
        section="Leave",
        chunk_id=str(middle["chunk_id"]),
        text=str(middle["text"]),
        retrieval_score=1.0,
        rerank_score=1.0,
        chunk_ordinal=int(middle["chunk_ordinal"]),  # type: ignore[call-overload]
    )
    answer = "The allowance is 42 days [1]."
    # Rebuilt from the passage alone (before the fix), the neighbour's "42" is unsupported.
    assert unsupported_numbers(answer, [chunk]) == 1

    def _trace(expanded: bool) -> Any:
        candidate = SimpleNamespace(point_id=chunk.point_id, neighbor_expanded=expanded)
        return SimpleNamespace(candidates=[candidate])

    trace = _trace(True)
    widened = _with_neighbour_text([chunk], trace, repository, settings)
    assert widened[0].expanded_text and "42 days" in widened[0].expanded_text
    assert unsupported_numbers(answer, widened) == 0
    # A passage the first prompt didn't widen is left as it was.
    kept = _with_neighbour_text([chunk], _trace(False), repository, settings)
    assert kept[0].expanded_text is None


def test_a_missing_key_for_the_default_provider_is_a_503_before_any_search() -> None:
    from fastapi.testclient import TestClient

    from api.dependencies import clear_dependency_caches, get_embedding_provider
    from tests.routes_http.support import hermetic_app
    from tests.routes_http.support import make_settings as route_settings

    class _NoEmbedding:
        def embed_query(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("searched before the provider was checked")

        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"embedding used: {name}")

    clear_dependency_caches()
    app = hermetic_app(
        route_settings(llm_provider="openai", openai_api_key=None), QdrantClient(":memory:")
    )
    app.dependency_overrides[get_embedding_provider] = lambda: _NoEmbedding()
    with TestClient(app) as client:
        for path in ("/questions", "/questions/stream"):
            response = client.post(path, json={"question": "alpha?"})
            assert response.status_code == 503, path
            assert "OPENAI_API_KEY" in response.json()["detail"]
        # A per-request provider the server can't use is still the request's fault.
        bad = client.post("/questions", json={"question": "alpha?", "llm_provider": "anthropic"})
        assert bad.status_code == 400
    clear_dependency_caches()


def test_a_client_leaving_while_a_continue_waits_for_the_model_ends_quietly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.answer_stream as answer_stream
    from api.continuation import ContinuePlan
    from api.run_coordination import QuestionAbandoned

    def _abandoned(settings: AppSettings, should_stop: Any) -> Any:
        yield {"type": "stage", "stage": "waiting_for_model"}
        raise QuestionAbandoned

    monkeypatch.setattr(answer_stream, "wait_for_model", _abandoned)
    trace = SimpleNamespace(
        answer="Alpha is",
        prompt_messages=[{"role": "user", "content": "q"}],
        config=None,
        candidates=[],
    )
    pipeline = SimpleNamespace()
    events = list(
        answer_stream.StreamingAnswers.continue_stream(
            pipeline,  # type: ignore[arg-type]
            ContinuePlan(trace, []),  # type: ignore[arg-type]
            make_settings(),
        )
    )
    assert [e["type"] for e in events] == ["sources", "stage"]


def test_a_cached_answer_is_served_when_every_question_slot_is_busy() -> None:
    from fastapi.testclient import TestClient

    from api.admission import QuestionSlots
    from api.dependencies import clear_dependency_caches, get_question_slots
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings

    clear_dependency_caches()
    slots = QuestionSlots(1)
    app = hermetic_app(route_settings(), QdrantClient(":memory:"))
    app.dependency_overrides[get_question_slots] = lambda: slots
    with TestClient(app) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        assert client.post("/questions", json={"question": "alpha?"}).status_code == 200
        held = slots.try_acquire()
        assert held is not None
        try:
            again = client.post("/questions", json={"question": "alpha?"})
            assert again.status_code == 200 and again.json()["cached"] is True
            streamed = client.post("/questions/stream", json={"question": "alpha?"})
            assert streamed.status_code == 200 and '"cached": true' in streamed.text
            # Nothing stored for this one: the server is still full.
            assert client.post("/questions", json={"question": "beta?"}).status_code == 429
            regenerate = client.post("/questions", json={"question": "alpha?", "use_cache": False})
            assert regenerate.status_code == 429
        finally:
            held()
    clear_dependency_caches()


def test_history_keeps_a_bracketed_number_no_answer_could_cite() -> None:
    from api.citations import strip_citation_markers

    text = "See the Policy [2024] and the table [1, 2] and [3-4]."
    assert strip_citation_markers(text, 20) == "See the Policy [2024] and the table and."
    assert strip_citation_markers(text) == "See the Policy and the table and."


def test_ragas_contexts_are_the_prompts_passages_with_their_neighbour_text() -> None:
    from eval.answer_eval import _contexts_from_prompt

    content = (
        "Question:\nHow long?\n\nContext:\n"
        "[1] filename=a.md; page=1; section=Leave\nBefore. Leave is 28 days. After.\n\n"
        "[2] filename=b.md; page=2; section=Pay\nPaid monthly.\n\n"
        "Instructions:\n- Use only the context above."
    )
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": content}]
    assert _contexts_from_prompt(messages) == [
        "Before. Leave is 28 days. After.",
        "Paid monthly.",
    ]
    assert _contexts_from_prompt([{"role": "user", "content": "no context here"}]) == []


def test_an_answer_baseline_from_another_setup_is_refused(tmp_path: Any) -> None:
    import json

    from eval.harness import _check_answer_baseline

    metrics = {"fact_hit_rate": 1.0, "unsupported_numbers_mean": 0.0}
    run = tmp_path / "run.json"
    baseline = tmp_path / "baseline.json"

    def write(path: Any, fingerprint: dict[str, Any], values: dict[str, float]) -> None:
        path.write_text(json.dumps({"fingerprint": fingerprint, "metrics": values}))

    write(run, {"question_count": 5, "platform_machine": "x86_64"}, metrics)
    write(baseline, {"question_count": 19, "platform_machine": "arm64"}, metrics)
    assert _check_answer_baseline(run, baseline, 0.0) == 1  # a --limit 5 run
    write(baseline, {"question_count": 5, "platform_machine": "arm64"}, metrics)
    assert _check_answer_baseline(run, baseline, 0.0) == 0  # only the machine differs
    write(run, {"question_count": 5}, {**metrics, "unsupported_numbers_mean": 0.5})
    assert _check_answer_baseline(run, baseline, 0.1) == 1  # now gated


def test_from_queries_sends_follow_ups_and_scoped_questions_to_review() -> None:
    from eval.generate import rows_from_queries

    common = {"status": "ok", "filenames": ["a.md"], "filenames_cited": True, "kind": "question"}
    rows, review = rows_from_queries(
        [
            {**common, "question": "How long is leave?", "created_at": 1},
            {
                **common,
                "question": "And for contractors?",
                "retrieval_query": "Contractor leave?",
                "created_at": 2,
            },
            {
                **common,
                "question": "What is it?",
                "scope": {"filenames": ["a.md"]},
                "created_at": 3,
            },
        ]
    )
    assert [r["question"] for r in rows] == ["How long is leave?"]
    assert {r["question"] for r in review} == {"And for contractors?", "What is it?"}
    assert next(r for r in review if "searched_as" in r)["searched_as"] == "Contractor leave?"


def test_a_full_server_lets_a_request_wait_for_an_identical_run_without_a_slot() -> None:
    import threading

    from api.admission import QuestionSlots
    from api.answer_cache import AnswerCache, CachedAnswer
    from api.routes.questions import question_slot_unless_cached
    from api.run_coordination import NoSlotForRun, drain, follow_only

    slots = QuestionSlots(1)
    held = slots.try_acquire()
    assert held is not None
    event = threading.Event()
    admitted = question_slot_unless_cached(slots, lambda: None, lambda: event)
    assert admitted.follow is event and admitted.cached is None
    admitted.release()  # releases nothing
    assert slots.try_acquire() is None

    cache = AnswerCache(max_entries=4, ttl_seconds=60)
    key = ("k",)
    answer = CachedAnswer.__new__(CachedAnswer)
    event.set()
    cache.put(key, answer)
    assert drain(follow_only(cache, key, event, None)) is answer
    # The run it waited for left no answer: a 429, never a slotless run.
    with pytest.raises(NoSlotForRun):
        drain(follow_only(cache, ("other",), event, None))
    held()


@pytest.mark.parametrize(
    ("language", "start"),
    [
        ("German", "Die bereitgestellten"),
        ("français", "Les documents"),
        ("", "I do not have"),
        ("match", "I do not have"),
        ("Klingon", "I do not have"),
    ],
)
def test_the_canned_not_found_answer_is_in_the_answer_language(language: str, start: str) -> None:
    from api.prompting import insufficient_context_answer

    assert insufficient_context_answer(make_settings(answer_language=language)).startswith(start)


def test_a_stopped_answer_keeps_its_prompt_and_can_be_continued() -> None:
    from api.pipeline import IngestService, RagPipeline
    from tests.test_pipeline import FakeReranker, ListTraceStore

    class _Slow:
        def complete(self, messages: Sequence[Any], settings: AppSettings) -> str:
            return "unused"

        def stream(self, messages: Sequence[Any], settings: AppSettings) -> Any:
            yield "Alpha is documented"
            yield " and more"

    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    store = ListTraceStore()
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=_Slow(),
        settings=settings,
        trace_store=store,
    )
    stream = pipeline.answer_stream("alpha?", use_cache=False)
    for event in stream:
        if event["type"] == "delta":
            break
    stream.close()  # the reader pressed Stop
    trace = store.traces[-1]
    assert trace.status == "error" and trace.answer and trace.prompt_messages
    plan = pipeline.continuable(trace.trace_id, trace, settings)
    assert [c.filename for c in plan.chunks] == ["guide.txt"]


def test_a_source_header_echoed_inside_a_sentence_is_cut_to_its_marker() -> None:
    from api.citations import strip_echoed_source_headers, unsupported_numbers
    from api.reranking import RerankedChunk

    echoed = (
        "Unused days carry over until the end of March "
        "[ [1] filename=handbook.md; page=1; section=Leave ]. Requests go in writing [1]."
    )
    cleaned = strip_echoed_source_headers(echoed)
    assert cleaned == (
        "Unused days carry over until the end of March [1]. Requests go in writing [1]."
    )
    chunk = RerankedChunk(
        "p",
        "handbook.md",
        1,
        "Leave",
        "c",
        "Unused days carry over until the end of March. Requests go in writing.",
        1.0,
        1.0,
    )
    assert unsupported_numbers(cleaned, [chunk]) == 0
    # A header on its own line still goes whole.
    assert strip_echoed_source_headers("[1] filename=a.md; page=1\nAnswer [1].") == "Answer [1]."


def test_identical_questions_sent_together_on_one_slot_all_get_the_one_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    import time

    from fastapi.testclient import TestClient

    import api.routes.questions as question_routes
    from api.admission import QuestionSlots
    from api.dependencies import clear_dependency_caches, get_generator, get_question_slots
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings

    # No settling wait: the claim at admission alone must make them followers.
    monkeypatch.setattr(question_routes, "IDENTICAL_RUN_SETTLE_SECONDS", 0.0)

    class _Slow:
        calls = 0

        def complete(self, messages: Sequence[Any], settings: AppSettings) -> str:
            type(self).calls += 1
            time.sleep(0.5)
            return "Alpha is documented [1]."

        def stream(self, messages: Sequence[Any], settings: AppSettings) -> Any:
            yield self.complete(messages, settings)

    clear_dependency_caches()
    app = hermetic_app(route_settings(followups_enabled=False), QdrantClient(":memory:"))
    slots = QuestionSlots(1)
    app.dependency_overrides[get_question_slots] = lambda: slots
    app.dependency_overrides[get_generator] = lambda: _Slow()
    with TestClient(app) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        codes: list[int] = []

        def ask() -> None:
            codes.append(client.post("/questions", json={"question": "alpha?"}).status_code)

        threads = [threading.Thread(target=ask) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(codes) == [200, 200, 200, 200]
        assert _Slow.calls == 1
        # Nothing left claimed: the same question again is a cache hit.
        assert client.post("/questions", json={"question": "alpha?"}).json()["cached"] is True
    clear_dependency_caches()
