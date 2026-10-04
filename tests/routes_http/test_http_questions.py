"""Question routes: /questions, /questions/stream, answer cache, admission, SSE keepalive."""

from __future__ import annotations

import threading
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.admission import QuestionSlots
from api.chunking import HeuristicTokenCounter
from api.dependencies import (
    clear_dependency_caches,
    get_app_settings,
    get_embedding_provider,
    get_generator,
    get_ollama_reachability_checker,
    get_qdrant_client,
    get_reranker,
    get_token_counter,
)
from api.main import create_app
from api.qdrant_schema import (
    EMBEDDING_MODEL_TAG_KEY,
    dense_vectors_config,
    sparse_vectors_config,
)
from tests.routes_http.support import (
    ApiTestContext,
    CapturingCompletionClient,
    FakeEmbeddingProvider,
    FakeGenerator,
    FakeReranker,
    count_generations,
    ingested_client,
    make_settings,
    parse_sse_events,
    raw_storage_client,
    upload_and_wait,
    without_stage_events,
)


def test_question_endpoint_runs_grounded_pipeline(api_context: ApiTestContext) -> None:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Alpha is documented [1]."
    assert payload["sources"] == [
        {
            "source_number": 1,
            "filename": "guide.txt",
            "page": 1,
            "section": "Intro",
            "chunk_id": payload["sources"][0]["chunk_id"],
            "text": "Intro alpha beta",
        }
    ]
    assert api_context.reranker.seen_documents == ["guide.txt › Intro\nIntro alpha beta"]
    assert "Intro alpha beta" in api_context.generator.messages[1]["content"]


def test_question_endpoint_rejects_blank_query(api_context: ApiTestContext) -> None:
    response = api_context.client.post("/questions", json={"question": "   "})

    assert response.status_code == 400
    assert "Query text is required" in response.json()["detail"]


def test_question_endpoint_rejects_empty_string_query(api_context: ApiTestContext) -> None:
    """Distinct from the whitespace case above: truly empty fails schema validation
    (min_length=1) before the pipeline's own strip-and-check ever runs."""

    response = api_context.client.post("/questions", json={"question": ""})

    assert response.status_code == 422


def test_question_endpoint_rejects_oversized_query(api_context: ApiTestContext) -> None:
    response = api_context.client.post("/questions", json={"question": "a" * 4001})

    assert response.status_code == 422


def test_question_endpoint_accepts_history(api_context: ApiTestContext) -> None:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post(
        "/questions",
        json={
            "question": "alpha",
            "history": [
                {"role": "user", "content": "Tell me about doc A."},
                {"role": "assistant", "content": "Doc A covers setup [1]."},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["timings"]["condense_ms"] > 0.0


def test_question_endpoint_rejects_more_than_twelve_history_messages(
    api_context: ApiTestContext,
) -> None:
    history = [{"role": "user", "content": f"turn {i}"} for i in range(13)]

    response = api_context.client.post("/questions", json={"question": "alpha", "history": history})

    assert response.status_code == 422


def test_question_endpoint_rejects_invalid_history_role(api_context: ApiTestContext) -> None:
    response = api_context.client.post(
        "/questions",
        json={"question": "alpha", "history": [{"role": "system", "content": "nope"}]},
    )

    assert response.status_code == 422


def test_question_endpoint_rejects_out_of_bounds_rerank_top_k(
    api_context: ApiTestContext,
) -> None:
    response = api_context.client.post(
        "/questions", json={"question": "alpha", "rerank_top_k": 999}
    )

    assert response.status_code == 422


def test_question_endpoint_rejects_max_context_chunks_above_rerank_top_k(
    api_context: ApiTestContext,
) -> None:
    response = api_context.client.post(
        "/questions",
        json={"question": "alpha", "rerank_top_k": 3, "max_context_chunks": 5},
    )

    assert response.status_code == 422


def test_question_endpoint_rejects_rerank_top_k_above_fused_top_n(
    api_context: ApiTestContext,
) -> None:
    """api_context's settings default to fused_top_n=5 — a business rule known only
    inside the pipeline, so this is a 400 (RetrievalConfigError), not the schema's 422."""

    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post("/questions", json={"question": "alpha", "rerank_top_k": 45})

    assert response.status_code == 400
    assert "fused_top_n" in response.json()["detail"]


def test_question_endpoint_accepts_valid_overrides_and_limits_sources(
    api_context: ApiTestContext,
) -> None:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post(
        "/questions",
        json={"question": "alpha", "rerank_top_k": 1, "max_context_chunks": 1},
    )

    assert response.status_code == 200
    assert len(response.json()["sources"]) <= 1


def test_question_endpoint_routes_to_openai_when_selected() -> None:
    clear_dependency_caches()
    settings = make_settings(openai_api_key="sk-test", openai_model="gpt-test")
    completion_client = CapturingCompletionClient()
    client, _ = ingested_client(settings, completion_client)

    response = client.post("/questions", json={"question": "alpha", "llm_provider": "openai"})

    assert response.status_code == 200
    assert completion_client.kwargs is not None
    assert completion_client.kwargs["model"] == "gpt-test"
    assert completion_client.kwargs["api_key"] == "sk-test"
    # Per-request override must not mutate the shared settings singleton.
    assert settings.llm_provider == "ollama"
    clear_dependency_caches()


def test_question_endpoint_openai_selected_without_key_returns_400() -> None:
    clear_dependency_caches()
    settings = make_settings(openai_api_key=None)
    completion_client = CapturingCompletionClient()
    client, _ = ingested_client(settings, completion_client)

    response = client.post("/questions", json={"question": "alpha", "llm_provider": "openai"})

    assert response.status_code == 400
    assert "OPENAI_API_KEY" in response.json()["detail"]
    # The provider never got called since routing rejected the request first.
    assert completion_client.kwargs is None
    clear_dependency_caches()


def test_question_endpoint_rejects_unknown_provider() -> None:
    clear_dependency_caches()
    settings = make_settings()
    completion_client = CapturingCompletionClient()
    client, _ = ingested_client(settings, completion_client)

    response = client.post("/questions", json={"question": "alpha", "llm_provider": "anthropic"})

    # Literal["ollama","openai"] rejects unknown values at the schema boundary.
    assert response.status_code == 422
    clear_dependency_caches()


def test_question_endpoint_returns_conflict_for_embedding_mismatch(
    api_context: ApiTestContext,
) -> None:
    api_context.qdrant.create_collection(
        collection_name=api_context.settings.qdrant_collection,
        vectors_config=dense_vectors_config(api_context.settings),
        sparse_vectors_config=sparse_vectors_config(api_context.settings),
        metadata={EMBEDDING_MODEL_TAG_KEY: "other-model"},
    )

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 409
    assert "Re-ingest documents" in response.json()["detail"]


def test_document_original_streams_with_its_length_and_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"x" * 200_000  # several stream blocks
    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert upload_and_wait(client, "big.txt", content)["state"] == "done"

        response = client.get("/documents/big.txt/original")

        assert response.status_code == 200
        assert response.content == content
        assert response.headers["content-length"] == str(len(content))
        assert response.headers["content-type"].startswith("text/plain")
        assert 'filename="big.txt"' in response.headers["content-disposition"]


def test_question_endpoint_includes_stage_timings(api_context: ApiTestContext) -> None:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 200
    timings = response.json()["timings"]
    assert timings is not None
    assert set(timings) == {
        "embed_ms",
        "search_ms",
        "rerank_ms",
        "generate_ms",
        "total_ms",
        "condense_ms",
        # Two separate stages: query-variant generation before embedding, neighbour
        # expansion after rerank. They were one ambiguous "expand_ms" that named the
        # former and was ordered like the latter.
        "query_expansion_ms",
        "context_expansion_ms",
        # The zero-citation retry's own LLM round trip, held apart from generate_ms —
        # the sync path used to fold it in while the streaming path dropped it entirely.
        "citation_retry_ms",
    }
    assert timings["condense_ms"] == 0.0
    # No history to condense and query expansion is off by default.
    assert timings["query_expansion_ms"] == 0.0


def test_question_endpoint_filters_by_filenames(api_context: ApiTestContext) -> None:
    assert upload_and_wait(api_context.client, "a.txt", b"alpha in a")["state"] == "done"
    assert upload_and_wait(api_context.client, "b.txt", b"alpha in b")["state"] == "done"

    response = api_context.client.post(
        "/questions", json={"question": "alpha", "filenames": ["b.txt"]}
    )

    assert response.status_code == 200
    assert [source["filename"] for source in response.json()["sources"]] == ["b.txt"]


def test_question_stream_endpoint_emits_sources_delta_done(
    api_context: ApiTestContext,
) -> None:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post("/questions/stream", json={"question": "alpha"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = without_stage_events(parse_sse_events(response.text))

    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "Alpha is documented [1]."
    assert "timings" in events[-1]


def test_question_stream_endpoint_done_timings_include_condense_ms(
    api_context: ApiTestContext,
) -> None:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post(
        "/questions/stream",
        json={
            "question": "What about the second one?",
            "history": [
                {"role": "user", "content": "Tell me about doc A."},
                {"role": "assistant", "content": "Doc A covers setup [1]."},
            ],
        },
    )
    events = parse_sse_events(response.text)

    done = events[-1]
    assert done["type"] == "done"
    assert done["timings"]["condense_ms"] > 0.0

    detail = api_context.client.get(f"/traces/{done['trace_id']}").json()
    assert detail["history_message_count"] == 2
    assert detail["condensed_question"]


def test_question_rejects_a_filenames_scope_above_the_cap(
    api_context: ApiTestContext,
) -> None:
    """filenames becomes a Qdrant MatchAny filter rebuilt once per query variant under
    query expansion, so an unbounded list is unbounded server work the client chooses.
    Both sibling list fields in QuestionRequest are already capped."""

    response = api_context.client.post(
        "/questions",
        json={"question": "alpha", "filenames": [f"doc-{i}.txt" for i in range(501)]},
    )

    assert response.status_code == 422


def test_question_stream_disables_proxy_buffering(api_context: ApiTestContext) -> None:
    assert upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha")["state"] == "done"

    response = api_context.client.post("/questions/stream", json={"question": "alpha"})

    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


def test_keepalive_pings_while_the_source_is_quiet_and_passes_errors_through() -> None:
    from api.main import _with_keepalive

    release = threading.Event()

    def slow() -> Generator[str]:
        yield "first"
        release.wait(5)
        yield "second"
        raise RuntimeError("boom")

    stream = _with_keepalive(slow(), 0.01)
    assert next(stream) == "first"
    assert next(stream) is None  # quiet: a heartbeat instead of blocking
    release.set()
    rest: list[str | None] = []
    with pytest.raises(RuntimeError, match="boom"):
        for item in stream:
            rest.append(item)
    assert "second" in rest


def test_keepalive_closes_the_source_when_the_client_goes_away() -> None:
    from api.main import _with_keepalive

    closed = threading.Event()
    produced = threading.Event()

    def endless() -> Generator[int]:
        try:
            count = 0
            while True:
                count += 1
                produced.set()
                yield count
        finally:
            closed.set()

    stream = _with_keepalive(endless(), 1.0)
    assert next(stream) == 1
    stream.close()

    assert closed.wait(2)


def test_a_repeated_question_is_answered_from_the_cache(api_context: ApiTestContext) -> None:
    uploaded = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert uploaded["state"] == "done"
    calls = count_generations(api_context.generator)

    first = api_context.client.post("/questions", json={"question": "alpha"}).json()
    second = api_context.client.post("/questions", json={"question": "  ALPHA "}).json()

    assert calls[0] == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]
    assert second["sources"] == first["sources"]
    # The inspector can still show how the original answer was retrieved.
    assert second["trace_id"] == first["trace_id"]


def test_regenerate_bypasses_the_cache_and_a_corpus_change_invalidates_it(
    api_context: ApiTestContext,
) -> None:
    client = api_context.client
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
    calls = count_generations(api_context.generator)

    client.post("/questions", json={"question": "alpha"})
    regenerated = client.post("/questions", json={"question": "alpha", "use_cache": False})
    assert regenerated.json()["cached"] is False
    assert calls[0] == 2

    assert upload_and_wait(client, "notes.txt", b"Intro\nalpha gamma")["state"] == "done"
    after_upload = client.post("/questions", json={"question": "alpha"}).json()
    assert after_upload["cached"] is False
    assert calls[0] == 3

    assert client.delete("/documents/notes.txt").status_code == 200
    assert client.post("/questions", json={"question": "alpha"}).json()["cached"] is False


def test_a_cached_answer_streams_as_sources_one_delta_and_done(
    api_context: ApiTestContext,
) -> None:
    client = api_context.client
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
    client.post("/questions", json={"question": "alpha"})

    events = without_stage_events(
        parse_sse_events(client.post("/questions/stream", json={"question": "alpha"}).text)
    )

    assert [event["type"] for event in events] == ["sources", "delta", "done"]
    assert events[-1]["cached"] is True
    assert events[1]["text"] == events[-1]["answer"]


def test_follow_up_questions_with_history_are_never_cached(api_context: ApiTestContext) -> None:
    client = api_context.client
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    client.post("/questions", json={"question": "alpha", "history": history})
    again = client.post("/questions", json={"question": "alpha", "history": history}).json()

    assert again["cached"] is False


def test_the_cache_can_be_turned_off() -> None:
    clear_dependency_caches()
    settings = make_settings(answer_cache_size=0)
    app = create_app()
    generator = FakeGenerator()
    qdrant = QdrantClient(":memory:")
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_generator] = lambda: generator
    app.dependency_overrides[get_token_counter] = lambda: HeuristicTokenCounter()
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: False
    try:
        with TestClient(app) as client:
            assert upload_and_wait(client, "guide.txt", b"Intro\nalpha")["state"] == "done"
            client.post("/questions", json={"question": "alpha"})
            assert client.post("/questions", json={"question": "alpha"}).json()["cached"] is False
    finally:
        clear_dependency_caches()


def test_tags_are_set_listed_preserved_on_re_upload_and_scope_questions(
    api_context: ApiTestContext,
) -> None:
    client = api_context.client
    assert upload_and_wait(client, "contract.txt", b"Intro\nalpha terms")["state"] == "done"
    assert upload_and_wait(client, "notes.txt", b"Intro\nalpha notes")["state"] == "done"

    response = client.patch(
        "/documents/contract.txt/tags", json={"tags": [" Legal ", "legal", "2026"]}
    )

    assert response.status_code == 200
    assert response.json() == {"filename": "contract.txt", "tags": ["Legal", "2026"]}
    assert client.get("/documents").json()["tags"] == {"contract.txt": ["Legal", "2026"]}

    scoped = client.post("/questions", json={"question": "alpha", "tags": ["Legal"]}).json()
    assert {source["filename"] for source in scoped["sources"]} == {"contract.txt"}

    # A re-upload replaces the points; the tags belong to the document and carry over.
    assert upload_and_wait(client, "contract.txt", b"Intro\nalpha v2")["state"] == "done"
    assert client.get("/documents").json()["tags"] == {"contract.txt": ["Legal", "2026"]}

    cleared = client.patch("/documents/contract.txt/tags", json={"tags": []})
    assert cleared.json()["tags"] == []
    assert client.get("/documents").json()["tags"] == {}


def test_questions_past_the_concurrency_cap_are_429_with_retry_after(
    api_context: ApiTestContext,
) -> None:
    from api.dependencies import get_question_slots

    client = api_context.client
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
    busy = QuestionSlots(1)
    held = busy.try_acquire()
    api_context.app.dependency_overrides[get_question_slots] = lambda: busy

    refused = client.post("/questions", json={"question": "alpha", "use_cache": False})
    refused_stream = client.post("/questions/stream", json={"question": "alpha"})

    assert refused.status_code == 429
    assert refused.headers["retry-after"] == "5"
    assert refused_stream.status_code == 429
    assert held is not None
    held()
    # Both answer paths give their slot back, so one-at-a-time questions keep working.
    for _ in range(2):
        assert (
            client.post("/questions", json={"question": "alpha", "use_cache": False}).status_code
            == 200
        )
        assert client.post("/questions/stream", json={"question": "alpha"}).status_code == 200
