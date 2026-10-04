"""Trace and feedback routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

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
from tests.routes_http.support import (
    ApiTestContext,
    FakeEmbeddingProvider,
    FakeGenerator,
    FakeReranker,
    hermetic_app,
    make_settings,
    parse_sse_events,
    upload_and_wait,
    without_stage_events,
)


def test_feedback_returns_404_when_disabled(api_context: ApiTestContext) -> None:
    # feedback_enabled is False by default (make_settings() doesn't set it) — same
    # 404 shape as any other absent route, not a distinct "feature unavailable" body.
    response = api_context.client.post(
        "/feedback",
        json={"question": "q", "answer_excerpt": "a", "rating": "up"},
    )

    assert response.status_code == 404


def test_feedback_records_a_rating_and_survives_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: feedback_enabled=True -> POST /feedback -> a brand-new
    FeedbackStore instance pointed at the same file can read it back — simulating a
    process restart between submission and any later inspection of the data.

    get_feedback_store() (api/dependencies.py) reads settings via a direct
    get_app_settings() call, not FastAPI's Depends — same pattern as the sqlite job
    store and raw-document-store tests above.
    """

    clear_dependency_caches()
    db_path = str(tmp_path / "feedback.db")
    settings = make_settings(feedback_enabled=True, feedback_store_path=db_path)
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)

    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_qdrant_client] = lambda: QdrantClient(":memory:")
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_generator] = lambda: FakeGenerator()
    app.dependency_overrides[get_token_counter] = lambda: HeuristicTokenCounter()
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: False

    try:
        with TestClient(app) as client:
            response = client.post(
                "/feedback",
                json={
                    "trace_id": "trace-1",
                    "question": "What is DocRAG?",
                    "answer_excerpt": "A self-hostable document Q&A system [1].",
                    "cited_filenames": ["docrag.md"],
                    "rating": "up",
                    "citation_source_number": 1,
                },
            )
            assert response.status_code == 201
            feedback_id = response.json()["id"]
    finally:
        monkeypatch.undo()
        clear_dependency_caches()

    from api.feedback import FeedbackStore

    reloaded = FeedbackStore(db_path).list_recent()
    assert len(reloaded) == 1
    assert reloaded[0].id == feedback_id
    assert reloaded[0].question == "What is DocRAG?"
    assert reloaded[0].cited_filenames == ["docrag.md"]
    assert reloaded[0].citation_source_number == 1


def test_feedback_rejects_an_invalid_rating(api_context: ApiTestContext) -> None:
    response = api_context.client.post(
        "/feedback",
        json={"question": "q", "answer_excerpt": "a", "rating": "sideways"},
    )

    assert response.status_code == 422


def test_question_endpoint_returns_a_trace_id(api_context: ApiTestContext) -> None:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 200
    assert response.json()["trace_id"]


def test_traces_endpoint_lists_and_fetches_a_recorded_trace(
    api_context: ApiTestContext,
) -> None:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"
    question_response = api_context.client.post("/questions", json={"question": "alpha"})
    trace_id = question_response.json()["trace_id"]

    list_response = api_context.client.get("/traces")
    assert list_response.status_code == 200
    summaries = list_response.json()["traces"]
    assert len(summaries) == 1
    assert summaries[0]["trace_id"] == trace_id
    assert summaries[0]["status"] == "ok"
    assert summaries[0]["mode"] == "sync"

    detail_response = api_context.client.get(f"/traces/{trace_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["trace_id"] == trace_id
    assert detail["question"] == "alpha"
    assert detail["answer"] == "Alpha is documented [1]."
    assert detail["cited_source_numbers"] == [1]
    assert len(detail["candidates"]) == 1
    assert detail["candidates"][0]["filename"] == "guide.txt"
    assert detail["candidates"][0]["kept"] is True
    assert detail["prompt_messages"] == api_context.generator.messages
    assert detail["config"]["llm_provider"] == "ollama"
    assert detail["timings"]["total_ms"] >= 0


def test_traces_endpoint_returns_404_for_unknown_trace_id(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/traces/does-not-exist")

    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_question_stream_endpoint_trace_id_matches_across_events_and_is_fetchable(
    api_context: ApiTestContext,
) -> None:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post("/questions/stream", json={"question": "alpha"})
    events = without_stage_events(parse_sse_events(response.text))

    trace_id = events[0]["trace_id"]
    assert trace_id
    assert events[-1]["trace_id"] == trace_id

    detail_response = api_context.client.get(f"/traces/{trace_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["mode"] == "stream"


def test_question_endpoint_trace_id_null_and_traces_empty_when_tracing_disabled() -> None:
    clear_dependency_caches()
    settings = make_settings(trace_enabled=False)
    qdrant = QdrantClient(":memory:")
    app = hermetic_app(settings, qdrant)
    with TestClient(app) as client:
        upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")
        response = client.post("/questions", json={"question": "alpha"})
        traces_response = client.get("/traces")
    clear_dependency_caches()

    assert response.json()["trace_id"] is None
    assert traces_response.json()["traces"] == []


def test_feedback_list_returns_404_when_disabled(api_context: ApiTestContext) -> None:
    assert api_context.client.get("/feedback").status_code == 404


def test_feedback_list_rejects_an_out_of_range_limit(api_context: ApiTestContext) -> None:
    assert api_context.client.get("/feedback", params={"limit": 0}).status_code == 422
    assert api_context.client.get("/feedback", params={"limit": 100000}).status_code == 422


def test_feedback_list_returns_recorded_ratings_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read side of the feature: FeedbackStore.list_recent was implemented and
    tested but had no route, so captured ratings were reachable only by opening the
    sqlite file by hand."""

    clear_dependency_caches()
    db_path = str(tmp_path / "feedback.db")
    settings = make_settings(feedback_enabled=True, feedback_store_path=db_path)
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)

    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_qdrant_client] = lambda: QdrantClient(":memory:")
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_generator] = lambda: FakeGenerator()
    app.dependency_overrides[get_token_counter] = lambda: HeuristicTokenCounter()
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: False

    try:
        with TestClient(app) as client:
            for rating in ("up", "down"):
                assert (
                    client.post(
                        "/feedback",
                        json={
                            "trace_id": f"trace-{rating}",
                            "question": f"question {rating}",
                            "answer_excerpt": f"answer {rating} [1].",
                            "cited_filenames": ["docrag.md"],
                            "rating": rating,
                        },
                    ).status_code
                    == 201
                )

            listed = client.get("/feedback").json()["feedback"]
            limited = client.get("/feedback", params={"limit": 1}).json()["feedback"]
    finally:
        monkeypatch.undo()
        clear_dependency_caches()

    assert [item["rating"] for item in listed] == ["down", "up"]
    assert listed[0]["question"] == "question down"
    assert listed[0]["cited_filenames"] == ["docrag.md"]
    assert listed[0]["trace_id"] == "trace-down"
    assert listed[0]["citation_source_number"] is None
    assert len(limited) == 1
