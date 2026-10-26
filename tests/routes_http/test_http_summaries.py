"""POST /questions/summary and /questions/summary/stream."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.admission import QuestionSlots
from api.dependencies import get_app_settings, get_question_slots
from tests.routes_http.support import (
    ApiTestContext,
    make_settings,
    parse_sse_events,
    upload_and_wait,
    without_stage_events,
)

GUIDE = b"Intro\nAlpha is the first letter.\n\nUsage\nBeta comes second."


def _ingest(client: TestClient, name: str = "guide.txt", content: bytes = GUIDE) -> None:
    assert upload_and_wait(client, name, content)["state"] == "done"


def test_a_document_is_summarized_with_citations_and_coverage(
    api_context: ApiTestContext,
) -> None:
    _ingest(api_context.client)

    response = api_context.client.post(
        "/questions/summary", json={"mode": "summarize", "filenames": ["guide.txt"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Alpha is documented [1]."
    assert body["answer_status"] == "answered"
    assert [source["filename"] for source in body["sources"]] == ["guide.txt"]
    assert body["sources"][0]["chunk_id"]
    assert body["chunks_read"] == body["chunks_total"] >= 1
    prompt = api_context.generator.messages[-1]["content"]
    assert 'Summarize the document "guide.txt"' in prompt
    # Every passage reaches the prompt, not just the top few retrieved.
    assert "Alpha is the first letter" in prompt and "Beta comes second" in prompt


def test_the_summary_streams_the_question_frames(api_context: ApiTestContext) -> None:
    _ingest(api_context.client)

    response = api_context.client.post(
        "/questions/summary/stream",
        json={"mode": "summarize", "filenames": ["guide.txt"], "instruction": "letters"},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[0] == {"type": "stage", "stage": "reading_documents"}
    assert [event["type"] for event in without_stage_events(events)] == [
        "sources",
        "delta",
        "done",
    ]
    done = events[-1]
    assert done["answer"] == "Alpha is documented [1]."
    assert done["chunks_read"] >= 1
    assert "Focus on: letters" in api_context.generator.messages[-1]["content"]


def test_compare_needs_two_different_documents(api_context: ApiTestContext) -> None:
    client = api_context.client
    for body in (
        {"mode": "compare", "filenames": ["a.txt"]},
        {"mode": "compare", "filenames": ["a.txt", "a.txt"]},
        {"mode": "summarize", "filenames": ["a.txt", "b.txt"]},
        {"mode": "outline", "filenames": ["a.txt"]},
    ):
        assert client.post("/questions/summary", json=body).status_code == 422


def test_compare_reads_both_documents(api_context: ApiTestContext) -> None:
    _ingest(api_context.client, "a.txt", b"Plan\nPlan A costs five.")
    _ingest(api_context.client, "b.txt", b"Plan\nPlan B costs seven.")

    response = api_context.client.post(
        "/questions/summary", json={"mode": "compare", "filenames": ["a.txt", "b.txt"]}
    )

    assert response.status_code == 200
    prompt = api_context.generator.messages[-1]["content"]
    assert "Plan A costs five" in prompt and "Plan B costs seven" in prompt
    assert response.json()["chunks_read"] == 2


def test_unknown_documents_and_unconfigured_providers_are_refused_before_a_slot(
    api_context: ApiTestContext,
) -> None:
    _ingest(api_context.client)
    slots = QuestionSlots(1)
    api_context.app.dependency_overrides[get_question_slots] = lambda: slots

    for route in ("/questions/summary", "/questions/summary/stream"):
        missing = api_context.client.post(
            route, json={"mode": "summarize", "filenames": ["missing.txt"]}
        )
        assert missing.status_code == 404
        padded = api_context.client.post(
            route, json={"mode": "summarize", "filenames": [" guide.txt"]}
        )
        assert padded.status_code == 404
        provider = api_context.client.post(
            route,
            json={"mode": "summarize", "filenames": ["guide.txt"], "llm_provider": "openai"},
        )
        assert provider.status_code == 400
        assert "OPENAI_API_KEY" in provider.json()["detail"]
    assert api_context.generator.messages == []
    # No slot was left taken by any refusal.
    release = slots.try_acquire()
    assert release is not None
    release()


def test_a_full_server_answers_429(api_context: ApiTestContext) -> None:
    _ingest(api_context.client)
    slots = QuestionSlots(1)
    api_context.app.dependency_overrides[get_question_slots] = lambda: slots
    held = slots.try_acquire()
    assert held is not None

    response = api_context.client.post(
        "/questions/summary", json={"mode": "summarize", "filenames": ["guide.txt"]}
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "5"
    held()


def test_the_read_only_key_may_summarize(api_context: ApiTestContext) -> None:
    _ingest(api_context.client)
    keyed = make_settings(api_key="full-key", api_read_key="read-key")
    api_context.app.dependency_overrides[get_app_settings] = lambda: keyed
    body = {"mode": "summarize", "filenames": ["guide.txt"]}

    assert api_context.client.post("/questions/summary", json=body).status_code == 401
    allowed = api_context.client.post(
        "/questions/summary", json=body, headers={"X-API-Key": "read-key"}
    )
    assert allowed.status_code == 200


def test_summaries_are_counted_in_metrics(api_context: ApiTestContext) -> None:
    _ingest(api_context.client)

    api_context.client.post(
        "/questions/summary", json={"mode": "summarize", "filenames": ["guide.txt"]}
    )

    body = api_context.client.get("/metrics").text
    assert 'docrag_summaries_total{mode="summarize",outcome="ok"}' in body


def test_comparing_with_a_document_that_has_no_readable_text_names_it(
    api_context: ApiTestContext,
) -> None:
    from qdrant_client import models

    _ingest(api_context.client, "a.txt", b"Plan\nPlan A costs five.")
    _ingest(api_context.client, "empty.txt", b"Plan\nSoon to be blank.")
    api_context.qdrant.set_payload(
        collection_name=api_context.settings.qdrant_collection,
        payload={"text": "  "},
        points=models.Filter(
            must=[models.FieldCondition(key="filename", match=models.MatchValue(value="empty.txt"))]
        ),
    )
    body = {"mode": "compare", "filenames": ["a.txt", "empty.txt"]}

    refused = api_context.client.post("/questions/summary", json=body)
    assert refused.status_code == 400
    assert "'empty.txt' has no readable text" in refused.json()["detail"]

    events = parse_sse_events(api_context.client.post("/questions/summary/stream", json=body).text)
    assert events[-1]["type"] == "error"
    assert "'empty.txt' has no readable text" in events[-1]["detail"]


def test_summaries_have_their_own_cap_and_questions_keep_the_rest(
    api_context: ApiTestContext,
) -> None:
    _ingest(api_context.client)
    slots = QuestionSlots(4, summary_limit=1)
    held = slots.try_acquire("summary")
    assert held is not None
    api_context.app.dependency_overrides[get_question_slots] = lambda: slots
    try:
        for path in ("/questions/summary", "/questions/summary/stream"):
            busy = api_context.client.post(
                path, json={"mode": "summarize", "filenames": ["guide.txt"]}
            )
            assert busy.status_code == 429, path
            assert "MAX_CONCURRENT_SUMMARIES" in busy.json()["detail"]
            assert busy.headers["retry-after"] == "5"
        assert api_context.client.post("/questions", json={"question": "alpha"}).status_code == 200
    finally:
        held()


def test_a_brief_takes_several_documents_and_the_modes_check_their_counts(
    api_context: ApiTestContext,
) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        _ingest(api_context.client, name, GUIDE)
    client = api_context.client
    brief = client.post(
        "/questions/summary", json={"mode": "brief", "filenames": ["a.txt", "b.txt", "c.txt"]}
    )
    assert brief.status_code == 200, brief.text
    assert "covering the documents" in api_context.generator.messages[-1]["content"]
    for body in (
        {"mode": "brief", "filenames": ["a.txt"]},
        {"mode": "summarize", "filenames": ["a.txt", "b.txt"]},
        {"mode": "compare", "filenames": ["a.txt", "b.txt", "c.txt"]},
        {"mode": "brief", "filenames": ["a.txt", "a.txt"]},
    ):
        assert client.post("/questions/summary", json=body).status_code == 422, body
    assert client.get("/config").json()["summary_max_documents"] == 5


def test_a_summary_the_documents_could_not_support_is_traced_as_such(
    api_context: ApiTestContext,
) -> None:
    """The trace status decides what the query log counts as uncovered."""

    _ingest(api_context.client)
    api_context.generator.answer = (
        "The provided documents do not contain enough information to summarize that."
    )

    body = api_context.client.post(
        "/questions/summary", json={"mode": "summarize", "filenames": ["guide.txt"]}
    ).json()
    trace = api_context.client.get(f"/traces/{body['trace_id']}").json()

    assert body["answer_status"] == "insufficient_context"
    assert trace["status"] == "insufficient_context"
    assert trace["answer_status"] == "insufficient_context"
