"""The operational metrics: answers by status, gauges, build info."""

from __future__ import annotations

import re
from pathlib import Path

from api.version import app_version
from tests.routes_http.support import ApiTestContext, keyed_client, upload_and_wait


def _value(body: str, sample: str) -> float:
    match = re.search(rf"^{re.escape(sample)} ([0-9.e+-]+)$", body, flags=re.MULTILINE)
    assert match is not None, f"{sample} not in /metrics"
    return float(match.group(1))


def test_answers_are_counted_by_grounding_status_including_cache_hits(
    api_context: ApiTestContext,
) -> None:
    assert upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
    before = api_context.client.get("/metrics").text
    answered = 'docrag_answers_total{status="answered"}'
    uncited = 'docrag_answers_total{status="uncited"}'
    start_answered = _value(before, answered) if answered in before else 0.0
    start_uncited = _value(before, uncited) if uncited in before else 0.0

    api_context.client.post("/questions", json={"question": "alpha"})
    api_context.client.post("/questions", json={"question": "alpha"})  # from the cache
    api_context.generator.answer = "Alpha, with no citation at all."
    api_context.client.post("/questions/stream", json={"question": "beta", "use_cache": False})

    body = api_context.client.get("/metrics").text
    assert _value(body, answered) == start_answered + 2
    assert _value(body, uncited) == start_uncited + 1


def test_build_info_and_gauges_are_exported(api_context: ApiTestContext) -> None:
    assert upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
    api_context.client.get("/documents")

    body = api_context.client.get("/metrics").text

    assert f'docrag_build_info{{version="{app_version()}"}} 1.0' in body
    assert _value(body, 'docrag_corpus_documents{workspace="default"}') == 1
    # Nothing is running or queued between requests.
    assert _value(body, "docrag_question_slots_in_use") == 0
    assert _value(body, "docrag_ingest_backlog_jobs") == 0
    assert _value(body, "docrag_ingest_backlog_bytes") == 0


def test_health_reports_the_running_version(api_context: ApiTestContext) -> None:
    assert api_context.client.get("/health").json()["version"] == app_version()


def test_feedback_ratings_are_counted(tmp_path: Path) -> None:
    from api.dependencies import get_feedback_store
    from api.feedback import FeedbackStore

    store = FeedbackStore(str(tmp_path / "f.db"))
    with keyed_client(
        "", extra_overrides={get_feedback_store: lambda: store}, feedback_enabled=True
    ) as client:
        sample = 'docrag_feedback_total{rating="down"}'
        before = client.get("/metrics").text
        start = _value(before, sample) if sample in before else 0.0
        rating = {"question": "q", "answer_excerpt": "a", "rating": "down"}
        assert client.post("/feedback", json=rating).status_code == 201
        assert _value(client.get("/metrics").text, sample) == start + 1
