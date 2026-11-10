"""Pass 16 tuning fixes: a re-run repeats the question as it was asked, the query log
keeps a question's scope and kind, and the cache keys carry every setting a workspace
can change."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches, get_query_log_store
from api.query_log import LoggedQuery, QueryLogStore
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait


def _client(**overrides: Any) -> Iterator[TestClient]:
    clear_dependency_caches()
    settings = make_settings(**overrides)
    with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as client:
        yield client
    clear_dependency_caches()


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield from _client(answer_cache_size=16)


@pytest.fixture
def logged(tmp_path: Path) -> Iterator[tuple[TestClient, QueryLogStore]]:
    clear_dependency_caches()
    store = QueryLogStore(str(tmp_path / "log.db"), max_rows=100)
    app = hermetic_app(make_settings(answer_cache_size=16), QdrantClient(":memory:"))
    app.dependency_overrides[get_query_log_store] = lambda: store
    with TestClient(app) as client:
        yield client, store
    clear_dependency_caches()


def _ask(client: TestClient, question: str, **body: Any) -> dict[str, Any]:
    response = client.post("/questions", json={"question": question, **body})
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_an_unchanged_rerun_repeats_the_question_as_asked(client: TestClient) -> None:
    body = b"# Guide\n" + b"\n\n".join(
        f"## Part {n}\nalpha beta gamma fact {n} {'word ' * 40}".encode() for n in range(6)
    )
    assert upload_and_wait(client, "guide.md", body)["state"] == "done"
    trace_id = _ask(
        client,
        "What is alpha?",
        max_context_chunks=1,
        rerank_top_k=2,
        strong_matches_only=True,
        passages_only=True,
    )["trace_id"]

    same = client.post(f"/traces/{trace_id}/rerun", json={}).json()

    assert same["diff"]["settings"] == []
    assert same["diff"]["candidates"] == []


def test_a_scoped_miss_is_logged_as_a_question_and_rechecked_in_its_scope(
    logged: tuple[TestClient, QueryLogStore],
) -> None:
    client, store = logged
    assert upload_and_wait(client, "a.txt", b"Intro\nalpha beta gamma")["state"] == "done"
    assert upload_and_wait(client, "b.txt", b"Intro\nunrelated words here")["state"] == "done"
    # Nothing on page 99: no candidates at all, still a question, logged with its scope.
    _ask(client, "What is alpha?", filenames=["a.txt"], page_min=99)
    (row,) = store.recent("", limit=10)
    assert row.kind == "question" and row.status == "insufficient_context"
    assert row.scope == {"filenames": ["a.txt"], "narrowing": {"page_min": 99}}

    # Asked over b.txt only and declined there; a whole-workspace search would find
    # a.txt on top and call it covered.
    store.add(
        LoggedQuery(
            2e9,
            "",
            "What is alpha?",
            "insufficient_context",
            0.99,
            "sync",
            None,
            top_filename="b.txt",
            scope={"filenames": ["b.txt"]},
        )
    )
    rechecked = client.post("/query-log/recheck").json()
    assert rechecked["checked"] == 2 and rechecked["covered_now"] == 0
    assert {item["top_filename"] for item in rechecked["items"]} == {"b.txt", None}


def test_a_cached_answer_is_logged_each_time_it_is_served(
    logged: tuple[TestClient, QueryLogStore],
) -> None:
    client, store = logged
    assert upload_and_wait(client, "a.txt", b"Intro\nalpha beta gamma")["state"] == "done"
    first = _ask(client, "What is alpha?")
    again = _ask(client, "What is alpha?")
    assert again["cached"] is True
    rows = store.recent("", limit=10)
    assert [row.cached for row in rows] == [True, False]
    assert rows[0].trace_id == first["trace_id"] and rows[0].question == "What is alpha?"

    summary = {"mode": "summarize", "filenames": ["a.txt"]}
    for _ in range(2):
        assert client.post("/questions/summary", json=summary).status_code == 200
    latest = store.recent("", limit=1)[0]
    assert latest.kind == "summary" and latest.cached


def test_a_workspace_can_refuse_weak_matches_and_answer_from_fewer_passages(
    client: TestClient,
) -> None:
    body = b"# Guide\n" + b"\n\n".join(
        f"## Part {n}\nalpha beta gamma fact {n} {'word ' * 40}".encode() for n in range(6)
    )
    assert upload_and_wait(client, "guide.md", body)["state"] == "done"
    saved = client.put(
        "/workspaces/default/settings",
        json={"rerank_min_score": 1.0, "rerank_fallback_count": 0, "max_context_chunks": 1},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["effective"]["max_context_chunks"] == 1

    refused = _ask(client, "What is alpha?")
    # Nothing clears 1.0 and the fallback is off here: not answered from weak matches.
    assert refused["weak_matches"] is False and refused["answer_status"] == "no_context"
    client.put("/workspaces/default/settings", json={"max_context_chunks": 1})
    answered = _ask(client, "What is alpha?")
    trace = client.get(f"/traces/{answered['trace_id']}").json()
    assert trace["config"]["max_context_chunks"] == 1
    refused = client.put("/workspaces/default/settings", json={"max_context_chunks": 0})
    assert refused.status_code == 422


def test_the_score_report_counts_only_traces_since_and_under_a_bar() -> None:
    from api.relevance import score_report
    from api.tracing import TraceSummary

    def summary(at: float, score: float, status: str, bar: float | None) -> TraceSummary:
        return TraceSummary(
            trace_id=f"t{at}-{score}",
            created_at=at,
            question="q",
            mode="sync",
            status="ok",
            llm_provider="ollama",
            total_ms=1.0,
            candidate_count=1,
            kept_count=1,
            answer_status=status,
            top_rerank_score=score,
            rerank_min_score=bar,
        )

    old = [summary(10, 0.9, "answered", 0.15), summary(11, 0.1, "insufficient_context", None)]
    new = [summary(100, 0.8, "answered", 0.3), summary(101, 0.2, "insufficient_context", 0.3)]
    everything = score_report([*old, *new], 0.3)
    assert everything.answered_count == 2 and everything.filtered_out == 0
    recent = score_report([*old, *new], 0.3, since=50)
    assert (recent.answered_count, recent.unanswered_count, recent.filtered_out) == (1, 1, 2)
    under = score_report([*old, *new], 0.3, bar=0.3)
    assert under.filtered_out == 2 and under.answered_count == 1


def test_an_archived_trace_keeps_its_bar(tmp_path: Path) -> None:
    import time

    from api.trace_archive import TraceArchive
    from api.tracing import QueryTrace, TraceConfig

    archive = TraceArchive(str(tmp_path / "t.db"), 100)
    config = TraceConfig("ollama", 8, 6, 0.0, 0.4, 50, None)
    archive.add(
        QueryTrace("t1", time.time(), "q", "sync", "ok", config, [], None, "a", [], None, None)
    )
    (row,) = archive.summaries("", 5, None)
    assert row.rerank_min_score == 0.4


def test_what_if_tries_a_bar_on_the_latest_traced_questions(client: TestClient) -> None:
    assert upload_and_wait(client, "guide.md", b"# Guide\nalpha beta gamma")["state"] == "done"
    for question in ("What is alpha?", "What is beta?"):
        _ask(client, question, use_cache=False)

    # Both answered; at a bar nothing clears, neither would be.
    strict = client.post(
        "/traces/what-if", json={"rerank_min_score": 1.0, "only_unanswered": False}
    )
    assert strict.status_code == 200, strict.text
    body = strict.json()
    assert (body["checked"], body["would_answer"], body["newly_unanswered"]) == (2, 0, 2)
    assert body["settings"] == {"rerank_min_score": 1.0}
    assert {item["question"] for item in body["items"]} == {"What is alpha?", "What is beta?"}
    # Only the unanswered ones (none here).
    assert client.post("/traces/what-if", json={}).json()["checked"] == 0
    assert client.post("/traces/what-if", json={"limit": 0}).status_code == 422


def test_the_cli_runs_a_what_if(client: TestClient) -> None:
    import io

    from cli.__main__ import main
    from cli.client import DocragClient

    assert upload_and_wait(client, "guide.md", b"# Guide\nalpha beta gamma")["state"] == "done"
    _ask(client, "What is alpha?")
    docrag = DocragClient("http://testserver", None, http=client, sleep=lambda _: None)
    out = io.StringIO()
    assert main(["traces", "what-if", "--min-score", "1.0", "--all"], client=docrag, out=out) == 0
    assert "0 of 1 would be answered (0 newly, 1 no longer)" in out.getvalue()


def test_a_server_run_records_what_the_server_reports(client: TestClient) -> None:
    from cli.client import DocragClient
    from eval.harness import server_fingerprint
    from eval.metrics import baseline_shape_problems

    docrag = DocragClient("http://testserver", None, http=client, sleep=lambda _: None)
    fingerprint = server_fingerprint(docrag, "http://testserver")
    config = client.get("/config").json()
    assert fingerprint["reranker_model"] == config["reranker_model"]
    assert fingerprint["rerank_min_score"] == config["rerank_min_score"]
    assert fingerprint["server_version"] == client.get("/health").json()["version"]
    assert fingerprint["workspace"] == "default"

    run = {
        "k_values": [1],
        "question_count": 1,
        "fingerprint": fingerprint,
        "aggregate": {"filename": {"hit@1": 1.0}},
    }
    # Another host, same settings: comparable; another bar: not.
    elsewhere = {**run, "fingerprint": {**fingerprint, "server": "http://other"}}
    assert baseline_shape_problems(run, elsewhere) == []
    stricter = {**run, "fingerprint": {**fingerprint, "rerank_min_score": 0.5}}
    assert baseline_shape_problems(run, stricter)
