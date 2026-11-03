"""Tuning the relevance bar: the score report, trace comparison and the re-check."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from api.dependencies import get_query_log_store
from api.query_log import LoggedQuery, QueryLogStore
from api.relevance import MIN_TRACES_PER_SIDE, score_report, trace_diff
from api.tracing import QueryTrace, TraceCandidate, TraceConfig, TraceSummary
from tests.routes_http.support import keyed_client, upload_and_wait

FULL = {"X-API-Key": "full-key"}
READ = {"X-API-Key": "read-key"}


def _summary(status: str, score: float | None) -> TraceSummary:
    return TraceSummary("t", 0.0, "q", "sync", "ok", None, None, 0, 0, status, score)


def test_a_bar_is_suggested_only_from_enough_questions_on_both_sides() -> None:
    few = score_report([_summary("answered", 0.9), _summary("no_context", 0.01)], 0.15)
    assert few.suggested_min_score is None and "Needs at least" in few.note
    assert few.answered_count == 1 and sum(few.answered) == 1

    answered = [_summary("answered", 0.30 + i / 100) for i in range(MIN_TRACES_PER_SIDE)]
    unanswered = [
        _summary("insufficient_context", 0.05 + i / 1000) for i in range(MIN_TRACES_PER_SIDE)
    ]
    # One answered question scoring low: it stays on the wrong side of any good bar.
    report = score_report([*answered, *unanswered, _summary("uncited", 0.02)], 0.15)
    assert report.suggested_min_score is not None
    # Midway across the gap between the groups (0.069 and 0.30).
    assert report.suggested_min_score == round((0.069 + 0.30) / 2, 4)
    assert report.answered_count == MIN_TRACES_PER_SIDE + 1
    # Traced scores between it and today's 0.15 — none here — are the questions it would
    # decide differently.
    assert report.would_change == 0
    # Traces with no score or no status are left out.
    assert score_report([_summary("answered", None)], 0.15).answered_count == 0


def _candidate(
    point: str, score: float | None, selected: bool, drop: str | None = None
) -> TraceCandidate:
    return TraceCandidate(
        point, "a.txt", 1, "S", f"c-{point}", 0.1, score, drop is None, drop, selected, False
    )


def _trace(candidates: list[TraceCandidate], answer: str, top_k: int) -> QueryTrace:
    config = TraceConfig(
        llm_provider="ollama",
        rerank_top_k=top_k,
        max_context_chunks=2,
        llm_temperature=0.0,
        rerank_min_score=0.15,
        fused_top_n=5,
        filenames=None,
    )
    return QueryTrace(
        trace_id="t",
        created_at=0.0,
        question="q",
        mode="sync",
        status="ok",
        config=config,
        candidates=candidates,
        prompt_messages=None,
        answer=answer,
        cited_source_numbers=[1],
        timings=None,
        error=None,
    )


def test_two_traces_are_compared_by_settings_and_moved_candidates() -> None:
    a = _trace([_candidate("p1", 0.9, True), _candidate("p2", 0.5, True)], "A [1].", 3)
    b = _trace(
        [_candidate("p1", 0.9, True), _candidate("p2", 0.4, False, "below_min_score")],
        "A [1].",
        5,
    )

    diff = trace_diff(a, b)

    assert diff.same_answer
    assert [(s.name, s.a, s.b) for s in diff.settings] == [("rerank_top_k", 3, 5)]
    # p1 kept its place: only p2 is listed.
    assert [c.point_id for c in diff.candidates] == ["p2"]
    (moved,) = diff.candidates
    assert (moved.selected_a, moved.selected_b, moved.drop_reason_b) == (
        True,
        False,
        "below_min_score",
    )
    assert trace_diff(a, replace(a, answer="Other.")).same_answer is False


def test_the_report_diff_and_recheck_routes(tmp_path: Path) -> None:
    store = QueryLogStore(str(tmp_path / "log.db"), max_rows=100)
    with keyed_client(
        "full-key", api_read_key="read-key", extra_overrides={get_query_log_store: lambda: store}
    ) as client:
        assert (
            upload_and_wait(client, "g.txt", b"Intro\nalpha beta", headers=FULL)["state"] == "done"
        )
        first = client.post("/questions", json={"question": "alpha"}, headers=FULL).json()
        second = client.post(
            "/questions",
            json={"question": "alpha", "use_cache": False, "rerank_top_k": 2},
            headers=FULL,
        ).json()

        report = client.get("/traces/score-report", headers=READ)
        assert report.status_code == 200
        assert report.json()["answered_count"] + report.json()["unanswered_count"] == 2
        assert report.json()["suggested_min_score"] is None

        diff = client.get(f"/traces/{first['trace_id']}/diff/{second['trace_id']}", headers=READ)
        assert diff.status_code == 200
        assert ("rerank_top_k", 3, 2) in [
            (s["name"], s["a"], s["b"]) for s in diff.json()["settings"]
        ]
        assert client.get(f"/traces/{first['trace_id']}/diff/nope", headers=READ).status_code == 404

        store.add(LoggedQuery(1.0, "", "what is alpha", "insufficient_context", 0.01, "sync", None))
        store.add(LoggedQuery(2.0, "", "What is  ALPHA", "no_context", None, "sync", None))
        assert client.post("/query-log/recheck", headers=READ).status_code == 403
        rechecked = client.post("/query-log/recheck", headers=FULL)
        assert rechecked.status_code == 200
        body = rechecked.json()
        # Asked twice, searched once; the document now answers it.
        assert body["checked"] == 1 and body["covered_now"] == 1
        assert body["items"][0]["top_filename"] == "g.txt"
