"""The opt-in query log (QUERY_LOG_ENABLED) and GET /query-log."""

from __future__ import annotations

from pathlib import Path

from api.dependencies import get_query_log_store
from api.query_log import LoggedQuery, QueryLogStore
from tests.routes_http.support import keyed_client, upload_and_wait

FULL = {"X-API-Key": "full-key"}
READ = {"X-API-Key": "read-key"}


def _row(question: str, status: str = "ok", workspace: str = "", at: float = 1.0) -> LoggedQuery:
    return LoggedQuery(at, workspace, question, status, 0.5, "sync", None)


def test_the_store_keeps_rows_per_workspace_newest_first_and_trims(tmp_path: Path) -> None:
    store = QueryLogStore(str(tmp_path / "log.db"), max_rows=3)
    for n in range(1, 5):
        store.add(_row(f"q{n}", at=float(n)))
    store.add(_row("elsewhere", workspace="legal", at=9.0))
    store.add(_row("not covered", status="insufficient_context", at=10.0))

    questions = [row.question for row in store.recent("", limit=10)]
    assert questions[0] == "not covered"
    assert "elsewhere" not in questions
    assert [row.question for row in store.recent("", limit=10, uncovered=True)] == ["not covered"]
    # Trimmed to the cap now and then (every 100 inserts, the first included).
    many = QueryLogStore(str(tmp_path / "many.db"), max_rows=5)
    for n in range(102):
        many.add(_row(f"q{n}", at=float(n)))
    assert len(many.recent("", limit=500)) <= 5 + 100


def test_questions_are_logged_and_listed_with_the_full_key_only(tmp_path: Path) -> None:
    store = QueryLogStore(str(tmp_path / "log.db"), max_rows=100)
    with keyed_client(
        "full-key",
        api_read_key="read-key",
        extra_overrides={get_query_log_store: lambda: store},
        trace_enabled=False,  # logged even with the trace buffer off
    ) as client:
        assert (
            upload_and_wait(client, "g.txt", b"Intro\nalpha beta", headers=FULL)["state"] == "done"
        )
        assert (
            client.post("/questions", json={"question": "alpha"}, headers=READ).status_code == 200
        )

        listed = client.get("/query-log", headers=FULL)
        assert listed.status_code == 200
        (entry,) = listed.json()["queries"]
        assert entry["question"] == "alpha" and entry["status"] == "ok"
        assert entry["top_score"] is not None
        assert client.get("/query-log", headers=READ).status_code == 403
        assert client.get("/query-log?uncovered=true", headers=FULL).json()["queries"] == []


def test_the_route_is_404_when_the_log_is_off() -> None:
    with keyed_client("full-key", extra_overrides={get_query_log_store: lambda: None}) as client:
        assert client.get("/query-log", headers=FULL).status_code == 404


def test_a_deleted_workspaces_questions_are_removed(tmp_path: Path) -> None:
    store = QueryLogStore(str(tmp_path / "log.db"), max_rows=100)
    store.add(_row("legal one", workspace="legal"))
    store.add(_row("default one"))

    assert store.delete_workspace("legal") == 1
    assert store.recent("legal", limit=10) == []
    assert [row.question for row in store.recent("", limit=10)] == ["default one"]


def test_rows_name_their_documents_and_filter_by_one(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "log.db"
    # A log from before documents were recorded: opened, it gains the column.
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE queries (created_at REAL NOT NULL, workspace TEXT NOT NULL "
            "DEFAULT '', question TEXT NOT NULL, status TEXT NOT NULL, top_score REAL, "
            "mode TEXT NOT NULL, trace_id TEXT)"
        )
        conn.execute("INSERT INTO queries VALUES (1, '', 'old', 'ok', 0.5, 'sync', NULL)")
    store = QueryLogStore(str(path), max_rows=100)
    store.add(LoggedQuery(2.0, "", "a?", "ok", 0.9, "sync", None, ("a.txt", "b.txt")))
    store.add(LoggedQuery(3.0, "", "aa?", "ok", 0.9, "sync", None, ("aa.txt",)))

    assert [r.question for r in store.recent("", limit=10, filename="a.txt")] == ["a?"]
    assert [r.filenames for r in store.recent("", limit=10)] == [
        ("aa.txt",),
        ("a.txt", "b.txt"),
        (),
    ]


def test_only_an_answered_question_is_drawn_from_its_documents() -> None:
    from dataclasses import replace

    from api.query_log import logged_query
    from api.tracing import QueryTrace, TraceCandidate

    candidate = TraceCandidate("p", "a.txt", 1, "S", "c", 0.1, 0.02, True, None, True, False)
    trace = QueryTrace(
        trace_id="t",
        created_at=1.0,
        question="q",
        mode="sync",
        status="ok",
        config=None,
        candidates=[candidate],
        prompt_messages=None,
        answer="A [1].",
        cited_source_numbers=[1],
        timings=None,
        error=None,
    )
    assert logged_query(trace).filenames == ("a.txt",)
    # Declined over weak matches: it read a.txt but drew nothing from it.
    assert logged_query(replace(trace, status="insufficient_context")).filenames == ()
