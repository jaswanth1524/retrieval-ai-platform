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
