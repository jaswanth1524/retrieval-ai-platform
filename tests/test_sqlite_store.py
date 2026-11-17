"""``add_missing_columns``: how every sqlite store upgrades a file an older version wrote."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from api import feedback, jobs, query_log, trace_archive
from api.sqlite_store import add_missing_columns


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_missing_columns_are_added_with_their_defaults_and_present_ones_left() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE things (id TEXT, kind TEXT)")
    conn.execute("INSERT INTO things VALUES ('a', 'old')")

    added = add_missing_columns(
        conn,
        "things",
        [("kind", "TEXT"), ("count", "INTEGER NOT NULL DEFAULT 3"), ("note", "TEXT")],
    )

    assert added == ["count", "note"]
    assert _columns(conn, "things") == ["id", "kind", "count", "note"]
    assert conn.execute("SELECT kind, count, note FROM things").fetchone() == ("old", 3, None)
    # Opening the same file again changes nothing.
    assert add_missing_columns(conn, "things", [("count", "INTEGER NOT NULL DEFAULT 3")]) == []


def test_a_not_null_column_without_a_default_is_refused_before_any_old_file_needs_it() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE things (id TEXT)")

    with pytest.raises(ValueError, match="needs a DEFAULT"):
        # Refused even though the column exists: a file without it would fail later.
        add_missing_columns(conn, "things", [("id", "TEXT NOT NULL")])


@pytest.mark.parametrize("name", ["things; DROP TABLE x", "1things", ""])
def test_names_that_are_not_identifiers_are_refused(name: str) -> None:
    conn = sqlite3.connect(":memory:")

    with pytest.raises(ValueError, match="not a table or column name"):
        add_missing_columns(conn, name, [])
    with pytest.raises(ValueError, match="not a table or column name"):
        add_missing_columns(conn, "things", [(name, "TEXT")])


@pytest.mark.parametrize(
    "columns",
    [
        jobs._ADDED_COLUMNS,
        query_log._ADDED_COLUMNS,
        trace_archive._SUMMARY_COLUMNS,
    ],
)
def test_every_store_s_added_columns_can_be_added_to_an_old_table(
    columns: tuple[tuple[str, str], ...],
) -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE old (rowid_only INTEGER)")
    conn.execute("INSERT INTO old VALUES (1)")

    assert add_missing_columns(conn, "old", columns) == [name for name, _ in columns]


def test_a_jobs_file_from_before_workspaces_opens_with_every_column(tmp_path: Path) -> None:
    path = tmp_path / "jobs.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, filename TEXT NOT NULL, state TEXT NOT NULL,"
        " chunks_total INTEGER NOT NULL, chunks_done INTEGER NOT NULL, error TEXT,"
        " result_json TEXT, created_at_wall REAL NOT NULL)"
    )
    conn.commit()
    conn.close()

    jobs.SqliteIngestJobStore(str(path), max_retained=10)

    conn = sqlite3.connect(path)
    assert set(_columns(conn, "jobs")) >= {name for name, _ in jobs._ADDED_COLUMNS}
    conn.close()


def test_a_feedback_file_without_kind_gains_it(tmp_path: Path) -> None:
    path = str(tmp_path / "feedback.db")
    store = feedback.FeedbackStore(path)
    conn = sqlite3.connect(path)
    assert "kind" in _columns(conn, "feedback")
    conn.close()
    assert store is not None
