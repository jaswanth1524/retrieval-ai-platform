"""Traces kept on disk as well as in memory (TRACE_STORE_PATH, off by default).

``TraceStore`` is a ring of the last ``TRACE_MAX_RETAINED`` traces, lost on restart. A
cached answer keeps its trace id for up to an hour and a feedback row keeps it for
good, so their "how was this retrieved" link stopped working after a busy hour or any
restart. With a path set, every trace is also written here as one JSON row (newest
``TRACE_STORE_MAX_ROWS`` kept), read back when the ring no longer has it, and
downloadable as JSON Lines (``GET /traces/export``). Holds prompts — document passages
and what users asked — so it is opt-in, and never part of a backup.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from contextlib import AbstractContextManager
from pathlib import Path
from threading import Lock

from api.sqlite_store import enable_wal, sqlite_transaction
from api.tracing import (
    QueryTrace,
    TraceSummary,
    summarize_trace,
    trace_from_json,
    trace_to_json,
)

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS traces (
        trace_id TEXT PRIMARY KEY,
        created_at REAL NOT NULL,
        workspace TEXT NOT NULL DEFAULT '',
        body TEXT NOT NULL
    )
"""
# Rows read per query while exporting: the download streams, it isn't built in memory.
_EXPORT_PAGE_ROWS = 200
# The trace list's columns, beside the body: listing parsed every body (prompts and
# candidates included) only to show a dozen fields, and couldn't filter on them.
_SUMMARY_COLUMNS = (
    ("question", "TEXT NOT NULL DEFAULT ''"),
    ("mode", "TEXT NOT NULL DEFAULT ''"),
    ("status", "TEXT NOT NULL DEFAULT ''"),
    ("llm_provider", "TEXT"),
    ("total_ms", "REAL"),
    ("candidate_count", "INTEGER NOT NULL DEFAULT 0"),
    ("kept_count", "INTEGER NOT NULL DEFAULT 0"),
    ("answer_status", "TEXT"),
    ("top_rerank_score", "REAL"),
    # 0 on rows written before these columns existed, until they are filled in.
    ("summarized", "INTEGER NOT NULL DEFAULT 0"),
)
_SUMMARY_FIELDS = tuple(name for name, _ in _SUMMARY_COLUMNS if name != "summarized")
# Rows backfilled per transaction when an older archive is opened.
_BACKFILL_ROWS = 500


def _summary_values(summary: TraceSummary) -> tuple[object, ...]:
    return tuple(getattr(summary, name) for name in _SUMMARY_FIELDS)


class TraceArchive:
    """Sqlite-backed trace rows; a connection per call under one lock, as the query
    log does (sqlite3 connections aren't shared across threads)."""

    def __init__(self, path: str, max_rows: int) -> None:
        self._path = path
        self._max_rows = max_rows
        self._lock = Lock()
        self._added = 0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        enable_wal(path)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS traces_workspace_created_at "
                "ON traces (workspace, created_at)"
            )
            present = {row[1] for row in conn.execute("PRAGMA table_info(traces)")}
            for name, declaration in _SUMMARY_COLUMNS:
                if name not in present:
                    conn.execute(f"ALTER TABLE traces ADD COLUMN {name} {declaration}")
        self._backfill_summaries()

    def _backfill_summaries(self) -> None:
        """Fill the summary columns of rows an older version wrote (once; a batch per
        transaction, so a big archive doesn't hold one long write lock)."""

        while True:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT rowid, body FROM traces WHERE summarized = 0 LIMIT ?",
                    (_BACKFILL_ROWS,),
                ).fetchall()
                if not rows:
                    return
                for rowid, body in rows:
                    try:
                        values = _summary_values(summarize_trace(trace_from_json(body)))
                    except Exception:  # noqa: BLE001 - an unreadable row lists as blank
                        conn.execute("UPDATE traces SET summarized = 1 WHERE rowid = ?", (rowid,))
                        continue
                    assignments = ", ".join(f"{name} = ?" for name in _SUMMARY_FIELDS)
                    conn.execute(
                        f"UPDATE traces SET {assignments}, summarized = 1 WHERE rowid = ?",
                        (*values, rowid),
                    )

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_transaction(self._path)

    def add(self, trace: QueryTrace) -> None:
        body = trace_to_json(trace)
        columns = ", ".join(_SUMMARY_FIELDS)
        marks = ", ".join("?" for _ in _SUMMARY_FIELDS)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO traces (trace_id, created_at, workspace, body, "
                f"{columns}, summarized) VALUES (?, ?, ?, ?, {marks}, 1)",
                (
                    trace.trace_id,
                    trace.created_at,
                    trace.workspace,
                    body,
                    *_summary_values(summarize_trace(trace)),
                ),
            )
            self._added += 1
            # Trimmed every so often, not on each insert (a count per question).
            if self._added % 100 == 1:
                conn.execute(
                    "DELETE FROM traces WHERE rowid <= (SELECT MAX(rowid) FROM traces) - ?",
                    (self._max_rows,),
                )

    def get(self, trace_id: str) -> QueryTrace | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT body FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        return trace_from_json(row[0]) if row is not None else None

    def recent(self, workspace: str | None, limit: int) -> list[QueryTrace]:
        """The newest ``limit`` traces (only ``workspace``'s when given), newest first."""

        where, params = ("WHERE workspace = ?", [workspace]) if workspace is not None else ("", [])
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT body FROM traces {where} ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [trace_from_json(row[0]) for row in rows]

    def summaries(
        self, workspace: str | None, limit: int, answer_statuses: AbstractSet[str] | None
    ) -> list[TraceSummary]:
        """The newest ``limit`` traces' list rows (only ``workspace``'s when given; only
        those with one of ``answer_statuses`` when given), newest first."""

        conditions: list[str] = []
        params: list[object] = []
        if workspace is not None:
            conditions.append("workspace = ?")
            params.append(workspace)
        if answer_statuses is not None:
            if not answer_statuses:
                return []
            conditions.append(f"answer_status IN ({', '.join('?' for _ in answer_statuses)})")
            params.extend(sorted(answer_statuses))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT trace_id, created_at, {', '.join(_SUMMARY_FIELDS)} FROM traces "
                f"{where} ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [
            TraceSummary(
                trace_id=row[0],
                created_at=row[1],
                **dict(zip(_SUMMARY_FIELDS, row[2:], strict=True)),
            )
            for row in rows
        ]

    def export_lines(self, workspace: str) -> Iterator[str]:
        """``workspace``'s traces as JSON lines, oldest first, a page at a time."""

        after: tuple[float, int] = (float("-inf"), -1)
        while True:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT created_at, rowid, body FROM traces WHERE workspace = ? AND "
                    "(created_at > ? OR (created_at = ? AND rowid > ?)) "
                    "ORDER BY created_at, rowid LIMIT ?",
                    (workspace, after[0], after[0], after[1], _EXPORT_PAGE_ROWS),
                ).fetchall()
            if not rows:
                return
            for _, _, body in rows:
                yield body + "\n"
            after = (float(rows[-1][0]), int(rows[-1][1]))
