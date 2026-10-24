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
from contextlib import AbstractContextManager
from pathlib import Path
from threading import Lock

from api.sqlite_store import enable_wal, sqlite_transaction
from api.tracing import QueryTrace, trace_from_json, trace_to_json

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

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_transaction(self._path)

    def add(self, trace: QueryTrace) -> None:
        body = trace_to_json(trace)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces (trace_id, created_at, workspace, body) "
                "VALUES (?, ?, ?, ?)",
                (trace.trace_id, trace.created_at, trace.workspace, body),
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
