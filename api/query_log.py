"""A slim, persistent log of what was asked (QUERY_LOG_ENABLED, off by default).

Traces (``api.tracing``) live in a 100-entry in-memory ring buffer, so "what did people
ask that the documents don't cover?" could only be answered for the last few minutes.
This keeps one small row per question or summary — the text, the workspace, how it
went and the best passage score — in its own sqlite file, newest first on
``GET /query-log`` (full key; ``?uncovered=true`` for the questions the documents
didn't answer). It holds what users typed: off unless the operator turns it on, capped
at ``QUERY_LOG_MAX_ROWS``, and never part of a backup.

Fed by the same sink the pipeline records traces through (``CompositeTraceSink``), so a
question is logged however it was asked (sync, streamed, a summary).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from api.sqlite_store import enable_wal, sqlite_transaction
from api.tracing import QueryTrace, TraceSink

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS queries (
        created_at REAL NOT NULL,
        workspace TEXT NOT NULL DEFAULT '',
        question TEXT NOT NULL,
        status TEXT NOT NULL,
        top_score REAL,
        mode TEXT NOT NULL,
        trace_id TEXT
    )
"""
# A status that means the documents didn't answer it.
UNCOVERED_STATUSES = ("insufficient_context",)


@dataclass(frozen=True)
class LoggedQuery:
    created_at: float
    workspace: str
    question: str
    # The trace's status: ok, insufficient_context (nothing cleared the bar, or the model
    # said the documents don't cover it), error.
    status: str
    # Best cross-encoder score among the candidates (0-1); None for a summary.
    top_score: float | None
    mode: str
    trace_id: str | None


class QueryLogStore:
    """Sqlite-backed query log; a connection per call under one lock, as the feedback
    store does (sqlite3 connections aren't shared across threads)."""

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
                "CREATE INDEX IF NOT EXISTS queries_workspace_created_at "
                "ON queries (workspace, created_at)"
            )

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_transaction(self._path)

    def add(self, query: LoggedQuery) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO queries (created_at, workspace, question, status, top_score, "
                "mode, trace_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    query.created_at,
                    query.workspace,
                    query.question,
                    query.status,
                    query.top_score,
                    query.mode,
                    query.trace_id,
                ),
            )
            self._added += 1
            # Trimmed every so often, not on each insert: a count over the table per
            # question would be the costliest part of logging it.
            if self._added % 100 == 1:
                conn.execute(
                    "DELETE FROM queries WHERE rowid <= (SELECT MAX(rowid) FROM queries) - ?",
                    (self._max_rows,),
                )

    def recent(self, workspace: str, *, limit: int, uncovered: bool = False) -> list[LoggedQuery]:
        """``workspace``'s logged questions, newest first."""

        where = "workspace = ?"
        params: list[object] = [workspace]
        if uncovered:
            where += f" AND status IN ({', '.join('?' for _ in UNCOVERED_STATUSES)})"
            params.extend(UNCOVERED_STATUSES)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT created_at, workspace, question, status, top_score, mode, trace_id "
                f"FROM queries WHERE {where} ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [
            LoggedQuery(
                created_at=float(row[0]),
                workspace=str(row[1]),
                question=str(row[2]),
                status=str(row[3]),
                top_score=None if row[4] is None else float(row[4]),
                mode=str(row[5]),
                trace_id=None if row[6] is None else str(row[6]),
            )
            for row in rows
        ]


def logged_query(trace: QueryTrace) -> LoggedQuery:
    """What the log keeps of a trace: no prompt, answer or candidates."""

    scores = [c.rerank_score for c in trace.candidates if c.rerank_score is not None]
    return LoggedQuery(
        created_at=trace.created_at,
        workspace=trace.workspace,
        question=trace.question,
        status=trace.status,
        top_score=max(scores) if scores else None,
        mode=trace.mode,
        trace_id=trace.trace_id,
    )


class QueryLogSink:
    """A ``TraceSink`` that logs each trace's question; failures are swallowed (a full
    disk must not fail the answer that was just given)."""

    def __init__(self, store: QueryLogStore) -> None:
        self._store = store

    def add(self, trace: QueryTrace) -> None:
        try:
            self._store.add(logged_query(trace))
        except sqlite3.Error:
            import logging

            logging.getLogger(__name__).warning("Could not log a question.", exc_info=True)


class CompositeTraceSink:
    """Hands each trace to every sink: the trace buffer and the query log."""

    def __init__(self, sinks: Sequence[TraceSink]) -> None:
        self._sinks = list(sinks)

    def add(self, trace: QueryTrace) -> None:
        for sink in self._sinks:
            sink.add(trace)
