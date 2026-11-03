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

import json
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
        trace_id TEXT,
        filenames_json TEXT
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
    # The documents its answer was drawn from (the passages put in its prompt; a
    # summary's documents), for answered questions only: what "questions that cited
    # this document" counts.
    filenames: tuple[str, ...] = ()


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
            # A log from before documents were recorded: its rows name none.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(queries)")}
            if "filenames_json" not in columns:
                conn.execute("ALTER TABLE queries ADD COLUMN filenames_json TEXT")
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
                "mode, trace_id, filenames_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    query.created_at,
                    query.workspace,
                    query.question,
                    query.status,
                    query.top_score,
                    query.mode,
                    query.trace_id,
                    json.dumps(list(query.filenames)),
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

    def delete_workspace(self, workspace: str) -> int:
        """Remove every row of ``workspace`` (deleted), so one created again under its
        name starts with an empty log."""

        with self._lock, self._connect() as conn:
            return conn.execute("DELETE FROM queries WHERE workspace = ?", (workspace,)).rowcount

    def recent(
        self,
        workspace: str,
        *,
        limit: int,
        uncovered: bool = False,
        filename: str | None = None,
    ) -> list[LoggedQuery]:
        """``workspace``'s logged questions, newest first: only unanswered ones with
        ``uncovered``, only those drawn from ``filename`` when it is given."""

        where = "workspace = ?"
        params: list[object] = [workspace]
        if uncovered:
            where += f" AND status IN ({', '.join('?' for _ in UNCOVERED_STATUSES)})"
            params.extend(UNCOVERED_STATUSES)
        if filename is not None:
            # The name as it appears inside the JSON list: an exact element, not a
            # substring of a longer name.
            where += " AND instr(filenames_json, ?) > 0"
            params.append(json.dumps(filename))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT created_at, workspace, question, status, top_score, mode, trace_id, "
                "filenames_json "
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
                filenames=_filenames(row[7]),
            )
            for row in rows
        ]


def logged_query(trace: QueryTrace) -> LoggedQuery:
    """What the log keeps of a trace: no prompt, answer or candidates."""

    scores = [c.rerank_score for c in trace.candidates if c.rerank_score is not None]
    # Only an answer counts as drawn from its documents: one that declined ("not in the
    # documents") read the closest passages and used none of them.
    drawn_from: dict[str, None] = {}
    if trace.status == "ok":
        drawn_from = dict.fromkeys(c.filename for c in trace.candidates if c.selected_for_context)
        if not drawn_from and trace.config is not None and trace.config.filenames and not scores:
            drawn_from = dict.fromkeys(trace.config.filenames)  # a summary: its documents
    return LoggedQuery(
        created_at=trace.created_at,
        workspace=trace.workspace,
        question=trace.question,
        status=trace.status,
        top_score=max(scores) if scores else None,
        mode=trace.mode,
        trace_id=trace.trace_id,
        filenames=tuple(drawn_from),
    )


def _filenames(value: object) -> tuple[str, ...]:
    try:
        names = json.loads(value) if isinstance(value, str) else []
    except ValueError:
        return ()
    return tuple(name for name in names if isinstance(name, str)) if isinstance(names, list) else ()


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
