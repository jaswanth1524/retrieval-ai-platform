"""Answer-feedback capture: thumbs up/down on a chat turn, optionally on one citation.

Off by default (``AppSettings.feedback_enabled``) — every prior release has no
feedback path at all. When enabled, ``POST /feedback`` (from ``ChatMessage.tsx``)
persists one row per rating to a dedicated sqlite3 file (stdlib, no new dependency),
independent of ``job_store_backend`` (see ``api.jobs``) — this feature works whether
or not the operator has also opted the job store into sqlite persistence.

Denormalized at write time deliberately: the question, an answer excerpt, and the
cited filenames are stored alongside ``trace_id`` rather than looked up through it,
because ``TraceStore`` (``api.tracing``) is a capped in-memory ring buffer — a
``trace_id``-only foreign key would go stale the moment that buffer rolls past it.
``trace_id`` is kept anyway as a best-effort join key, allowed to dangle.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Literal, get_args

from api.sqlite_store import enable_wal, sqlite_transaction

FeedbackRating = Literal["up", "down"]
# What was rated: an answer to a question, a suggested follow-up question, or a summary
# or comparison. Only "question" ratings label retrieval (eval/feedback_dataset.py).
FeedbackKind = Literal["question", "follow_up", "summary"]
FEEDBACK_KINDS: tuple[str, ...] = get_args(FeedbackKind)


@dataclass(frozen=True)
class Feedback:
    """One recorded rating of an assistant answer, or of one of its citations."""

    id: str
    trace_id: str | None
    question: str
    answer_excerpt: str
    cited_filenames: list[str]
    rating: FeedbackRating
    # None for feedback on the answer as a whole; set when the rating targets one
    # specific citation (its `[n]` source_number as shown in the answer).
    citation_source_number: int | None
    created_at: float = field(default_factory=time.time)
    # The workspace the rated answer came from ("" = the default). Not part of a
    # backup's rows: a backup holds one workspace, and its rows join the workspace
    # it is restored into.
    workspace: str = ""
    kind: FeedbackKind = "question"


# Where a page of rows_oldest_first ended: (created_at, rowid) of its last row.
FeedbackCursor = tuple[float, int]


# Keyed by (workspace, id): the same rating restored into two workspaces is two rows,
# and restoring a backup into the workspace it came from twice still adds nothing.
_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS {name} (
        id TEXT NOT NULL,
        trace_id TEXT,
        question TEXT NOT NULL,
        answer_excerpt TEXT NOT NULL,
        cited_filenames_json TEXT NOT NULL,
        rating TEXT NOT NULL,
        citation_source_number INTEGER,
        created_at REAL NOT NULL,
        workspace TEXT NOT NULL DEFAULT '',
        kind TEXT NOT NULL DEFAULT 'question',
        PRIMARY KEY (workspace, id)
    )
"""
_COLUMNS = (
    "id, trace_id, question, answer_excerpt, cited_filenames_json, rating, "
    "citation_source_number, created_at"
)


def _migrate_to_workspace_key(conn: sqlite3.Connection) -> None:
    """Rebuild a table from before workspaces (keyed by id alone, no workspace column)
    in place, in the same transaction; its rows land in the default workspace, in their
    original order."""

    info = {row[1]: row[5] for row in conn.execute("PRAGMA table_info(feedback)")}
    if info.get("workspace"):  # already part of the primary key
        return
    workspace = "workspace" if "workspace" in info else "''"
    # sqlite3 runs DDL outside a transaction unless one is open: open it, so a failure
    # anywhere below leaves the old table as it was (committed when the caller's block ends).
    if not conn.in_transaction:
        conn.execute("BEGIN")
    conn.execute(_CREATE_TABLE.format(name="feedback_migrated"))
    conn.execute(
        f"INSERT INTO feedback_migrated ({_COLUMNS}, workspace) "
        f"SELECT {_COLUMNS}, {workspace} FROM feedback ORDER BY rowid"
    )
    conn.execute("DROP TABLE feedback")
    conn.execute("ALTER TABLE feedback_migrated RENAME TO feedback")


def _add_kind_column(conn: sqlite3.Connection) -> None:
    """Add ``kind`` to a table from before it was recorded, in place: every rating
    then was of an answer to a question."""

    columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
    if "kind" not in columns:
        conn.execute("ALTER TABLE feedback ADD COLUMN kind TEXT NOT NULL DEFAULT 'question'")


class FeedbackStore:
    """Sqlite-backed feedback store.

    A short-lived connection is opened per call, inside the lock guarding every
    other operation on this instance — sqlite3 connections aren't safe to share
    across threads, and requests can arrive concurrently.
    """

    def __init__(self, path: str, max_rows: int | None = None) -> None:
        self._path = path
        # Oldest rows are pruned past this on insert. POST /feedback is open whenever
        # API_KEY is unset, so without a cap any caller could grow the file forever.
        self._max_rows = max_rows
        self._lock = Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        enable_wal(path)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE.format(name="feedback"))
            _migrate_to_workspace_key(conn)
            _add_kind_column(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS feedback_created_at ON feedback (created_at)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS feedback_workspace_created_at "
                "ON feedback (workspace, created_at)"
            )

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_transaction(self._path)

    def _trim(self, conn: sqlite3.Connection, workspace: str) -> None:
        """Keep ``workspace``'s newest ``max_rows`` by when they were given, not by
        insert order: rows restored from a backup are inserted last but can be the
        oldest.

        Per workspace: across all of them, a busy workspace's ratings pushed out a quiet
        one's (and a restore into one workspace could empty another).
        """

        if self._max_rows is None:
            return
        # Counted first: the old NOT IN subquery ordered the whole table on every insert,
        # even when (as nearly always) nothing was over the cap.
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE workspace = ?", (workspace,)
        ).fetchone()
        excess = count - self._max_rows
        if excess <= 0:
            return
        conn.execute(
            "DELETE FROM feedback WHERE rowid IN (SELECT rowid FROM feedback "
            "WHERE workspace = ? ORDER BY created_at ASC, rowid ASC LIMIT ?)",
            (workspace, excess),
        )

    def add(
        self,
        *,
        trace_id: str | None,
        question: str,
        answer_excerpt: str,
        cited_filenames: list[str],
        rating: FeedbackRating,
        citation_source_number: int | None,
        workspace: str = "",
        kind: FeedbackKind = "question",
    ) -> Feedback:
        feedback = Feedback(
            id=str(uuid.uuid4()),
            trace_id=trace_id,
            question=question,
            answer_excerpt=answer_excerpt,
            cited_filenames=cited_filenames,
            rating=rating,
            citation_source_number=citation_source_number,
            workspace=workspace,
            kind=kind,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (id, trace_id, question, answer_excerpt, "
                "cited_filenames_json, rating, citation_source_number, created_at, workspace, "
                "kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback.id,
                    feedback.trace_id,
                    feedback.question,
                    feedback.answer_excerpt,
                    json.dumps(feedback.cited_filenames),
                    feedback.rating,
                    feedback.citation_source_number,
                    feedback.created_at,
                    feedback.workspace,
                    feedback.kind,
                ),
            )
            self._trim(conn, feedback.workspace)
        return feedback

    def import_rows(self, rows: Sequence[Feedback], workspace: str = "") -> int:
        """Insert rows from a backup into ``workspace``, keeping their ids and timestamps;
        rows whose id is already here are skipped, so restoring the same backup twice adds
        nothing. Returns how many were added."""

        added = 0
        with self._lock, self._connect() as conn:
            for row in sorted(rows, key=lambda item: item.created_at):
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO feedback (id, trace_id, question, answer_excerpt, "
                    "cited_filenames_json, rating, citation_source_number, created_at, "
                    "workspace, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row.id,
                        row.trace_id,
                        row.question,
                        row.answer_excerpt,
                        json.dumps(row.cited_filenames),
                        row.rating,
                        row.citation_source_number,
                        row.created_at,
                        workspace,
                        row.kind,
                    ),
                )
                added += cursor.rowcount
            self._trim(conn, workspace)
        return added

    def rows_oldest_first(
        self, after: FeedbackCursor | None, limit: int, workspace: str | None = None
    ) -> tuple[list[Feedback], FeedbackCursor | None]:
        """Up to ``limit`` rows after ``after``, oldest first, and the cursor of the last;
        only ``workspace``'s rows when it is given (None = every workspace).

        Keyset paging on (created_at, rowid), the order a backup lists rows in, so an
        export reads one page at a time instead of the whole table.
        """

        created_at, rowid = after if after is not None else (float("-inf"), -1)
        scope, params = _workspace_filter(workspace)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT rowid, {_FEEDBACK_COLUMNS} FROM feedback "
                f"WHERE (created_at > ? OR (created_at = ? AND rowid > ?)){scope} "
                "ORDER BY created_at ASC, rowid ASC LIMIT ?",
                (created_at, created_at, rowid, *params, limit),
            ).fetchall()
        page = [_feedback(row[1:]) for row in rows]
        return page, ((rows[-1][8], rows[-1][0]) if rows else None)

    def list_recent(self, limit: int = 100, workspace: str | None = None) -> list[Feedback]:
        """Return the most recent feedback, newest first by when it was given (a row
        restored from a backup keeps its original time); only ``workspace``'s when it is
        given (None = every workspace)."""

        scope, params = _workspace_filter(workspace)
        where = f"WHERE {scope.removeprefix(' AND ')} " if scope else ""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_FEEDBACK_COLUMNS} FROM feedback {where}"
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_feedback(row) for row in rows]


# The columns a Feedback is read from, in this order (both queries select them so).
_FEEDBACK_COLUMNS = (
    "id, trace_id, question, answer_excerpt, cited_filenames_json, rating, "
    "citation_source_number, created_at, workspace, kind"
)


def _feedback(row: Sequence[Any]) -> Feedback:
    return Feedback(
        id=row[0],
        trace_id=row[1],
        question=row[2],
        answer_excerpt=row[3],
        cited_filenames=json.loads(row[4]),
        rating=row[5],
        citation_source_number=row[6],
        created_at=row[7],
        workspace=row[8],
        kind=_kind(row[9]),
    )


def _kind(value: object) -> FeedbackKind:
    """A stored kind, read back; anything unrecognized (written by a newer version) as
    "question", the meaning every row had before kinds existed."""

    if value == "follow_up":
        return "follow_up"
    if value == "summary":
        return "summary"
    return "question"


def _workspace_filter(workspace: str | None) -> tuple[str, tuple[str, ...]]:
    """An `` AND workspace = ?`` clause and its parameter, or nothing for None."""

    return ("", ()) if workspace is None else (" AND workspace = ?", (workspace,))
