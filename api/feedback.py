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
from typing import Literal

from api.sqlite_store import enable_wal, sqlite_transaction

FeedbackRating = Literal["up", "down"]


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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    trace_id TEXT,
                    question TEXT NOT NULL,
                    answer_excerpt TEXT NOT NULL,
                    cited_filenames_json TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    citation_source_number INTEGER,
                    created_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_transaction(self._path)

    def add(
        self,
        *,
        trace_id: str | None,
        question: str,
        answer_excerpt: str,
        cited_filenames: list[str],
        rating: FeedbackRating,
        citation_source_number: int | None,
    ) -> Feedback:
        feedback = Feedback(
            id=str(uuid.uuid4()),
            trace_id=trace_id,
            question=question,
            answer_excerpt=answer_excerpt,
            cited_filenames=cited_filenames,
            rating=rating,
            citation_source_number=citation_source_number,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (id, trace_id, question, answer_excerpt, "
                "cited_filenames_json, rating, citation_source_number, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback.id,
                    feedback.trace_id,
                    feedback.question,
                    feedback.answer_excerpt,
                    json.dumps(feedback.cited_filenames),
                    feedback.rating,
                    feedback.citation_source_number,
                    feedback.created_at,
                ),
            )
            if self._max_rows is not None:
                conn.execute(
                    "DELETE FROM feedback WHERE rowid <= "
                    "(SELECT rowid FROM feedback ORDER BY rowid DESC LIMIT 1 OFFSET ?)",
                    (self._max_rows,),
                )
        return feedback

    def import_rows(self, rows: Sequence[Feedback]) -> int:
        """Insert rows from a backup, keeping their ids and timestamps; rows whose id is
        already here are skipped, so restoring the same backup twice adds nothing.
        Returns how many were added."""

        added = 0
        with self._lock, self._connect() as conn:
            for row in sorted(rows, key=lambda item: item.created_at):
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO feedback (id, trace_id, question, answer_excerpt, "
                    "cited_filenames_json, rating, citation_source_number, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row.id,
                        row.trace_id,
                        row.question,
                        row.answer_excerpt,
                        json.dumps(row.cited_filenames),
                        row.rating,
                        row.citation_source_number,
                        row.created_at,
                    ),
                )
                added += cursor.rowcount
            if self._max_rows is not None:
                conn.execute(
                    "DELETE FROM feedback WHERE rowid <= "
                    "(SELECT rowid FROM feedback ORDER BY rowid DESC LIMIT 1 OFFSET ?)",
                    (self._max_rows,),
                )
        return added

    def list_recent(self, limit: int = 100) -> list[Feedback]:
        """Return the most recently recorded feedback, newest first."""

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, trace_id, question, answer_excerpt, cited_filenames_json, "
                "rating, citation_source_number, created_at FROM feedback "
                "ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            Feedback(
                id=row[0],
                trace_id=row[1],
                question=row[2],
                answer_excerpt=row[3],
                cited_filenames=json.loads(row[4]),
                rating=row[5],
                citation_source_number=row[6],
                created_at=row[7],
            )
            for row in rows
        ]
