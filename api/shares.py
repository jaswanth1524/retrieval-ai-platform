"""Shared answers (SHARES_ENABLED): a frozen copy of a conversation, or of one answer,
that another key-holder opens from a link (``#s=<id>`` in the UI).

A snapshot, not a sync: what was shared is what is shown — the answer, its citations
and the passages' text as they were — even after the documents change. A share belongs
to the workspace it was made in: opening it needs a key that may use that workspace
(API_KEYS_FILE), so a link reaching the wrong person shows them nothing. Shares expire
after SHARE_TTL_DAYS (0: never) and can be deleted by the key that made them or a full
key. Stored in one sqlite file (SHARE_STORE_PATH); not part of backups.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from api.sqlite_store import enable_wal, sqlite_transaction

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS shares (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL,
    created_by TEXT NOT NULL,
    title TEXT NOT NULL,
    turns INTEGER NOT NULL,
    payload_json TEXT NOT NULL
)
"""


class ShareLimitError(Exception):
    """The workspace already holds SHARE_MAX_PER_WORKSPACE shares."""


@dataclass(frozen=True)
class Share:
    id: str
    workspace: str
    created_at: float
    expires_at: float | None
    created_by: str
    title: str
    turns: int
    payload: list[dict[str, Any]] | None = None


class ShareStore:
    """Sqlite-backed; a short-lived connection per call, under one lock (as
    ``FeedbackStore``). ``clock`` is for tests."""

    def __init__(
        self,
        path: str,
        *,
        max_per_workspace: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._max = max_per_workspace
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        enable_wal(path)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS shares_workspace_created ON shares "
                "(workspace, created_at)"
            )

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_transaction(self._path)

    def _purge(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "DELETE FROM shares WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (self._clock(),),
        )

    def add(
        self, *, workspace: str, created_by: str, title: str, turns: list[dict[str, Any]]
    ) -> Share:
        now = self._clock()
        share = Share(
            id=secrets.token_urlsafe(16),
            workspace=workspace or "default",
            created_at=now,
            expires_at=now + self._ttl if self._ttl > 0 else None,
            created_by=created_by,
            title=title,
            turns=len(turns),
        )
        with self._lock, self._connect() as conn:
            self._purge(conn)
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM shares WHERE workspace = ?", (share.workspace,)
            ).fetchone()
            if count >= self._max:
                raise ShareLimitError
            conn.execute(
                "INSERT INTO shares (id, workspace, created_at, expires_at, created_by, "
                "title, turns, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    share.id,
                    share.workspace,
                    share.created_at,
                    share.expires_at,
                    share.created_by,
                    share.title,
                    share.turns,
                    json.dumps(turns, ensure_ascii=False),
                ),
            )
        return share

    def get(self, share_id: str) -> Share | None:
        """The share with its turns; None when there is none or it has expired."""

        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, workspace, created_at, expires_at, created_by, title, turns, "
                "payload_json FROM shares WHERE id = ?",
                (share_id,),
            ).fetchone()
        if row is None:
            return None
        share = _share(row[:7])
        if share.expires_at is not None and share.expires_at <= self._clock():
            return None
        try:
            payload = json.loads(row[7])
        except ValueError:
            payload = []
        return Share(**{**share.__dict__, "payload": payload if isinstance(payload, list) else []})

    def list(self, workspace: str, *, created_by: str | None = None) -> list[Share]:
        """``workspace``'s shares, newest first, without their turns (only those
        ``created_by`` made, when given)."""

        where, params = (
            "workspace = ? AND (expires_at IS NULL OR expires_at > ?)",
            [
                workspace or "default",
                self._clock(),
            ],
        )
        if created_by is not None:
            where += " AND created_by = ?"
            params.append(created_by)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, workspace, created_at, expires_at, created_by, title, turns "
                f"FROM shares WHERE {where} ORDER BY created_at DESC",
                params,
            ).fetchall()
        return [_share(row) for row in rows]

    def delete(self, share_id: str) -> bool:
        with self._lock, self._connect() as conn:
            return conn.execute("DELETE FROM shares WHERE id = ?", (share_id,)).rowcount > 0

    def delete_workspace(self, workspace: str) -> int:
        with self._lock, self._connect() as conn:
            return conn.execute(
                "DELETE FROM shares WHERE workspace = ?", (workspace or "default",)
            ).rowcount


def _share(row: tuple[Any, ...]) -> Share:
    return Share(
        id=str(row[0]),
        workspace=str(row[1]),
        created_at=float(row[2]),
        expires_at=None if row[3] is None else float(row[3]),
        created_by=str(row[4]),
        title=str(row[5]),
        turns=int(row[6]),
    )
