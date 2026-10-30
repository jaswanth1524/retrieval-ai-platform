"""When each URL-fetched document was last checked by the scheduled refresh, how that
went, and when it is next due (api/url_refresh.py).

Kept in memory unless URL_REFRESH_STATE_PATH names a sqlite file: then a restart picks
up where it left off, instead of treating every document as checked when it was last
indexed — a page that only ever answered 304 then looked overdue after every restart,
and every such page was fetched in the first passes. Not part of a backup: it is only
scheduling, rebuilt from the documents' upload times when missing.
"""

from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from api.sqlite_store import enable_wal, sqlite_transaction


@dataclass(frozen=True)
class RefreshState:
    """The last scheduled check of one document."""

    last_checked: float
    # unchanged, queued, failed (see UrlRefresher).
    last_outcome: str
    # Failed checks in a row; each one waits longer before the next (backoff).
    failures: int
    next_due: float


class RefreshStateStore(Protocol):
    def get_all(self, collection: str) -> dict[str, RefreshState]: ...

    def record(self, collection: str, filename: str, state: RefreshState) -> None: ...

    def forget(self, collection: str, filenames: list[str]) -> None: ...

    def forget_collections_except(self, collections: set[str]) -> None: ...


class MemoryRefreshStateStore:
    """The default: lost on restart."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._rows: dict[tuple[str, str], RefreshState] = {}

    def get_all(self, collection: str) -> dict[str, RefreshState]:
        with self._lock:
            return {
                name: state for (owner, name), state in self._rows.items() if owner == collection
            }

    def record(self, collection: str, filename: str, state: RefreshState) -> None:
        with self._lock:
            self._rows[(collection, filename)] = state

    def forget(self, collection: str, filenames: list[str]) -> None:
        with self._lock:
            for name in filenames:
                self._rows.pop((collection, name), None)

    def forget_collections_except(self, collections: set[str]) -> None:
        with self._lock:
            for key in [key for key in self._rows if key[0] not in collections]:
                del self._rows[key]


_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS url_refresh_state (
        collection TEXT NOT NULL,
        filename TEXT NOT NULL,
        last_checked REAL NOT NULL,
        last_outcome TEXT NOT NULL,
        failures INTEGER NOT NULL,
        next_due REAL NOT NULL,
        PRIMARY KEY (collection, filename)
    )
"""


class SqliteRefreshStateStore:
    """URL_REFRESH_STATE_PATH: a connection per call under one lock, as the other
    sqlite stores do."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        enable_wal(path)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_transaction(self._path)

    def get_all(self, collection: str) -> dict[str, RefreshState]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT filename, last_checked, last_outcome, failures, next_due "
                "FROM url_refresh_state WHERE collection = ?",
                (collection,),
            ).fetchall()
        return {row[0]: RefreshState(row[1], row[2], int(row[3]), row[4]) for row in rows}

    def record(self, collection: str, filename: str, state: RefreshState) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO url_refresh_state VALUES (?, ?, ?, ?, ?, ?)",
                (
                    collection,
                    filename,
                    state.last_checked,
                    state.last_outcome,
                    state.failures,
                    state.next_due,
                ),
            )

    def forget(self, collection: str, filenames: list[str]) -> None:
        if not filenames:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                "DELETE FROM url_refresh_state WHERE collection = ? AND filename = ?",
                [(collection, name) for name in filenames],
            )

    def forget_collections_except(self, collections: set[str]) -> None:
        marks = ", ".join("?" for _ in collections)
        where = f"WHERE collection NOT IN ({marks})" if collections else ""
        with self._lock, self._connect() as conn:
            conn.execute(f"DELETE FROM url_refresh_state {where}", tuple(sorted(collections)))
