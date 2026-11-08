"""Shared connection handling for the small sqlite-backed stores (jobs, feedback, the
query log, the trace archive)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from threading import Lock


def enable_wal(path: str) -> None:
    """Switch the database to WAL once; the mode persists in the file itself.

    Readers (1s job polls, GET /feedback) then stop contending with the ingest worker's
    progress writes, and with ``synchronous=NORMAL`` a commit no longer fsyncs twice.
    """

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
    finally:
        connection.close()


@contextmanager
def sqlite_transaction(path: str) -> Iterator[sqlite3.Connection]:
    """One short-lived connection: commit on success, roll back on error, always close.

    ``with sqlite3.connect(...) as conn`` commits but never closes the connection —
    that was left to garbage collection on every call.
    """

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA synchronous=NORMAL")
        with connection:
            yield connection
    finally:
        connection.close()


# Rows added between trims of a capped table: a count over the table on every insert
# would be the costliest part of adding a row.
_TRIM_EVERY = 100


class CappedRowLog:
    """A sqlite table of rows kept newest ``max_rows`` (by rowid), each with a
    ``workspace`` column: a connection per call under one lock (sqlite3 connections
    aren't shared across threads), trimmed every ``_TRIM_EVERY`` inserts. The query log
    and the trace archive are both this."""

    _table: str

    def __init__(self, path: str, max_rows: int) -> None:
        self._path = path
        self._max_rows = max_rows
        self._lock = Lock()
        self._added = 0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        enable_wal(path)

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_transaction(self._path)

    def _added_one(self, conn: sqlite3.Connection) -> None:
        """Count an insert made on ``conn`` (under the lock), trimming every so often."""

        self._added += 1
        if self._added % _TRIM_EVERY == 1:
            conn.execute(
                f"DELETE FROM {self._table} WHERE rowid <= "  # noqa: S608 - a class constant
                f"(SELECT MAX(rowid) FROM {self._table}) - ?",
                (self._max_rows,),
            )

    def delete_workspace(self, workspace: str) -> int:
        """Remove every row of ``workspace`` (deleted), so one created again under its
        name starts empty; how many went."""

        with self._lock, self._connect() as conn:
            return int(
                conn.execute(
                    f"DELETE FROM {self._table} WHERE workspace = ?",  # noqa: S608
                    (workspace,),
                ).rowcount
            )
