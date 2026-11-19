"""Shared connection handling and column upgrades for the small sqlite-backed stores
(jobs, feedback, the query log, the trace archive)."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from threading import Lock

_NOT_NULL_RE = re.compile(r"\bNOT\s+NULL\b", re.I)
_DEFAULT_RE = re.compile(r"\bDEFAULT\b", re.I)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def add_missing_columns(
    conn: sqlite3.Connection, table: str, columns: Iterable[tuple[str, str]]
) -> list[str]:
    """Add each ``(name, declaration)`` column ``table`` lacks; the names added.

    How every store upgrades a file an older version wrote: it looks at what is there
    rather than a version number, so a file at any earlier state (or one a newer
    version already upgraded) opens the same way, and running it twice is harmless.
    Rows already in the table take the column's default.
    """

    columns = list(columns)
    # Names go into the statement itself (sqlite takes no parameters there).
    for identifier in (table, *(name for name, _ in columns)):
        if not _IDENTIFIER_RE.fullmatch(identifier):
            raise ValueError(f"not a table or column name: {identifier!r}")
    for name, declaration in columns:
        # sqlite refuses ADD COLUMN ... NOT NULL without a default, but only on a table
        # that has to be upgraded — caught here, before any file is that old.
        if _NOT_NULL_RE.search(declaration) and not _DEFAULT_RE.search(declaration):
            raise ValueError(f"{table}.{name}: an added NOT NULL column needs a DEFAULT")
    present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    added: list[str] = []
    for name, declaration in columns:
        if name not in present:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
            added.append(name)
    return added


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
