"""Shared connection handling for the small sqlite-backed stores (jobs, feedback)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager


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
