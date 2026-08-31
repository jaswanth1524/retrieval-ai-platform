"""Background job tracking for document ingestion.

Two backends, selected by ``AppSettings.job_store_backend`` (see
``api.dependencies.get_ingest_job_store``, the sole call site that picks between them):

- ``IngestJobStore`` (default, "memory"): process-local, lost on restart. Acceptable
  for a single-instance self-hosted deployment where the process rarely restarts.
- ``SqliteIngestJobStore`` ("sqlite"): persists to a local sqlite3 file (stdlib, no
  new dependency) so a client polling a job survives the process restarting —
  otherwise it gets a 404 (``JobNotFoundError``) even though the ingest already
  succeeded in Qdrant.

Both implement the ``JobStore`` protocol below; every other module (``api.main``,
``api.dependencies``) types against that, not either concrete class.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol

from api.metrics import ingest_jobs_evicted_total

JobState = Literal["queued", "parsing", "embedding", "done", "failed"]


class JobNotFoundError(RuntimeError):
    """Raised when a job id has no known job (never existed, or was pruned)."""

TERMINAL_STATES: frozenset[JobState] = frozenset({"done", "failed"})


@dataclass
class IngestJob:
    """Status of one background document-ingestion job."""

    id: str
    filename: str
    state: JobState = "queued"
    chunks_total: int = 0
    chunks_done: int = 0
    error: str | None = None
    result: dict[str, object] | None = None
    # Process-relative (time.monotonic): safe for in-process duration math, but
    # meaningless across a restart, so it can't order jobs persisted to disk.
    created_at: float = field(default_factory=time.monotonic)
    # Wall-clock equivalent used only for oldest-first ordering when a job store
    # backend needs that to survive a restart (SqliteIngestJobStore). Additive field
    # with a default, so existing IngestJobStore construction/tests are unaffected.
    created_at_wall: float = field(default_factory=time.time)


class JobStore(Protocol):
    """Minimal surface ``api.main`` needs to track background ingest jobs."""

    def create(self, filename: str) -> IngestJob: ...

    def get(self, job_id: str) -> IngestJob | None: ...

    def update(self, job_id: str, **fields: object) -> None: ...


class IngestJobStore:
    """Thread-safe in-memory store for background ingest job status."""

    def __init__(self, max_retained: int) -> None:
        # Finished jobs beyond this count are pruned oldest-first, so a long-running
        # process doesn't accumulate unbounded job history in memory — mirrors
        # TraceStore's max_retained (api.tracing), sized from AppSettings.
        self._max_retained = max_retained
        self._lock = Lock()
        self._jobs: OrderedDict[str, IngestJob] = OrderedDict()

    def create(self, filename: str) -> IngestJob:
        job = IngestJob(id=str(uuid.uuid4()), filename=filename)
        with self._lock:
            self._jobs[job.id] = job
            self._prune_finished_locked()
        return job

    def get(self, job_id: str) -> IngestJob | None:
        """Return a snapshot of a job's current status, or None if unknown/pruned.

        Returns a copy (not the live object) so callers never observe a
        partially-updated job while a background thread is mid-``update``.
        """

        with self._lock:
            job = self._jobs.get(job_id)
            return replace(job) if job is not None else None

    def update(self, job_id: str, **fields: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def _prune_finished_locked(self) -> None:
        """Evict oldest-first to stay within the cap, finished jobs first.

        Finished jobs go first because nobody is polling them any more. But the store
        must not grow without bound when none have finished: uploads are client-driven
        and the executor has only two workers, so a burst leaves a long queue of
        non-terminal jobs. Past the cap those are evicted too, oldest first — a client
        polling an evicted job gets a 404 (``JobNotFoundError``), which is the same
        answer it already gets for a job pruned after completion.
        """

        overflow = len(self._jobs) - self._max_retained
        if overflow <= 0:
            return
        finished_ids = [
            job_id for job_id, job in self._jobs.items() if job.state in TERMINAL_STATES
        ]
        to_evict = finished_ids[:overflow]
        for job_id in to_evict:
            del self._jobs[job_id]
        if to_evict:
            ingest_jobs_evicted_total.inc(len(to_evict))

        still_over = len(self._jobs) - self._max_retained
        if still_over <= 0:
            return
        # Insertion order is creation order, so this drops the least recently created.
        still_evicting = list(self._jobs)[:still_over]
        for job_id in still_evicting:
            del self._jobs[job_id]
        ingest_jobs_evicted_total.inc(len(still_evicting))


_JOB_COLUMNS = "id, filename, state, chunks_total, chunks_done, error, result_json, created_at_wall"


def _row_to_job(row: tuple[Any, ...]) -> IngestJob:
    job_id, filename, state, chunks_total, chunks_done, error, result_json, created_at_wall = row
    return IngestJob(
        id=str(job_id),
        filename=str(filename),
        state=state,
        chunks_total=int(chunks_total),
        chunks_done=int(chunks_done),
        error=None if error is None else str(error),
        result=None if result_json is None else json.loads(result_json),
        # Loaded from disk, not measured in this process — meaningless for duration
        # math (nothing reads IngestJob.created_at; see the module docstring), only
        # needs to exist to satisfy the dataclass.
        created_at=time.monotonic(),
        created_at_wall=float(created_at_wall),
    )


class SqliteIngestJobStore:
    """Sqlite-backed job store — same public surface as ``IngestJobStore``, but
    persists across process restarts. Selected via
    ``AppSettings.job_store_backend="sqlite"`` (see ``api.dependencies.get_ingest_job_store``).

    A short-lived connection is opened per call, inside the same lock guarding every
    other operation on this instance: sqlite3 connections aren't safe to share across
    threads, and this store is written from the ingest executor's worker threads while
    read from request threads.
    """

    def __init__(self, path: str, max_retained: int) -> None:
        self._path = path
        self._max_retained = max_retained
        self._lock = Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    state TEXT NOT NULL,
                    chunks_total INTEGER NOT NULL,
                    chunks_done INTEGER NOT NULL,
                    error TEXT,
                    result_json TEXT,
                    created_at_wall REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def create(self, filename: str) -> IngestJob:
        job = IngestJob(id=str(uuid.uuid4()), filename=filename)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, filename, state, chunks_total, chunks_done, "
                "error, result_json, created_at_wall) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.id,
                    job.filename,
                    job.state,
                    job.chunks_total,
                    job.chunks_done,
                    job.error,
                    None,
                    job.created_at_wall,
                ),
            )
            self._prune_finished_locked(conn)
        return job

    def get(self, job_id: str) -> IngestJob | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return None if row is None else _row_to_job(row)

    def update(self, job_id: str, **fields: object) -> None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return
            job = _row_to_job(row)
            for key, value in fields.items():
                setattr(job, key, value)
            conn.execute(
                "UPDATE jobs SET filename=?, state=?, chunks_total=?, chunks_done=?, "
                "error=?, result_json=? WHERE id=?",
                (
                    job.filename,
                    job.state,
                    job.chunks_total,
                    job.chunks_done,
                    job.error,
                    None if job.result is None else json.dumps(job.result),
                    job_id,
                ),
            )

    def _prune_finished_locked(self, conn: sqlite3.Connection) -> None:
        """Mirrors ``IngestJobStore._prune_finished_locked``: evict oldest-first (by
        insertion order, i.e. ``rowid``) to stay within the cap, finished jobs first,
        then any job regardless of state if the cap is still exceeded."""

        (total,) = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
        overflow = total - self._max_retained
        if overflow <= 0:
            return
        finished_states = ", ".join("?" for _ in TERMINAL_STATES)
        finished_ids = [
            row[0]
            for row in conn.execute(
                f"SELECT id FROM jobs WHERE state IN ({finished_states}) "
                "ORDER BY rowid ASC LIMIT ?",
                (*TERMINAL_STATES, overflow),
            ).fetchall()
        ]
        if finished_ids:
            placeholders = ", ".join("?" for _ in finished_ids)
            conn.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", finished_ids)
            ingest_jobs_evicted_total.inc(len(finished_ids))

        (total,) = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
        still_over = total - self._max_retained
        if still_over <= 0:
            return
        still_evicting = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM jobs ORDER BY rowid ASC LIMIT ?", (still_over,)
            ).fetchall()
        ]
        placeholders = ", ".join("?" for _ in still_evicting)
        conn.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", still_evicting)
        ingest_jobs_evicted_total.inc(len(still_evicting))
