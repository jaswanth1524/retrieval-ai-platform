"""Background job tracking for document ingestion.

Two backends, selected by ``AppSettings.job_store_backend`` (see
``api.dependencies.get_ingest_job_store``, the sole call site that picks between them):

- ``IngestJobStore`` (default, "memory"): process-local, lost on restart. Acceptable
  for a single-instance self-hosted deployment where the process rarely restarts.
- ``SqliteIngestJobStore`` ("sqlite"): persists to a local sqlite3 file (stdlib, no
  new dependency) so a client polling a job survives the process restarting —
  otherwise it gets a 404 (``JobNotFoundError``) even though the ingest already
  succeeded in Qdrant.

Both implement the ``JobStore`` protocol below; every other module (``api.ingest_jobs``,
``api.dependencies``) types against that, not either concrete class.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections import OrderedDict
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol

from api.metrics import ingest_backlog_bytes, ingest_backlog_jobs, ingest_jobs_evicted_total
from api.sqlite_store import enable_wal, sqlite_transaction

JobState = Literal["queued", "parsing", "embedding", "done", "failed"]


class JobNotFoundError(RuntimeError):
    """Raised when a job id has no known job (never existed, or was pruned)."""


TERMINAL_STATES: frozenset[JobState] = frozenset({"done", "failed"})

INTERRUPTED_JOB_ERROR = (
    "The server restarted before this document finished indexing. Upload it again."
)


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
    # Wall clock, for oldest-first ordering that survives a restart
    # (SqliteIngestJobStore).
    created_at_wall: float = field(default_factory=time.time)
    # The workspace the document is indexed into ("" = the default): a job is only
    # visible (and cancellable) from its own workspace, as its filename and error are.
    workspace: str = ""
    # Why a failed job failed, coarsely (api.ingest_jobs.ingest_error_type): "document"
    # is the file itself (unparseable, unsupported, empty), which sending it again won't
    # change; the rest are the server's state at the time.
    error_type: str | None = None
    # What started it — upload, url (from-url or a refresh), reindex, restore (a
    # version), import, watch — and what it was given, so a failed one can be retried
    # (POST /documents/jobs/{id}/retry) as it was asked for.
    kind: str = "upload"
    source_url: str | None = None
    tags: tuple[str, ...] | None = None


JOB_KINDS = ("upload", "url", "reindex", "restore", "import", "watch")


class IngestBacklog:
    """Byte and job budget for uploads accepted but not yet finished ingesting.

    ``try_reserve`` never blocks: the upload route answers 503 instead of waiting. An
    empty backlog always admits, so a single upload larger than the budget still runs.

    ``max_jobs`` caps how many jobs can be in flight at once. It is sized to the job
    store's retention cap: past it the store would evict a job that is still queued, and
    a client polling that job would see a 404 for work that is still going to run.
    """

    def __init__(self, max_bytes: int, max_jobs: int | None = None) -> None:
        self._max_bytes = max_bytes
        self._max_jobs = max_jobs
        self._pending = 0
        self._jobs = 0
        self._lock = Lock()

    def try_reserve(self, size: int) -> bool:
        with self._lock:
            if not self._fits(size):
                return False
            self._pending += size
            self._jobs += 1
            self._publish()
            return True

    def has_room(self, size: int) -> bool:
        """Whether ``try_reserve(size)`` would succeed right now, without reserving.

        Lets a bulk re-index skip reading an original it couldn't queue anyway; the
        real reservation still happens when the job is enqueued.
        """

        with self._lock:
            return self._fits(size)

    def _fits(self, size: int) -> bool:
        # Under the lock. An empty backlog takes anything, however large.
        if self._pending and self._pending + size > self._max_bytes:
            return False
        return self._max_jobs is None or self._jobs < self._max_jobs

    def release(self, size: int) -> None:
        with self._lock:
            self._pending = max(0, self._pending - size)
            self._jobs = max(0, self._jobs - 1)
            self._publish()

    def _publish(self) -> None:
        # Under the lock, so the gauges never show a half-applied change.
        ingest_backlog_bytes.set(self._pending)
        ingest_backlog_jobs.set(self._jobs)

    @property
    def pending_bytes(self) -> int:
        with self._lock:
            return self._pending


class JobStore(Protocol):
    """Minimal surface ``api.ingest_jobs`` needs to track background ingest jobs."""

    def create(
        self,
        filename: str,
        workspace: str = "",
        *,
        kind: str = "upload",
        source_url: str | None = None,
        tags: tuple[str, ...] | None = None,
    ) -> IngestJob: ...

    def get(self, job_id: str) -> IngestJob | None: ...

    def update(self, job_id: str, **fields: object) -> None: ...

    def list_active(self, workspace: str) -> list[IngestJob]: ...

    def list_finished(
        self, workspace: str, *, failed_only: bool = False, limit: int = 100
    ) -> list[IngestJob]: ...


class IngestJobStore:
    """Thread-safe in-memory store for background ingest job status."""

    def __init__(self, max_retained: int) -> None:
        # Finished jobs beyond this count are pruned oldest-first, so a long-running
        # process doesn't accumulate unbounded job history in memory — mirrors
        # TraceStore's max_retained (api.tracing), sized from AppSettings.
        self._max_retained = max_retained
        self._lock = Lock()
        self._jobs: OrderedDict[str, IngestJob] = OrderedDict()

    def create(
        self,
        filename: str,
        workspace: str = "",
        *,
        kind: str = "upload",
        source_url: str | None = None,
        tags: tuple[str, ...] | None = None,
    ) -> IngestJob:
        job = IngestJob(
            id=str(uuid.uuid4()),
            filename=filename,
            workspace=workspace,
            kind=kind,
            source_url=source_url,
            tags=tags,
        )
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

    def list_active(self, workspace: str) -> list[IngestJob]:
        """``workspace``'s queued and running jobs, oldest first (copies)."""

        with self._lock:
            return [
                replace(job)
                for job in self._jobs.values()
                if job.workspace == workspace and job.state not in TERMINAL_STATES
            ]

    def list_finished(
        self, workspace: str, *, failed_only: bool = False, limit: int = 100
    ) -> list[IngestJob]:
        """``workspace``'s finished jobs still retained, newest first (copies): only the
        failed ones with ``failed_only``."""

        with self._lock:
            jobs = [
                replace(job)
                for job in reversed(self._jobs.values())
                if job.workspace == workspace
                and job.state in TERMINAL_STATES
                and (job.state == "failed" or not failed_only)
            ]
        return jobs[:limit]

    def _prune_finished_locked(self) -> None:
        """Evict oldest-first to stay within the cap, finished jobs first.

        Finished jobs go first because nobody is polling them any more. Non-terminal
        jobs are evicted only as a last resort: ``IngestBacklog``'s job cap (sized to
        this one in ``api.dependencies``) keeps in-flight jobs within it, so this only
        fires for a store built without that cap. A client polling an evicted job gets
        a 404 (``JobNotFoundError``).
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


_JOB_COLUMNS = (
    "id, filename, state, chunks_total, chunks_done, error, result_json, created_at_wall, "
    "workspace, error_type, kind, source_url, tags_json"
)


def _row_to_job(row: tuple[Any, ...]) -> IngestJob:
    (
        job_id,
        filename,
        state,
        chunks_total,
        chunks_done,
        error,
        result_json,
        created_at_wall,
        workspace,
        error_type,
        kind,
        source_url,
        tags_json,
    ) = row
    tags = json.loads(tags_json) if tags_json else None
    return IngestJob(
        id=str(job_id),
        filename=str(filename),
        state=state,
        chunks_total=int(chunks_total),
        chunks_done=int(chunks_done),
        error=None if error is None else str(error),
        result=None if result_json is None else json.loads(result_json),
        created_at_wall=float(created_at_wall),
        workspace=str(workspace),
        error_type=None if error_type is None else str(error_type),
        kind=str(kind or "upload"),
        source_url=None if source_url is None else str(source_url),
        tags=tuple(str(tag) for tag in tags) if isinstance(tags, list) else None,
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
        enable_wal(path)
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
                    created_at_wall REAL NOT NULL,
                    workspace TEXT NOT NULL DEFAULT '',
                    error_type TEXT
                )
                """
            )
            # A file from before workspaces: its jobs were all the default workspace's.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "workspace" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN workspace TEXT NOT NULL DEFAULT ''")
            if "error_type" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN error_type TEXT")
            # Older rows read as uploads: the one kind a retry never re-runs by itself.
            if "kind" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'upload'")
            if "source_url" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN source_url TEXT")
            if "tags_json" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN tags_json TEXT")
            # Only this process runs jobs (one uvicorn worker), so anything still
            # non-terminal at startup belonged to a process that died mid-ingest. Left
            # as-is, a client polling it would wait forever.
            terminal = ", ".join("?" for _ in TERMINAL_STATES)
            conn.execute(
                f"UPDATE jobs SET state = 'failed', error = ? WHERE state NOT IN ({terminal})",
                (INTERRUPTED_JOB_ERROR, *TERMINAL_STATES),
            )

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_transaction(self._path)

    def create(
        self,
        filename: str,
        workspace: str = "",
        *,
        kind: str = "upload",
        source_url: str | None = None,
        tags: tuple[str, ...] | None = None,
    ) -> IngestJob:
        job = IngestJob(
            id=str(uuid.uuid4()),
            filename=filename,
            workspace=workspace,
            kind=kind,
            source_url=source_url,
            tags=tags,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, filename, state, chunks_total, chunks_done, "
                "error, result_json, created_at_wall, workspace, kind, source_url, tags_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.id,
                    job.filename,
                    job.state,
                    job.chunks_total,
                    job.chunks_done,
                    job.error,
                    None,
                    job.created_at_wall,
                    job.workspace,
                    job.kind,
                    job.source_url,
                    json.dumps(list(tags)) if tags is not None else None,
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

    def list_active(self, workspace: str) -> list[IngestJob]:
        terminal = ", ".join("?" for _ in TERMINAL_STATES)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM jobs WHERE workspace = ? "
                f"AND state NOT IN ({terminal}) ORDER BY created_at_wall, rowid",
                (workspace, *TERMINAL_STATES),
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    def list_finished(
        self, workspace: str, *, failed_only: bool = False, limit: int = 100
    ) -> list[IngestJob]:
        states = ("failed",) if failed_only else tuple(TERMINAL_STATES)
        placeholders = ", ".join("?" for _ in states)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM jobs WHERE workspace = ? "
                f"AND state IN ({placeholders}) ORDER BY created_at_wall DESC, rowid DESC "
                "LIMIT ?",
                (workspace, *states, limit),
            ).fetchall()
        return [_row_to_job(row) for row in rows]

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
                "error=?, result_json=?, error_type=? WHERE id=?",
                (
                    job.filename,
                    job.state,
                    job.chunks_total,
                    job.chunks_done,
                    job.error,
                    None if job.result is None else json.dumps(job.result),
                    job.error_type,
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
