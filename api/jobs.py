"""In-memory background job tracking for document ingestion.

Jobs are process-local and lost on restart — acceptable for a single-instance
self-hosted deployment; there is no requirement for durable job persistence across
process restarts (unlike the indexed documents themselves, which live in Qdrant).
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Literal

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
    created_at: float = field(default_factory=time.monotonic)


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
        for job_id in finished_ids[:overflow]:
            del self._jobs[job_id]

        still_over = len(self._jobs) - self._max_retained
        if still_over <= 0:
            return
        # Insertion order is creation order, so this drops the least recently created.
        for job_id in list(self._jobs)[:still_over]:
            del self._jobs[job_id]
