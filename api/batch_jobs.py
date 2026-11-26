"""Batches of questions answered in the background (``POST /questions/batch/jobs``).

``POST /questions/batch`` answers its questions inside one request: 20 questions at a
local model's pace ran past any proxy's timeout, and a client that dropped got nothing,
not even the answers already given. A job answers them on a thread of its own, one at
a time as question slots come free (it waits for one rather than failing with a 429),
and keeps each answer as it comes: ``GET /questions/batch/jobs/{id}`` reads the answers
so far, and a client that lost its connection reads them again.

Jobs live in memory, per process: a restart forgets them (their answers are in the
answer cache and the traces as any question's are).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

BatchState = Literal["queued", "running", "done", "cancelled", "failed"]

# Jobs remembered at once (the oldest finished one is forgotten first), how long a
# finished one is kept, and how many run at the same time.
MAX_BATCH_JOBS = 50
BATCH_JOB_KEEP_SECONDS = 24 * 3600
MAX_RUNNING_BATCH_JOBS = 2
# How often a job waiting for a question slot looks again.
_SLOT_POLL_SECONDS = 0.5


@dataclass
class BatchJob:
    id: str
    # The collection (workspace) it asks in: another workspace's client can't read it.
    collection: str
    questions: list[str]
    answers: list[Any] = field(default_factory=list)
    state: BatchState = "queued"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def finished(self) -> bool:
        return self.state in ("done", "cancelled", "failed")


class BatchJobStore:
    """The jobs of this process, bounded in number and age."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._jobs: OrderedDict[str, BatchJob] = OrderedDict()
        self._lock = threading.Lock()
        self._clock = clock

    def add(self, collection: str, questions: list[str]) -> BatchJob | None:
        """A new job, or None when MAX_RUNNING_BATCH_JOBS are going already."""

        with self._lock:
            self._forget_old()
            if sum(not job.finished for job in self._jobs.values()) >= MAX_RUNNING_BATCH_JOBS:
                return None
            job = BatchJob(uuid.uuid4().hex, collection, list(questions))
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_BATCH_JOBS:
                oldest = next((j for j in self._jobs.values() if j.finished), None)
                if oldest is None:
                    break
                del self._jobs[oldest.id]
            return job

    def get(self, job_id: str, collection: str) -> BatchJob | None:
        with self._lock:
            self._forget_old()
            job = self._jobs.get(job_id)
        return job if job is not None and job.collection == collection else None

    def _forget_old(self) -> None:
        now = self._clock()
        for job in list(self._jobs.values()):
            if job.finished_at is not None and now - job.finished_at > BATCH_JOB_KEEP_SECONDS:
                del self._jobs[job.id]


_store = BatchJobStore()


def batch_job_store() -> BatchJobStore:
    return _store


def run_batch_job(
    job: BatchJob,
    take_slot: Callable[[], Callable[[], None] | None],
    answer: Callable[[str, Callable[[], bool]], Any],
    failed: Callable[[str, Exception], Any],
) -> None:
    """Answer ``job``'s questions in order: for each a question slot (waited for), then
    ``answer`` (the question and a should-stop) — its result kept — or, when it raises,
    ``failed``'s. Stops before the next question once ``job.cancel`` is set."""

    job.state = "running"
    try:
        for question in job.questions:
            release = None
            while release is None:
                if job.cancel.is_set():
                    break
                release = take_slot()
                if release is None:
                    job.cancel.wait(_SLOT_POLL_SECONDS)
            if release is None:
                break
            try:
                try:
                    result = answer(question, job.cancel.is_set)
                except Exception as exc:  # noqa: BLE001 - reported on this question
                    if job.cancel.is_set():
                        break
                    result = failed(question, exc)
            finally:
                release()
            job.answers.append(result)
        job.state = "cancelled" if job.cancel.is_set() else "done"
    except Exception as exc:  # noqa: BLE001 - the job's own failure, kept for its reader
        job.state = "failed"
        job.error = str(exc)[:300]
    finally:
        job.finished_at = time.time()


__all__ = [
    "BATCH_JOB_KEEP_SECONDS",
    "MAX_BATCH_JOBS",
    "MAX_RUNNING_BATCH_JOBS",
    "BatchJob",
    "BatchJobStore",
    "batch_job_store",
    "run_batch_job",
]
