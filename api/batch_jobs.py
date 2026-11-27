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
# finished one is kept, and how many run at the same time: in one workspace, and in all
# of them (each is a thread).
MAX_BATCH_JOBS = 50
BATCH_JOB_KEEP_SECONDS = 24 * 3600
MAX_RUNNING_BATCH_JOBS = 2
MAX_RUNNING_BATCH_JOBS_TOTAL = 6
# How often a job waiting for a question slot looks again.
_SLOT_POLL_SECONDS = 0.5
# Left between a batch's questions, longer than the poll: a job that took the slot back
# at once left another waiting for it until its own last question.
BATCH_YIELD_SECONDS = _SLOT_POLL_SECONDS * 1.5


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
    # Stop before the next question (a client's cancel).
    cancel: threading.Event = field(default_factory=threading.Event)
    # Stop the question being answered too: its workspace or the server is going.
    abort: threading.Event = field(default_factory=threading.Event)

    @property
    def finished(self) -> bool:
        return self.state in ("done", "cancelled", "failed")

    def stop_now(self) -> None:
        self.abort.set()
        self.cancel.set()


class BatchJobStore:
    """The jobs of this process, bounded in number and age."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._jobs: OrderedDict[str, BatchJob] = OrderedDict()
        self._lock = threading.Lock()
        self._clock = clock

    def add(self, collection: str, questions: list[str]) -> BatchJob | None:
        """A new job, or None when as many as may run are going already: in this
        workspace (MAX_RUNNING_BATCH_JOBS) or in all (MAX_RUNNING_BATCH_JOBS_TOTAL)."""

        with self._lock:
            self._forget_old()
            running = [job.collection for job in self._jobs.values() if not job.finished]
            if (
                len(running) >= MAX_RUNNING_BATCH_JOBS_TOTAL
                or running.count(collection) >= MAX_RUNNING_BATCH_JOBS
            ):
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

    def abort(self, collection: str | None = None) -> None:
        """Stop every unfinished job (of ``collection``, when given), the question it
        is answering included."""

        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if not job.finished and collection in (None, job.collection):
                job.stop_now()

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
    pause_seconds: float = 0.0,
) -> None:
    """Answer ``job``'s questions in order: for each a question slot (waited for), then
    ``answer`` (the question and a should-stop) — its result kept — or, when it raises,
    ``failed``'s. Stops before the next question once ``job.cancel`` is set; ``answer``'s
    should-stop is ``job.abort``. ``pause_seconds`` is left between questions for others
    to take the slot."""

    job.state = "running"
    try:
        for index, question in enumerate(job.questions):
            if index and pause_seconds:
                job.cancel.wait(pause_seconds)
            release = None
            while not job.cancel.is_set():
                release = take_slot()
                if release is not None:
                    break
                if release is None:
                    job.cancel.wait(_SLOT_POLL_SECONDS)
            if release is None:
                break
            try:
                try:
                    result = answer(question, job.abort.is_set)
                except Exception as exc:  # noqa: BLE001 - reported on this question
                    if job.abort.is_set():
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
    "BATCH_YIELD_SECONDS",
    "MAX_BATCH_JOBS",
    "MAX_RUNNING_BATCH_JOBS",
    "MAX_RUNNING_BATCH_JOBS_TOTAL",
    "BatchJob",
    "BatchJobStore",
    "batch_job_store",
    "run_batch_job",
]
