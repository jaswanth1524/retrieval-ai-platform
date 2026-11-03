"""What the server's background workers share: a loop on a daemon thread, and the jobs a
worker queued itself.

The folder watcher (api/watch.py), the stale re-indexer (api/auto_reindex.py) and the
URL refresher (api/url_refresh.py) each run one pass every so often until the app stops
them, and the last two count only their own jobs against their batch.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from api.jobs import TERMINAL_STATES, IngestJob, JobStore


class BackgroundLoop:
    """Runs ``run_pass`` on a daemon thread every ``interval`` seconds until ``stop``.

    ``wake`` runs the next pass at once. With ``first_pass_delayed`` the first pass waits
    one interval (the app has just started). A pass that raises is logged with
    ``failure_message`` on ``log`` (the worker's own logger) and the loop goes on: one
    bad pass must not end it.
    """

    def __init__(
        self,
        thread_name: str,
        interval: float,
        failure_message: str,
        log: logging.Logger,
        *,
        first_pass_delayed: bool = False,
        join_timeout: float = 5.0,
    ) -> None:
        self._thread_name = thread_name
        self._interval = interval
        self._failure_message = failure_message
        self._log = log
        self._first_pass_delayed = first_pass_delayed
        self._join_timeout = join_timeout
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def run_pass(self) -> object:
        raise NotImplementedError

    def before_first_pass(self) -> None:
        """Work to do on the loop's thread before its first pass (nothing by default)."""

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=self._thread_name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout)

    def wake(self) -> None:
        self._wake.set()

    def _wait(self) -> bool:
        """Wait one interval (or for ``wake``); whether the loop should go on."""

        self._wake.wait(self._interval)
        self._wake.clear()
        return not self._stop.is_set()

    def _run(self) -> None:
        self.before_first_pass()
        if self._first_pass_delayed and not self._wait():
            return
        while not self._stop.is_set():
            try:
                self.run_pass()
            except Exception:  # noqa: BLE001 - one failed pass must not end the loop
                self._log.exception(self._failure_message)
            if not self._wait():
                return


class OwnedJobs[T]:
    """The jobs a background worker queued, each with what it needs to know when the job
    ends, until it ends."""

    def __init__(self) -> None:
        self._jobs: dict[tuple[JobStore, str], T] = {}

    def add(self, job_store: JobStore, job_id: str, info: T) -> None:
        self._jobs[(job_store, job_id)] = info

    def running(self, on_finished: Callable[[T, IngestJob], None] | None = None) -> int:
        """How many are still queued or running. Each one found finished is handed to
        ``on_finished`` once and forgotten; one the job store no longer has is just
        forgotten."""

        still: dict[tuple[JobStore, str], T] = {}
        for (job_store, job_id), info in self._jobs.items():
            job = job_store.get(job_id)
            if job is not None and job.state not in TERMINAL_STATES:
                still[(job_store, job_id)] = info
            elif job is not None and on_finished is not None:
                on_finished(info, job)
        self._jobs = still
        return len(still)
