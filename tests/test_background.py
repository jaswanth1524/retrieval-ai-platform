"""api/background.py: the loop the watcher, re-indexer and URL refresher share, and the
jobs a worker owns."""

from __future__ import annotations

import logging
import threading

import pytest

from api.background import BackgroundLoop, OwnedJobs
from api.jobs import IngestJobStore
from tests.factories import wait_until


class Counting(BackgroundLoop):
    def __init__(self, *, first_pass_delayed: bool = False, fail_first: bool = False) -> None:
        super().__init__(
            "test-loop",
            3600,
            "A test pass failed.",
            logging.getLogger("tests.background"),
            first_pass_delayed=first_pass_delayed,
        )
        self.passes = 0
        self.fail_first = fail_first
        self.ran = threading.Event()

    def run_pass(self) -> None:
        self.passes += 1
        self.ran.set()
        if self.fail_first and self.passes == 1:
            raise RuntimeError("boom")


def test_a_pass_runs_at_once_again_on_wake_and_a_failure_does_not_end_the_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="tests.background")
    loop = Counting(fail_first=True)
    loop.start()
    try:
        wait_until(lambda: loop.passes == 1)
        loop.wake()
        wait_until(lambda: loop.passes == 2)
    finally:
        loop.stop()
    assert "A test pass failed." in caplog.text


def test_a_delayed_loop_waits_an_interval_first_and_stops_promptly() -> None:
    loop = Counting(first_pass_delayed=True)
    loop.start()
    assert not loop.ran.wait(0.2)
    loop.stop()  # wakes it: no pass, and the thread ends without waiting an hour
    assert loop.passes == 0
    assert loop._thread is not None and not loop._thread.is_alive()  # noqa: SLF001


def test_owned_jobs_report_each_finished_job_once() -> None:
    store = IngestJobStore(10)
    jobs: OwnedJobs[str] = OwnedJobs()
    running = store.create("a.txt")
    finished = store.create("b.txt")
    jobs.add(store, running.id, "a")
    jobs.add(store, finished.id, "b")
    store.update(finished.id, state="failed", error="bad")
    seen: list[tuple[str, str]] = []

    assert jobs.running(lambda info, job: seen.append((info, job.state))) == 1
    assert jobs.running(lambda info, job: seen.append((info, job.state))) == 1
    assert seen == [("b", "failed")]
