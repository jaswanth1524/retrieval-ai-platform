"""Re-index stale documents in the background (AUTO_REINDEX_STALE, off by default).

After an upgrade that changes chunking (a ``CHUNKER_VERSION`` bump) or a change to the
chunk settings, every document is "older chunking" until someone clicks re-index all —
and then again for whatever the full backlog deferred. With this on, a background pass
every ``interval`` seconds queues the stale documents of the default workspace and every
other one through the same path as ``POST /documents/reindex``
(``document_ops.reindex_stale``), keeping at most ``AUTO_REINDEX_BATCH`` of its own jobs
queued or running so uploads still get through. It needs the stored originals
(RAW_DOCUMENT_DIR).

A workspace whose index was built for another embedding setup is skipped (re-indexing
can't write to it); one being deleted refuses the jobs, which the next pass retries.

A document whose re-index failed isn't queued again at once: the pass walks the stale
documents in name order, so the first few failing for good (an original that no longer
parses) took every place in the batch, pass after pass, and the rest never ran. One the
file itself failed ("document" error) waits until its original changes; any other
failure is retried after a backoff (``FAILURE_BACKOFF_SECONDS`` doubling, at most
``MAX_FAILURE_BACKOFF_SECONDS``), or at once when the original changes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from api.audit import audit
from api.background import BackgroundLoop, OwnedJobs
from api.backoff import doubled_wait
from api.document_ops import reindex_stale
from api.jobs import JobStore
from api.metrics import stale_documents
from api.qdrant_schema import CollectionSchemaError
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.settings import AppSettings
from api.workspace_target import WorkspaceTarget
from api.workspaces import workspace_of

logger = logging.getLogger(__name__)

AUTO_REINDEX_INTERVAL_SECONDS = 60.0
FAILURE_BACKOFF_SECONDS = 3600.0
MAX_FAILURE_BACKOFF_SECONDS = 86400.0


@dataclass(frozen=True)
class _QueuedJob:
    workspace: str
    filename: str
    # The original's (size, mtime) when queued: a failure counts against that file only.
    original: tuple[int, int] | None


@dataclass(frozen=True)
class _Failure:
    original: tuple[int, int] | None
    failures: int
    # Retry no sooner than this (time.monotonic); None = not until the original changes.
    retry_at: float | None


class StaleReindexer(BackgroundLoop):
    """The background loop; ``run_once`` is one pass (tests call it directly)."""

    def __init__(
        self,
        settings: AppSettings,
        repository: VectorRepository,
        targets: Callable[[], Sequence[WorkspaceTarget]],
        job_store: JobStore,
        token_counter: object,
        *,
        interval: float = AUTO_REINDEX_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._batch = int(settings.auto_reindex_batch)
        self._repository = repository
        self._targets = targets
        self._job_store = job_store
        self._token_counter = token_counter
        super().__init__(
            "docrag-auto-reindex", interval, "Background re-index pass failed.", logger
        )
        self._clock = clock
        self._jobs: OwnedJobs[_QueuedJob] = OwnedJobs()
        self._failures: dict[tuple[str, str], _Failure] = {}
        self._skipped: set[str] = set()

    def before_first_pass(self) -> None:
        # Staleness compares each point's token counter with the one in use now; asked
        # before the tokenizer has loaded, every HF-counted document could look stale.
        count = getattr(self._token_counter, "count", None)
        if callable(count):
            try:
                count("warmup")
            except Exception:  # noqa: BLE001 - the counter falls back on its own
                logger.debug("Token counter warmup failed.", exc_info=True)

    def run_pass(self) -> int:
        return self.run_once()

    def _running(self) -> int:
        """How many of its own jobs are still queued or running, noting how the
        finished ones ended."""

        return self._jobs.running(
            lambda queued, job: self._note_outcome(queued, job.state, job.error_type, job.error),
            # How it ended is unknown: backed off as a transient failure. One that did
            # index is no longer stale, so the back-off never holds it.
            lambda queued: self._note_outcome(queued, "failed", "evicted", None),
        )

    def _note_outcome(
        self, queued: _QueuedJob, state: str, error_type: str | None, error: str | None
    ) -> None:
        key = (queued.workspace, queued.filename)
        if state != "failed" or error_type == "cancelled":
            self._failures.pop(key, None)
            return
        previous = self._failures.get(key)
        failures = (
            1
            if previous is None or previous.original != queued.original
            else (previous.failures + 1)
        )
        if error_type == "document":
            retry_at = None
            logger.warning(
                "Not re-indexing %r (workspace %r) again until its original changes: %s",
                queued.filename,
                queued.workspace or "default",
                error,
            )
        else:
            delay = doubled_wait(FAILURE_BACKOFF_SECONDS, failures - 1, MAX_FAILURE_BACKOFF_SECONDS)
            retry_at = self._clock() + delay
        self._failures[key] = _Failure(queued.original, failures, retry_at)

    def _held_back(self, workspace: str, raw_store: RawDocumentStore) -> Callable[[str], bool]:
        def skip(filename: str) -> bool:
            failure = self._failures.get((workspace, filename))
            if failure is None:
                return False
            if raw_store.identity(filename) != failure.original:
                # A new original: whatever failed before may work now.
                del self._failures[(workspace, filename)]
                return False
            return failure.retry_at is None or self._clock() < failure.retry_at

        return skip

    def run_once(self) -> int:
        """Queue what fits of every workspace's stale documents; the jobs queued."""

        if getattr(self._token_counter, "mode", "hf") == "unknown":
            return 0
        queued = 0
        stale = 0
        for target in self._targets():
            name = workspace_of(target.settings) or "default"
            store, queue = target.raw_store, target.queue
            if store is None or queue is None:
                continue  # no originals kept there: nothing to re-index from
            room = self._batch - self._running()
            try:
                outcome = reindex_stale(
                    self._repository,
                    target.settings,
                    queue,
                    store,
                    self._token_counter,
                    limit=max(room, 0),
                    skip=self._held_back(workspace_of(target.settings), store),
                )
            except CollectionSchemaError as exc:
                if name not in self._skipped:
                    self._skipped.add(name)
                    logger.warning("Not re-indexing workspace %r: %s", name, exc)
                continue
            self._skipped.discard(name)
            stale += (
                len(outcome.jobs)
                + len(outcome.deferred)
                + len(outcome.in_progress)
                + len(outcome.failed)
                + len(outcome.skipped)
                + len(outcome.no_original)
            )
            for job in outcome.jobs:
                self._jobs.add(
                    self._job_store,
                    job.job_id,
                    _QueuedJob(
                        workspace_of(target.settings),
                        job.filename,
                        store.identity(job.filename),
                    ),
                )
            queued += len(outcome.jobs)
            for job in outcome.jobs:
                audit(
                    "document.reindex",
                    workspace=workspace_of(target.settings),
                    filename=job.filename,
                    job_id=job.job_id,
                    scheduled=True,
                )
            if outcome.jobs:
                logger.info(
                    "Re-indexing %d stale document(s) in workspace %r (%d waiting).",
                    len(outcome.jobs),
                    name,
                    len(outcome.deferred),
                )
        stale_documents.set(stale)
        return queued
