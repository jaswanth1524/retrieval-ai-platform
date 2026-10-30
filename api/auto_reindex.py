"""Re-index stale documents in the background (AUTO_REINDEX_STALE, off by default).

After an upgrade that changes chunking (a ``CHUNKER_VERSION`` bump) or a change to the
chunk settings, every document is "older chunking" until someone clicks re-index all —
and then again for whatever the full backlog deferred. With this on, a background pass
every ``interval`` seconds queues the stale documents of the default workspace and every
other one through the same path as ``POST /documents/reindex``
(``ingest_jobs.reindex_stale``), keeping at most ``AUTO_REINDEX_BATCH`` of its own jobs
queued or running so uploads still get through. It needs the stored originals
(RAW_DOCUMENT_DIR).

A workspace whose index was built for another embedding setup is skipped (re-indexing
can't write to it); one being deleted refuses the jobs, which the next pass retries.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from api.audit import audit
from api.ingest_jobs import IngestQueue, reindex_stale
from api.jobs import JobStore
from api.metrics import stale_documents
from api.qdrant_schema import CollectionSchemaError
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.settings import AppSettings
from api.workspaces import workspace_of

logger = logging.getLogger(__name__)

AUTO_REINDEX_INTERVAL_SECONDS = 60.0
_ACTIVE_STATES = frozenset({"queued", "parsing", "embedding"})


@dataclass(frozen=True)
class ReindexTarget:
    """One workspace to keep current: its settings, original store and ingest queue."""

    settings: AppSettings
    raw_store: RawDocumentStore
    queue: IngestQueue


class StaleReindexer:
    """The background loop; ``run_once`` is one pass (tests call it directly)."""

    def __init__(
        self,
        settings: AppSettings,
        repository: VectorRepository,
        targets: Callable[[], Sequence[ReindexTarget]],
        job_store: JobStore,
        token_counter: object,
        *,
        interval: float = AUTO_REINDEX_INTERVAL_SECONDS,
    ) -> None:
        self._batch = int(settings.auto_reindex_batch)
        self._repository = repository
        self._targets = targets
        self._job_store = job_store
        self._token_counter = token_counter
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jobs: set[str] = set()
        self._skipped: set[str] = set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="docrag-auto-reindex", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        # Staleness compares each point's token counter with the one in use now; asked
        # before the tokenizer has loaded, every HF-counted document could look stale.
        count = getattr(self._token_counter, "count", None)
        if callable(count):
            try:
                count("warmup")
            except Exception:  # noqa: BLE001 - the counter falls back on its own
                logger.debug("Token counter warmup failed.", exc_info=True)
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:  # noqa: BLE001 - one failed pass must not end the loop
                logger.exception("Background re-index pass failed.")
            if self._stop.wait(self._interval):
                return

    def _running(self) -> int:
        running = set()
        for job_id in self._jobs:
            job = self._job_store.get(job_id)
            if job is not None and job.state in _ACTIVE_STATES:
                running.add(job_id)
        self._jobs = running
        return len(running)

    def run_once(self) -> int:
        """Queue what fits of every workspace's stale documents; the jobs queued."""

        if getattr(self._token_counter, "mode", "hf") == "unknown":
            return 0
        queued = 0
        stale = 0
        for target in self._targets():
            name = workspace_of(target.settings) or "default"
            room = self._batch - self._running()
            try:
                outcome = reindex_stale(
                    self._repository,
                    target.settings,
                    target.queue,
                    target.raw_store,
                    self._token_counter,
                    limit=max(room, 0),
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
            )
            self._jobs.update(job.job_id for job in outcome.jobs)
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
