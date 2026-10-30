"""Refresh URL-fetched documents in the background (URL_REFRESH_INTERVAL_MINUTES, off by
default).

A document added from a URL (POST /documents/from-url) keeps where it came from and the
server's validators (ETag, Last-Modified). Once a minute this checks every workspace for
such documents whose last check is URL_REFRESH_INTERVAL_MINUTES old and fetches each
again the way POST /documents/{f}/refresh does: conditionally, so an unchanged page is
a 304 and nothing is downloaded, and through the same fetch slots and backlog as the
routes. At most URL_REFRESH_BATCH are fetched per pass.

When each document was last checked, and how, is kept by a ``RefreshStateStore``
(api/url_refresh_state.py): in memory, or in URL_REFRESH_STATE_PATH to survive a
restart. A document never checked counts as checked when it was last indexed. A
document can set its own interval (``refresh_minutes``; 0 = never), and a failing
one waits twice as long after each failure, up to a day.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from api.audit import audit
from api.ingest_jobs import IngestQueue, enqueue_refreshed, indexing_now
from api.jobs import JobStore
from api.metrics import url_refresh_total
from api.qdrant_schema import CollectionSchemaError
from api.raw_documents import RawDocumentStore
from api.repository import DocumentMetadata, VectorRepository
from api.settings import AppSettings
from api.url_fetch import BacklogFullError, FetchedDocument, fetch_document, fetch_slots
from api.url_refresh_state import MemoryRefreshStateStore, RefreshState, RefreshStateStore
from api.workspaces import workspace_of

logger = logging.getLogger(__name__)

URL_REFRESH_PASS_SECONDS = 60.0
# The longest a failing document waits between checks (unless its interval is longer).
MAX_BACKOFF_SECONDS = 86_400.0
_ACTIVE_STATES = frozenset({"queued", "parsing", "embedding"})

Fetch = Callable[..., FetchedDocument]


def refresh_interval_seconds(meta: DocumentMetadata, default_seconds: float) -> float | None:
    """How often ``meta``'s document is checked: its own interval, else the server's;
    None when it is never checked (its interval is 0, or the refresh is off)."""

    minutes = meta.source_refresh_minutes
    seconds = default_seconds if minutes is None else float(minutes) * 60.0
    return seconds if seconds > 0 else None


def refresh_due(
    meta: DocumentMetadata, state: RefreshState | None, default_seconds: float
) -> float | None:
    """When ``meta``'s document is next checked (Unix seconds), or None for never.

    From its last check when there was one, after a wait doubled for each failure in a
    row (at most a day, or its interval when that is longer); from its upload otherwise.
    """

    interval = refresh_interval_seconds(meta, default_seconds)
    if interval is None or not meta.source_url:
        return None
    if state is None:
        return (meta.uploaded_at or 0.0) + interval
    wait = interval
    if state.failures > 0:
        wait = min(interval * 2 ** min(state.failures, 20), max(MAX_BACKOFF_SECONDS, interval))
    return state.last_checked + wait


@dataclass(frozen=True)
class RefreshTarget:
    """One workspace whose fetched documents are kept current."""

    settings: AppSettings
    raw_store: RawDocumentStore | None
    queue: IngestQueue


class UrlRefresher:
    """The background loop; ``run_once`` is one pass (tests call it directly)."""

    def __init__(
        self,
        settings: AppSettings,
        repository: VectorRepository,
        targets: Callable[[], Sequence[RefreshTarget]],
        *,
        fetch: Fetch = fetch_document,
        clock: Callable[[], float] = time.time,
        interval: float = URL_REFRESH_PASS_SECONDS,
        state: RefreshStateStore | None = None,
    ) -> None:
        self._every = float(settings.url_refresh_interval_minutes) * 60.0
        self._batch = int(settings.url_refresh_batch)
        self._repository = repository
        self._targets = targets
        self._fetch = fetch
        self._clock = clock
        self._interval = interval
        self._state: RefreshStateStore = state if state is not None else MemoryRefreshStateStore()
        # (job store, job id) of the refreshes it queued, until they finish.
        self._jobs: set[tuple[JobStore, str]] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="docrag-url-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.run_once()
            except Exception:  # noqa: BLE001 - one failed pass must not end the loop
                logger.exception("Background URL refresh pass failed.")

    def _running(self) -> int:
        """How many of the jobs this refresher queued are still queued or running."""

        running = set()
        for job_store, job_id in self._jobs:
            job = job_store.get(job_id)
            if job is not None and job.state in _ACTIVE_STATES:
                running.add((job_store, job_id))
        self._jobs = running
        return len(running)

    def run_once(self) -> dict[str, int]:
        """Check what is due in every workspace, earliest due first; how many of each
        outcome.

        At most URL_REFRESH_BATCH fetches a pass, and none while that many of its own
        jobs are still indexing: a run of changed pages must not fill the backlog the
        uploads share. The first fetch deferred (no backlog room or fetch slot) ends
        the pass — the rest would be deferred for the same reason. The state of
        documents (and workspaces) that are gone is dropped.
        """

        outcomes = {"unchanged": 0, "queued": 0, "failed": 0, "deferred": 0}
        now = self._clock()
        room = self._batch - self._running()
        targets = list(self._targets())
        self._state.forget_collections_except(
            {target.settings.qdrant_collection for target in targets}
        )
        for target in targets:
            if outcomes["deferred"] or sum(outcomes.values()) >= room:
                break
            try:
                metadata = self._repository.filename_metadata(target.settings)
            except CollectionSchemaError as exc:
                logger.debug("Not refreshing workspace: %s", exc)
                continue
            collection = target.settings.qdrant_collection
            states = self._state.get_all(collection)
            self._state.forget(
                collection,
                [name for name in states if name not in metadata or not metadata[name].source_url],
            )
            busy = indexing_now(target.settings)
            due: list[tuple[float, str, DocumentMetadata]] = []
            for name, meta in metadata.items():
                when = refresh_due(meta, states.get(name), self._every)
                if when is not None and when <= now and name not in busy and meta.source_url:
                    due.append((when, name, meta))
            for _, name, meta in sorted(due):
                if sum(outcomes.values()) >= room:
                    break
                assert meta.source_url is not None
                outcome = self._refresh(target, name, meta.source_url, meta)
                outcomes[outcome] += 1
                url_refresh_total.labels(outcome=outcome).inc()
                if outcome == "deferred":
                    break
                previous = states.get(name)
                failures = (
                    (previous.failures if previous is not None else 0) + 1
                    if outcome == "failed"
                    else 0
                )
                checked = RefreshState(now, outcome, failures, 0.0)
                self._state.record(
                    collection,
                    name,
                    dataclasses.replace(
                        checked, next_due=refresh_due(meta, checked, self._every) or 0.0
                    ),
                )
        return outcomes

    def _refresh(self, target: RefreshTarget, name: str, url: str, meta: DocumentMetadata) -> str:
        if not target.queue.backlog.has_room(1) or not fetch_slots.acquire(blocking=False):
            return "deferred"  # the routes' fetches and uploads come first
        try:
            fetched = self._fetch(
                url,
                target.settings,
                filename=name,
                has_room=target.queue.backlog.has_room,
                etag=meta.source_etag,
                last_modified=meta.source_last_modified,
            )
        except BacklogFullError:
            return "deferred"
        except Exception as exc:  # noqa: BLE001 - a fetch error is this document's
            logger.warning("Scheduled refresh of %s failed: %s", name, exc)
            return "failed"
        finally:
            fetch_slots.release()
        if fetched.not_modified:
            return "unchanged"
        try:
            accepted = enqueue_refreshed(
                name,
                url,
                fetched,
                target.queue,
                target.raw_store,
                expected=meta,
                repository=self._repository,
            )
        except Exception as exc:  # noqa: BLE001 - a full backlog, a closing workspace
            logger.warning("Scheduled refresh of %s could not be queued: %s", name, exc)
            return "deferred"
        self._jobs.add((target.queue.job_store, accepted.job_id))
        audit(
            "document.refresh",
            workspace=workspace_of(target.settings),
            filename=name,
            job_id=accepted.job_id,
            scheduled=True,
        )
        return "queued"
