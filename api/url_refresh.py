"""Refresh URL-fetched documents in the background (URL_REFRESH_INTERVAL_MINUTES, off by
default).

A document added from a URL (POST /documents/from-url) keeps where it came from and the
server's validators (ETag, Last-Modified). Once a minute this checks every workspace for
such documents whose last check is URL_REFRESH_INTERVAL_MINUTES old and fetches each
again the way POST /documents/{f}/refresh does: conditionally, so an unchanged page is
a 304 and nothing is downloaded, and through the same fetch slots and backlog as the
routes. At most URL_REFRESH_BATCH are fetched per pass.

When a document was last checked is kept in memory, seeded from its upload time: after
a restart every document counts as checked when it was last indexed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from api.audit import audit
from api.ingest_jobs import IngestQueue, enqueue_refreshed, indexing_now
from api.metrics import url_refresh_total
from api.qdrant_schema import CollectionSchemaError
from api.raw_documents import RawDocumentStore
from api.repository import DocumentMetadata, VectorRepository
from api.settings import AppSettings
from api.url_fetch import BacklogFullError, FetchedDocument, fetch_document, fetch_slots
from api.workspaces import workspace_of

logger = logging.getLogger(__name__)

URL_REFRESH_PASS_SECONDS = 60.0

Fetch = Callable[..., FetchedDocument]


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
    ) -> None:
        self._every = float(settings.url_refresh_interval_minutes) * 60.0
        self._batch = int(settings.url_refresh_batch)
        self._repository = repository
        self._targets = targets
        self._fetch = fetch
        self._clock = clock
        self._interval = interval
        self._checked: dict[tuple[str, str], float] = {}
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

    def run_once(self) -> dict[str, int]:
        """Check what is due in every workspace; how many of each outcome."""

        outcomes = {"unchanged": 0, "queued": 0, "failed": 0, "deferred": 0}
        now = self._clock()
        for target in self._targets():
            if sum(outcomes.values()) - outcomes["deferred"] >= self._batch:
                break
            try:
                metadata = self._repository.filename_metadata(target.settings)
            except CollectionSchemaError as exc:
                logger.debug("Not refreshing workspace: %s", exc)
                continue
            busy = indexing_now(target.settings)
            collection = target.settings.qdrant_collection
            for name, meta in sorted(metadata.items()):
                if not meta.source_url or name in busy:
                    continue
                key = (collection, name)
                last = self._checked.get(key, meta.uploaded_at or 0.0)
                if now - last < self._every:
                    continue
                if sum(outcomes.values()) - outcomes["deferred"] >= self._batch:
                    break
                outcome = self._refresh(target, name, meta.source_url, meta)
                outcomes[outcome] += 1
                url_refresh_total.labels(outcome=outcome).inc()
                if outcome != "deferred":
                    self._checked[key] = now
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
            accepted = enqueue_refreshed(name, url, fetched, target.queue, target.raw_store)
        except Exception as exc:  # noqa: BLE001 - a full backlog, a closing workspace
            logger.warning("Scheduled refresh of %s could not be queued: %s", name, exc)
            return "deferred"
        audit(
            "document.refresh",
            workspace=workspace_of(target.settings),
            filename=name,
            job_id=accepted.job_id,
            scheduled=True,
        )
        return "queued"
