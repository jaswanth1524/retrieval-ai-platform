"""One workspace as a background worker goes over it (the stale re-indexer, the URL
refresher, scheduled backups): its settings, its originals store (None when originals
aren't kept) and an ingest queue for its documents. Built per pass by
``main.workspace_targets``."""

from __future__ import annotations

from dataclasses import dataclass

from api.ingest_jobs import IngestQueue
from api.raw_documents import RawDocumentStore
from api.settings import AppSettings


@dataclass(frozen=True)
class WorkspaceTarget:
    settings: AppSettings
    raw_store: RawDocumentStore | None
    # None only for a worker that never indexes (scheduled backups).
    queue: IngestQueue | None = None
