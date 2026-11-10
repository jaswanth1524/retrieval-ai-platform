"""A read-only check of how the index and the stored originals agree
(``GET /admin/consistency``, ``docrag check``).

The report changes nothing: each finding has its own remedy (re-index, re-upload,
delete), and which one is right is the operator's call. ``repair`` applies the ones
asked for (``POST /admin/consistency/repair``), each checked again under the document's
lock. What it looks at:

* originals with no indexed document (an ingest that failed after its original was
  kept by an older release, or a document deleted while its original couldn't be);
* indexed documents with no stored original, when originals are kept (they can't be
  re-indexed, restored from a backup or downloaded);
* folders of earlier versions whose document is gone;
* documents missing passages (a write cut off part way), stale ones, and documents
  with identical bytes under different names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from api.document_names import indexing_now
from api.document_ops import reindex_stale, staleness
from api.ingest_jobs import IngestQueue
from api.ingestion import filename_write_lock
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.schemas import DocumentJobAcceptedResponse
from api.settings import AppSettings
from api.workspaces import document_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsistencyReport:
    documents: int
    originals_kept: bool
    originals_without_document: list[str] = field(default_factory=list)
    documents_without_original: list[str] = field(default_factory=list)
    versions_without_document: list[str] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    duplicate_groups: list[list[str]] = field(default_factory=list)
    # Being indexed or deleted right now: left out of the findings above, which they
    # would otherwise show half-way.
    in_flight: list[str] = field(default_factory=list)

    @property
    def problems(self) -> int:
        return (
            len(self.originals_without_document)
            + len(self.documents_without_original)
            + len(self.versions_without_document)
            + len(self.incomplete)
        )


def consistency_report(
    repository: VectorRepository,
    settings: AppSettings,
    raw_store: RawDocumentStore | None,
    token_counter: object,
) -> ConsistencyReport:
    """``settings``' workspace as it stands (see the module docstring)."""

    from api.routes.documents_read import duplicate_groups

    busy = indexing_now(settings)
    metadata = repository.filename_metadata(settings)
    indexed = set(metadata) - busy
    is_stale = staleness(settings, token_counter)
    stored = raw_store.stored_names() - busy if raw_store is not None else set()
    versioned = raw_store.versioned_names() - busy if raw_store is not None else set()
    return ConsistencyReport(
        documents=len(metadata),
        originals_kept=raw_store is not None,
        originals_without_document=sorted(stored - indexed),
        documents_without_original=sorted(indexed - stored) if raw_store is not None else [],
        versions_without_document=sorted(versioned - indexed - stored),
        incomplete=sorted(name for name in indexed if metadata[name].incomplete),
        stale=sorted(name for name in indexed if is_stale(metadata[name])),
        duplicate_groups=duplicate_groups({name: metadata[name] for name in indexed}),
        in_flight=sorted(busy),
    )


@dataclass(frozen=True)
class RepairOutcome:
    originals_deleted: list[str] = field(default_factory=list)
    versions_deleted: list[str] = field(default_factory=list)
    reindex_jobs: list[DocumentJobAcceptedResponse] = field(default_factory=list)
    # Not touched: indexed (or being indexed) by the time it was looked at again, or it
    # couldn't be removed (logged); or couldn't be queued for a re-index now.
    left: list[str] = field(default_factory=list)


def repair(
    repository: VectorRepository,
    settings: AppSettings,
    raw_store: RawDocumentStore | None,
    queue: IngestQueue,
    token_counter: object,
    *,
    delete_orphan_originals: bool = False,
    delete_orphan_versions: bool = False,
    reindex_incomplete: bool = False,
) -> RepairOutcome:
    """Apply the asked-for remedies to what ``consistency_report`` finds now. An
    original (or a folder of versions) is deleted only when, under its document's
    write lock, no document of that name is indexed or in flight; incomplete documents
    are re-indexed from their originals (``reindex_stale`` for those names only)."""

    report = consistency_report(repository, settings, raw_store, token_counter)
    outcome = RepairOutcome()
    if raw_store is not None:
        targets: list[tuple[str, list[str]]] = []
        if delete_orphan_originals:
            targets.extend(
                (name, outcome.originals_deleted) for name in report.originals_without_document
            )
        if delete_orphan_versions:
            targets.extend(
                (name, outcome.versions_deleted) for name in report.versions_without_document
            )
        for name, done in targets:
            if _remove_orphan(repository, settings, raw_store, name):
                done.append(name)
            else:
                outcome.left.append(name)
    if reindex_incomplete and report.incomplete and raw_store is not None:
        wanted = set(report.incomplete)
        queued = reindex_stale(
            repository,
            settings,
            queue,
            raw_store,
            token_counter,
            skip=lambda name: name not in wanted,
        )
        outcome.reindex_jobs.extend(queued.jobs)
        outcome.left.extend([*queued.deferred, *queued.failed, *queued.no_original])
    return outcome


def _remove_orphan(
    repository: VectorRepository, settings: AppSettings, store: RawDocumentStore, name: str
) -> bool:
    with filename_write_lock(document_key(settings, name)):
        if name in indexing_now(settings) or repository.document_metadata(settings, name):
            return False
        try:
            store.delete(name)
        except OSError:
            logger.warning("Could not remove the orphaned original %s.", name, exc_info=True)
            return False
    return True
