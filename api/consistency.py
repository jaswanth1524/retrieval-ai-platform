"""A read-only check of how the index and the stored originals agree
(``GET /admin/consistency``, ``docrag check``).

Nothing here changes anything: each finding has its own remedy (re-index, re-upload,
delete), and which one is right is the operator's call. What it looks at:

* originals with no indexed document (an ingest that failed after its original was
  kept by an older release, or a document deleted while its original couldn't be);
* indexed documents with no stored original, when originals are kept (they can't be
  re-indexed, restored from a backup or downloaded);
* folders of earlier versions whose document is gone;
* documents missing passages (a write cut off part way), stale ones, and documents
  with identical bytes under different names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from api.document_names import indexing_now
from api.document_ops import staleness
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.settings import AppSettings


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
