"""Vector-store repository — the only module that talks to Qdrant for search/writes.

Ingestion depends on the ``WriteRepository`` protocol it declares locally, which
``VectorRepository`` satisfies by structural typing, so callers never import
``qdrant_client`` directly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal
from weakref import WeakKeyDictionary, WeakSet

import grpc
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models.models import QueryResponse

from api.corpus import changed_since, corpus_generation
from api.documents import CHUNKER_VERSION, DocumentChunk
from api.embeddings import EmbeddedText
from api.qdrant_schema import (
    VectorStoreUnavailableError,
    ensure_collection,
    forget_collection,
    recreate_if_empty_and_mismatched,
)
from api.settings import AppSettings

logger = logging.getLogger(__name__)

# Status codes from a server-side RRF query that indicate "this Qdrant server/client
# combination doesn't support prefetch+RRF" rather than a real query failure — only
# these should trigger the manual-fusion fallback; anything else is a real error.
_RRF_UNSUPPORTED_STATUS_CODES = frozenset({400, 404, 501})

# Clients whose server has already rejected the server-side hybrid query. Remembered
# per process so every later question goes straight to manual fusion instead of paying
# a failed round trip first; keyed weakly by client so a new client re-probes.
_server_side_hybrid_unsupported: WeakSet[QdrantClient] = WeakSet()


# filename_metadata's cache: client -> collection -> (corpus generation, time, result).
# How long an entry lives is DOCUMENT_LISTING_TTL_SECONDS (writes from another process).
# Past this many changed documents, one full scan beats a filtered re-read of each.
_METADATA_PATCH_MAX_FILENAMES = 50
_metadata_lock = Lock()
_metadata_cache: WeakKeyDictionary[
    QdrantClient, dict[str, tuple[int, float, dict[str, DocumentMetadata]]]
]
_metadata_cache = WeakKeyDictionary()
# One scan at a time per (client, collection): after every invalidation (each finished
# job) the UI's concurrent GET /documents calls each scrolled the whole collection.
_metadata_scan_locks: WeakKeyDictionary[QdrantClient, dict[str, Lock]] = WeakKeyDictionary()


def clear_metadata_cache() -> None:
    """Forget every cached document listing (tests, reloads)."""

    with _metadata_lock:
        _metadata_cache.clear()


def forget_metadata(client: QdrantClient, collection: str) -> None:
    """Forget one collection's cached listing and its scan lock (a deleted workspace):
    the other workspaces' listings stay warm, and a lock per collection ever deleted
    is no longer kept for the life of the process."""

    with _metadata_lock:
        _metadata_cache.get(client, {}).pop(collection, None)
        _metadata_scan_locks.get(client, {}).pop(collection, None)


def clear_hybrid_fallback_cache() -> None:
    """Forget which clients needed the manual-fusion fallback (tests, reloads)."""

    _server_side_hybrid_unsupported.clear()


# Payload fields a search hit is read for (retrieval.scored_point_to_chunk). Fetching
# the whole payload sent each of 50 candidates' tags, fingerprint, sizes and stamps
# over the wire for nothing.
SEARCH_PAYLOAD_FIELDS = [
    "filename",
    "page",
    "page_end",
    "section",
    "chunk_id",
    "text",
    "chunk_ordinal",
    # RECENCY_BOOST_WEIGHT orders kept passages by it.
    "uploaded_at",
]
# Neighbour expansion reads only these.
NEIGHBOR_PAYLOAD_FIELDS = ["chunk_ordinal", "text", "chunk_id", "page", "page_end", "section"]
# Distinct tags one facet call returns before stored_tags falls back to a full scan.
TAG_FACET_LIMIT = 1000


@dataclass(frozen=True)
class SeedPassage:
    """A stored passage a similar-passages search starts from: its own vectors are
    the query (both legs, so the search stays hybrid)."""

    chunk_id: str
    filename: str
    section: str
    text: str
    embedding: EmbeddedText


@dataclass(frozen=True)
class DocumentMetadata:
    """Per-filename aggregate derived from its indexed points' payloads."""

    chunk_count: int
    page_count: int
    # None when every one of the filename's points predates byte_size/uploaded_at
    # being stamped at ingest time (see api.documents.DocumentChunk) — nullable, not
    # forced by a re-ingest.
    byte_size: int | None
    uploaded_at: float | None
    # Oldest chunker version among the filename's points; None when any point predates
    # the field (legacy). A re-ingest that failed partway can leave old-version points
    # behind, so the minimum — not whichever point came first — decides staleness.
    chunker_version: int | None = None
    # Union of tags across the filename's points (they are normally all the same).
    tags: tuple[str, ...] = ()
    # Every chunking fingerprint among the points (empty when none carries one), and
    # whether any point was sized by the heuristic token counter.
    chunking_fingerprints: frozenset[str] = frozenset()
    heuristic_chunked: bool = False
    # The uploaded bytes' SHA-256 when every point carries the same one, else None, and
    # the chunk count the ingest that wrote them meant to write (when they agree).
    content_sha256: str | None = None
    document_chunks: int | None = None
    # Where a URL-ingested document came from (any point's; they are all the same).
    source_url: str | None = None
    source_etag: str | None = None
    source_last_modified: str | None = None
    # A fetched document's own refresh interval (None: the server's; 0: never).
    source_refresh_minutes: int | None = None
    # Points from more than one ingest, or fewer than the ingest that wrote them meant
    # to write: a write cut off part way (a killed container) or a failed clean-up of
    # the previous version's points. Never set for points from before those fields.
    incomplete: bool = False

    def is_stale(self, fingerprint: str, counter_mode: str) -> bool:
        """Chunked differently from how it would be today, so re-indexing changes it.

        Stale when any point predates ``CHUNKER_VERSION`` (or the field), when any point
        was chunked under other settings, or when the heuristic sized it while the real
        tokenizer is available now. A point with no fingerprint but a current version
        counts as current, so upgrading doesn't flag the whole corpus at once.
        """

        if self.chunker_version is None or self.chunker_version < CHUNKER_VERSION:
            return True
        if self.chunking_fingerprints - {fingerprint}:
            return True
        return self.heuristic_chunked and counter_mode == "hf"


class _MetadataFold:
    """One document's metadata, folded point by point during a listing scroll.

    ``page_count`` is the highest page seen (pages are 1-based). ``byte_size``,
    ``uploaded_at`` and the source come from the first point that has them (every chunk
    of one ingest carries the same values); a document ingested before those fields
    existed has none. The chunker version is the lowest seen, None when any point has
    none (legacy: stale).
    """

    def __init__(self) -> None:
        self.count = 0
        self.max_page = 0
        self.byte_size: int | None = None
        self.uploaded_at: float | None = None
        self.min_version: int | None = None
        self.unversioned = False
        self.tags: dict[str, None] = {}
        self.fingerprints: set[str] = set()
        self.heuristic = False
        # None stands for a point without one.
        self.hashes: set[str | None] = set()
        self.totals: set[int | None] = set()
        self.source: tuple[str, str | None, str | None, int | None] | None = None

    def add(self, payload: dict[str, Any]) -> None:
        self.count += 1
        page = payload.get("page")
        if isinstance(page, int):
            self.max_page = max(self.max_page, page)
        size = payload.get("byte_size")
        if isinstance(size, int) and self.byte_size is None:
            self.byte_size = size
        stamp = payload.get("uploaded_at")
        if isinstance(stamp, int | float) and self.uploaded_at is None:
            self.uploaded_at = float(stamp)
        version = payload.get("chunker_version")
        if isinstance(version, int) and not isinstance(version, bool):
            lowest = self.min_version
            self.min_version = version if lowest is None else min(lowest, version)
        else:
            self.unversioned = True
        point_tags = payload.get("tags")
        if isinstance(point_tags, list):
            self.tags.update((tag, None) for tag in point_tags if isinstance(tag, str))
        fingerprint = payload.get("chunking_fingerprint")
        if isinstance(fingerprint, str):
            self.fingerprints.add(fingerprint)
        if payload.get("token_counter") == "heuristic":
            self.heuristic = True
        digest = payload.get("content_sha256")
        self.hashes.add(digest if isinstance(digest, str) else None)
        total = payload.get("document_chunks")
        self.totals.add(total if isinstance(total, int) and not isinstance(total, bool) else None)
        url = payload.get("source_url")
        if isinstance(url, str) and self.source is None:
            etag = payload.get("source_etag")
            modified = payload.get("source_last_modified")
            minutes = payload.get("source_refresh_minutes")
            self.source = (
                url,
                etag if isinstance(etag, str) else None,
                modified if isinstance(modified, str) else None,
                minutes if isinstance(minutes, int) and not isinstance(minutes, bool) else None,
            )

    def result(self) -> DocumentMetadata:
        url, etag, modified, minutes = (
            self.source if self.source is not None else (None, None, None, None)
        )
        return DocumentMetadata(
            chunk_count=self.count,
            page_count=self.max_page,
            byte_size=self.byte_size,
            uploaded_at=self.uploaded_at,
            chunker_version=None if self.unversioned else self.min_version,
            tags=tuple(self.tags),
            chunking_fingerprints=frozenset(self.fingerprints),
            heuristic_chunked=self.heuristic,
            content_sha256=_single_hash(self.hashes),
            document_chunks=_single_value(self.totals),
            source_url=url,
            source_etag=etag,
            source_last_modified=modified,
            source_refresh_minutes=minutes,
            incomplete=self._incomplete(),
        )

    def _incomplete(self) -> bool:
        # Every point one ingest writes carries the same hash and chunk total, so any
        # mix means two ingests' points side by side. A document whose points all
        # predate both fields (all None) is never flagged.
        if len(self.hashes) > 1 or len(self.totals) > 1:
            return True
        total = _single_value(self.totals)
        return total is not None and total != self.count


# Points per scroll page. Ids and a handful of payload fields are small: at 256 a 50k-point
# document took ~200 round trips (measured: ids 2.5-3.5x and the metadata scan ~2x
# faster at 2048). Pages carrying whole payloads (chunk text) stay small.
_SCROLL_PAGE_SIZE = 2048
_FULL_PAYLOAD_PAGE_SIZE = 256
# Points fetched per requested ordinal by fetch_neighbors (see there).
_NEIGHBOR_COPIES_ALLOWED = 4


def _dense_search_params(settings: AppSettings) -> models.SearchParams | None:
    """With quantization on: search the int8 copy, then re-score the best
    ``oversampling`` times the limit against the full vectors."""

    if settings.qdrant_quantization != "scalar":
        return None
    return models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=True, oversampling=float(settings.qdrant_quantization_oversampling)
        )
    )


def _single_hash(hashes: set[str | None]) -> str | None:
    """The one hash every point agrees on; None when any lacks it or they differ (a
    partial re-ingest)."""

    return next(iter(hashes)) if len(hashes) == 1 else None


def _single_value(values: set[int | None]) -> int | None:
    return next(iter(values)) if len(values) == 1 else None


def _single_filename_filter(filename: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="filename", match=models.MatchValue(value=filename))]
    )


class VectorRepository:
    """Owns collection lifecycle, hybrid search, and writes against Qdrant."""

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def _call(self, settings: AppSettings, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a Qdrant client method, recovering once from a collection deleted underneath.

        Readiness is cached per process, so a collection dropped out of band (a reset,
        a manual delete) used to 500 every request until a restart. Now the cache entry
        is dropped, the collection recreated empty, and the call retried once.
        """

        try:
            return getattr(self._client, method)(*args, **kwargs)
        except (UnexpectedResponse, grpc.RpcError, ValueError) as exc:
            if not _is_missing_collection(exc):
                raise
            # The default collection is recreated; a workspace's is not (a deleted
            # workspace stays deleted: ensure_ready raises WorkspaceGoneError).
            logger.warning("Collection %r disappeared while in use.", settings.qdrant_collection)
            forget_collection(self._client, settings)
            self.ensure_ready(settings)
            return getattr(self._client, method)(*args, **kwargs)

    def ensure_ready(self, settings: AppSettings, *, strict: bool = True) -> None:
        """Create or validate the collection. Cheap after the first call (cached).

        ``strict=False`` (reads, tags, deletes) accepts a collection built for another
        embedding setup — see ``qdrant_schema.ensure_collection``.
        """

        ensure_collection(self._client, settings, strict=strict)

    def ensure_writable(self, settings: AppSettings, *, create_workspace: bool = False) -> None:
        """``ensure_ready`` for a write of new vectors: an empty collection built for
        another embedding setup is recreated for this one; a non-empty one refuses.
        ``create_workspace``: make a missing workspace collection (creating the
        workspace) instead of refusing it."""

        recreate_if_empty_and_mismatched(self._client, settings, create_workspace=create_workspace)
        ensure_collection(self._client, settings, create_workspace=create_workspace)

    def hybrid_search(
        self,
        settings: AppSettings,
        query_embedding: EmbeddedText,
        filenames: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        narrow: PassageFilter | None = None,
    ) -> list[models.ScoredPoint]:
        """Fused dense+sparse candidates: server-side RRF, falling back to manual fusion.

        ``filenames`` and ``tags``, when given, restrict both the dense and sparse legs
        before fusion (payload filters on the indexed ``filename`` / ``tags`` fields):
        a document must match one of the filenames *and* carry one of the tags.
        ``narrow`` restricts them further (``PassageFilter``).
        """

        self.ensure_ready(settings)
        query_filter = _scope_filter(filenames, tags, narrow)
        try:
            if self._client in _server_side_hybrid_unsupported:
                return self._manual_hybrid_query(settings, query_embedding, query_filter)
            try:
                return self._server_side_hybrid_query(settings, query_embedding, query_filter)
            except UnexpectedResponse as exc:
                if exc.status_code not in _RRF_UNSUPPORTED_STATUS_CODES:
                    # A real query error (auth, malformed request, server fault) —
                    # falling back to manual fusion would only produce a second,
                    # more confusing failure and hide the actual cause.
                    raise
                # Older Qdrant servers may not support server-side prefetch + RRF.
                # Remembered only once manual fusion works: a 400 for any other reason
                # (a vector of the wrong size) fails that too, and remembering it
                # switched this client off server-side hybrid until a restart.
                points = self._manual_hybrid_query(settings, query_embedding, query_filter)
                _server_side_hybrid_unsupported.add(self._client)
                return points
        except ResponseHandlingException as exc:
            logger.warning("Qdrant is unreachable.", exc_info=True)
            raise VectorStoreUnavailableError("Qdrant is unreachable.") from exc

    def upsert(self, settings: AppSettings, points: Sequence[models.PointStruct]) -> None:
        """Index points into the collection."""

        self.ensure_ready(settings)
        self._call(
            settings,
            "upsert",
            collection_name=settings.qdrant_collection,
            points=list(points),
            wait=True,
        )

    def _scroll_all(
        self,
        settings: AppSettings,
        *,
        with_payload: bool | list[str],
        scroll_filter: models.Filter | None = None,
    ) -> Iterator[models.Record]:
        """Yield every point matching ``scroll_filter``, paging through the collection."""

        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = self._call(
                settings,
                "scroll",
                collection_name=settings.qdrant_collection,
                scroll_filter=scroll_filter,
                with_payload=with_payload,
                with_vectors=False,
                limit=_FULL_PAYLOAD_PAGE_SIZE if with_payload is True else _SCROLL_PAGE_SIZE,
                offset=offset,
            )
            yield from points
            if offset is None:
                return

    def point_ids_for_filename(self, settings: AppSettings, filename: str) -> list[str]:
        """Return all point IDs currently indexed for a filename.

        Used to compute which points are stale *after* a re-ingest upsert, rather than
        deleting by filename before the upsert — see ``ingest_chunks`` for why ordering
        matters.
        """

        self.ensure_ready(settings, strict=False)
        return [
            str(point.id)
            for point in self._scroll_all(
                settings, scroll_filter=_single_filename_filter(filename), with_payload=False
            )
        ]

    def stored_vectors(
        self, settings: AppSettings, point_ids: Sequence[str]
    ) -> dict[str, tuple[list[float], models.SparseVector, str | None]]:
        """The dense and sparse vectors and the chunking fingerprint of each of
        ``point_ids`` that is stored, read in batches (ingestion reuses them)."""

        self.ensure_ready(settings)
        found: dict[str, tuple[list[float], models.SparseVector, str | None]] = {}
        dense_name = settings.qdrant_dense_vector_name
        sparse_name = settings.qdrant_sparse_vector_name
        ids = list(point_ids)
        for start in range(0, len(ids), _FULL_PAYLOAD_PAGE_SIZE):
            points = self._call(
                settings,
                "retrieve",
                collection_name=settings.qdrant_collection,
                ids=ids[start : start + _FULL_PAYLOAD_PAGE_SIZE],
                with_payload=[DocumentChunk.PAYLOAD_FINGERPRINT_KEY],
                with_vectors=[dense_name, sparse_name],
            )
            for point in points:
                vectors = point.vector if isinstance(point.vector, dict) else {}
                dense = vectors.get(dense_name)
                sparse = vectors.get(sparse_name)
                if not isinstance(dense, list) or not isinstance(sparse, models.SparseVector):
                    continue
                fingerprint = (point.payload or {}).get(DocumentChunk.PAYLOAD_FINGERPRINT_KEY)
                found[str(point.id)] = (
                    [float(value) for value in dense],
                    sparse,
                    fingerprint if isinstance(fingerprint, str) else None,
                )
        return found

    def chunks_for_filename(self, settings: AppSettings, filename: str) -> list[dict[str, object]]:
        """Return all payloads for a filename, ordered by ``chunk_ordinal``.

        Backs the document-content endpoint (reconstruct a document from its chunks for
        the source viewer). Points with no ``chunk_ordinal`` sort last, stably.
        """

        self.ensure_ready(settings, strict=False)
        payloads: list[dict[str, object]] = [
            point.payload
            for point in self._scroll_all(
                settings, scroll_filter=_single_filename_filter(filename), with_payload=True
            )
            if point.payload
        ]
        payloads.sort(key=_chunk_ordinal_sort_key)
        return payloads

    def chunk_count_for_filename(self, settings: AppSettings, filename: str) -> int:
        """How many points one document has (an exact count, no payloads read)."""

        self.ensure_ready(settings, strict=False)
        result = self._call(
            settings,
            "count",
            collection_name=settings.qdrant_collection,
            count_filter=_single_filename_filter(filename),
            exact=True,
        )
        return int(result.count)

    def chunk_ordinal_of(self, settings: AppSettings, filename: str, chunk_id: str) -> int | None:
        """The ordinal of one chunk of a document, or None when it doesn't exist."""

        self.ensure_ready(settings, strict=False)
        points, _ = self._call(
            settings,
            "scroll",
            collection_name=settings.qdrant_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="filename", match=models.MatchValue(value=filename)),
                    models.FieldCondition(key="chunk_id", match=models.MatchValue(value=chunk_id)),
                ]
            ),
            with_payload=["chunk_ordinal"],
            with_vectors=False,
            limit=1,
        )
        ordinal = points[0].payload.get("chunk_ordinal") if points and points[0].payload else None
        return ordinal if isinstance(ordinal, int) and not isinstance(ordinal, bool) else None

    def seed_passage(self, settings: AppSettings, chunk_id: str) -> SeedPassage | None:
        """One stored passage with its dense and sparse vectors (a "more like this"
        search starts from it), or None when no passage has that id."""

        self.ensure_ready(settings)
        dense_name = settings.qdrant_dense_vector_name
        sparse_name = settings.qdrant_sparse_vector_name
        points, _ = self._call(
            settings,
            "scroll",
            collection_name=settings.qdrant_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="chunk_id", match=models.MatchValue(value=chunk_id))
                ]
            ),
            with_payload=["filename", "section", "text"],
            with_vectors=[dense_name, sparse_name],
            limit=1,
        )
        if not points:
            return None
        payload = points[0].payload or {}
        vectors = points[0].vector if isinstance(points[0].vector, dict) else {}
        dense = vectors.get(dense_name)
        sparse = vectors.get(sparse_name)
        if not isinstance(dense, list) or not isinstance(sparse, models.SparseVector):
            return None
        return SeedPassage(
            chunk_id=chunk_id,
            filename=str(payload.get("filename") or ""),
            section=str(payload.get("section") or ""),
            text=str(payload.get("text") or ""),
            embedding=EmbeddedText(dense=[float(value) for value in dense], sparse=sparse),
        )

    def chunks_in_ordinal_range(
        self, settings: AppSettings, filename: str, start: int, end: int
    ) -> list[dict[str, object]]:
        """Payloads of chunks ``start``..``end`` (inclusive, 1-based ordinals), in order.

        One page of a document for the source viewer: a range filter on the indexed
        ``chunk_ordinal`` rather than the whole document scrolled and sliced.
        """

        self.ensure_ready(settings, strict=False)
        payloads = [
            point.payload
            for point in self._scroll_all(
                settings,
                with_payload=True,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename", match=models.MatchValue(value=filename)
                        ),
                        models.FieldCondition(
                            key="chunk_ordinal", range=models.Range(gte=start, lte=end)
                        ),
                    ]
                ),
            )
            if point.payload
        ]
        payloads.sort(key=_chunk_ordinal_sort_key)
        return payloads

    def filename_metadata(self, settings: AppSettings) -> dict[str, DocumentMetadata]:
        """Every document's metadata, served from a per-collection cache between changes.

        Reading it means scrolling every point's payload — the whole collection — and the
        UI asks after every upload, delete and page load. The result depends only on the
        corpus, so it is kept until the corpus generation moves (any ingest, delete or
        tag change in this process) or ``document_listing_ttl_seconds`` passes (another
        process's writes). When the generation moved for a few named documents, only
        those are re-read and patched in: a bulk import used to rescan the whole
        collection once per finished document.
        """

        collection = settings.qdrant_collection
        ttl = float(settings.document_listing_ttl_seconds)
        cached = self._cached_metadata(collection, corpus_generation(collection), ttl)
        if cached is not None:
            return dict(cached)
        with _metadata_lock:
            scan_lock = _metadata_scan_locks.setdefault(self._client, {}).setdefault(
                collection, Lock()
            )
        with scan_lock:
            # Whoever held the lock may have just stored a fresh listing.
            generation = corpus_generation(collection)
            cached = self._cached_metadata(collection, generation, ttl)
            if cached is not None:
                return dict(cached)
            patched = self._patched_metadata(settings, generation)
            if patched is not None:
                return dict(patched)
            # The generation is read before the scan, so a change landing during it
            # leaves the entry stale; the time is taken after, so a scan slower than
            # the TTL doesn't store an entry that has already expired.
            metadata = self._scan_filename_metadata(settings)
            with _metadata_lock:
                _metadata_cache.setdefault(self._client, {})[collection] = (
                    generation,
                    time.monotonic(),
                    metadata,
                )
        return dict(metadata)

    def _patched_metadata(
        self, settings: AppSettings, generation: int
    ) -> dict[str, DocumentMetadata] | None:
        """The cached listing with just the documents changed since it was read
        re-scanned, or None when that isn't possible (no fresh entry, too many or
        unknown changes). Caller holds the scan lock and read ``generation`` first."""

        collection = settings.qdrant_collection
        with _metadata_lock:
            cached = _metadata_cache.get(self._client, {}).get(collection)
        if cached is None:
            return None
        cached_generation, stored_at, metadata = cached
        if time.monotonic() - stored_at >= float(settings.document_listing_ttl_seconds):
            return None
        changed = changed_since(cached_generation, collection)
        if changed is None or len(changed) > _METADATA_PATCH_MAX_FILENAMES:
            return None
        patched = {name: meta for name, meta in metadata.items() if name not in changed}
        if changed:
            patched.update(self._scan_filename_metadata(settings, only=changed))
        with _metadata_lock:
            # The original time: the TTL still forces a full scan for writes made by
            # another process, which no generation here records.
            _metadata_cache.setdefault(self._client, {})[collection] = (
                generation,
                stored_at,
                patched,
            )
        return patched

    def _cached_metadata(
        self, collection: str, generation: int, ttl: float
    ) -> dict[str, DocumentMetadata] | None:
        with _metadata_lock:
            cached = _metadata_cache.get(self._client, {}).get(collection)
        if cached is None:
            return None
        cached_generation, stored_at, metadata = cached
        fresh = time.monotonic() - stored_at < ttl
        return metadata if cached_generation == generation and fresh else None

    def _scan_filename_metadata(
        self, settings: AppSettings, only: frozenset[str] | None = None
    ) -> dict[str, DocumentMetadata]:
        """Return each indexed filename's chunk count, page count, byte size, and
        upload time — one scroll, folded together rather than a separate query per
        field (which would turn a page-count add into an N+1 against the corpus list).

        ``page_count`` is ``max(page)`` seen across the filename's points — pages are
        1-based per api.documents' parsers, so this is a direct count, not an index.
        ``byte_size``/``uploaded_at`` come from whichever point has them (every chunk
        of one ingest carries the same values — see api.documents.DocumentChunk); a
        filename ingested before those fields existed has neither in any point's
        payload, so both stay ``None``.

        ``only`` limits the scroll to those filenames (a patch of a cached listing).
        """

        self.ensure_ready(settings, strict=False)
        folds: dict[str, _MetadataFold] = {}
        for point in self._scroll_all(
            settings,
            scroll_filter=(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename", match=models.MatchAny(any=sorted(only))
                        )
                    ]
                )
                if only is not None
                else None
            ),
            with_payload=[
                "filename",
                "page",
                "byte_size",
                "uploaded_at",
                "chunker_version",
                "tags",
                "chunking_fingerprint",
                "token_counter",
                "content_sha256",
                "document_chunks",
                "source_url",
                "source_etag",
                "source_last_modified",
                "source_refresh_minutes",
            ],
        ):
            if not point.payload:
                continue
            filename = point.payload.get("filename")
            if not isinstance(filename, str):
                continue
            folds.setdefault(filename, _MetadataFold()).add(point.payload)
        return {filename: fold.result() for filename, fold in folds.items()}

    def content_sha256_of(self, settings: AppSettings, filename: str) -> str | None:
        """The content hash one point of ``filename`` carries (one point read), or None
        when it isn't indexed or has none."""

        self.ensure_ready(settings, strict=False)
        points, _ = self._call(
            settings,
            "scroll",
            collection_name=settings.qdrant_collection,
            scroll_filter=_single_filename_filter(filename),
            with_payload=["content_sha256"],
            with_vectors=False,
            limit=1,
        )
        stored = points[0].payload.get("content_sha256") if points and points[0].payload else None
        return stored if isinstance(stored, str) else None

    def content_identity(
        self, settings: AppSettings, filename: str
    ) -> tuple[str | None, str | None] | None:
        """(content hash, source URL) one point of ``filename`` carries (one point read),
        or None when it isn't indexed: what a refresh checks before it writes."""

        self.ensure_ready(settings, strict=False)
        points, _ = self._call(
            settings,
            "scroll",
            collection_name=settings.qdrant_collection,
            scroll_filter=_single_filename_filter(filename),
            with_payload=["content_sha256", "source_url"],
            with_vectors=False,
            limit=1,
        )
        if not points:
            return None
        payload = points[0].payload or {}
        digest = payload.get("content_sha256")
        url = payload.get("source_url")
        return (
            digest if isinstance(digest, str) else None,
            url if isinstance(url, str) else None,
        )

    def document_metadata(self, settings: AppSettings, filename: str) -> DocumentMetadata | None:
        """One document's metadata, read now (not from the cached listing), or None
        when it isn't indexed. For decisions made under its filename lock."""

        return self._scan_filename_metadata(settings, only=frozenset({filename})).get(filename)

    def set_uploaded_at(self, settings: AppSettings, filename: str, uploaded_at: float) -> None:
        """Restamp every point of ``filename`` with ``uploaded_at``. Caller holds its
        filename lock."""

        self.ensure_ready(settings)
        self._call(
            settings,
            "set_payload",
            collection_name=settings.qdrant_collection,
            payload={"uploaded_at": uploaded_at},
            points=_single_filename_filter(filename),
            wait=True,
        )

    def set_source_details(
        self,
        settings: AppSettings,
        filename: str,
        etag: str | None,
        last_modified: str | None,
        refresh_minutes: int | None,
    ) -> None:
        """Replace the ETag, Last-Modified and refresh interval every point of
        ``filename`` carries (the same bytes refreshed under new validators, or its
        refresh setting changed): payload only, nothing re-embedded. A None value is
        removed. Caller holds its filename lock."""

        self.ensure_ready(settings)
        values: dict[str, object] = {
            "source_etag": etag,
            "source_last_modified": last_modified,
            "source_refresh_minutes": refresh_minutes,
        }
        kept = {key: value for key, value in values.items() if value is not None}
        dropped = [key for key, value in values.items() if value is None]
        if kept:
            self._call(
                settings,
                "set_payload",
                collection_name=settings.qdrant_collection,
                payload=kept,
                points=_single_filename_filter(filename),
                wait=True,
            )
        if dropped:
            self._call(
                settings,
                "delete_payload",
                collection_name=settings.qdrant_collection,
                keys=dropped,
                points=_single_filename_filter(filename),
                wait=True,
            )

    def uploaded_at_for_filename(self, settings: AppSettings, filename: str) -> float | None:
        """The stored upload time of an indexed document (one point read), or None."""

        self.ensure_ready(settings, strict=False)
        points, _ = self._call(
            settings,
            "scroll",
            collection_name=settings.qdrant_collection,
            scroll_filter=_single_filename_filter(filename),
            with_payload=["uploaded_at"],
            with_vectors=False,
            limit=1,
        )
        stamp = points[0].payload.get("uploaded_at") if points and points[0].payload else None
        return float(stamp) if isinstance(stamp, int | float) else None

    def stored_tags(self, settings: AppSettings) -> set[str]:
        """Every distinct tag in the collection, as stored.

        From the metadata cache when it is warm, else one facet call on the ``tags``
        keyword index. The scan behind ``filename_metadata`` reads every point's payload,
        and a tag-scoped question used to pay it, before embedding, whenever the cache
        had gone cold. A facet that comes back full may be truncated — a stored spelling
        missing from it would silently mis-scope the question — so it falls back to the
        scan then, and when the server can't facet at all.
        """

        cached = self._cached_metadata(
            settings.qdrant_collection,
            corpus_generation(settings.qdrant_collection),
            float(settings.document_listing_ttl_seconds),
        )
        if cached is not None:
            return {tag for metadata in cached.values() for tag in metadata.tags}
        try:
            response = self._call(
                settings,
                "facet",
                collection_name=settings.qdrant_collection,
                key="tags",
                limit=TAG_FACET_LIMIT,
                exact=True,
            )
        except (UnexpectedResponse, ResponseHandlingException, grpc.RpcError, ValueError):
            logger.debug("Tag facet unavailable; reading tags from metadata.", exc_info=True)
        else:
            if len(response.hits) < TAG_FACET_LIMIT:
                return {hit.value for hit in response.hits if isinstance(hit.value, str)}
        return {
            tag for metadata in self.filename_metadata(settings).values() for tag in metadata.tags
        }

    def tags_for_filename(self, settings: AppSettings, filename: str) -> tuple[str, ...]:
        """The tags stored on an indexed document (one point read), or ``()``."""

        self.ensure_ready(settings, strict=False)
        points, _ = self._call(
            settings,
            "scroll",
            collection_name=settings.qdrant_collection,
            scroll_filter=_single_filename_filter(filename),
            with_payload=["tags"],
            with_vectors=False,
            limit=1,
        )
        stored = points[0].payload.get("tags") if points and points[0].payload else None
        if not isinstance(stored, list):
            return ()
        return tuple(tag for tag in stored if isinstance(tag, str))

    def set_tags(self, settings: AppSettings, filename: str, tags: Sequence[str]) -> None:
        """Replace a document's tags on every one of its points. Payload-only: no re-embed."""

        self.ensure_ready(settings, strict=False)
        self._call(
            settings,
            "set_payload",
            collection_name=settings.qdrant_collection,
            payload={"tags": list(tags)},
            points=_single_filename_filter(filename),
            wait=True,
        )

    def delete_filename(self, settings: AppSettings, filename: str) -> int:
        """Remove every point of ``filename``; how many there were. Caller holds its
        filename lock, so nothing is written to it between the count and the delete.

        By filter, in one call: listing the ids first and deleting those took twice as
        long under the lock (measured: 0.48 s against 0.24 s at 50k points)."""

        self.ensure_ready(settings, strict=False)
        selector = _single_filename_filter(filename)
        count = self._call(
            settings,
            "count",
            collection_name=settings.qdrant_collection,
            count_filter=selector,
            exact=True,
        ).count
        if count:
            self._call(
                settings,
                "delete",
                collection_name=settings.qdrant_collection,
                points_selector=models.FilterSelector(filter=selector),
                wait=True,
            )
        return int(count)

    def delete_by_ids(self, settings: AppSettings, point_ids: Sequence[str]) -> None:
        """Remove specific points by id — a single call regardless of how many
        filenames those ids originally belonged to (no per-filename round-trips)."""

        if not point_ids:
            return
        self.ensure_ready(settings, strict=False)
        self._call(
            settings,
            "delete",
            collection_name=settings.qdrant_collection,
            points_selector=models.PointIdsList(points=list(point_ids)),
            wait=True,
        )

    def _server_side_hybrid_query(
        self,
        settings: AppSettings,
        query_embedding: EmbeddedText,
        query_filter: models.Filter | None,
    ) -> list[models.ScoredPoint]:
        response = self._call(
            settings,
            "query_points",
            collection_name=settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=query_embedding.dense,
                    using=settings.qdrant_dense_vector_name,
                    limit=int(settings.dense_retrieval_limit),
                    filter=query_filter,
                    params=_dense_search_params(settings),
                ),
                models.Prefetch(
                    query=query_embedding.sparse,
                    using=settings.qdrant_sparse_vector_name,
                    limit=int(settings.sparse_retrieval_limit),
                    filter=query_filter,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=int(settings.rrf_k))),
            limit=int(settings.fused_top_n),
            with_payload=SEARCH_PAYLOAD_FIELDS,
            with_vectors=False,
        )
        return _response_points(response)

    def search_leg(
        self,
        settings: AppSettings,
        query_embedding: EmbeddedText,
        leg: Literal["dense", "sparse"],
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        """One side of the hybrid search on its own — for fusion, and for evaluation."""

        request = _leg_request(settings, query_embedding, leg, query_filter)
        response = self._call(
            settings,
            "query_points",
            collection_name=settings.qdrant_collection,
            query=request.query,
            using=request.using,
            limit=request.limit,
            query_filter=request.filter,
            search_params=request.params,
            with_payload=request.with_payload,
            with_vectors=False,
        )
        return _response_points(response)

    def _manual_hybrid_query(
        self,
        settings: AppSettings,
        query_embedding: EmbeddedText,
        query_filter: models.Filter | None,
    ) -> list[models.ScoredPoint]:
        # Both legs in one round trip; fused here with the same pinned rrf_k.
        responses = self._call(
            settings,
            "query_batch_points",
            collection_name=settings.qdrant_collection,
            requests=[
                _leg_request(settings, query_embedding, "dense", query_filter),
                _leg_request(settings, query_embedding, "sparse", query_filter),
            ],
        )
        return reciprocal_rank_fusion(
            [_response_points(response) for response in responses],
            k=int(settings.rrf_k),
            limit=int(settings.fused_top_n),
        )

    def fetch_neighbors(
        self,
        settings: AppSettings,
        filename: str,
        ordinals: Sequence[int],
    ) -> list[models.ScoredPoint]:
        """Fetch chunks for a filename at specific ordinals (small-to-big expansion)."""

        if not ordinals:
            return []
        self.ensure_ready(settings)
        points, _ = self._call(
            settings,
            "scroll",
            collection_name=settings.qdrant_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="filename", match=models.MatchValue(value=filename)),
                    models.FieldCondition(
                        key="chunk_ordinal", match=models.MatchAny(any=list(ordinals))
                    ),
                ]
            ),
            with_payload=NEIGHBOR_PAYLOAD_FIELDS,
            with_vectors=False,
            # Room for duplicates: an ingest that failed partway leaves the previous
            # version's points next to the new ones, sharing ordinals, and an exact limit
            # let those copies crowd out the ordinals that were asked for.
            limit=len(ordinals) * _NEIGHBOR_COPIES_ALLOWED,
        )
        return [
            models.ScoredPoint(id=point.id, version=0, score=0.0, payload=point.payload)
            for point in points
        ]


def _is_missing_collection(exc: BaseException) -> bool:
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code == 404 and b"doesn't exist" in (exc.content or b"")
    if isinstance(exc, grpc.RpcError):
        code = getattr(exc, "code", None)
        return callable(code) and code() == grpc.StatusCode.NOT_FOUND
    # The in-process (":memory:") client says it this way.
    message = str(exc)
    return (
        isinstance(exc, ValueError) and message.startswith("Collection ") and "not found" in message
    )


@dataclass(frozen=True)
class PassageFilter:
    """Narrows a search (POST /search) or a question to passages by where they are: a
    page range, one section (its exact label), documents left out, or picked passages.
    Pages, sections and chunk ids are indexed payload fields, so these are plain Qdrant
    filters."""

    page_min: int | None = None
    page_max: int | None = None
    section: str | None = None
    exclude_filenames: tuple[str, ...] = ()
    # Only these passages (chunk ids): a question about passages picked in the Search
    # panel. Still searched and reranked like any other scope.
    chunk_ids: tuple[str, ...] = ()
    # Passages left out (an answer asked again without one of its sources).
    exclude_chunk_ids: tuple[str, ...] = ()
    # Documents uploaded in this range (Unix seconds).
    uploaded_after: float | None = None
    uploaded_before: float | None = None

    def admits_neighbour(self, payload: Mapping[str, Any]) -> bool:
        """Whether a neighbouring passage of a kept one (same document, so the same
        filename, tags and upload time) is inside this narrowing: neighbour text is
        prompt context, and a page or section narrowing that let the next page in let
        the model cite page 5 for a fact from page 6."""

        if payload.get("chunk_id") in self.exclude_chunk_ids:
            return False
        if self.section is not None and payload.get("section") != self.section:
            return False
        page = payload.get("page")
        if isinstance(page, int) and not isinstance(page, bool):
            end = payload.get("page_end")
            last = end if isinstance(end, int) and not isinstance(end, bool) else page
            if self.page_min is not None and max(page, last) < self.page_min:
                return False
            if self.page_max is not None and page > self.page_max:
                return False
        return True


def _scope_filter(
    filenames: Sequence[str] | None,
    tags: Sequence[str] | None = None,
    narrow: PassageFilter | None = None,
) -> models.Filter | None:
    """Build a payload filter restricting search to filenames and/or tags, or None."""

    must: list[models.Condition] = []
    must_not: list[models.Condition] = []
    if filenames:
        must.append(
            models.FieldCondition(key="filename", match=models.MatchAny(any=list(filenames)))
        )
    if tags:
        must.append(models.FieldCondition(key="tags", match=models.MatchAny(any=list(tags))))
    if narrow is not None:
        if narrow.page_max is not None:
            must.append(models.FieldCondition(key="page", range=models.Range(lte=narrow.page_max)))
        if narrow.page_min is not None:
            # Overlapping the range: starting in it, or starting before and running onto
            # it. page_end is stored only on passages that cross a page break.
            must.append(
                models.Filter(
                    should=[
                        models.FieldCondition(key="page", range=models.Range(gte=narrow.page_min)),
                        models.FieldCondition(
                            key="page_end", range=models.Range(gte=narrow.page_min)
                        ),
                    ]
                )
            )
        if narrow.uploaded_after is not None or narrow.uploaded_before is not None:
            must.append(
                models.FieldCondition(
                    key="uploaded_at",
                    range=models.Range(gte=narrow.uploaded_after, lte=narrow.uploaded_before),
                )
            )
        if narrow.section:
            must.append(
                models.FieldCondition(key="section", match=models.MatchValue(value=narrow.section))
            )
        if narrow.chunk_ids:
            must.append(
                models.FieldCondition(
                    key="chunk_id", match=models.MatchAny(any=list(narrow.chunk_ids))
                )
            )
        if narrow.exclude_filenames:
            must_not.append(
                models.FieldCondition(
                    key="filename", match=models.MatchAny(any=list(narrow.exclude_filenames))
                )
            )
        if narrow.exclude_chunk_ids:
            must_not.append(
                models.FieldCondition(
                    key="chunk_id", match=models.MatchAny(any=list(narrow.exclude_chunk_ids))
                )
            )
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def _chunk_ordinal_sort_key(payload: dict[str, object]) -> tuple[int, int]:
    """Sort by chunk_ordinal; payloads without an int ordinal sort last, stably."""

    ordinal = payload.get("chunk_ordinal")
    if isinstance(ordinal, int) and not isinstance(ordinal, bool):
        return (0, ordinal)
    return (1, 0)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[models.ScoredPoint]],
    *,
    k: int,
    limit: int,
) -> list[models.ScoredPoint]:
    """Fuse ranked lists with Qdrant-compatible RRF scoring."""

    scores: dict[str, float] = {}
    best_points: dict[str, models.ScoredPoint] = {}
    best_ranks: dict[str, int] = {}

    for ranked_list in ranked_lists:
        for rank, point in enumerate(ranked_list):
            point_id = str(point.id)
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank)
            best_points.setdefault(point_id, point)
            best_ranks[point_id] = min(best_ranks.get(point_id, rank), rank)

    ordered_ids = sorted(
        scores,
        key=lambda point_id: (-scores[point_id], best_ranks[point_id], point_id),
    )

    fused: list[models.ScoredPoint] = []
    for point_id in ordered_ids[:limit]:
        point = best_points[point_id]
        fused.append(point.model_copy(update={"score": scores[point_id]}))
    return fused


def _response_points(response: QueryResponse) -> list[models.ScoredPoint]:
    return list(response.points)


def _leg_request(
    settings: AppSettings,
    query_embedding: EmbeddedText,
    leg: Literal["dense", "sparse"],
    query_filter: models.Filter | None,
) -> models.QueryRequest:
    """One retrieval leg (dense or BM25) as a query request."""

    if leg == "dense":
        return models.QueryRequest(
            query=query_embedding.dense,
            using=settings.qdrant_dense_vector_name,
            limit=int(settings.dense_retrieval_limit),
            filter=query_filter,
            params=_dense_search_params(settings),
            with_payload=SEARCH_PAYLOAD_FIELDS,
            with_vector=False,
        )
    return models.QueryRequest(
        query=query_embedding.sparse,
        using=settings.qdrant_sparse_vector_name,
        limit=int(settings.sparse_retrieval_limit),
        filter=query_filter,
        with_payload=SEARCH_PAYLOAD_FIELDS,
        with_vector=False,
    )
