"""Vector-store repository — the only module that talks to Qdrant for search/writes.

Ingestion depends on the ``WriteRepository`` protocol it declares locally, which
``VectorRepository`` satisfies by structural typing, so callers never import
``qdrant_client`` directly.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal
from weakref import WeakKeyDictionary, WeakSet

import grpc
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models.models import QueryResponse

from api.corpus import corpus_generation
from api.documents import CHUNKER_VERSION
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
_METADATA_TTL_SECONDS = 30.0
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


def clear_hybrid_fallback_cache() -> None:
    """Forget which clients needed the manual-fusion fallback (tests, reloads)."""

    _server_side_hybrid_unsupported.clear()


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


_SCROLL_PAGE_SIZE = 256


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
            logger.warning(
                "Collection %r disappeared while in use; recreating it.",
                settings.qdrant_collection,
            )
            forget_collection(self._client, settings)
            self.ensure_ready(settings)
            return getattr(self._client, method)(*args, **kwargs)

    def ensure_ready(self, settings: AppSettings, *, strict: bool = True) -> None:
        """Create or validate the collection. Cheap after the first call (cached).

        ``strict=False`` (reads, tags, deletes) accepts a collection built for another
        embedding setup — see ``qdrant_schema.ensure_collection``.
        """

        ensure_collection(self._client, settings, strict=strict)

    def ensure_writable(self, settings: AppSettings) -> None:
        """``ensure_ready`` for a write of new vectors: an empty collection built for
        another embedding setup is recreated for this one; a non-empty one refuses."""

        recreate_if_empty_and_mismatched(self._client, settings)
        ensure_collection(self._client, settings)

    def hybrid_search(
        self,
        settings: AppSettings,
        query_embedding: EmbeddedText,
        filenames: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> list[models.ScoredPoint]:
        """Fused dense+sparse candidates: server-side RRF, falling back to manual fusion.

        ``filenames`` and ``tags``, when given, restrict both the dense and sparse legs
        before fusion (payload filters on the indexed ``filename`` / ``tags`` fields):
        a document must match one of the filenames *and* carry one of the tags.
        """

        self.ensure_ready(settings)
        query_filter = _scope_filter(filenames, tags)
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
                limit=_SCROLL_PAGE_SIZE,
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
        tag change in this process) or ``_METADATA_TTL_SECONDS`` passes (another
        process's writes).
        """

        collection = settings.qdrant_collection
        cached = self._cached_metadata(collection, corpus_generation())
        if cached is not None:
            return dict(cached)
        with _metadata_lock:
            scan_lock = _metadata_scan_locks.setdefault(self._client, {}).setdefault(
                collection, Lock()
            )
        with scan_lock:
            # Whoever held the lock may have just stored a fresh listing.
            generation = corpus_generation()
            cached = self._cached_metadata(collection, generation)
            if cached is not None:
                return dict(cached)
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

    def _cached_metadata(
        self, collection: str, generation: int
    ) -> dict[str, DocumentMetadata] | None:
        with _metadata_lock:
            cached = _metadata_cache.get(self._client, {}).get(collection)
        if cached is None:
            return None
        cached_generation, stored_at, metadata = cached
        fresh = time.monotonic() - stored_at < _METADATA_TTL_SECONDS
        return metadata if cached_generation == generation and fresh else None

    def _scan_filename_metadata(self, settings: AppSettings) -> dict[str, DocumentMetadata]:
        """Return each indexed filename's chunk count, page count, byte size, and
        upload time — one scroll, folded together rather than a separate query per
        field (which would turn a page-count add into an N+1 against the corpus list).

        ``page_count`` is ``max(page)`` seen across the filename's points — pages are
        1-based per api.documents' parsers, so this is a direct count, not an index.
        ``byte_size``/``uploaded_at`` come from whichever point has them (every chunk
        of one ingest carries the same values — see api.documents.DocumentChunk); a
        filename ingested before those fields existed has neither in any point's
        payload, so both stay ``None``.
        """

        self.ensure_ready(settings, strict=False)
        chunk_counts: Counter[str] = Counter()
        max_page: dict[str, int] = {}
        byte_size: dict[str, int] = {}
        uploaded_at: dict[str, float] = {}
        min_version: dict[str, int] = {}
        unversioned: set[str] = set()
        tags: dict[str, dict[str, None]] = {}
        fingerprints: dict[str, set[str]] = {}
        heuristic: set[str] = set()
        for point in self._scroll_all(
            settings,
            with_payload=[
                "filename",
                "page",
                "byte_size",
                "uploaded_at",
                "chunker_version",
                "tags",
                "chunking_fingerprint",
                "token_counter",
            ],
        ):
            if not point.payload:
                continue
            filename = point.payload.get("filename")
            if not isinstance(filename, str):
                continue
            chunk_counts[filename] += 1
            page = point.payload.get("page")
            if isinstance(page, int):
                max_page[filename] = max(max_page.get(filename, 0), page)
            size = point.payload.get("byte_size")
            if isinstance(size, int) and filename not in byte_size:
                byte_size[filename] = size
            stamp = point.payload.get("uploaded_at")
            if isinstance(stamp, int | float) and filename not in uploaded_at:
                uploaded_at[filename] = float(stamp)
            version = point.payload.get("chunker_version")
            if isinstance(version, int) and not isinstance(version, bool):
                min_version[filename] = min(min_version.get(filename, version), version)
            else:
                unversioned.add(filename)
            point_tags = point.payload.get("tags")
            if isinstance(point_tags, list):
                ordered = tags.setdefault(filename, {})
                ordered.update((tag, None) for tag in point_tags if isinstance(tag, str))
            fingerprint = point.payload.get("chunking_fingerprint")
            if isinstance(fingerprint, str):
                fingerprints.setdefault(filename, set()).add(fingerprint)
            if point.payload.get("token_counter") == "heuristic":
                heuristic.add(filename)
        return {
            filename: DocumentMetadata(
                chunk_count=count,
                page_count=max_page.get(filename, 0),
                byte_size=byte_size.get(filename),
                uploaded_at=uploaded_at.get(filename),
                chunker_version=None if filename in unversioned else min_version.get(filename),
                tags=tuple(tags.get(filename, {})),
                chunking_fingerprints=frozenset(fingerprints.get(filename, ())),
                heuristic_chunked=filename in heuristic,
            )
            for filename, count in chunk_counts.items()
        }

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
            with_payload=True,
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

        if leg == "dense":
            response = self._call(
                settings,
                "query_points",
                collection_name=settings.qdrant_collection,
                query=query_embedding.dense,
                using=settings.qdrant_dense_vector_name,
                limit=int(settings.dense_retrieval_limit),
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
        else:
            response = self._call(
                settings,
                "query_points",
                collection_name=settings.qdrant_collection,
                query=query_embedding.sparse,
                using=settings.qdrant_sparse_vector_name,
                limit=int(settings.sparse_retrieval_limit),
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
        return _response_points(response)

    def _manual_hybrid_query(
        self,
        settings: AppSettings,
        query_embedding: EmbeddedText,
        query_filter: models.Filter | None,
    ) -> list[models.ScoredPoint]:
        return reciprocal_rank_fusion(
            [
                self.search_leg(settings, query_embedding, "dense", query_filter),
                self.search_leg(settings, query_embedding, "sparse", query_filter),
            ],
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
            with_payload=True,
            with_vectors=False,
            limit=len(ordinals),
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


def _scope_filter(
    filenames: Sequence[str] | None, tags: Sequence[str] | None = None
) -> models.Filter | None:
    """Build a payload filter restricting search to filenames and/or tags, or None."""

    must: list[models.Condition] = []
    if filenames:
        must.append(
            models.FieldCondition(key="filename", match=models.MatchAny(any=list(filenames)))
        )
    if tags:
        must.append(models.FieldCondition(key="tags", match=models.MatchAny(any=list(tags))))
    return models.Filter(must=must) if must else None


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
