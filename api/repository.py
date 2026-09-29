"""Vector-store repository — the only module that talks to Qdrant for search/writes.

Retrieval and ingestion depend on the ``SearchRepository``/``WriteRepository``
protocols they declare locally; ``VectorRepository`` satisfies both by structural
typing, so callers never import ``qdrant_client`` directly.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from weakref import WeakSet

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models.models import QueryResponse

from api.embeddings import EmbeddedText
from api.qdrant_schema import VectorStoreUnavailableError, ensure_collection
from api.settings import AppSettings

# Status codes from a server-side RRF query that indicate "this Qdrant server/client
# combination doesn't support prefetch+RRF" rather than a real query failure — only
# these should trigger the manual-fusion fallback; anything else is a real error.
_RRF_UNSUPPORTED_STATUS_CODES = frozenset({400, 404, 501})

# Clients whose server has already rejected the server-side hybrid query. Remembered
# per process so every later question goes straight to manual fusion instead of paying
# a failed round trip first; keyed weakly by client so a new client re-probes.
_server_side_hybrid_unsupported: WeakSet[QdrantClient] = WeakSet()


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


_SCROLL_PAGE_SIZE = 256


def _single_filename_filter(filename: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="filename", match=models.MatchValue(value=filename))]
    )


class VectorRepository:
    """Owns collection lifecycle, hybrid search, and writes against Qdrant."""

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def ensure_ready(self, settings: AppSettings) -> None:
        """Create or validate the collection. Cheap after the first call (cached)."""

        ensure_collection(self._client, settings)

    def hybrid_search(
        self,
        settings: AppSettings,
        query_embedding: EmbeddedText,
        filenames: Sequence[str] | None = None,
    ) -> list[models.ScoredPoint]:
        """Fused dense+sparse candidates: server-side RRF, falling back to manual fusion.

        ``filenames``, when given, restricts both the dense and sparse legs to those
        documents before fusion (a payload filter on the indexed ``filename`` field).
        """

        self.ensure_ready(settings)
        query_filter = _filename_filter(filenames)
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
                _server_side_hybrid_unsupported.add(self._client)
                return self._manual_hybrid_query(settings, query_embedding, query_filter)
        except ResponseHandlingException as exc:
            raise VectorStoreUnavailableError(f"Qdrant is unreachable: {exc}") from exc

    def upsert(self, settings: AppSettings, points: Sequence[models.PointStruct]) -> None:
        """Index points into the collection."""

        self.ensure_ready(settings)
        self._client.upsert(
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
            points, offset = self._client.scroll(
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

        self.ensure_ready(settings)
        return [
            str(point.id)
            for point in self._scroll_all(
                settings, scroll_filter=_single_filename_filter(filename), with_payload=False
            )
        ]

    def chunks_for_filename(
        self, settings: AppSettings, filename: str
    ) -> list[dict[str, object]]:
        """Return all payloads for a filename, ordered by ``chunk_ordinal``.

        Backs the document-content endpoint (reconstruct a document from its chunks for
        the source viewer). Points with no ``chunk_ordinal`` sort last, stably.
        """

        self.ensure_ready(settings)
        payloads: list[dict[str, object]] = [
            point.payload
            for point in self._scroll_all(
                settings, scroll_filter=_single_filename_filter(filename), with_payload=True
            )
            if point.payload
        ]
        payloads.sort(key=_chunk_ordinal_sort_key)
        return payloads

    def filename_chunk_counts(self, settings: AppSettings) -> dict[str, int]:
        """Return each indexed filename's chunk count, for the corpus panel and its footer total."""

        self.ensure_ready(settings)
        counts: Counter[str] = Counter()
        for point in self._scroll_all(settings, with_payload=["filename"]):
            if point.payload and isinstance(point.payload.get("filename"), str):
                counts[point.payload["filename"]] += 1
        return dict(counts)

    def filename_metadata(self, settings: AppSettings) -> dict[str, DocumentMetadata]:
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

        self.ensure_ready(settings)
        chunk_counts: Counter[str] = Counter()
        max_page: dict[str, int] = {}
        byte_size: dict[str, int] = {}
        uploaded_at: dict[str, float] = {}
        min_version: dict[str, int] = {}
        unversioned: set[str] = set()
        for point in self._scroll_all(
            settings,
            with_payload=["filename", "page", "byte_size", "uploaded_at", "chunker_version"],
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
        return {
            filename: DocumentMetadata(
                chunk_count=count,
                page_count=max_page.get(filename, 0),
                byte_size=byte_size.get(filename),
                uploaded_at=uploaded_at.get(filename),
                chunker_version=None if filename in unversioned else min_version.get(filename),
            )
            for filename, count in chunk_counts.items()
        }

    def uploaded_at_for_filename(self, settings: AppSettings, filename: str) -> float | None:
        """The stored upload time of an indexed document (one point read), or None."""

        self.ensure_ready(settings)
        points, _ = self._client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=_single_filename_filter(filename),
            with_payload=["uploaded_at"],
            with_vectors=False,
            limit=1,
        )
        stamp = points[0].payload.get("uploaded_at") if points and points[0].payload else None
        return float(stamp) if isinstance(stamp, int | float) else None

    def list_filenames(self, settings: AppSettings) -> list[str]:
        """Return the distinct filenames currently indexed, for per-document filtering."""

        return sorted(self.filename_chunk_counts(settings))

    def delete_by_ids(self, settings: AppSettings, point_ids: Sequence[str]) -> None:
        """Remove specific points by id — a single call regardless of how many
        filenames those ids originally belonged to (no per-filename round-trips)."""

        if not point_ids:
            return
        self.ensure_ready(settings)
        self._client.delete(
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
        response = self._client.query_points(
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

    def _manual_hybrid_query(
        self,
        settings: AppSettings,
        query_embedding: EmbeddedText,
        query_filter: models.Filter | None,
    ) -> list[models.ScoredPoint]:
        dense_response = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_embedding.dense,
            using=settings.qdrant_dense_vector_name,
            limit=int(settings.dense_retrieval_limit),
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
        sparse_response = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_embedding.sparse,
            using=settings.qdrant_sparse_vector_name,
            limit=int(settings.sparse_retrieval_limit),
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
        return reciprocal_rank_fusion(
            [_response_points(dense_response), _response_points(sparse_response)],
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
        points, _ = self._client.scroll(
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


def _filename_filter(filenames: Sequence[str] | None) -> models.Filter | None:
    """Build a payload filter restricting search to specific filenames, or None."""

    if not filenames:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="filename", match=models.MatchAny(any=list(filenames)))]
    )


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
