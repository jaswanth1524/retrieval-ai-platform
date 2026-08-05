"""Vector-store repository — the only module that talks to Qdrant for search/writes.

Retrieval and ingestion depend on the ``SearchRepository``/``WriteRepository``
protocols they declare locally; ``VectorRepository`` satisfies both by structural
typing, so callers never import ``qdrant_client`` directly.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

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
            try:
                return self._server_side_hybrid_query(settings, query_embedding, query_filter)
            except UnexpectedResponse as exc:
                if exc.status_code not in _RRF_UNSUPPORTED_STATUS_CODES:
                    # A real query error (auth, malformed request, server fault) —
                    # falling back to manual fusion would only produce a second,
                    # more confusing failure and hide the actual cause.
                    raise
                # Older Qdrant servers may not support server-side prefetch + RRF.
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

    def point_ids_for_filename(self, settings: AppSettings, filename: str) -> list[str]:
        """Return all point IDs currently indexed for a filename.

        Used to compute which points are stale *after* a re-ingest upsert, rather than
        deleting by filename before the upsert — see ``ingest_chunks`` for why ordering
        matters.
        """

        self.ensure_ready(settings)
        ids: list[str] = []
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = self._client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename),
                        )
                    ]
                ),
                with_payload=False,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            ids.extend(str(point.id) for point in points)
            if offset is None:
                break
        return ids

    def chunks_for_filename(
        self, settings: AppSettings, filename: str
    ) -> list[dict[str, object]]:
        """Return all payloads for a filename, ordered by ``chunk_ordinal``.

        Backs the document-content endpoint (reconstruct a document from its chunks for
        the source viewer). Points with no ``chunk_ordinal`` sort last, stably.
        """

        self.ensure_ready(settings)
        payloads: list[dict[str, object]] = []
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = self._client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename),
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            payloads.extend(point.payload for point in points if point.payload)
            if offset is None:
                break
        payloads.sort(key=_chunk_ordinal_sort_key)
        return payloads

    def filename_chunk_counts(self, settings: AppSettings) -> dict[str, int]:
        """Return each indexed filename's chunk count, for the corpus panel and its footer total."""

        self.ensure_ready(settings)
        counts: Counter[str] = Counter()
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = self._client.scroll(
                collection_name=settings.qdrant_collection,
                with_payload=["filename"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in points:
                if point.payload and isinstance(point.payload.get("filename"), str):
                    counts[point.payload["filename"]] += 1
            if offset is None:
                break
        return dict(counts)

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
