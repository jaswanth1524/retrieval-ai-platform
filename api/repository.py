"""Vector-store repository — the only module that talks to Qdrant for search/writes.

Retrieval and ingestion depend on the ``SearchRepository``/``WriteRepository``
protocols they declare locally; ``VectorRepository`` satisfies both by structural
typing, so callers never import ``qdrant_client`` directly.
"""

from __future__ import annotations

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
    ) -> list[models.ScoredPoint]:
        """Fused dense+sparse candidates: server-side RRF, falling back to manual fusion."""

        self.ensure_ready(settings)
        try:
            try:
                return self._server_side_hybrid_query(settings, query_embedding)
            except UnexpectedResponse as exc:
                if exc.status_code not in _RRF_UNSUPPORTED_STATUS_CODES:
                    # A real query error (auth, malformed request, server fault) —
                    # falling back to manual fusion would only produce a second,
                    # more confusing failure and hide the actual cause.
                    raise
                # Older Qdrant servers may not support server-side prefetch + RRF.
                return self._manual_hybrid_query(settings, query_embedding)
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

    def delete_by_filename(self, settings: AppSettings, filenames: Sequence[str]) -> None:
        """Remove any prior points for these filenames so re-ingest leaves no orphans."""

        self.ensure_ready(settings)
        for filename in sorted(set(filenames)):
            self._client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="filename",
                                match=models.MatchValue(value=filename),
                            )
                        ]
                    )
                ),
                wait=True,
            )

    def _server_side_hybrid_query(
        self,
        settings: AppSettings,
        query_embedding: EmbeddedText,
    ) -> list[models.ScoredPoint]:
        response = self._client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=query_embedding.dense,
                    using=settings.qdrant_dense_vector_name,
                    limit=int(settings.dense_retrieval_limit),
                ),
                models.Prefetch(
                    query=query_embedding.sparse,
                    using=settings.qdrant_sparse_vector_name,
                    limit=int(settings.sparse_retrieval_limit),
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
    ) -> list[models.ScoredPoint]:
        dense_response = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_embedding.dense,
            using=settings.qdrant_dense_vector_name,
            limit=int(settings.dense_retrieval_limit),
            with_payload=True,
            with_vectors=False,
        )
        sparse_response = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_embedding.sparse,
            using=settings.qdrant_sparse_vector_name,
            limit=int(settings.sparse_retrieval_limit),
            with_payload=True,
            with_vectors=False,
        )
        return reciprocal_rank_fusion(
            [_response_points(dense_response), _response_points(sparse_response)],
            k=int(settings.rrf_k),
            limit=int(settings.fused_top_n),
        )


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
