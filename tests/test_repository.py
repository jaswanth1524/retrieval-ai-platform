from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient, models

from api.repository import VectorRepository, reciprocal_rank_fusion
from api.settings import AppSettings


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "repository_documents",
        "qdrant_dense_vector_name": "dense",
        "qdrant_sparse_vector_name": "sparse",
        "qdrant_dense_vector_size": 3,
        "embedding_model_tag": "test-embedding:v1",
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


def scored_point(chunk_id: str, score: float) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=str(uuid5(NAMESPACE_URL, chunk_id)),
        version=0,
        score=score,
        payload={
            "filename": "guide.md",
            "page": 1,
            "section": "Setup",
            "chunk_id": chunk_id,
            "text": f"text {chunk_id}",
        },
    )


def make_point(point_id: str, filename: str) -> models.PointStruct:
    return models.PointStruct(
        id=str(uuid5(NAMESPACE_URL, point_id)),
        vector={
            "dense": [0.1, 0.2, 0.3],
            "sparse": models.SparseVector(indices=[1], values=[0.5]),
        },
        payload={
            "filename": filename,
            "page": 1,
            "section": "Intro",
            "chunk_id": point_id,
            "text": "hello",
        },
    )


def test_reciprocal_rank_fusion_combines_duplicate_points() -> None:
    point_a = scored_point("a", 0.9)
    point_b = scored_point("b", 0.8)
    point_c = scored_point("c", 0.7)

    fused = reciprocal_rank_fusion(
        [[point_a, point_b], [point_b, point_c]],
        k=60,
        limit=3,
    )

    assert [str(point.id) for point in fused] == [str(point_b.id), str(point_a.id), str(point_c.id)]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 60)


def test_vector_repository_upsert_indexes_and_ensures_collection() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)

    repository.upsert(settings, [make_point("p1", "guide.md")])

    assert client.collection_exists(settings.qdrant_collection)
    records, _ = client.scroll(
        collection_name=settings.qdrant_collection, with_payload=True, with_vectors=False
    )
    assert len(records) == 1
    assert records[0].payload is not None
    assert records[0].payload["filename"] == "guide.md"


def test_vector_repository_point_ids_for_filename_returns_only_that_files_ids() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    guide_point = make_point("p1", "guide.md")
    repository.upsert(settings, [guide_point, make_point("p2", "other.md")])

    ids = repository.point_ids_for_filename(settings, "guide.md")

    assert ids == [str(guide_point.id)]


def test_vector_repository_point_ids_for_filename_empty_for_unknown_file() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(settings, [make_point("p1", "guide.md")])

    assert repository.point_ids_for_filename(settings, "missing.md") == []


def test_vector_repository_delete_by_ids_removes_only_those_points() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    guide_point = make_point("p1", "guide.md")
    other_point = make_point("p2", "other.md")
    repository.upsert(settings, [guide_point, other_point])

    repository.delete_by_ids(settings, [str(guide_point.id)])

    records, _ = client.scroll(
        collection_name=settings.qdrant_collection, with_payload=True, with_vectors=False
    )
    assert [record.payload["filename"] for record in records if record.payload] == ["other.md"]


def test_vector_repository_delete_by_ids_is_a_noop_for_empty_list() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(settings, [make_point("p1", "guide.md")])

    repository.delete_by_ids(settings, [])

    records, _ = client.scroll(
        collection_name=settings.qdrant_collection, with_payload=True, with_vectors=False
    )
    assert len(records) == 1
