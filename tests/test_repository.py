from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient, models

from api.repository import DocumentMetadata, VectorRepository, reciprocal_rank_fusion
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


def make_point(
    point_id: str,
    filename: str,
    page: int = 1,
    byte_size: int | None = None,
    uploaded_at: float | None = None,
) -> models.PointStruct:
    payload: dict[str, str | int | float] = {
        "filename": filename,
        "page": page,
        "section": "Intro",
        "chunk_id": point_id,
        "text": "hello",
    }
    if byte_size is not None:
        payload["byte_size"] = byte_size
    if uploaded_at is not None:
        payload["uploaded_at"] = uploaded_at
    return models.PointStruct(
        id=str(uuid5(NAMESPACE_URL, point_id)),
        vector={
            "dense": [0.1, 0.2, 0.3],
            "sparse": models.SparseVector(indices=[1], values=[0.5]),
        },
        payload=payload,
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


def test_vector_repository_list_filenames_returns_distinct_sorted_names() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(
        settings,
        [make_point("p1", "b.md"), make_point("p2", "a.md"), make_point("p3", "b.md")],
    )

    assert repository.list_filenames(settings) == ["a.md", "b.md"]


def test_vector_repository_list_filenames_empty_for_empty_collection() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.ensure_ready(settings)

    assert repository.list_filenames(settings) == []


def test_vector_repository_filename_chunk_counts_counts_points_per_file() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(
        settings,
        [make_point("p1", "b.md"), make_point("p2", "a.md"), make_point("p3", "b.md")],
    )

    assert repository.filename_chunk_counts(settings) == {"a.md": 1, "b.md": 2}


def test_vector_repository_filename_chunk_counts_empty_for_empty_collection() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.ensure_ready(settings)

    assert repository.filename_chunk_counts(settings) == {}


def test_vector_repository_filename_metadata_aggregates_page_count_and_stamped_fields() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(
        settings,
        [
            make_point("p1", "guide.md", page=1, byte_size=2048, uploaded_at=1700000000.0),
            make_point("p2", "guide.md", page=3, byte_size=2048, uploaded_at=1700000000.0),
            make_point("p3", "guide.md", page=2, byte_size=2048, uploaded_at=1700000000.0),
        ],
    )

    metadata = repository.filename_metadata(settings)

    assert metadata == {
        "guide.md": DocumentMetadata(
            chunk_count=3, page_count=3, byte_size=2048, uploaded_at=1700000000.0
        )
    }


def test_vector_repository_filename_metadata_nulls_stamped_fields_for_legacy_points() -> None:
    """A point ingested before byte_size/uploaded_at existed has neither key in its
    payload at all — this must not crash, and both fields must come back None rather
    than some default like 0."""

    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(settings, [make_point("p1", "old.md", page=1)])

    metadata = repository.filename_metadata(settings)

    assert metadata["old.md"] == DocumentMetadata(
        chunk_count=1, page_count=1, byte_size=None, uploaded_at=None
    )


def test_vector_repository_filename_metadata_empty_for_empty_collection() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.ensure_ready(settings)

    assert repository.filename_metadata(settings) == {}


def make_ordinal_point(
    point_id: str, filename: str, ordinal: int, text: str
) -> models.PointStruct:
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
            "text": text,
            "chunk_ordinal": ordinal,
        },
    )


def test_vector_repository_fetch_neighbors_returns_matching_ordinals() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(
        settings,
        [
            make_ordinal_point("p1", "guide.md", 1, "one"),
            make_ordinal_point("p2", "guide.md", 2, "two"),
            make_ordinal_point("p3", "guide.md", 3, "three"),
            make_ordinal_point("p4", "other.md", 2, "other-two"),
        ],
    )

    neighbors = repository.fetch_neighbors(settings, "guide.md", [1, 3])

    texts = sorted(point.payload["text"] for point in neighbors if point.payload)
    assert texts == ["one", "three"]


def test_vector_repository_fetch_neighbors_empty_for_no_ordinals() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)

    assert repository.fetch_neighbors(settings, "guide.md", []) == []


def test_vector_repository_chunks_for_filename_sorted_by_ordinal() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(
        settings,
        [
            make_ordinal_point("p3", "guide.md", 3, "three"),
            make_ordinal_point("p1", "guide.md", 1, "one"),
            make_ordinal_point("p2", "guide.md", 2, "two"),
            make_ordinal_point("x1", "other.md", 1, "other-one"),
        ],
    )

    chunks = repository.chunks_for_filename(settings, "guide.md")

    assert [payload["chunk_ordinal"] for payload in chunks] == [1, 2, 3]
    assert all(payload["filename"] == "guide.md" for payload in chunks)


def test_vector_repository_chunks_for_filename_empty_for_unknown_file() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(settings, [make_point("p1", "guide.md")])

    assert repository.chunks_for_filename(settings, "missing.md") == []
