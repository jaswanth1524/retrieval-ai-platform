from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from api.qdrant_schema import (
    EMBEDDING_MODEL_TAG_KEY,
    PAYLOAD_INDEXES,
    CollectionSchemaError,
    EmbeddingModelMismatchError,
    VectorStoreUnavailableError,
    clear_readiness_cache,
    collection_metadata,
    dense_vectors_config,
    ensure_collection,
    read_collection_metadata,
    sparse_vectors_config,
)
from api.settings import AppSettings


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "test_documents",
        "qdrant_dense_vector_name": "dense",
        "qdrant_sparse_vector_name": "sparse",
        "qdrant_dense_vector_size": 4,
        "dense_embedding_model": "local-dense",
        "sparse_embedding_model": "local-sparse",
        "embedding_model_tag": "local-dense:v1",
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


def test_vector_configs_use_named_dense_and_sparse_vectors() -> None:
    settings = make_settings(qdrant_dense_vector_name="semantic", qdrant_sparse_vector_name="bm25")

    dense_config = dense_vectors_config(settings)
    sparse_config = sparse_vectors_config(settings)

    assert set(dense_config) == {"semantic"}
    assert dense_config["semantic"].size == 4
    assert dense_config["semantic"].distance == models.Distance.COSINE
    assert set(sparse_config) == {"bm25"}


def test_collection_metadata_records_embedding_vector_space() -> None:
    settings = make_settings()

    metadata = collection_metadata(settings)

    assert metadata[EMBEDDING_MODEL_TAG_KEY] == "local-dense:v1"
    assert metadata["dense_embedding_model"] == "local-dense"
    assert metadata["sparse_embedding_model"] == "local-sparse"


def test_ensure_collection_creates_collection_with_metadata() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")

    result = ensure_collection(client, settings)
    info = client.get_collection(settings.qdrant_collection)

    assert result.created is True
    assert result.embedding_model_tag == "local-dense:v1"
    assert read_collection_metadata(info)[EMBEDDING_MODEL_TAG_KEY] == "local-dense:v1"
    assert isinstance(info.config.params.vectors, dict)
    assert settings.qdrant_dense_vector_name in info.config.params.vectors
    assert settings.qdrant_sparse_vector_name in (info.config.params.sparse_vectors or {})


def test_existing_collection_with_matching_schema_is_reused() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=dense_vectors_config(settings),
        sparse_vectors_config=sparse_vectors_config(settings),
        metadata=collection_metadata(settings),
    )

    result = ensure_collection(client, settings)

    assert result.created is False


def test_ensure_collection_caches_readiness_and_skips_repeat_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    first = ensure_collection(client, settings)

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Qdrant should not be queried again on a cache hit.")

    monkeypatch.setattr(client, "collection_exists", fail_if_called)
    monkeypatch.setattr(client, "get_collection", fail_if_called)

    second = ensure_collection(client, settings)

    assert second == first


def test_clear_readiness_cache_forces_revalidation() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    ensure_collection(client, settings)

    clear_readiness_cache()

    # Cache cleared: this call must hit Qdrant again and correctly see the collection
    # as already existing (created=False), not replay the cached created=True.
    result = ensure_collection(client, settings)

    assert result.created is False


def test_ensure_collection_cache_never_masks_a_changed_embedding_tag() -> None:
    client = QdrantClient(":memory:")
    settings_a = make_settings(qdrant_collection="shared", embedding_model_tag="tag-a")
    ensure_collection(client, settings_a)

    settings_b = make_settings(qdrant_collection="shared", embedding_model_tag="tag-b")

    with pytest.raises(EmbeddingModelMismatchError):
        ensure_collection(client, settings_b)


def test_existing_collection_with_embedding_tag_mismatch_is_rejected() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=dense_vectors_config(settings),
        sparse_vectors_config=sparse_vectors_config(settings),
        metadata={EMBEDDING_MODEL_TAG_KEY: "other-tag"},
    )

    with pytest.raises(EmbeddingModelMismatchError, match="Re-ingest documents"):
        ensure_collection(client, settings)


def test_existing_collection_with_sparse_model_mismatch_is_rejected() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    metadata = collection_metadata(settings)
    metadata["sparse_embedding_model"] = "different-sparse"
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=dense_vectors_config(settings),
        sparse_vectors_config=sparse_vectors_config(settings),
        metadata=metadata,
    )

    with pytest.raises(CollectionSchemaError, match="sparse embedding model"):
        ensure_collection(client, settings)


def test_existing_collection_without_expected_dense_vector_is_rejected() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={"other": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config=sparse_vectors_config(settings),
        metadata=collection_metadata(settings),
    )

    with pytest.raises(CollectionSchemaError, match="missing dense vector"):
        ensure_collection(client, settings)


def test_ensure_collection_wraps_connection_failure_as_service_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")

    def fail_connection(*args: Any, **kwargs: Any) -> Any:
        raise ResponseHandlingException(ConnectionError("connection refused"))

    monkeypatch.setattr(client, "collection_exists", fail_connection)

    with pytest.raises(VectorStoreUnavailableError, match="unreachable"):
        ensure_collection(client, settings)


def test_ensure_collection_repairs_missing_payload_indexes_on_existing_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards against a collection left index-less by a crash between
    create_collection and create_payload_indexes (they are not atomic) — validation
    must notice and repair missing indexes, not just check vectors/tags forever."""

    settings = make_settings()
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=dense_vectors_config(settings),
        sparse_vectors_config=sparse_vectors_config(settings),
        metadata=collection_metadata(settings),
    )
    # Deliberately skip create_payload_indexes to simulate the crash window.

    created_indexes: list[str] = []
    original_create_index = client.create_payload_index

    def tracking_create_index(*args: Any, **kwargs: Any) -> Any:
        created_indexes.append(kwargs["field_name"])
        return original_create_index(*args, **kwargs)

    monkeypatch.setattr(client, "create_payload_index", tracking_create_index)

    ensure_collection(client, settings)

    assert set(created_indexes) == {field_name for field_name, _ in PAYLOAD_INDEXES}


def test_ensure_collection_concurrent_cold_calls_create_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent first-ever requests (FastAPI's threadpool) must not both see
    'not exists' and both call create_collection — the lock must serialize them."""

    settings = make_settings()
    client = QdrantClient(":memory:")
    create_calls = 0
    original_create = client.create_collection

    def counting_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal create_calls
        create_calls += 1
        return original_create(*args, **kwargs)

    monkeypatch.setattr(client, "create_collection", counting_create)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: ensure_collection(client, settings), range(4)))

    # The lock serializes access: exactly one thread actually creates the collection;
    # the rest hit the cache the winner populated (all four share that cached result,
    # which is why every result reports created=True here, not just one).
    assert create_calls == 1
    assert all(result.created for result in results)
