from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from api import qdrant_schema
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
    recreate_if_empty_and_mismatched,
    sparse_vectors_config,
)
from api.settings import AppSettings
from tests.factories import make_test_settings


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
    return make_test_settings(**defaults)


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

    with pytest.raises(EmbeddingModelMismatchError, match="delete every document"):
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


def test_make_qdrant_client_sends_the_configured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.qdrant_schema as qdrant_schema

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(qdrant_schema, "QdrantClient", lambda **kwargs: captured.append(kwargs))

    qdrant_schema.make_qdrant_client(make_test_settings(qdrant_api_key="qdrant-secret"))
    qdrant_schema.make_qdrant_client(make_test_settings())

    assert captured[0]["api_key"] == "qdrant-secret"
    assert captured[1]["api_key"] is None


def test_new_collections_score_bm25_with_idf() -> None:
    clear_readiness_cache()
    client = QdrantClient(":memory:")
    settings = make_settings()

    ensure_collection(client, settings)

    sparse = client.get_collection(settings.qdrant_collection).config.params.sparse_vectors
    assert sparse is not None
    assert sparse[settings.qdrant_sparse_vector_name].modifier == models.Modifier.IDF


def test_an_existing_collection_without_idf_is_migrated_in_place() -> None:
    """Collections created before the modifier keep their points and gain IDF — no
    re-embedding and no refusal, since the stored vectors are unchanged."""

    clear_readiness_cache()
    client = QdrantClient(":memory:")
    settings = make_settings()
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=dense_vectors_config(settings),
        sparse_vectors_config={settings.qdrant_sparse_vector_name: models.SparseVectorParams()},
        metadata=collection_metadata(settings),
    )
    client.upsert(
        settings.qdrant_collection,
        [
            models.PointStruct(
                id=1,
                vector={
                    settings.qdrant_dense_vector_name: [1.0, 0.0, 0.0, 0.0],
                    settings.qdrant_sparse_vector_name: models.SparseVector(
                        indices=[1], values=[1.0]
                    ),
                },
            )
        ],
    )

    ready = ensure_collection(client, settings)

    info = client.get_collection(settings.qdrant_collection)
    assert ready.created is False
    assert info.points_count == 1
    assert info.config.params.sparse_vectors is not None
    assert (
        info.config.params.sparse_vectors[settings.qdrant_sparse_vector_name].modifier
        == models.Modifier.IDF
    )
    clear_readiness_cache()


def test_existing_collection_with_a_different_dense_model_is_rejected() -> None:
    """Same tag, same dimension, another dense model: the stored vectors are in a
    different space, and only the recorded model name shows it."""

    settings = make_settings()
    client = QdrantClient(":memory:")
    metadata = collection_metadata(settings)
    metadata["dense_embedding_model"] = "another-384-dim-model"
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=dense_vectors_config(settings),
        sparse_vectors_config=sparse_vectors_config(settings),
        metadata=metadata,
    )

    with pytest.raises(CollectionSchemaError, match="dense embedding model"):
        ensure_collection(client, settings)


def test_a_collection_without_a_recorded_dense_model_is_accepted() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    metadata = collection_metadata(settings)
    del metadata["dense_embedding_model"]
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=dense_vectors_config(settings),
        sparse_vectors_config=sparse_vectors_config(settings),
        metadata=metadata,
    )

    assert ensure_collection(client, settings).created is False


def test_the_empty_collection_rebuild_never_deletes_a_collection_with_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first check saw a mismatch; by the time the lock was held another worker had
    fixed the schema and written a point. The rebuild must stand down, not delete."""

    settings = make_settings()
    client = QdrantClient(":memory:")
    ensure_collection(client, settings)
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            models.PointStruct(
                id=1,
                vector={
                    "dense": [0.1, 0.2, 0.3, 0.4],
                    "sparse": models.SparseVector(indices=[1], values=[1.0]),
                },
                payload={},
            )
        ],
    )
    clear_readiness_cache()
    calls = 0
    real_strict = qdrant_schema._ensure_collection_strict

    def first_check_sees_a_mismatch(
        client: QdrantClient, settings: AppSettings, *, create_workspace: bool = False
    ) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CollectionSchemaError("stale view")
        return real_strict(client, settings, create_workspace=create_workspace)

    monkeypatch.setattr(qdrant_schema, "_ensure_collection_strict", first_check_sees_a_mismatch)

    assert recreate_if_empty_and_mismatched(client, settings) is False
    assert client.count(settings.qdrant_collection, exact=True).count == 1


def test_the_client_dials_the_configured_grpc_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """It always used 6334, whatever QDRANT_GRPC_PORT said."""

    import api.qdrant_schema as qdrant_schema
    from tests.factories import make_test_settings

    seen: dict[str, object] = {}
    monkeypatch.setattr(qdrant_schema, "QdrantClient", lambda **kwargs: seen.update(kwargs))

    qdrant_schema.make_qdrant_client(
        make_test_settings(
            qdrant_url="http://qdrant:6333", qdrant_prefer_grpc=True, qdrant_grpc_port=7334
        )
    )

    assert seen["grpc_port"] == 7334
    assert seen["prefer_grpc"] is True


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("sparse_language", "german", "BM25 stemmer language"),
        ("dense_passage_prefix", "passage: ", "passage prefix"),
    ],
)
def test_a_changed_multilingual_setting_is_refused_until_reindexed(
    setting: str, value: str, message: str
) -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    ensure_collection(client, settings)

    changed = settings.model_copy(update={setting: value})
    qdrant_schema.clear_readiness_cache()
    with pytest.raises(CollectionSchemaError, match=message):
        ensure_collection(client, changed)


def test_a_collection_from_before_the_multilingual_settings_reads_as_the_defaults() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    legacy = collection_metadata(settings)
    del legacy["sparse_language"], legacy["dense_passage_prefix"]
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=dense_vectors_config(settings),
        sparse_vectors_config=sparse_vectors_config(settings),
        metadata=legacy,
    )

    ensure_collection(client, settings)  # accepted: English stemming, no prefix
    qdrant_schema.clear_readiness_cache()
    with pytest.raises(CollectionSchemaError, match="BM25 stemmer language"):
        ensure_collection(client, settings.model_copy(update={"sparse_language": "french"}))


@pytest.mark.parametrize("stored", ["English", " english ", ""])
def test_an_equivalent_stored_sparse_language_is_accepted(stored: str) -> None:
    settings = make_settings(sparse_language=" English ")
    assert settings.sparse_language == "english"
    assert make_settings(sparse_language="").sparse_language == "english"
    client = QdrantClient(":memory:")
    metadata = collection_metadata(settings)
    metadata["sparse_language"] = stored  # as a collection recorded it before normalizing
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=dense_vectors_config(settings),
        sparse_vectors_config=sparse_vectors_config(settings),
        metadata=metadata,
    )

    ensure_collection(client, settings)  # the same English index: no re-ingest asked for


def test_quantization_is_only_changed_when_the_setting_differs() -> None:
    from unittest.mock import MagicMock

    from qdrant_client import models

    from api.qdrant_schema import ensure_quantization, quantization_config

    assert quantization_config(make_test_settings()) is None
    scalar = make_test_settings(qdrant_quantization="scalar")
    assert isinstance(quantization_config(scalar), models.ScalarQuantization)

    def info(quantized: bool) -> MagicMock:
        collection = MagicMock()
        collection.config.quantization_config = quantization_config(scalar) if quantized else None
        return collection

    client = MagicMock()
    ensure_quantization(client, scalar, info(quantized=True))
    ensure_quantization(client, make_test_settings(), info(quantized=False))
    client.update_collection.assert_not_called()
    ensure_quantization(client, scalar, info(quantized=False))
    assert client.update_collection.call_args.kwargs["quantization_config"] is not None
    ensure_quantization(client, make_test_settings(), info(quantized=True))
    assert client.update_collection.call_args.kwargs["quantization_config"] == (
        models.Disabled.DISABLED
    )
