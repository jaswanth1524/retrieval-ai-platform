"""Qdrant collection schema and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary

from qdrant_client import QdrantClient, models

from api.settings import AppSettings

EMBEDDING_MODEL_TAG_KEY = "embedding_model_tag"
PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    ("filename", models.PayloadSchemaType.KEYWORD),
    ("page", models.PayloadSchemaType.INTEGER),
    ("section", models.PayloadSchemaType.KEYWORD),
    ("chunk_id", models.PayloadSchemaType.KEYWORD),
)


class CollectionSchemaError(RuntimeError):
    """Raised when an existing Qdrant collection does not match DocRAG's schema."""


class EmbeddingModelMismatchError(CollectionSchemaError):
    """Raised when the configured embedding model tag differs from collection metadata."""

    def __init__(self, *, collection_name: str, expected: str, actual: str | None) -> None:
        self.collection_name = collection_name
        self.expected = expected
        self.actual = actual
        actual_label = actual if actual is not None else "<missing>"
        super().__init__(
            f"Collection '{collection_name}' uses embedding model tag '{actual_label}', "
            f"but the configured tag is '{expected}'. Re-ingest documents before querying."
        )


@dataclass(frozen=True)
class CollectionReady:
    """Result returned after checking or creating the Qdrant collection."""

    collection_name: str
    created: bool
    embedding_model_tag: str


def make_qdrant_client(settings: AppSettings) -> QdrantClient:
    """Create a Qdrant client from application settings."""

    return QdrantClient(url=settings.qdrant_url)


def dense_vectors_config(settings: AppSettings) -> dict[str, models.VectorParams]:
    """Return the named dense vector config for the DocRAG collection."""

    return {
        settings.qdrant_dense_vector_name: models.VectorParams(
            size=int(settings.qdrant_dense_vector_size),
            distance=models.Distance.COSINE,
        )
    }


def sparse_vectors_config(settings: AppSettings) -> dict[str, models.SparseVectorParams]:
    """Return the named sparse vector config for the DocRAG collection."""

    return {settings.qdrant_sparse_vector_name: models.SparseVectorParams()}


def collection_metadata(settings: AppSettings) -> dict[str, str]:
    """Return metadata that records the embedding vector space used by the collection."""

    return {
        EMBEDDING_MODEL_TAG_KEY: settings.embedding_model_tag,
        "dense_embedding_model": settings.dense_embedding_model,
        "sparse_embedding_model": settings.sparse_embedding_model,
    }


def create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create payload indexes for source citation metadata."""

    for field_name, field_schema in PAYLOAD_INDEXES:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )


# Per-client cache of already-validated collections, keyed by (collection, embedding
# tag, sparse model) so a config change is never masked by a stale cache entry. Only
# successful readiness is cached; validation failures always re-check against Qdrant so
# a mismatch keeps refusing until it is actually fixed (re-ingest), not until restart.
_ReadinessKey = tuple[str, str, str]
_readiness_cache: WeakKeyDictionary[QdrantClient, dict[_ReadinessKey, CollectionReady]]
_readiness_cache = WeakKeyDictionary()


def ensure_collection(client: QdrantClient, settings: AppSettings) -> CollectionReady:
    """Create or validate the configured Qdrant collection, caching success per client."""

    cache_key: _ReadinessKey = (
        settings.qdrant_collection,
        settings.embedding_model_tag,
        settings.sparse_embedding_model,
    )
    cached = _readiness_cache.get(client, {}).get(cache_key)
    if cached is not None:
        return cached

    ready = _ensure_collection_uncached(client, settings)
    _readiness_cache.setdefault(client, {})[cache_key] = ready
    return ready


def clear_readiness_cache() -> None:
    """Clear the process-wide collection-readiness cache (used by tests)."""

    _readiness_cache.clear()


def _ensure_collection_uncached(client: QdrantClient, settings: AppSettings) -> CollectionReady:
    """Create or validate the configured Qdrant collection against Qdrant itself."""

    collection_name = settings.qdrant_collection
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=dense_vectors_config(settings),
            sparse_vectors_config=sparse_vectors_config(settings),
            metadata=collection_metadata(settings),
        )
        create_payload_indexes(client, collection_name)
        return CollectionReady(
            collection_name=collection_name,
            created=True,
            embedding_model_tag=settings.embedding_model_tag,
        )

    collection_info = client.get_collection(collection_name)
    validate_collection_schema(collection_info, settings)
    return CollectionReady(
        collection_name=collection_name,
        created=False,
        embedding_model_tag=settings.embedding_model_tag,
    )


def validate_collection_schema(
    collection_info: models.CollectionInfo,
    settings: AppSettings,
) -> None:
    """Validate vector names, dense dimensions, sparse config, and embedding tag."""

    metadata = read_collection_metadata(collection_info)
    actual_tag = metadata.get(EMBEDDING_MODEL_TAG_KEY)
    if actual_tag != settings.embedding_model_tag:
        raise EmbeddingModelMismatchError(
            collection_name=settings.qdrant_collection,
            expected=settings.embedding_model_tag,
            actual=actual_tag,
        )

    # The dense-only tag does not capture the sparse model, but changing it also
    # invalidates the stored vectors. Validate the recorded sparse model too.
    actual_sparse = metadata.get("sparse_embedding_model")
    if actual_sparse != settings.sparse_embedding_model:
        raise CollectionSchemaError(
            f"Collection '{settings.qdrant_collection}' uses sparse embedding model "
            f"'{actual_sparse if actual_sparse is not None else '<missing>'}', but the "
            f"configured model is '{settings.sparse_embedding_model}'. "
            "Re-ingest documents before querying."
        )

    params = collection_info.config.params
    vectors = params.vectors
    if not isinstance(vectors, dict) or settings.qdrant_dense_vector_name not in vectors:
        raise CollectionSchemaError(
            f"Collection '{settings.qdrant_collection}' is missing dense vector "
            f"'{settings.qdrant_dense_vector_name}'. Re-ingest documents."
        )

    dense_config = vectors[settings.qdrant_dense_vector_name]
    if dense_config.size != settings.qdrant_dense_vector_size:
        raise CollectionSchemaError(
            f"Collection '{settings.qdrant_collection}' dense vector size is "
            f"{dense_config.size}, expected {settings.qdrant_dense_vector_size}. "
            "Re-ingest documents."
        )
    if dense_config.distance != models.Distance.COSINE:
        raise CollectionSchemaError(
            f"Collection '{settings.qdrant_collection}' dense vector distance is "
            f"{dense_config.distance}, expected {models.Distance.COSINE}. Re-ingest documents."
        )

    sparse_vectors = params.sparse_vectors or {}
    if settings.qdrant_sparse_vector_name not in sparse_vectors:
        raise CollectionSchemaError(
            f"Collection '{settings.qdrant_collection}' is missing sparse vector "
            f"'{settings.qdrant_sparse_vector_name}'. Re-ingest documents."
        )


def read_collection_metadata(collection_info: models.CollectionInfo) -> dict[str, str]:
    """Read collection metadata across qdrant-client response shapes."""

    metadata = getattr(collection_info.config, "metadata", None)
    if isinstance(metadata, dict):
        return {str(key): str(value) for key, value in metadata.items()}

    dumped = collection_info.model_dump()
    dumped_metadata = dumped.get("metadata")
    if isinstance(dumped_metadata, dict):
        return {str(key): str(value) for key, value in dumped_metadata.items()}

    config = dumped.get("config")
    if isinstance(config, dict):
        config_metadata = config.get("metadata")
        if isinstance(config_metadata, dict):
            return {str(key): str(value) for key, value in config_metadata.items()}

    return {}
