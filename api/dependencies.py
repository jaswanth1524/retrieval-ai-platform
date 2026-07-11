"""FastAPI dependency factories."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient

from api.embeddings import LocalEmbeddingProvider
from api.generation import LiteLLMGenerator
from api.pipeline import IngestService, RagPipeline
from api.provider_health import check_ollama_reachable
from api.qdrant_schema import clear_readiness_cache
from api.repository import VectorRepository
from api.reranking import LocalCrossEncoderReranker, RerankingError
from api.settings import AppSettings


@lru_cache
def get_app_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings()


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant client."""

    return QdrantClient(url=get_app_settings().qdrant_url)


@lru_cache
def get_embedding_provider() -> LocalEmbeddingProvider:
    """Return a cached local embedding provider.

    Intentionally hardcoded to local — embeddings stay local by design regardless of
    the selected generation provider (see CLAUDE.md's provider rules), so there is no
    settings-driven switch here.
    """

    return LocalEmbeddingProvider(get_app_settings())


@lru_cache
def get_reranker() -> LocalCrossEncoderReranker:
    """Return a cached local cross-encoder reranker."""

    try:
        return LocalCrossEncoderReranker(get_app_settings())
    except ValueError as exc:
        raise RerankingError(str(exc)) from exc


@lru_cache
def get_generator() -> LiteLLMGenerator:
    """Return a cached LiteLLM generator."""

    return LiteLLMGenerator(get_app_settings())


def get_ollama_reachability_checker() -> Callable[[AppSettings], bool]:
    """Return the Ollama-reachability probe.

    Not cached: Ollama can start or stop between requests, so ``/config`` must probe
    live each time. Exposed as a dependency (rather than a direct import in main.py)
    so tests can override it without hitting a real localhost:11434.
    """

    return check_ollama_reachable


def get_vector_repository(
    client: Annotated[QdrantClient, Depends(get_qdrant_client)],
) -> VectorRepository:
    """Return the vector-store repository seam, built from the cached Qdrant client.

    Expressed as a FastAPI sub-dependency (not a plain call to ``get_qdrant_client()``)
    so overriding ``get_qdrant_client`` in tests cascades through this and every seam
    built on top of it.
    """

    return VectorRepository(client)


def get_rag_pipeline(
    repository: Annotated[VectorRepository, Depends(get_vector_repository)],
    embedding_provider: Annotated[LocalEmbeddingProvider, Depends(get_embedding_provider)],
    reranker: Annotated[LocalCrossEncoderReranker, Depends(get_reranker)],
    generator: Annotated[LiteLLMGenerator, Depends(get_generator)],
    settings: Annotated[AppSettings, Depends(get_app_settings)],
) -> RagPipeline:
    """Return the query orchestration seam: retrieve -> rerank -> generate."""

    return RagPipeline(
        repository=repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
        generator=generator,
        settings=settings,
    )


def get_ingest_service(
    repository: Annotated[VectorRepository, Depends(get_vector_repository)],
    embedding_provider: Annotated[LocalEmbeddingProvider, Depends(get_embedding_provider)],
    settings: Annotated[AppSettings, Depends(get_app_settings)],
) -> IngestService:
    """Return the ingest orchestration seam: parse -> chunk -> embed -> index."""

    return IngestService(
        repository=repository,
        embedding_provider=embedding_provider,
        settings=settings,
    )


def clear_dependency_caches() -> None:
    """Clear dependency caches for tests and process reloads."""

    get_app_settings.cache_clear()
    get_qdrant_client.cache_clear()
    get_embedding_provider.cache_clear()
    get_reranker.cache_clear()
    get_generator.cache_clear()
    clear_readiness_cache()
