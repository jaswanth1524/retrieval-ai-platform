"""FastAPI dependency factories."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from qdrant_client import QdrantClient

from api.chunking import TokenCounter, make_token_counter
from api.embeddings import LocalEmbeddingProvider
from api.feedback import FeedbackStore
from api.generation import LiteLLMGenerator
from api.jobs import IngestJobStore, JobStore, SqliteIngestJobStore
from api.pipeline import IngestService, RagPipeline
from api.provider_health import check_ollama_reachable, check_qdrant_reachable
from api.qdrant_schema import clear_readiness_cache, make_qdrant_client
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.reranking import LocalCrossEncoderReranker, RerankingError
from api.settings import AppSettings
from api.tracing import TraceStore


@lru_cache
def get_app_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings()


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant client."""

    return make_qdrant_client(get_app_settings())


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


def get_qdrant_reachability_checker() -> Callable[[QdrantClient], bool]:
    """Return the Qdrant-reachability probe.

    Not cached, mirroring ``get_ollama_reachability_checker`` — exposed as a
    dependency so tests can override it without needing a real Qdrant instance.
    """

    return check_qdrant_reachable


def require_api_key(
    settings: Annotated[AppSettings, Depends(get_app_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Enforce the optional shared API key (see AppSettings.api_key).

    No-op when ``api_key`` is empty (the default) — every route stays fully open,
    preserving the zero-config self-host story. When set, the request must carry a
    matching ``X-API-Key`` header, compared with ``secrets.compare_digest`` to avoid a
    timing side channel. Not applied to /health or /health/ready, so healthchecks and
    readiness probes keep working regardless of whether a key is configured.

    The comparison is on *bytes*, not str: ``compare_digest`` raises TypeError on a
    non-ASCII str, and Starlette hands header values over latin-1-decoded, so any
    unauthenticated request carrying a non-ASCII byte in X-API-Key used to surface as a
    500 instead of a 401. Re-encoding the header with latin-1 recovers the exact wire
    bytes, which also lets a non-ASCII configured key (utf-8 out of the environment)
    match a client sending it as utf-8 rather than mismatching via mojibake.
    """

    if not settings.api_key:
        return
    if not x_api_key or not secrets.compare_digest(
        x_api_key.encode("latin-1", "replace"), settings.api_key.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )


def get_vector_repository(
    client: Annotated[QdrantClient, Depends(get_qdrant_client)],
) -> VectorRepository:
    """Return the vector-store repository seam, built from the cached Qdrant client.

    Expressed as a FastAPI sub-dependency (not a plain call to ``get_qdrant_client()``)
    so overriding ``get_qdrant_client`` in tests cascades through this and every seam
    built on top of it.
    """

    return VectorRepository(client)


@lru_cache
def get_trace_store() -> TraceStore:
    """Return the process-wide per-query debug trace store.

    Reads settings directly (not through FastAPI's DI) the same way
    ``get_ingest_executor``'s worker count does — the retention cap is sized once at
    first use, not per request. A test wanting a non-default cap should override this
    dependency directly, or unit-test ``TraceStore`` in isolation.
    """

    return TraceStore(max_retained=int(get_app_settings().trace_max_retained))


def get_rag_pipeline(
    repository: Annotated[VectorRepository, Depends(get_vector_repository)],
    embedding_provider: Annotated[LocalEmbeddingProvider, Depends(get_embedding_provider)],
    reranker: Annotated[LocalCrossEncoderReranker, Depends(get_reranker)],
    generator: Annotated[LiteLLMGenerator, Depends(get_generator)],
    settings: Annotated[AppSettings, Depends(get_app_settings)],
    trace_store: Annotated[TraceStore, Depends(get_trace_store)],
) -> RagPipeline:
    """Return the query orchestration seam: retrieve -> rerank -> generate."""

    return RagPipeline(
        repository=repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
        generator=generator,
        settings=settings,
        # The gate reads the *injected* settings (not a direct get_app_settings()
        # call) so a test overriding get_app_settings controls trace_enabled too.
        trace_store=trace_store if settings.trace_enabled else None,
    )


@lru_cache
def get_token_counter() -> TokenCounter:
    """Return the process-wide token counter for chunk sizing.

    Built from the dense embedding model name so chunking counts the same subword
    tokens the model will. Lazy-loads the tokenizer on first use and degrades to a
    word heuristic if it can't be fetched (see api/chunking.py).
    """

    return make_token_counter(get_app_settings().dense_embedding_model)


def get_ingest_service(
    repository: Annotated[VectorRepository, Depends(get_vector_repository)],
    embedding_provider: Annotated[LocalEmbeddingProvider, Depends(get_embedding_provider)],
    settings: Annotated[AppSettings, Depends(get_app_settings)],
    token_counter: Annotated[TokenCounter, Depends(get_token_counter)],
) -> IngestService:
    """Return the ingest orchestration seam: parse -> chunk -> embed -> index."""

    return IngestService(
        repository=repository,
        embedding_provider=embedding_provider,
        settings=settings,
        token_counter=token_counter,
    )


@lru_cache
def get_ingest_job_store() -> JobStore:
    """Return the process-wide background ingest job status store.

    Reads settings directly (not through FastAPI's DI), the same pattern as
    ``get_trace_store`` — the retention cap is sized once at first use. The backend
    (in-memory vs sqlite-persisted) is picked here, the only place that needs to know
    which concrete class is in play — everything else types against ``JobStore``.
    """

    settings = get_app_settings()
    if settings.job_store_backend == "sqlite":
        return SqliteIngestJobStore(
            path=settings.job_store_path,
            max_retained=int(settings.ingest_jobs_max_retained),
        )
    return IngestJobStore(max_retained=int(settings.ingest_jobs_max_retained))


@lru_cache
def get_raw_document_store() -> RawDocumentStore | None:
    """Return the optional original-bytes store, or None when the feature is off.

    Same direct-settings-read pattern as ``get_ingest_job_store``. An empty string
    (the ``.env.example`` default's literal value) means off, same as unset.
    """

    directory = get_app_settings().raw_document_dir
    return RawDocumentStore(directory) if directory else None


@lru_cache
def get_feedback_store() -> FeedbackStore | None:
    """Return the optional feedback store, or None when the feature is off.

    Same direct-settings-read pattern as ``get_raw_document_store``.
    """

    settings = get_app_settings()
    return FeedbackStore(settings.feedback_store_path) if settings.feedback_enabled else None


@lru_cache
def get_ingest_executor() -> ThreadPoolExecutor:
    """Return the process-wide executor background ingest jobs run on.

    Two workers: enough to overlap ingest of a couple of documents without letting an
    unbounded queue of uploads exhaust memory competing with query-time model calls.
    """

    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="docrag-ingest")


def clear_dependency_caches() -> None:
    """Clear dependency caches for tests and process reloads."""

    get_app_settings.cache_clear()
    get_qdrant_client.cache_clear()
    get_embedding_provider.cache_clear()
    get_reranker.cache_clear()
    get_generator.cache_clear()
    get_token_counter.cache_clear()
    get_ingest_job_store.cache_clear()
    get_raw_document_store.cache_clear()
    get_feedback_store.cache_clear()
    get_ingest_executor.cache_clear()
    get_trace_store.cache_clear()
    clear_readiness_cache()
