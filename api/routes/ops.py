"""Health, readiness, metrics and public configuration routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, Header
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.dependencies import (
    key_access,
    require_api_key,
    require_full_key,
    supplied_key_bytes,
)
from api.routes.deps import OllamaCheckDep, QdrantCheckDep, QdrantClientDep, SettingsDep
from api.routes.questions import allowed_request_providers
from api.schemas import (
    REQUEST_MAX_CONTEXT_CHUNKS_MAX,
    REQUEST_RERANK_TOP_K_MAX,
    REQUEST_TEMPERATURE_MAX,
    AccessResponse,
    HealthResponse,
    PublicConfigResponse,
    ReadinessResponse,
)
from api.settings import AppSettings


def public_config(settings: AppSettings, *, ollama_available: bool) -> PublicConfigResponse:
    """Build a non-secret config response."""

    return PublicConfigResponse(
        qdrant_collection=settings.qdrant_collection,
        dense_embedding_model=settings.dense_embedding_model,
        sparse_embedding_model=settings.sparse_embedding_model,
        reranker_model=settings.reranker_model,
        embedding_model_tag=settings.embedding_model_tag,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_temperature=float(settings.llm_temperature),
        openai_model=settings.openai_model,
        openai_available=bool(settings.openai_api_key),
        ollama_available=ollama_available,
        rrf_k=int(settings.rrf_k),
        dense_retrieval_limit=int(settings.dense_retrieval_limit),
        sparse_retrieval_limit=int(settings.sparse_retrieval_limit),
        fused_top_n=int(settings.fused_top_n),
        rerank_top_k=int(settings.rerank_top_k),
        max_context_chunks=int(settings.max_context_chunks),
        rerank_min_score=float(settings.rerank_min_score),
        max_upload_bytes=int(settings.max_upload_bytes),
        rerank_top_k_limit=min(REQUEST_RERANK_TOP_K_MAX, int(settings.fused_top_n)),
        max_context_chunks_limit=REQUEST_MAX_CONTEXT_CHUNKS_MAX,
        llm_temperature_max=REQUEST_TEMPERATURE_MAX,
        feedback_enabled=settings.feedback_enabled,
        raw_documents_enabled=bool(settings.raw_document_dir),
        openai_compatible_available=bool(
            settings.openai_compatible_base_url and settings.openai_compatible_model
        ),
        openai_compatible_model=settings.openai_compatible_model,
        allowed_request_providers=allowed_request_providers(settings),
    )


def register(app: FastAPI) -> None:
    """Register health, readiness, metrics and config routes."""

    # async: it touches nothing blocking, and a sync handler waits for a threadpool
    # token — so a burst of slow questions holding every token failed the healthcheck.
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/health/ready")
    def health_ready(
        settings: SettingsDep,
        check_qdrant: QdrantCheckDep,
        qdrant_client: QdrantClientDep,
        check_ollama: OllamaCheckDep,
    ) -> Response:
        # Plain `def`, not `async def`: both probes below block (an httpx sync call,
        # a Qdrant client call), so this runs on FastAPI's request threadpool instead
        # of the event loop — a slow Ollama probe (~1.5s worst case) never stalls
        # other in-flight requests the way it would inside an async handler.
        qdrant_ok = check_qdrant(qdrant_client)
        provider = settings.llm_provider.lower().strip()
        if provider == "ollama":
            provider_ok = check_ollama(settings)
        elif provider == "openai_compatible":
            provider_ok = bool(
                settings.openai_compatible_base_url and settings.openai_compatible_model
            )
        else:
            provider_ok = bool(settings.openai_api_key)
        body = ReadinessResponse(
            status="ok" if qdrant_ok and provider_ok else "degraded",
            qdrant=qdrant_ok,
            generation_provider=provider_ok,
            llm_provider=settings.llm_provider,
        )
        status_code = 200 if qdrant_ok and provider_ok else 503
        return JSONResponse(status_code=status_code, content=body.model_dump())

    # Guarded when a key is configured. The labels carry no content (only `stage` and
    # `outcome`, see api/metrics.py), but the counters still disclose usage volume and
    # rough corpus growth, and an operator who bothered to set a key did not intend to
    # publish those. A Prometheus scraper needs the header threaded into its scrape
    # config; /health and /health/ready stay open for liveness either way.
    @app.get("/metrics", dependencies=[Depends(require_full_key)])
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/config", response_model=PublicConfigResponse)
    def config(settings: SettingsDep, check_ollama: OllamaCheckDep) -> PublicConfigResponse:
        return public_config(settings, ollama_available=check_ollama(settings))

    # Guarded (any valid key): /config is open, so it can't say which key the caller
    # holds. The UI asks here to hide what the read-only key would be refused (403).
    @app.get("/access", response_model=AccessResponse, dependencies=[Depends(require_api_key)])
    def access(
        settings: SettingsDep,
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> AccessResponse:
        if not settings.api_key:
            return AccessResponse(access="open")
        read_only = key_access(settings, supplied_key_bytes(x_api_key)) == "read"
        return AccessResponse(access="read" if read_only else "full")
