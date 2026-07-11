"""FastAPI application for DocRAG."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from api.dependencies import (
    get_app_settings,
    get_embedding_provider,
    get_ingest_service,
    get_ollama_reachability_checker,
    get_rag_pipeline,
    get_reranker,
)
from api.documents import DocumentError
from api.embeddings import EmbeddedText, EmbeddingError
from api.generation import GenerationConfigError, GenerationError
from api.ingestion import IngestionError
from api.pipeline import AnswerOverrides, IngestService, RagPipeline
from api.qdrant_schema import CollectionSchemaError, VectorStoreUnavailableError
from api.reranking import RerankingError
from api.retrieval import RetrievalError, RetrievalPayloadError
from api.schemas import (
    REQUEST_MAX_CONTEXT_CHUNKS_MAX,
    REQUEST_RERANK_TOP_K_MAX,
    REQUEST_TEMPERATURE_MAX,
    CitationResponse,
    DocumentIngestResponse,
    HealthResponse,
    PublicConfigResponse,
    QuestionRequest,
    QuestionResponse,
)
from api.settings import AppSettings
from api.upload import UploadTooLargeError, read_upload_within_limit

logger = logging.getLogger(__name__)

SettingsDep = Annotated[AppSettings, Depends(get_app_settings)]
IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]
RagPipelineDep = Annotated[RagPipeline, Depends(get_rag_pipeline)]
OllamaCheckDep = Annotated[Callable[[AppSettings], bool], Depends(get_ollama_reachability_checker)]


FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class WarmupEmbeddingProvider(Protocol):
    """Minimal embedding surface the warmup step needs."""

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]: ...


class WarmupReranker(Protocol):
    """Minimal reranker surface the warmup step needs."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


def run_model_warmup(
    embedding_provider: WarmupEmbeddingProvider,
    reranker: WarmupReranker,
) -> None:
    """Force both lazy-loaded models to load once, logging duration or failure.

    Both models are constructed with ``lazy_load=True``, so the real download/load
    cost is paid on first use, not at construction — this runs that first use at boot
    instead of on a user's first request. Deliberately never raises: a warmup failure
    (e.g. a transient network blip fetching a model) must not prevent the app from
    starting and serving requests that might succeed once the model is retried lazily;
    it only means the misconfiguration is now visible in the startup log instead of
    silently deferred.
    """

    start = time.monotonic()
    try:
        embedding_provider.embed_texts(["warmup"])
        reranker.score("warmup", ["warmup"])
    except Exception:
        logger.exception(
            "Model warmup failed — this will resurface as an error on the first real "
            "request instead. Check DENSE_EMBEDDING_MODEL/SPARSE_EMBEDDING_MODEL/"
            "RERANKER_MODEL."
        )
    else:
        logger.info("Model warmup completed in %.1fs", time.monotonic() - start)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Optionally warm up lazy-loaded models once at startup (see AppSettings.warmup_models)."""

    settings = get_app_settings()
    if settings.warmup_models:
        await run_in_threadpool(run_model_warmup, get_embedding_provider(), get_reranker())
    yield


def create_app() -> FastAPI:
    """Create the FastAPI app."""

    app = FastAPI(
        title="DocRAG API",
        version="0.1.0",
        description="Self-hostable document Q&A API with hybrid retrieval and citations.",
        lifespan=lifespan,
    )
    settings = get_app_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    register_exception_handlers(app)
    register_routes(app)

    # Registered last: Starlette matches routes in registration order, so the API
    # routes above are matched before this catch-all mount ever sees the request.
    # Guarded by existence so a plain `uv run uvicorn api.main:app` (no frontend
    # build) still boots instead of crashing on StaticFiles' directory check.
    if FRONTEND_DIST_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")

    return app


def register_routes(app: FastAPI) -> None:
    """Register API routes."""

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/config", response_model=PublicConfigResponse)
    def config(settings: SettingsDep, check_ollama: OllamaCheckDep) -> PublicConfigResponse:
        return public_config(settings, ollama_available=check_ollama(settings))

    @app.post("/documents", response_model=DocumentIngestResponse)
    async def upload_document(
        ingest_service: IngestServiceDep,
        settings: SettingsDep,
        file: Annotated[UploadFile, File()],
    ) -> DocumentIngestResponse:
        content = await read_upload_within_limit(file, int(settings.max_upload_bytes))
        filename = file.filename or ""
        # Parsing, embedding, and upsert are blocking; run off the event loop so a
        # single upload cannot stall the whole server.
        outcome = await run_in_threadpool(ingest_service.ingest, filename, content)
        return DocumentIngestResponse(
            filename=outcome.filename,
            sections_parsed=outcome.sections_parsed,
            chunks_ingested=outcome.chunks_ingested,
            collection_name=outcome.collection_name,
        )

    @app.post("/questions", response_model=QuestionResponse)
    def answer_question(
        request: QuestionRequest,
        pipeline: RagPipelineDep,
    ) -> QuestionResponse:
        grounded = pipeline.answer(
            request.question,
            AnswerOverrides(
                llm_provider=request.llm_provider,
                rerank_top_k=request.rerank_top_k,
                max_context_chunks=request.max_context_chunks,
                llm_temperature=request.llm_temperature,
            ),
        )
        return QuestionResponse(
            answer=grounded.answer,
            sources=[
                CitationResponse(
                    source_number=source.source_number,
                    filename=source.filename,
                    page=source.page,
                    section=source.section,
                    chunk_id=source.chunk_id,
                    text=source.text,
                )
                for source in grounded.sources
            ],
        )


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
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register domain exception handlers."""

    app.add_exception_handler(DocumentError, bad_request_handler)
    app.add_exception_handler(UploadTooLargeError, payload_too_large_handler)
    app.add_exception_handler(IngestionError, bad_gateway_handler)
    app.add_exception_handler(RetrievalPayloadError, internal_error_handler)
    app.add_exception_handler(RetrievalError, bad_request_handler)
    app.add_exception_handler(CollectionSchemaError, conflict_handler)
    app.add_exception_handler(VectorStoreUnavailableError, service_unavailable_handler)
    app.add_exception_handler(GenerationConfigError, bad_request_handler)
    app.add_exception_handler(GenerationError, bad_gateway_handler)
    app.add_exception_handler(RerankingError, bad_gateway_handler)
    app.add_exception_handler(EmbeddingError, internal_error_handler)


async def bad_request_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 400 response for user-correctable request errors."""

    return error_response(400, exc)


async def conflict_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 409 response for collection/index compatibility errors."""

    return error_response(409, exc)


async def bad_gateway_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 502 response for provider failures."""

    return error_response(502, exc)


async def service_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 503 response when a required backing service is unreachable."""

    return error_response(503, exc)


async def payload_too_large_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 413 response when an upload exceeds the configured size limit."""

    return error_response(413, exc)


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 500 response for unexpected pipeline state."""

    return error_response(500, exc)


def error_response(status_code: int, exc: Exception) -> JSONResponse:
    """Build a consistent error response."""

    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


app = create_app()
