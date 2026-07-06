"""FastAPI application for DocRAG."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from api.dependencies import get_app_settings, get_ingest_service, get_rag_pipeline
from api.documents import DocumentError
from api.embeddings import EmbeddingError
from api.generation import GenerationConfigError, GenerationError
from api.pipeline import IngestService, RagPipeline
from api.qdrant_schema import CollectionSchemaError
from api.reranking import RerankingError
from api.retrieval import RetrievalError, RetrievalPayloadError
from api.schemas import (
    CitationResponse,
    DocumentIngestResponse,
    HealthResponse,
    PublicConfigResponse,
    QuestionRequest,
    QuestionResponse,
)
from api.settings import AppSettings

SettingsDep = Annotated[AppSettings, Depends(get_app_settings)]
IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]
RagPipelineDep = Annotated[RagPipeline, Depends(get_rag_pipeline)]


FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def create_app() -> FastAPI:
    """Create the FastAPI app."""

    app = FastAPI(
        title="DocRAG API",
        version="0.1.0",
        description="Self-hostable document Q&A API with hybrid retrieval and citations.",
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
    def config(settings: SettingsDep) -> PublicConfigResponse:
        return public_config(settings)

    @app.post("/documents", response_model=DocumentIngestResponse)
    async def upload_document(
        ingest_service: IngestServiceDep,
        file: Annotated[UploadFile, File()],
    ) -> DocumentIngestResponse:
        content = await file.read()
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
        grounded = pipeline.answer(request.question)
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


def public_config(settings: AppSettings) -> PublicConfigResponse:
    """Build a non-secret config response."""

    return PublicConfigResponse(
        qdrant_collection=settings.qdrant_collection,
        embedding_provider=settings.embedding_provider,
        dense_embedding_model=settings.dense_embedding_model,
        sparse_embedding_model=settings.sparse_embedding_model,
        reranker_model=settings.reranker_model,
        embedding_model_tag=settings.embedding_model_tag,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        openai_model=settings.openai_model,
        rrf_k=int(settings.rrf_k),
        dense_retrieval_limit=int(settings.dense_retrieval_limit),
        sparse_retrieval_limit=int(settings.sparse_retrieval_limit),
        fused_top_n=int(settings.fused_top_n),
        rerank_top_k=int(settings.rerank_top_k),
        max_context_chunks=int(settings.max_context_chunks),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register domain exception handlers."""

    app.add_exception_handler(DocumentError, bad_request_handler)
    app.add_exception_handler(RetrievalPayloadError, internal_error_handler)
    app.add_exception_handler(RetrievalError, bad_request_handler)
    app.add_exception_handler(CollectionSchemaError, conflict_handler)
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


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 500 response for unexpected pipeline state."""

    return error_response(500, exc)


def error_response(status_code: int, exc: Exception) -> JSONResponse:
    """Build a consistent error response."""

    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


app = create_app()
