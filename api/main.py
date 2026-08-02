"""FastAPI application for DocRAG."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from qdrant_client import QdrantClient
from starlette.staticfiles import StaticFiles

from api.dependencies import (
    get_app_settings,
    get_embedding_provider,
    get_ingest_executor,
    get_ingest_job_store,
    get_ingest_service,
    get_ollama_reachability_checker,
    get_qdrant_client,
    get_qdrant_reachability_checker,
    get_rag_pipeline,
    get_reranker,
    get_trace_store,
    get_vector_repository,
    require_api_key,
)
from api.documents import DocumentError, DocumentNotFoundError
from api.embeddings import EmbeddedText, EmbeddingError
from api.generation import ChatMessage, GenerationConfigError, GenerationError, StageTimings
from api.ingestion import IngestionError
from api.jobs import IngestJobStore, JobNotFoundError
from api.logging_config import configure_logging
from api.metrics import (
    ingest_chunks_total,
    ingest_jobs_total,
    observe_question_timings,
    questions_total,
)
from api.pipeline import AnswerOverrides, IngestService, RagPipeline
from api.qdrant_schema import CollectionSchemaError, VectorStoreUnavailableError
from api.repository import VectorRepository
from api.reranking import RerankingError
from api.retrieval import RetrievalError, RetrievalPayloadError
from api.schemas import (
    REQUEST_MAX_CONTEXT_CHUNKS_MAX,
    REQUEST_RERANK_TOP_K_MAX,
    REQUEST_TEMPERATURE_MAX,
    CitationResponse,
    DocumentChunkResponse,
    DocumentContentResponse,
    DocumentDeleteResponse,
    DocumentIngestResponse,
    DocumentJobAcceptedResponse,
    DocumentJobStatusResponse,
    DocumentListResponse,
    HealthResponse,
    HistoryMessageRequest,
    PromptMessageResponse,
    PublicConfigResponse,
    QuestionRequest,
    QuestionResponse,
    ReadinessResponse,
    TimingsResponse,
    TraceCandidateResponse,
    TraceConfigResponse,
    TraceDetailResponse,
    TraceListResponse,
    TraceSummaryResponse,
)
from api.settings import AppSettings
from api.tracing import QueryTrace, TraceNotFoundError, TraceStore
from api.upload import UploadTooLargeError, read_upload_within_limit

logger = logging.getLogger(__name__)

SettingsDep = Annotated[AppSettings, Depends(get_app_settings)]
IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]
RagPipelineDep = Annotated[RagPipeline, Depends(get_rag_pipeline)]
OllamaCheckDep = Annotated[Callable[[AppSettings], bool], Depends(get_ollama_reachability_checker)]
QdrantClientDep = Annotated[QdrantClient, Depends(get_qdrant_client)]
QdrantCheckDep = Annotated[
    Callable[[QdrantClient], bool], Depends(get_qdrant_reachability_checker)
]
VectorRepositoryDep = Annotated[VectorRepository, Depends(get_vector_repository)]
IngestJobStoreDep = Annotated[IngestJobStore, Depends(get_ingest_job_store)]
IngestExecutorDep = Annotated[ThreadPoolExecutor, Depends(get_ingest_executor)]
TraceStoreDep = Annotated[TraceStore, Depends(get_trace_store)]


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
    configure_logging(settings.log_level)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["GET", "POST", "DELETE"],
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


def _run_ingest_job(
    job_store: IngestJobStore,
    job_id: str,
    ingest_service: IngestService,
    filename: str,
    content: bytes,
) -> None:
    """Background-executor entry point: parse/chunk/embed/index and update job status.

    Runs on ``get_ingest_executor()``'s worker threads, outside any request's
    lifetime — exceptions are caught and recorded on the job rather than raised,
    since there is no HTTP response left to attach them to.
    """

    job_store.update(job_id, state="parsing")

    def on_progress(done: int, total: int) -> None:
        job_store.update(job_id, state="embedding", chunks_done=done, chunks_total=total)

    try:
        outcome = ingest_service.ingest(filename, content, on_progress)
    except Exception as exc:
        logger.warning("Ingest job %s for %r failed: %s", job_id, filename, exc)
        ingest_jobs_total.labels(outcome="failed").inc()
        job_store.update(job_id, state="failed", error=str(exc))
        return

    ingest_jobs_total.labels(outcome="done").inc()
    ingest_chunks_total.inc(outcome.chunks_ingested)
    job_store.update(
        job_id,
        state="done",
        chunks_done=outcome.chunks_ingested,
        chunks_total=outcome.chunks_ingested,
        result={
            "filename": outcome.filename,
            "sections_parsed": outcome.sections_parsed,
            "chunks_ingested": outcome.chunks_ingested,
            "collection_name": outcome.collection_name,
        },
    )


def _sse_event(payload: dict[str, object]) -> str:
    """Format one Server-Sent Events ``data:`` frame."""

    return f"data: {json.dumps(payload)}\n\n"


def _timings_response(timings: StageTimings | None) -> TimingsResponse | None:
    """Build the response's timings field from a ``StageTimings`` (or None)."""

    if timings is None:
        return None
    return TimingsResponse(
        embed_ms=timings.embed_ms,
        search_ms=timings.search_ms,
        rerank_ms=timings.rerank_ms,
        generate_ms=timings.generate_ms,
        total_ms=timings.total_ms,
        condense_ms=timings.condense_ms,
        expand_ms=timings.expand_ms,
    )


def _history_messages(history: list[HistoryMessageRequest] | None) -> list[ChatMessage] | None:
    """Convert the request's history schema into pipeline-layer chat messages."""

    if not history:
        return None
    return [{"role": message.role, "content": message.content} for message in history]


def _payload_optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _content_chunk(payload: dict[str, object]) -> DocumentChunkResponse:
    """Build one source-viewer chunk from a stored Qdrant payload, tolerating gaps."""

    page = _payload_optional_int(payload.get("page"))
    return DocumentChunkResponse(
        chunk_id=str(payload.get("chunk_id", "")),
        page=page if page is not None else 0,
        section=str(payload.get("section", "")),
        text=str(payload.get("text", "")),
        chunk_ordinal=_payload_optional_int(payload.get("chunk_ordinal")),
    )


def _trace_summary_response(trace: QueryTrace) -> TraceSummaryResponse:
    """Build one row of the trace list from a stored trace."""

    return TraceSummaryResponse(
        trace_id=trace.trace_id,
        created_at=trace.created_at,
        question=trace.question,
        mode=trace.mode,
        status=trace.status,
        llm_provider=trace.config.llm_provider if trace.config is not None else None,
        total_ms=trace.timings.get("total_ms") if trace.timings is not None else None,
        candidate_count=len(trace.candidates),
        kept_count=sum(1 for candidate in trace.candidates if candidate.kept),
    )


def _trace_detail_response(trace: QueryTrace) -> TraceDetailResponse:
    """Build the full trace detail response from a stored trace."""

    return TraceDetailResponse(
        trace_id=trace.trace_id,
        created_at=trace.created_at,
        question=trace.question,
        mode=trace.mode,
        status=trace.status,
        config=(
            TraceConfigResponse(
                llm_provider=trace.config.llm_provider,
                rerank_top_k=trace.config.rerank_top_k,
                max_context_chunks=trace.config.max_context_chunks,
                llm_temperature=trace.config.llm_temperature,
                rerank_min_score=trace.config.rerank_min_score,
                fused_top_n=trace.config.fused_top_n,
                filenames=trace.config.filenames,
            )
            if trace.config is not None
            else None
        ),
        candidates=[
            TraceCandidateResponse(
                point_id=candidate.point_id,
                filename=candidate.filename,
                page=candidate.page,
                section=candidate.section,
                chunk_id=candidate.chunk_id,
                retrieval_score=candidate.retrieval_score,
                rerank_score=candidate.rerank_score,
                kept=candidate.kept,
                drop_reason=candidate.drop_reason,
                selected_for_context=candidate.selected_for_context,
                neighbor_expanded=candidate.neighbor_expanded,
            )
            for candidate in trace.candidates
        ],
        prompt_messages=(
            [
                PromptMessageResponse(role=message["role"], content=message["content"])
                for message in trace.prompt_messages
            ]
            if trace.prompt_messages is not None
            else None
        ),
        answer=trace.answer,
        cited_source_numbers=trace.cited_source_numbers,
        timings=TimingsResponse(**trace.timings) if trace.timings is not None else None,
        error=trace.error,
        condensed_question=trace.condensed_question,
        history_message_count=trace.history_message_count,
        query_variants=trace.query_variants,
        citation_retry_used=trace.citation_retry_used,
    )


def _timings_dict(timings: StageTimings) -> dict[str, float]:
    return {
        "embed_ms": timings.embed_ms,
        "search_ms": timings.search_ms,
        "rerank_ms": timings.rerank_ms,
        "generate_ms": timings.generate_ms,
        "total_ms": timings.total_ms,
        "condense_ms": timings.condense_ms,
        "expand_ms": timings.expand_ms,
    }


def _log_question(
    *, provider: str | None, source_count: int, timings: dict[str, float]
) -> None:
    """Emit one structured log line per answered question for latency observability."""

    logger.info(
        "question answered provider=%s sources=%d condense_ms=%.1f embed_ms=%.1f "
        "search_ms=%.1f rerank_ms=%.1f generate_ms=%.1f total_ms=%.1f",
        provider or "default",
        source_count,
        timings.get("condense_ms", 0.0),
        timings["embed_ms"],
        timings["search_ms"],
        timings["rerank_ms"],
        timings["generate_ms"],
        timings["total_ms"],
    )


def register_routes(app: FastAPI) -> None:
    """Register API routes."""

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
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
        provider_ok = (
            check_ollama(settings) if provider == "ollama" else bool(settings.openai_api_key)
        )
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
    @app.get("/metrics", dependencies=[Depends(require_api_key)])
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/config", response_model=PublicConfigResponse)
    def config(settings: SettingsDep, check_ollama: OllamaCheckDep) -> PublicConfigResponse:
        return public_config(settings, ollama_available=check_ollama(settings))

    @app.get(
        "/documents",
        response_model=DocumentListResponse,
        dependencies=[Depends(require_api_key)],
    )
    def list_documents(
        repository: VectorRepositoryDep, settings: SettingsDep
    ) -> DocumentListResponse:
        return DocumentListResponse(filenames=repository.list_filenames(settings))

    @app.post(
        "/documents",
        response_model=DocumentJobAcceptedResponse,
        status_code=202,
        dependencies=[Depends(require_api_key)],
    )
    async def upload_document(
        ingest_service: IngestServiceDep,
        settings: SettingsDep,
        job_store: IngestJobStoreDep,
        executor: IngestExecutorDep,
        file: Annotated[UploadFile, File()],
    ) -> DocumentJobAcceptedResponse:
        content = await read_upload_within_limit(file, int(settings.max_upload_bytes))
        filename = file.filename or ""
        job = job_store.create(filename)
        # Ingestion runs on a dedicated executor (not FastAPI's request threadpool) so
        # it survives independently of this request/response cycle — the client can
        # disconnect and poll the job later without interrupting the work.
        executor.submit(_run_ingest_job, job_store, job.id, ingest_service, filename, content)
        return DocumentJobAcceptedResponse(job_id=job.id, filename=filename, state="queued")

    @app.delete(
        "/documents/{filename}",
        response_model=DocumentDeleteResponse,
        dependencies=[Depends(require_api_key)],
    )
    def delete_document(
        filename: str, repository: VectorRepositoryDep, settings: SettingsDep
    ) -> DocumentDeleteResponse:
        # Filenames are always basenames (normalize_filename strips any path
        # component at upload time), so a plain path segment is sufficient — a
        # filename containing "/" simply can't match this route, which is fine
        # since one can never have been uploaded.
        point_ids = repository.point_ids_for_filename(settings, filename)
        if not point_ids:
            raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
        repository.delete_by_ids(settings, point_ids)
        return DocumentDeleteResponse(filename=filename, points_deleted=len(point_ids))

    # Guarded like its sibling document routes: the response carries the filename (and
    # on failure the error text), and the frontend polls this URL once a second during
    # every upload, so the job id lands in browser history and any intermediate proxy
    # log — places a party who never held the key can read it from.
    @app.get(
        "/documents/jobs/{job_id}",
        response_model=DocumentJobStatusResponse,
        dependencies=[Depends(require_api_key)],
    )
    def get_document_job(job_id: str, job_store: IngestJobStoreDep) -> DocumentJobStatusResponse:
        job = job_store.get(job_id)
        if job is None:
            raise JobNotFoundError(f"No ingestion job found with id '{job_id}'.")
        result = (
            DocumentIngestResponse.model_validate(job.result) if job.result is not None else None
        )
        return DocumentJobStatusResponse(
            job_id=job.id,
            filename=job.filename,
            state=job.state,
            chunks_total=job.chunks_total,
            chunks_done=job.chunks_done,
            error=job.error,
            result=result,
        )

    # Guarded alongside GET /documents: this returns every chunk's raw text, so leaving
    # it open would let an unauthenticated caller list the corpus and then dump it in
    # full — the key would protect Q&A while the documents themselves stayed readable.
    @app.get(
        "/documents/{filename}/content",
        response_model=DocumentContentResponse,
        dependencies=[Depends(require_api_key)],
    )
    def document_content(
        filename: str, repository: VectorRepositoryDep, settings: SettingsDep
    ) -> DocumentContentResponse:
        payloads = repository.chunks_for_filename(settings, filename)
        if not payloads:
            raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
        chunks = [_content_chunk(payload) for payload in payloads]
        return DocumentContentResponse(filename=filename, chunks=chunks)

    @app.get("/traces", response_model=TraceListResponse, dependencies=[Depends(require_api_key)])
    def list_traces(trace_store: TraceStoreDep) -> TraceListResponse:
        return TraceListResponse(
            traces=[_trace_summary_response(trace) for trace in trace_store.list_traces()]
        )

    @app.get(
        "/traces/{trace_id}",
        response_model=TraceDetailResponse,
        dependencies=[Depends(require_api_key)],
    )
    def get_trace(trace_id: str, trace_store: TraceStoreDep) -> TraceDetailResponse:
        trace = trace_store.get(trace_id)
        if trace is None:
            raise TraceNotFoundError(f"No trace found with id '{trace_id}'.")
        return _trace_detail_response(trace)

    @app.post(
        "/questions", response_model=QuestionResponse, dependencies=[Depends(require_api_key)]
    )
    def answer_question(
        request: QuestionRequest,
        pipeline: RagPipelineDep,
    ) -> QuestionResponse:
        try:
            grounded = pipeline.answer(
                request.question,
                AnswerOverrides(
                    llm_provider=request.llm_provider,
                    rerank_top_k=request.rerank_top_k,
                    max_context_chunks=request.max_context_chunks,
                    llm_temperature=request.llm_temperature,
                ),
                filenames=request.filenames,
                history=_history_messages(request.history),
            )
        except Exception:
            questions_total.labels(outcome="error").inc()
            raise
        questions_total.labels(outcome="ok").inc()
        if grounded.timings is not None:
            timings = _timings_dict(grounded.timings)
            observe_question_timings(timings)
            _log_question(
                provider=request.llm_provider,
                source_count=len(grounded.sources),
                timings=timings,
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
            timings=_timings_response(grounded.timings),
            trace_id=grounded.trace_id,
        )

    @app.post("/questions/stream", dependencies=[Depends(require_api_key)])
    def answer_question_stream(
        request: QuestionRequest,
        pipeline: RagPipelineDep,
    ) -> StreamingResponse:
        overrides = AnswerOverrides(
            llm_provider=request.llm_provider,
            rerank_top_k=request.rerank_top_k,
            max_context_chunks=request.max_context_chunks,
            llm_temperature=request.llm_temperature,
        )

        def event_source() -> Iterator[str]:
            # The try/except below turns a mid-stream domain error into a clean SSE
            # `error` event instead of a raw 500 the client would otherwise never see
            # — the HTTP status is already committed by the time an error can occur
            # here (partway through an already-started streamed response).
            try:
                for event in pipeline.answer_stream(
                    request.question,
                    overrides,
                    filenames=request.filenames,
                    history=_history_messages(request.history),
                ):
                    if event["type"] == "done":
                        questions_total.labels(outcome="ok").inc()
                        observe_question_timings(event["timings"])
                        _log_question(
                            provider=request.llm_provider,
                            source_count=len(event["sources"]),
                            timings=event["timings"],
                        )
                    yield _sse_event(dict(event))
            except Exception as exc:
                questions_total.labels(outcome="error").inc()
                yield _sse_event({"type": "error", "detail": str(exc)})

        return StreamingResponse(event_source(), media_type="text/event-stream")


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
    app.add_exception_handler(JobNotFoundError, not_found_handler)
    app.add_exception_handler(TraceNotFoundError, not_found_handler)
    app.add_exception_handler(DocumentNotFoundError, not_found_handler)


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


async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 404 response for a referenced resource that doesn't exist."""

    return error_response(404, exc)


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
