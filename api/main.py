"""FastAPI application for DocRAG."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Protocol

import grpc
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from starlette.staticfiles import StaticFiles

from api.dependencies import (
    get_app_settings,
    get_embedding_provider,
    get_feedback_store,
    get_ingest_backlog,
    get_ingest_executor,
    get_ingest_job_store,
    get_ingest_service,
    get_ollama_reachability_checker,
    get_qdrant_client,
    get_qdrant_reachability_checker,
    get_rag_pipeline,
    get_raw_document_store,
    get_reranker,
    get_token_counter,
    get_trace_store,
    get_vector_repository,
    require_api_key,
    shutdown_executors,
)
from api.documents import (
    CHUNKER_VERSION,
    DocumentError,
    DocumentNotFoundError,
    normalize_filename,
    validate_upload_filename,
)
from api.embeddings import EmbeddedText, EmbeddingError
from api.feedback import FeedbackStore
from api.generation import ChatMessage, GenerationConfigError, GenerationError, StageTimings
from api.ingestion import IngestionError, filename_write_lock
from api.jobs import INTERRUPTED_JOB_ERROR, IngestBacklog, JobNotFoundError, JobStore
from api.logging_config import configure_logging
from api.metrics import (
    NO_ERROR_TYPE,
    ingest_chunks_total,
    ingest_job_seconds,
    ingest_jobs_total,
    observe_question_timings,
    questions_total,
)
from api.pipeline import AnswerOverrides, IngestService, RagPipeline, timings_dict
from api.qdrant_schema import CollectionSchemaError, VectorStoreUnavailableError
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.reranking import RerankingError
from api.retrieval import RetrievalConfigError, RetrievalError, RetrievalPayloadError
from api.schemas import (
    FEEDBACK_LIST_MAX_LIMIT,
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
    DocumentReindexAllResponse,
    FeedbackItemResponse,
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
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
IngestJobStoreDep = Annotated[JobStore, Depends(get_ingest_job_store)]
RawDocumentStoreDep = Annotated[RawDocumentStore | None, Depends(get_raw_document_store)]
FeedbackStoreDep = Annotated[FeedbackStore | None, Depends(get_feedback_store)]
IngestExecutorDep = Annotated[ThreadPoolExecutor, Depends(get_ingest_executor)]
IngestBacklogDep = Annotated[IngestBacklog, Depends(get_ingest_backlog)]
TraceStoreDep = Annotated[TraceStore, Depends(get_trace_store)]


FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# The largest reranker_batch_size with a measured peak (2.66 GB over 50 candidates of
# ~448 tokens) equal to the loaded-model baseline — i.e. the point past which scoring
# starts costing memory on top of just having the model resident. See the field's
# docstring in api/settings.py for the full 8 / 16 / 50 -> 2.66 / 3.53 / 6.33 GB ladder.
MEASURED_SAFE_RERANKER_BATCH_SIZE = 8


def warn_on_risky_reranker_config(settings: AppSettings) -> None:
    """Log a startup warning for reranker batch settings that have OOM-killed before.

    Warnings, not validation errors, and deliberately so: peak reranker memory is
    absolute (see MEASURED_SAFE_RERANKER_BATCH_SIZE), but how much memory is *available*
    is deployment-specific — a 64 GB host is entitled to batch 50, and refusing to boot
    there would be wrong. What an operator actually lacks is the signal, since the config
    looks reasonable field-by-field and the failure only shows up as exit 137 partway
    through the first whole-corpus question.
    """

    batch_size = int(settings.reranker_batch_size)
    candidates = int(settings.rerank_candidates)

    if batch_size > MEASURED_SAFE_RERANKER_BATCH_SIZE:
        logger.warning(
            "RERANKER_BATCH_SIZE=%d exceeds the measured-safe %d. Peak reranker memory "
            "scales with this value alone (measured over 50 candidates of ~448 tokens: "
            "8 -> 2.66 GB, 16 -> 3.53 GB, 50 -> 6.33 GB) and 6.33 GB OOM-kills a default "
            "~6 GB Docker Desktop. Latency is flat across the range, so this buys no "
            "speed — confirm the container has the headroom.",
            batch_size,
            MEASURED_SAFE_RERANKER_BATCH_SIZE,
        )

    if batch_size > candidates:
        logger.warning(
            "RERANKER_BATCH_SIZE=%d exceeds RERANK_CANDIDATES=%d, so batching is a no-op "
            "— every pair goes through the cross-encoder in one forward pass. Lower the "
            "batch size to at most %d.",
            batch_size,
            candidates,
            candidates,
        )


class WarmupEmbeddingProvider(Protocol):
    """Minimal embedding surface the warmup step needs."""

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]: ...


class WarmupReranker(Protocol):
    """Minimal reranker surface the warmup step needs."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class WarmupTokenCounter(Protocol):
    """Minimal token-counter surface the warmup step needs."""

    def count(self, text: str) -> int: ...


def run_model_warmup(
    embedding_provider: WarmupEmbeddingProvider,
    reranker: WarmupReranker,
    token_counter: WarmupTokenCounter | None = None,
) -> None:
    """Force the lazy-loaded models to load once, logging duration or failure.

    Both models are constructed with ``lazy_load=True``, so the real download/load
    cost is paid on first use, not at construction — this runs that first use at boot
    instead of on a user's first request. The chunking tokenizer is a separate fetch
    (the first upload paid it otherwise). All three load concurrently, and each
    failure is logged on its own so one bad model name doesn't leave the others cold.
    Deliberately never raises: a warmup failure (e.g. a transient network blip
    fetching a model) must not prevent the app from starting and serving requests that
    might succeed once the model is retried lazily; it only means the
    misconfiguration is now visible in the startup log instead of silently deferred.
    """

    tasks: dict[str, Callable[[], object]] = {
        "embedding": lambda: embedding_provider.embed_texts(["warmup"]),
        "reranker": lambda: reranker.score("warmup", ["warmup"]),
    }
    if token_counter is not None:
        tasks["tokenizer"] = lambda: token_counter.count("warmup")

    start = time.monotonic()
    failed = False
    with ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="docrag-warmup") as pool:
        futures = {name: pool.submit(task) for name, task in tasks.items()}
    for name, future in futures.items():
        exc = future.exception()
        if exc is not None:
            failed = True
            logger.error(
                "Model warmup failed for the %s — this will resurface as an error on the "
                "first real request instead. Check DENSE_EMBEDDING_MODEL/"
                "SPARSE_EMBEDDING_MODEL/RERANKER_MODEL.",
                name,
                exc_info=exc,
            )
    if not failed:
        logger.info("Model warmup completed in %.1fs", time.monotonic() - start)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Optionally warm up lazy-loaded models once at startup (see AppSettings.warmup_models)."""

    settings = get_app_settings()
    if settings.warmup_models:
        await run_in_threadpool(
            run_model_warmup, get_embedding_provider(), get_reranker(), get_token_counter()
        )
    yield
    shutdown_executors()


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
    warn_on_risky_reranker_config(settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["GET", "POST", "DELETE"],
        # X-API-Key is not a CORS-simple header, so a cross-origin request carrying it
        # is preflighted — omitting it here made the middleware answer that preflight
        # with "Disallowed CORS headers" and every authenticated request failed before
        # it reached a route. Configuring API_KEY and CORS_ALLOW_ORIGINS together (the
        # separately-hosted-frontend fallback documented in .env.example) was therefore
        # impossible. The single-origin Docker path never preflights, which is why this
        # went unnoticed.
        allow_headers=["Content-Type", "X-API-Key"],
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


# Domain errors IngestService.ingest can actually raise (documents.py's DocumentError
# tree, ingestion.py, embeddings.py, qdrant_schema.py) — these carry a message that's
# safe and useful to return to a polling client. Anything else is an exception the
# ingest path wasn't built to anticipate, so its str() may carry internal detail
# (a path, a config value) that shouldn't leave the server; it gets a generic message
# instead, with the real detail already captured by the logger.warning below.
_INGEST_USER_FACING_ERRORS: tuple[type[Exception], ...] = (
    DocumentError,
    IngestionError,
    EmbeddingError,
    CollectionSchemaError,
    VectorStoreUnavailableError,
)


_SHUTTING_DOWN_ERROR = "The server is shutting down and could not start indexing. Retry the upload."


def _job_error_message(exc: Exception) -> str:
    if isinstance(exc, _INGEST_USER_FACING_ERRORS):
        return str(exc)
    return "Ingestion failed due to an unexpected server error."


def _ingest_error_type(exc: Exception) -> str:
    """Coarse error_type label for docrag_ingest_jobs_total{outcome="failed"}."""

    if isinstance(exc, DocumentError):
        return "document"
    if isinstance(exc, IngestionError):
        return "ingestion"
    if isinstance(exc, EmbeddingError):
        return "embedding"
    if isinstance(exc, CollectionSchemaError):
        return "collection_schema"
    if isinstance(exc, VectorStoreUnavailableError):
        return "vector_store_unavailable"
    return "unexpected"


# Ordered most-specific-first: several of these are subclasses of one another
# (RetrievalPayloadError/RetrievalConfigError < RetrievalError,
# GenerationConfigError < GenerationError), so a parent check must not run first.
_QUESTION_ERROR_TYPES: tuple[tuple[type[Exception], str], ...] = (
    (RetrievalPayloadError, "retrieval_payload"),
    (RetrievalConfigError, "retrieval_config"),
    (RetrievalError, "retrieval"),
    (GenerationConfigError, "generation_config"),
    (GenerationError, "generation"),
    (RerankingError, "reranking"),
    (EmbeddingError, "embedding"),
    (VectorStoreUnavailableError, "vector_store_unavailable"),
    (CollectionSchemaError, "collection_schema"),
)


def _question_error_type(exc: Exception) -> str:
    """Coarse error_type label for docrag_questions_total{outcome="error"}."""

    for exc_type, label in _QUESTION_ERROR_TYPES:
        if isinstance(exc, exc_type):
            return label
    return "unexpected"


# Derived from _QUESTION_ERROR_TYPES above, which now has two jobs: it orders the
# metric labels AND decides which messages are safe to return. Adding a type there for
# labeling silently widens what the streaming route is allowed to send verbatim, so
# only add types whose str() is an authored, client-safe message.
#
# The same domain errors register_exception_handlers maps to authored status codes, so
# their str() is already what a client sees on the synchronous /questions route. The
# streaming route needs this set spelled out separately: /questions lets an unanticipated
# exception fall through to Starlette's default 500 ("Internal Server Error", nothing
# leaked), but /questions/stream has already committed a 200 by the time it can fail, so
# it must catch everything — and without an allowlist that means putting an arbitrary
# str(exc) (a path, a config value) on the wire.
_QUESTION_USER_FACING_ERRORS: tuple[type[Exception], ...] = tuple(
    exc_type for exc_type, _ in _QUESTION_ERROR_TYPES
)


# The gRPC status codes that mean "couldn't reach it / gave up waiting" rather than
# "it answered and said no" — the gRPC-transport equivalent of the REST client's
# ResponseHandlingException.
_GRPC_UNAVAILABLE_CODES = frozenset(
    {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}
)


def _is_vector_store_unavailable(exc: Exception) -> bool:
    if isinstance(exc, ResponseHandlingException):
        return True
    # grpc.RpcError itself declares no code(); the concrete _InactiveRpcError raised in
    # practice does, so probe rather than assume.
    code = getattr(exc, "code", None)
    return callable(code) and code() in _GRPC_UNAVAILABLE_CODES


def _question_error_message(exc: Exception) -> str:
    if isinstance(exc, _QUESTION_USER_FACING_ERRORS):
        return str(exc)
    # A raw transport failure carries no authored message, but "unexpected server
    # error" would hide an actionable cause the synchronous route names properly.
    if _is_vector_store_unavailable(exc):
        return "The vector store is unavailable. Try again shortly."
    return "The question could not be answered due to an unexpected server error."


def _run_ingest_job(
    job_store: JobStore,
    job_id: str,
    ingest_service: IngestService,
    filename: str,
    content: bytes,
    raw_store: RawDocumentStore | None = None,
    uploaded_at: float | None = None,
    precondition: Callable[[], None] | None = None,
) -> None:
    """Background-executor entry point: parse/chunk/embed/index and update job status.

    Runs on ``get_ingest_executor()``'s worker threads, outside any request's
    lifetime — exceptions are caught and recorded on the job rather than raised,
    since there is no HTTP response left to attach them to. The original bytes (when
    ``raw_store`` is set) are saved only after indexing succeeds and under the same
    filename lock: saving them at upload time left orphans for uploads that then
    failed, and overwrote the original of a still-indexed older version.
    """

    def on_progress(done: int, total: int) -> None:
        job_store.update(job_id, state="embedding", chunks_done=done, chunks_total=total)

    def save_original() -> None:
        if raw_store is None:
            return
        try:
            raw_store.save(filename, content)
        except OSError:
            # The index is already updated; failing the job would report a document
            # as not indexed when it is. The original is an opt-in extra — but a
            # previous version's original must not outlive the index it described, or
            # /original would serve bytes that no longer match what is indexed. No
            # original (404) beats a wrong one.
            logger.exception("Indexed %r but could not store its original bytes.", filename)
            try:
                raw_store.delete(filename)
            except OSError:
                logger.exception(
                    "Could not remove the outdated original of %r either; "
                    "GET /documents/%s/original may serve the previous version.",
                    filename,
                    filename,
                )

    started = time.monotonic()
    try:
        job_store.update(job_id, state="parsing")
        outcome = ingest_service.ingest(
            filename,
            content,
            on_progress,
            save_original,
            uploaded_at=uploaded_at,
            precondition=precondition,
        )
    except Exception as exc:
        logger.warning("Ingest job %s for %r failed: %s", job_id, filename, exc, exc_info=True)
        ingest_job_seconds.labels(outcome="failed").observe(time.monotonic() - started)
        ingest_jobs_total.labels(outcome="failed", error_type=_ingest_error_type(exc)).inc()
        job_store.update(job_id, state="failed", error=_job_error_message(exc))
        return

    ingest_job_seconds.labels(outcome="done").observe(time.monotonic() - started)
    ingest_jobs_total.labels(outcome="done", error_type=NO_ERROR_TYPE).inc()
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


_ORIGINAL_CHUNK_BYTES = 64 * 1024

_BACKLOG_FULL_ERROR = (
    "Too many documents are already waiting to be indexed. Retry once some of them finish."
)


def _enqueue_ingest(
    *,
    job_store: JobStore,
    executor: ThreadPoolExecutor,
    backlog: IngestBacklog,
    ingest_service: IngestService,
    filename: str,
    content: bytes,
    raw_store: RawDocumentStore | None,
    uploaded_at: float | None = None,
    precondition: Callable[[], None] | None = None,
) -> DocumentJobAcceptedResponse:
    """Admit ``content`` against the backlog, create its job, and start it. Blocking.

    Shared by upload and re-index so both get the same backlog 503, the same
    shutting-down handling, and the same done-callback that frees the bytes.
    """

    size = len(content)
    if not backlog.try_reserve(size):
        raise HTTPException(
            status_code=503, detail=_BACKLOG_FULL_ERROR, headers={"Retry-After": "15"}
        )
    try:
        job = job_store.create(filename)
    except BaseException:
        backlog.release(size)
        raise
    try:
        # Ingestion runs on a dedicated executor (not FastAPI's request threadpool) so
        # it survives independently of this request/response cycle — the client can
        # disconnect and poll the job later without interrupting the work.
        future = executor.submit(
            _run_ingest_job,
            job_store,
            job.id,
            ingest_service,
            filename,
            content,
            raw_store,
            uploaded_at,
            precondition,
        )
    except RuntimeError as exc:
        # The executor is shut down (the process is stopping). The job already exists,
        # so fail it rather than leave it "queued" forever for a poller.
        backlog.release(size)
        job_store.update(job.id, state="failed", error=_SHUTTING_DOWN_ERROR)
        raise HTTPException(
            status_code=503, detail=_SHUTTING_DOWN_ERROR, headers={"Retry-After": "15"}
        ) from exc
    except BaseException:
        backlog.release(size)
        job_store.update(job.id, state="failed", error="Could not start indexing.")
        raise
    future.add_done_callback(
        lambda done: _finish_ingest_future(done, backlog, size, job_store, job.id)
    )
    return DocumentJobAcceptedResponse(job_id=job.id, filename=filename, state="queued")


def _original_still_stored(store: RawDocumentStore, filename: str) -> Callable[[], None]:
    """A re-index precondition: fail if the document was deleted while queued.

    Checked under the filename lock DELETE also takes (and DELETE removes the original
    under it), so "original gone" means the delete landed first — re-indexing anyway
    would silently bring the deleted document back.
    """

    def check() -> None:
        if store.path(filename) is None:
            raise IngestionError(
                f"'{filename}' was deleted before its re-index ran; nothing was re-indexed."
            )

    return check


def _require_raw_store(raw_store: RawDocumentStore | None) -> RawDocumentStore:
    """The raw store, or a 409: re-indexing reads originals only kept when it's on."""

    if raw_store is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Re-indexing needs stored originals: set RAW_DOCUMENT_DIR, then upload "
                "documents again once so their originals are kept."
            ),
        )
    return raw_store


def _finish_ingest_future(
    future: Future[None], backlog: IngestBacklog, size: int, job_store: JobStore, job_id: str
) -> None:
    """Release the job's backlog bytes, and surface anything that escaped the job.

    ``_run_ingest_job`` records failures on the job, but the job store itself can fail
    while doing so (sqlite disk full); nothing observed the future, so that left the
    job stuck in a non-terminal state with nothing in the log.

    A cancelled future is a job shutdown dropped from the queue before it started; it
    is failed here so a client still polling this process doesn't see "queued" forever.
    """

    backlog.release(size)
    if future.cancelled():
        try:
            job_store.update(job_id, state="failed", error=INTERRUPTED_JOB_ERROR)
        except Exception:
            logger.exception("Could not mark cancelled ingest job %s as failed.", job_id)
        return
    exc = future.exception()
    if exc is not None:
        logger.error("Ingest job %s crashed outside its own error handling.", job_id, exc_info=exc)


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
        query_expansion_ms=timings.query_expansion_ms,
        context_expansion_ms=timings.context_expansion_ms,
        citation_retry_ms=timings.citation_retry_ms,
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


def _log_question(
    *, provider: str | None, source_count: int, timings: dict[str, float]
) -> None:
    """Emit one structured log line per answered question for latency observability."""

    logger.info(
        "question answered provider=%s sources=%d condense_ms=%.1f embed_ms=%.1f "
        "search_ms=%.1f rerank_ms=%.1f context_expansion_ms=%.1f generate_ms=%.1f "
        "citation_retry_ms=%.1f total_ms=%.1f",
        provider or "default",
        source_count,
        timings.get("condense_ms", 0.0),
        timings["embed_ms"],
        timings["search_ms"],
        timings["rerank_ms"],
        timings.get("context_expansion_ms", 0.0),
        timings["generate_ms"],
        timings.get("citation_retry_ms", 0.0),
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
        repository: VectorRepositoryDep, settings: SettingsDep, raw_store: RawDocumentStoreDep
    ) -> DocumentListResponse:
        metadata = repository.filename_metadata(settings)
        return DocumentListResponse(
            filenames=sorted(metadata),
            chunk_counts={name: meta.chunk_count for name, meta in metadata.items()},
            page_counts={name: meta.page_count for name, meta in metadata.items()},
            byte_sizes={
                name: meta.byte_size
                for name, meta in metadata.items()
                if meta.byte_size is not None
            },
            uploaded_ats={
                name: meta.uploaded_at
                for name, meta in metadata.items()
                if meta.uploaded_at is not None
            },
            stale_filenames=sorted(
                name
                for name, meta in metadata.items()
                if meta.chunker_version is None or meta.chunker_version < CHUNKER_VERSION
            ),
            reindexable_filenames=sorted(
                name
                for name in metadata
                if raw_store is not None and raw_store.path(name) is not None
            ),
        )

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
        backlog: IngestBacklogDep,
        raw_store: RawDocumentStoreDep,
        file: Annotated[UploadFile, File()],
    ) -> DocumentJobAcceptedResponse:
        # Rejected here, not in the background job: an unusable name or type used to
        # be accepted with a 202 and only fail once the client polled the job.
        filename = validate_upload_filename(file.filename or "")
        content = await read_upload_within_limit(file, int(settings.max_upload_bytes))
        # job_store.create is a sqlite INSERT under JOB_STORE_BACKEND=sqlite; off the
        # event loop so it can't stall other in-flight requests.
        return await run_in_threadpool(
            _enqueue_ingest,
            job_store=job_store,
            executor=executor,
            backlog=backlog,
            ingest_service=ingest_service,
            filename=filename,
            content=content,
            raw_store=raw_store,
        )

    # Re-chunks and re-embeds a document from its stored original (RAW_DOCUMENT_DIR),
    # so a chunking/embedding change applies without the user re-uploading. Same job
    # flow as an upload: 202 + a job id to poll.
    @app.post(
        "/documents/reindex",
        response_model=DocumentReindexAllResponse,
        status_code=202,
        dependencies=[Depends(require_api_key)],
    )
    def reindex_stale_documents(
        repository: VectorRepositoryDep,
        ingest_service: IngestServiceDep,
        settings: SettingsDep,
        job_store: IngestJobStoreDep,
        executor: IngestExecutorDep,
        backlog: IngestBacklogDep,
        raw_store: RawDocumentStoreDep,
    ) -> DocumentReindexAllResponse:
        store = _require_raw_store(raw_store)
        metadata = repository.filename_metadata(settings)
        jobs: list[DocumentJobAcceptedResponse] = []
        deferred: list[str] = []
        for name, meta in sorted(metadata.items()):
            if meta.chunker_version is not None and meta.chunker_version >= CHUNKER_VERSION:
                continue
            content = store.read(name)
            if content is None:
                continue
            try:
                jobs.append(
                    _enqueue_ingest(
                        job_store=job_store,
                        executor=executor,
                        backlog=backlog,
                        ingest_service=ingest_service,
                        filename=name,
                        content=content,
                        raw_store=None,
                        uploaded_at=meta.uploaded_at,
                        precondition=_original_still_stored(store, name),
                    )
                )
            except HTTPException as exc:
                if exc.status_code != 503:
                    raise
                deferred.append(name)
        return DocumentReindexAllResponse(jobs=jobs, deferred=deferred)

    @app.post(
        "/documents/{filename}/reindex",
        response_model=DocumentJobAcceptedResponse,
        status_code=202,
        dependencies=[Depends(require_api_key)],
    )
    def reindex_document(
        filename: str,
        repository: VectorRepositoryDep,
        ingest_service: IngestServiceDep,
        settings: SettingsDep,
        job_store: IngestJobStoreDep,
        executor: IngestExecutorDep,
        backlog: IngestBacklogDep,
        raw_store: RawDocumentStoreDep,
    ) -> DocumentJobAcceptedResponse:
        store = _require_raw_store(raw_store)
        name = normalize_filename(filename)
        content = store.read(name)
        if content is None:
            raise DocumentNotFoundError(f"No stored original for '{name}' to re-index.")
        return _enqueue_ingest(
            job_store=job_store,
            executor=executor,
            backlog=backlog,
            ingest_service=ingest_service,
            filename=name,
            content=content,
            # The original is already stored, byte-identical. Passing the store would
            # re-save it, and a failed save deletes it — losing the only copy.
            raw_store=None,
            uploaded_at=repository.uploaded_at_for_filename(settings, name),
            precondition=_original_still_stored(store, name),
        )

    @app.delete(
        "/documents/{filename}",
        response_model=DocumentDeleteResponse,
        dependencies=[Depends(require_api_key)],
    )
    def delete_document(
        filename: str,
        repository: VectorRepositoryDep,
        settings: SettingsDep,
        raw_store: RawDocumentStoreDep,
    ) -> DocumentDeleteResponse:
        # Filenames are always basenames (normalize_filename strips any path
        # component at upload time), so a plain path segment is sufficient — a
        # filename containing "/" simply can't match this route, which is fine
        # since one can never have been uploaded.
        #
        # Same lock the ingest path takes: snapshot-then-delete is a read-modify-write,
        # so without it a delete landing between a concurrent ingest's upsert and its
        # stale-cleanup would remove the points that ingest just wrote, and the ingest
        # would still report success. The raw copy (when stored) is removed under the
        # same lock, or a deleted document's original bytes would accumulate forever.
        with filename_write_lock(filename):
            point_ids = repository.point_ids_for_filename(settings, filename)
            if point_ids:
                repository.delete_by_ids(settings, point_ids)
            # Also when no points remain: an original orphaned by an earlier failed
            # ingest was otherwise undeletable, since the 404 came first.
            removed_original = raw_store.delete(filename) if raw_store is not None else False
            if not point_ids and not removed_original:
                raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
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

    # Guarded at least as strictly as GET /documents/{filename}/content above: raw
    # original bytes are strictly more sensitive than derived chunk text. Returns 404
    # both when raw_document_dir is unset (feature off) and when it's set but this
    # particular filename predates the feature or was never stored — the client sees
    # the same "not found" either way, not a distinct "feature unavailable" state.
    @app.get("/documents/{filename}/original", dependencies=[Depends(require_api_key)])
    def document_original(filename: str, raw_store: RawDocumentStoreDep) -> Response:
        path = raw_store.path(filename) if raw_store is not None else None
        if path is None:
            raise DocumentNotFoundError(f"No stored original for '{filename}'.")
        # Open before building the response. FileResponse re-stats and opens the path
        # only when the response is sent, so a DELETE landing in between turned this
        # 404 into a 500; an already-open handle keeps streaming the bytes it opened
        # even if the file is unlinked meanwhile.
        try:
            handle = path.open("rb")
        except FileNotFoundError:
            raise DocumentNotFoundError(f"No stored original for '{filename}'.") from None
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        # Headers from FileResponse (given the stat, it does no I/O of its own): it
        # RFC 5987-encodes the name — a hand-built latin-1 header 500'd on any
        # non-Latin-1 filename and broke on one containing a quote.
        template = FileResponse(
            path,
            media_type=media_type,
            filename=normalize_filename(filename),
            stat_result=os.fstat(handle.fileno()),
            headers={"X-Content-Type-Options": "nosniff"},
        )
        headers = {
            name: value
            for name, value in template.headers.items()
            # Ranges need seeking the streamed body can't do; its own type is set below.
            if name not in ("accept-ranges", "content-type")
        }

        def chunks() -> Iterator[bytes]:
            with handle:
                while block := handle.read(_ORIGINAL_CHUNK_BYTES):
                    yield block

        return StreamingResponse(chunks(), media_type=media_type, headers=headers)

    # Off by default (AppSettings.feedback_enabled) — a deployment that hasn't
    # opted in gets the same 404 shape as any other absent route, not a distinct
    # "feature unavailable" response.
    @app.post(
        "/feedback",
        response_model=FeedbackResponse,
        status_code=201,
        dependencies=[Depends(require_api_key)],
    )
    def submit_feedback(
        request: FeedbackRequest, feedback_store: FeedbackStoreDep
    ) -> FeedbackResponse:
        if feedback_store is None:
            raise HTTPException(status_code=404, detail="Feedback capture is not enabled.")
        feedback = feedback_store.add(
            trace_id=request.trace_id,
            question=request.question,
            answer_excerpt=request.answer_excerpt,
            cited_filenames=request.cited_filenames,
            rating=request.rating,
            citation_source_number=request.citation_source_number,
        )
        return FeedbackResponse(id=feedback.id)

    # The read side of the same feature: without it, captured ratings are reachable
    # only by opening the sqlite file by hand, which defeats the point of capturing
    # them (thumbs-down turns are meant to become negative retrieval examples).
    @app.get(
        "/feedback",
        response_model=FeedbackListResponse,
        dependencies=[Depends(require_api_key)],
    )
    def list_feedback(
        feedback_store: FeedbackStoreDep,
        limit: Annotated[int, Query(ge=1, le=FEEDBACK_LIST_MAX_LIMIT)] = 100,
    ) -> FeedbackListResponse:
        if feedback_store is None:
            raise HTTPException(status_code=404, detail="Feedback capture is not enabled.")
        return FeedbackListResponse(
            feedback=[
                FeedbackItemResponse(
                    id=item.id,
                    trace_id=item.trace_id,
                    question=item.question,
                    answer_excerpt=item.answer_excerpt,
                    cited_filenames=item.cited_filenames,
                    rating=item.rating,
                    citation_source_number=item.citation_source_number,
                    created_at=item.created_at,
                )
                for item in feedback_store.list_recent(limit)
            ]
        )

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
        except Exception as exc:
            questions_total.labels(outcome="error", error_type=_question_error_type(exc)).inc()
            raise
        questions_total.labels(outcome="ok", error_type=NO_ERROR_TYPE).inc()
        if grounded.timings is not None:
            timings = timings_dict(grounded.timings)
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
                        questions_total.labels(outcome="ok", error_type=NO_ERROR_TYPE).inc()
                        observe_question_timings(event["timings"])
                        _log_question(
                            provider=request.llm_provider,
                            source_count=len(event["sources"]),
                            timings=event["timings"],
                        )
                    yield _sse_event(dict(event))
            except Exception as exc:
                questions_total.labels(
                    outcome="error", error_type=_question_error_type(exc)
                ).inc()
                logger.warning("Streamed question failed: %s", exc, exc_info=True)
                yield _sse_event({"type": "error", "detail": _question_error_message(exc)})

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
        feedback_enabled=settings.feedback_enabled,
        raw_documents_enabled=bool(settings.raw_document_dir),
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
    app.add_exception_handler(ResponseHandlingException, vector_store_transport_handler)
    # grpcio is a hard dependency of qdrant-client and nothing else here speaks gRPC,
    # so this is effectively "any Qdrant gRPC failure" despite the broad type.
    app.add_exception_handler(grpc.RpcError, vector_store_transport_handler)
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


async def vector_store_transport_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map a raw Qdrant transport failure to 503 instead of a bare 500.

    Two gaps, one handler. First, VectorRepository.hybrid_search translates
    ResponseHandlingException into VectorStoreUnavailableError itself but no other
    repository method does, and ensure_ready can't cover for them because
    ensure_collection caches success per client — so an outage starting after the
    first successful request made GET /documents, GET /documents/{f}/content and
    DELETE /documents/{f} return bare 500s.

    Second, and found only by running this against a real container: those existing
    translations are REST-only. Under QDRANT_PREFER_GRPC=true — which docker-compose
    sets by default — an unreachable Qdrant raises grpc.RpcError, which
    ResponseHandlingException never matches, so even /questions returned a bare 500 on
    the project's own default deployment path.

    Messages are fixed rather than str(exc): this is a blanket handler over an
    exception type the app doesn't own, so it also catches bugs that merely surface as
    one, where naming the vector store as the cause would be wrong. The real cause is
    logged.
    """

    logger.warning("Vector store transport failure: %s", exc, exc_info=True)
    if _is_vector_store_unavailable(exc):
        return JSONResponse(
            status_code=503,
            content={"detail": "The vector store is unavailable. Try again shortly."},
        )
    # A gRPC error that isn't a transport failure (a malformed query, say) is a bug,
    # not an outage — still redacted and logged, but not mislabeled as unavailability.
    return JSONResponse(
        status_code=500,
        content={"detail": "The request failed due to an unexpected vector store error."},
    )


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
