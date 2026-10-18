"""FastAPI application for DocRAG."""

from __future__ import annotations

import logging
import mimetypes
import time
from collections.abc import AsyncIterator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import (
    get_app_settings,
    get_embedding_provider,
    get_ingest_backlog,
    get_ingest_executor,
    get_ingest_job_store,
    get_qdrant_client,
    get_raw_document_store,
    get_reranker,
    get_token_counter,
    shutdown_executors,
)
from api.embeddings import EmbeddedText
from api.error_handlers import register_exception_handlers
from api.ingest_jobs import (
    delete_indexed_document,
    enqueue_ingest,
    known_filenames,
    refuse_near_duplicate_name,
)
from api.ingest_service import IngestService
from api.llm import GenerationConfigError, completion_model_and_kwargs
from api.logging_config import configure_logging
from api.metrics import build_info
from api.qdrant_schema import check_dense_dimension
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.request_guard import RequestGuardMiddleware
from api.request_id import RequestIdMiddleware
from api.routes import documents as documents_routes
from api.routes import feedback as feedback_routes
from api.routes import ops as ops_routes
from api.routes import questions as questions_routes
from api.routes import search as search_routes
from api.routes import traces as traces_routes
from api.routes import workspaces as workspaces_routes
from api.security_headers import SecurityHeadersMiddleware
from api.settings import AppSettings
from api.static_files import FrontendFiles
from api.storage import check_writable_storage
from api.version import app_version
from api.watch import FolderWatcher
from api.workspaces import (
    WorkspaceError,
    settings_for,
    validate_name,
    workspace_exists,
    workspace_of,
)

logger = logging.getLogger(__name__)


FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
# The UI's bundled fonts. python:3.12-slim has no /etc/mime.types, and Python's own table
# lacks .woff2, so StaticFiles served them as application/octet-stream.
mimetypes.add_type("font/woff2", ".woff2")


# The largest reranker_batch_size with a measured peak (2.66 GB over 50 candidates of
# ~448 tokens) equal to the loaded-model baseline — i.e. the point past which scoring
# starts costing memory on top of just having the model resident. See the field's
# docstring in api/settings.py for the full 8 / 16 / 50 -> 2.66 / 3.53 / 6.33 GB ladder.
MEASURED_SAFE_RERANKER_BATCH_SIZE = 8


def warn_on_unconfigured_provider(settings: AppSettings) -> None:
    """Log at startup when the default LLM provider can't be used as configured
    (``LLM_PROVIDER=openai`` with no key, an OpenAI-compatible server with no URL).

    A warning, not a refusal: upload, indexing and passage search never need the LLM,
    and another provider may still be picked per question. Unwarned, the first sign was
    every question failing.
    """

    try:
        completion_model_and_kwargs(settings)
    except GenerationConfigError as exc:
        logger.warning("Questions will fail until this is fixed: %s", exc)


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
    embedding_provider: WarmupEmbeddingProvider | None,
    reranker: WarmupReranker | None,
    token_counter: WarmupTokenCounter | None = None,
    settings: AppSettings | None = None,
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

    def warm_embedding() -> None:
        assert embedding_provider is not None
        embedded = embedding_provider.embed_texts(["warmup"])
        # A wrong QDRANT_DENSE_VECTOR_SIZE named at boot, not at the first upload.
        if settings is not None:
            check_dense_dimension(len(embedded[0].dense), settings)

    tasks: dict[str, Callable[[], object]] = {}
    if embedding_provider is not None:
        tasks["embedding"] = warm_embedding
    if reranker is not None:
        tasks["reranker"] = lambda: reranker.score("warmup", ["warmup"])
    if token_counter is not None:
        tasks["tokenizer"] = lambda: token_counter.count("warmup")
    if not tasks:
        return

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


def _build_for_warmup[T](name: str, factory: Callable[[], T]) -> T | None:
    """Construct one warmup model, logging instead of raising (a misspelled model name
    must not stop the app from booting — the same error reaches the first request)."""

    try:
        return factory()
    except Exception:  # noqa: BLE001 - warmup never raises, see run_model_warmup
        logger.error(
            "Could not construct the %s for warmup; check DENSE_EMBEDDING_MODEL/"
            "SPARSE_EMBEDDING_MODEL/RERANKER_MODEL.",
            name,
            exc_info=True,
        )
        return None


def watch_target(settings: AppSettings) -> AppSettings:
    """The settings of the workspace WATCH_DIR indexes into (WATCH_WORKSPACE; the app's
    own for the default). A bad name, or a workspace with workspaces off, stops startup
    with the setting named — rather than a watcher quietly indexing somewhere else."""

    name = settings.watch_workspace.strip()
    if not name:
        return settings
    if not settings.workspaces_enabled:
        raise RuntimeError("WATCH_WORKSPACE needs WORKSPACES_ENABLED=true.")
    try:
        validate_name(name)
    except WorkspaceError as exc:
        raise RuntimeError(f"WATCH_WORKSPACE: {exc}") from exc
    return settings_for(settings, name)


def start_folder_watcher(settings: AppSettings) -> FolderWatcher:
    """Start indexing WATCH_DIR into WATCH_WORKSPACE (api/watch.py)."""

    target = watch_target(settings)
    repository = VectorRepository(get_qdrant_client())
    raw_store = get_raw_document_store()
    if target is not settings:
        # Named in the configuration, so made here if it doesn't exist yet (a request
        # never creates a workspace by naming it).
        name = workspace_of(target)
        if not workspace_exists(get_qdrant_client(), settings.qdrant_collection, name):
            logger.info("Creating workspace %r for WATCH_DIR.", name)
        repository.ensure_writable(target, create_workspace=True)
        if raw_store is not None and target.raw_document_dir:
            raw_store = RawDocumentStore(target.raw_document_dir)
    settings = target
    service = IngestService(repository, get_embedding_provider(), settings, get_token_counter())

    def list_indexed() -> dict[str, float | None]:
        metadata = repository.filename_metadata(settings)
        return {name: meta.uploaded_at for name, meta in metadata.items()}

    def enqueue(filename: str, content: bytes, mtime: float) -> None:
        if raw_store is not None:
            # As an upload is: with originals stored, Report.pdf and report.pdf would
            # overwrite each other's original on a case-insensitive disk. The
            # FilenameConflictError is logged by the watcher and the file skipped.
            refuse_near_duplicate_name(filename, known_filenames(repository, settings))
        enqueue_ingest(
            job_store=get_ingest_job_store(),
            executor=get_ingest_executor(),
            backlog=get_ingest_backlog(),
            ingest_service=service,
            filename=filename,
            content=content,
            raw_store=raw_store,
            # The file's mtime, not the time the job runs: it is how the watcher tells
            # an indexed document is this exact version of the file (api/watch.py).
            uploaded_at=mtime,
        )

    def delete(filename: str) -> None:
        delete_indexed_document(repository, settings, raw_store, filename)

    watcher = FolderWatcher(settings, list_indexed, enqueue, delete)
    watcher.start()
    return watcher


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Optionally warm up lazy-loaded models once at startup (see AppSettings.warmup_models)."""

    settings = get_app_settings()
    check_writable_storage(settings)
    if settings.warmup_models:
        await run_in_threadpool(
            run_model_warmup,
            _build_for_warmup("embedding models", get_embedding_provider),
            _build_for_warmup("reranker", get_reranker),
            get_token_counter(),
            settings,
        )
    watcher = start_folder_watcher(settings) if settings.watch_dir else None
    try:
        yield
    finally:
        if watcher is not None:
            watcher.stop()
    shutdown_executors()


def create_app() -> FastAPI:
    """Create the FastAPI app."""

    build_info.info({"version": app_version()})
    app = FastAPI(
        title="DocRAG API",
        version=app_version(),
        description="Self-hostable document Q&A API with hybrid retrieval and citations.",
        lifespan=lifespan,
    )
    settings = get_app_settings()
    configure_logging(settings.log_level, settings.log_format)
    warn_on_risky_reranker_config(settings)
    warn_on_unconfigured_provider(settings)
    # Inside CORS (add_middleware wraps outward), so its 401/413 answers still carry CORS
    # headers for a separately-hosted frontend.
    app.add_middleware(RequestGuardMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        # X-API-Key is not a CORS-simple header, so a cross-origin request carrying it
        # is preflighted — omitting it here made the middleware answer that preflight
        # with "Disallowed CORS headers" and every authenticated request failed before
        # it reached a route. Configuring API_KEY and CORS_ALLOW_ORIGINS together (the
        # separately-hosted-frontend fallback documented in .env.example) was therefore
        # impossible. The single-origin Docker path never preflights, which is why this
        # went unnoticed.
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID", "X-Workspace"],
        expose_headers=["X-Request-ID"],
    )
    # Outside CORS and the guard, so their rejections carry the headers too.
    app.add_middleware(SecurityHeadersMiddleware)
    # Outermost, so even a CORS or guard rejection carries the id it was logged under.
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    register_routes(app)

    # Registered last: Starlette matches routes in registration order, so the API
    # routes above are matched before this catch-all mount ever sees the request.
    # Guarded by existence so a plain `uv run uvicorn api.main:app` (no frontend
    # build) still boots instead of crashing on StaticFiles' directory check.
    if FRONTEND_DIST_DIR.is_dir():
        app.mount("/", FrontendFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")

    return app


def register_routes(app: FastAPI) -> None:
    """Register API routes, in the order they match (the static frontend mount comes last)."""

    ops_routes.register(app)
    documents_routes.register(app)
    feedback_routes.register(app)
    traces_routes.register(app)
    questions_routes.register(app)
    search_routes.register(app)
    workspaces_routes.register(app)


app = create_app()
