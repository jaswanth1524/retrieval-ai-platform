"""FastAPI application for DocRAG."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from api.dependencies import (
    get_app_settings,
    get_embedding_provider,
    get_reranker,
    get_token_counter,
    shutdown_executors,
)
from api.embeddings import EmbeddedText
from api.error_handlers import register_exception_handlers
from api.logging_config import configure_logging
from api.request_guard import RequestGuardMiddleware
from api.routes import documents as documents_routes
from api.routes import feedback as feedback_routes
from api.routes import ops as ops_routes
from api.routes import questions as questions_routes
from api.routes import search as search_routes
from api.routes import traces as traces_routes
from api.settings import AppSettings
from api.version import app_version

logger = logging.getLogger(__name__)


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
        version=app_version(),
        description="Self-hostable document Q&A API with hybrid retrieval and citations.",
        lifespan=lifespan,
    )
    settings = get_app_settings()
    configure_logging(settings.log_level)
    warn_on_risky_reranker_config(settings)
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


def register_routes(app: FastAPI) -> None:
    """Register API routes, in the order they match (the static frontend mount comes last)."""

    ops_routes.register(app)
    documents_routes.register(app)
    feedback_routes.register(app)
    traces_routes.register(app)
    questions_routes.register(app)
    search_routes.register(app)


app = create_app()
