"""FastAPI application for DocRAG."""

from __future__ import annotations

import logging
import mimetypes
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from api.audit import audit, set_audit_store
from api.auto_reindex import StaleReindexer
from api.batch_jobs import batch_job_store
from api.chunking import TokenCounter
from api.dependencies import (
    get_app_settings,
    get_audit_log,
    get_embedding_provider,
    get_feedback_store,
    get_ingest_backlog,
    get_ingest_executor,
    get_ingest_job_store,
    get_qdrant_client,
    get_raw_document_store,
    get_reranker,
    get_token_counter,
    get_url_refresh_state_store,
    raw_store_for,
    shutdown_executors,
)
from api.doctor import diagnose
from api.document_names import (
    known_filenames,
    refuse_near_duplicate_name,
)
from api.document_ops import (
    delete_indexed_document,
    enqueue_refetched,
    refuse_if_replaced,
    reindex_from_original,
    retry_refusal,
)
from api.documents import normalize_tags
from api.embeddings import EmbeddedText
from api.error_handlers import register_exception_handlers
from api.imports import leftover, spool_directory
from api.imports import shutdown as shutdown_background_imports
from api.imports import start as start_import
from api.ingest_jobs import (
    JOB_CANCELLED_ERROR,
    IngestQueue,
)
from api.ingest_service import IngestService
from api.llm import GenerationConfigError, completion_model_and_kwargs
from api.logging_config import configure_logging
from api.metrics import backup_enabled, build_info, watch_index, watch_storage
from api.model_health import ModelHealth, ModelState
from api.notify import Notifier, notify, set_notifier
from api.qdrant_schema import check_dense_dimension
from api.repository import VectorRepository
from api.request_guard import RequestGuardMiddleware
from api.request_id import RequestIdMiddleware
from api.restore import DrillReport
from api.routes import backup as backup_routes
from api.routes import documents as documents_routes
from api.routes import documents_write as documents_write_routes
from api.routes import feedback as feedback_routes
from api.routes import ops as ops_routes
from api.routes import questions as questions_routes
from api.routes import search as search_routes
from api.routes import traces as traces_routes
from api.routes import watch as watch_routes
from api.routes import workspaces as workspaces_routes
from api.scheduled_backup import ScheduledBackup
from api.security_headers import SecurityHeadersMiddleware
from api.settings import AppSettings
from api.sitemap import shutdown_imports
from api.static_files import FrontendFiles
from api.storage import check_writable_storage, enabled_store_directories, require_free_space
from api.url_fetch import fetch_document, fetch_slots
from api.url_refresh import UrlRefresher
from api.version import app_version
from api.watch import FolderWatcher, WatchedJob, watch_folders
from api.workspace_target import WorkspaceTarget
from api.workspaces import (
    WorkspaceError,
    list_workspaces,
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
) -> dict[str, ModelState]:
    """Force the lazy-loaded models to load once, logging duration or failure; each
    warmed model's state (``api/model_health.py``).

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
        return {}

    start = time.monotonic()
    failed = False
    states: dict[str, ModelState] = {}
    with ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="docrag-warmup") as pool:
        futures = {name: pool.submit(task) for name, task in tasks.items()}
    for name, future in futures.items():
        exc = future.exception()
        states[name] = "ok" if exc is None else "failed"
        if exc is not None:
            failed = True
            logger.error(
                "Model warmup failed for the %s — this will resurface as an error on the "
                "first real request instead. Check DENSE_EMBEDDING_MODEL/"
                "SPARSE_EMBEDDING_MODEL/RERANKER_MODEL.",
                name,
                exc_info=exc,
            )
    # The HF tokenizer falls back to the heuristic counter instead of raising.
    if token_counter is not None and getattr(token_counter, "mode", None) == "heuristic":
        states["tokenizer"] = "fallback"
    if not failed:
        logger.info("Model warmup completed in %.1fs", time.monotonic() - start)
    return states


def warm_models(
    settings: AppSettings, names: frozenset[str] | None = None
) -> dict[str, ModelState]:
    """Build and warm the models (``names`` only, when given). A model that can't even
    be constructed counts as failed."""

    def wanted(name: str) -> bool:
        return names is None or name in names

    embedding = (
        _build_for_warmup("embedding models", get_embedding_provider)
        if wanted("embedding")
        else None
    )
    reranker = _build_for_warmup("reranker", get_reranker) if wanted("reranker") else None
    token_counter = get_token_counter() if wanted("tokenizer") else None
    states = run_model_warmup(embedding, reranker, token_counter, settings)
    if wanted("embedding") and embedding is None:
        states["embedding"] = "failed"
    if wanted("reranker") and reranker is None:
        states["reranker"] = "failed"
    return states


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
        # Reserved names too: "default" here would make a workspace called "default",
        # not use the default one (leave the setting empty for that).
        validate_name(name, new=True)
    except WorkspaceError as exc:
        raise RuntimeError(f"WATCH_WORKSPACE: {exc}") from exc
    return settings_for(settings, name)


def start_folder_watcher(settings: AppSettings) -> FolderWatcher:
    """Start indexing WATCH_DIR into WATCH_WORKSPACE (api/watch.py)."""

    target = watch_target(settings)
    repository = VectorRepository(get_qdrant_client())
    if target is not settings:
        # Named in the configuration, so made here if it doesn't exist yet (a request
        # never creates a workspace by naming it).
        name = workspace_of(target)
        if not workspace_exists(get_qdrant_client(), settings.qdrant_collection, name):
            logger.info("Creating workspace %r for WATCH_DIR.", name)
        repository.ensure_writable(target, create_workspace=True)
    # The same store requests use for that workspace (one per process, not a second).
    raw_store = raw_store_for(target)
    settings = target
    folder = str(Path(settings.watch_dir).resolve())
    service = IngestService(
        repository,
        get_embedding_provider(),
        settings,
        get_token_counter(),
        watched_from=folder,
    )

    def list_indexed() -> dict[str, float | None]:
        metadata = repository.filename_metadata(settings)
        return {name: meta.uploaded_at for name, meta in metadata.items()}

    def list_owned() -> set[str]:
        # Indexed from this folder in an earlier run: still the watcher's to delete
        # when the file is gone, though this run hasn't seen it (deleted while the
        # API was down).
        metadata = repository.filename_metadata(settings)
        return {name for name, meta in metadata.items() if meta.watched_from == folder}

    def enqueue(filename: str, content: bytes, mtime: float) -> str:
        # As an upload is: nothing more kept on a nearly full disk (a 507, which the
        # watcher waits out like a full backlog).
        require_free_space(settings, settings.raw_document_dir)
        if raw_store is not None:
            # As an upload is: with originals stored, Report.pdf and report.pdf would
            # overwrite each other's original on a case-insensitive disk. The
            # FilenameConflictError is logged by the watcher and the file skipped.
            refuse_near_duplicate_name(filename, known_filenames(repository, settings))
        queue = IngestQueue(
            get_ingest_job_store(), get_ingest_executor(), get_ingest_backlog(), service
        )
        accepted = queue.enqueue(
            filename=filename,
            content=content,
            raw_store=raw_store,
            # The file's mtime, not the time the job runs: it is how the watcher tells
            # an indexed document is this exact version of the file (api/watch.py).
            uploaded_at=mtime,
            # A file only touched (or copied over with the same bytes) isn't re-embedded:
            # its document just takes the new mtime.
            dedupe=True,
            kind="watch",
        )
        audit(
            "document.upload",
            workspace=workspace_of(settings),
            filename=filename,
            size=len(content),
            job_id=accepted.job_id,
            source="watch",
        )
        return accepted.job_id

    def job_lookup(job_id: str) -> WatchedJob | None:
        job = get_ingest_job_store().get(job_id)
        if job is None:
            return None
        return WatchedJob(
            job.state,
            job.error,
            job.error == JOB_CANCELLED_ERROR,
            # The file itself is the problem: retrying the same bytes can't help.
            permanent=job.error_type == "document",
        )

    def delete(filename: str) -> None:
        delete_indexed_document(repository, settings, raw_store, filename)

    watcher = FolderWatcher(
        settings, list_indexed, enqueue, delete, job_lookup, list_owned=list_owned
    )
    watcher.start()
    return watcher


def workspace_target(
    settings: AppSettings, repository: VectorRepository, token_counter: TokenCounter
) -> WorkspaceTarget:
    """What a background worker needs to index into ``settings``' workspace."""

    service = IngestService(repository, get_embedding_provider(), settings, token_counter)
    queue = IngestQueue(
        get_ingest_job_store(), get_ingest_executor(), get_ingest_backlog(), service
    )
    return WorkspaceTarget(settings, raw_store_for(settings), queue)


def workspace_targets(
    settings: AppSettings, repository: VectorRepository, token_counter: TokenCounter
) -> Callable[[], list[WorkspaceTarget]]:
    """The default workspace and every other one (with WORKSPACES_ENABLED), listed again
    on each call: what the stale re-indexer and the URL refresher go over each pass.
    Originals stores are the cached ones requests use, not a new store per pass."""

    def targets() -> list[WorkspaceTarget]:
        found = [workspace_target(settings, repository, token_counter)]
        if settings.workspaces_enabled:
            for name in list_workspaces(get_qdrant_client(), settings.qdrant_collection):
                found.append(
                    workspace_target(settings_for(settings, name), repository, token_counter)
                )
        return found

    return targets


def start_scheduled_backup(settings: AppSettings) -> ScheduledBackup:
    """Start BACKUP_DIR's backups (api/scheduled_backup.py) of every workspace."""

    repository = VectorRepository(get_qdrant_client())
    targets = workspace_targets(settings, repository, get_token_counter())
    counter = get_token_counter()

    def drill(path: Path, workspace_settings: AppSettings) -> DrillReport:
        return backup_routes.drill_file(
            path,
            workspace_settings,
            counter,
            int(workspace_settings.backup_drill_sample),
            originals_only=True,
        )

    backup = ScheduledBackup(
        settings,
        repository,
        targets,
        get_feedback_store(),
        backup_routes._export_running,
        drill=drill,
    )
    backup.start()
    backup_enabled.set(1)
    return backup


def requeue_interrupted_jobs(settings: AppSettings) -> list[str]:
    """Start again, once, each re-index and URL fetch the last restart cut off
    (REQUEUE_INTERRUPTED_JOBS; the sqlite job store remembers which they were): the new
    jobs' ids. Uploads, imports and version restores can't be (their bytes weren't
    kept); a job that can't be queued now is logged and left failed, for its Retry."""

    interrupted = getattr(get_ingest_job_store(), "interrupted_at_open", ())
    if not interrupted:
        return []
    store = get_ingest_job_store()
    repository = VectorRepository(get_qdrant_client())
    token_counter = get_token_counter()
    started: list[str] = []
    for job_id in interrupted:
        job = store.get(job_id)
        if job is None or job.kind not in ("reindex", "url") or retry_refusal(job) is not None:
            continue
        workspace_settings = settings_for(settings, job.workspace) if job.workspace else settings
        target = workspace_target(workspace_settings, repository, token_counter)
        queue = target.queue
        if queue is None:  # pragma: no cover - workspace_target always builds one
            continue
        try:
            if job.kind == "reindex":
                accepted = reindex_from_original(
                    job.filename, repository, workspace_settings, queue, target.raw_store
                )
            else:
                if not settings.url_ingest_enabled or job.source_url is None:
                    continue
                meta = repository.document_metadata(workspace_settings, job.filename)
                refuse_if_replaced(job, meta)
                with fetch_slots:
                    fetched = fetch_document(
                        job.source_url,
                        workspace_settings,
                        filename=job.filename,
                        has_room=queue.backlog.has_room,
                    )
                accepted = enqueue_refetched(
                    job,
                    repository,
                    workspace_settings,
                    queue,
                    target.raw_store,
                    fetched,
                    meta,
                )
        except Exception as exc:  # noqa: BLE001 - one job's failure must not stop the rest
            logger.warning(
                "Could not start the interrupted %s of %r again: %s", job.kind, job.filename, exc
            )
            continue
        audit(
            "job.retry",
            workspace=job.workspace,
            job_id=job.id,
            filename=job.filename,
            new_job_id=accepted.job_id,
            scheduled=True,
        )
        started.append(accepted.job_id)
    if started:
        logger.info("Started %d job(s) a restart had cut off again.", len(started))
    return started


def start_stale_reindexer(settings: AppSettings) -> StaleReindexer | None:
    """Start AUTO_REINDEX_STALE's background passes (api/auto_reindex.py), or None
    when there are no stored originals to re-index from."""

    if get_raw_document_store() is None:
        return None
    repository = VectorRepository(get_qdrant_client())
    token_counter = get_token_counter()
    targets = workspace_targets(settings, repository, token_counter)
    reindexer = StaleReindexer(settings, repository, targets, get_ingest_job_store(), token_counter)
    reindexer.start()
    return reindexer


def start_url_refresher(settings: AppSettings) -> UrlRefresher:
    """Start URL_REFRESH_INTERVAL_MINUTES's background passes (api/url_refresh.py) over
    the default workspace and every other one."""

    repository = VectorRepository(get_qdrant_client())
    targets = workspace_targets(settings, repository, get_token_counter())
    refresher = UrlRefresher(settings, repository, targets, state=get_url_refresh_state_store())
    refresher.start()
    return refresher


def resume_imports(settings: AppSettings) -> list[str]:
    """Run again each background import a restart cut off (IMPORT_SPOOL_DIR keeps its
    file and a record of what it was): the new imports' ids. One whose workspace is
    gone, or that can't start now, is dropped with its file and logged."""

    directory = spool_directory(settings.import_spool_dir)
    if directory is None:
        return []
    repository = VectorRepository(get_qdrant_client())
    token_counter = get_token_counter()
    started: list[str] = []
    for record_path, record, spool in leftover(directory):
        kind, name = str(record.get("kind")), str(record.get("name") or "import")
        workspace = str(record.get("workspace") or "")

        def drop(record_path: Path = record_path, spool: Path | None = spool) -> None:
            record_path.unlink(missing_ok=True)
            if spool is not None:
                spool.unlink(missing_ok=True)

        try:
            target_settings = settings_for(settings, workspace) if workspace else settings
            target = workspace_target(target_settings, repository, token_counter)
            assert target.queue is not None
            if kind == "archive" and spool is not None:
                raw_tags = record.get("tags")
                tags = (
                    normalize_tags([tag for tag in raw_tags if isinstance(tag, str)])
                    if isinstance(raw_tags, list)
                    else None
                )
                status_id = documents_write_routes.start_archive_import(
                    spool,
                    name,
                    repository,
                    target_settings,
                    target.queue,
                    target.raw_store,
                    tags,
                    record=record_path,
                    resumed=True,
                    keep_paths=record.get("keep_paths") is True,
                ).id
            elif kind == "restore":
                stored = record.get("stored")
                path = (
                    backup_routes._stored_backup(target_settings, str(stored)) if stored else spool
                )
                if path is None:
                    raise ValueError("its record names no file")
                status_id = (
                    backup_routes._restore_in_background(
                        path,
                        name,
                        repository,
                        target_settings,
                        target.queue,
                        target.raw_store,
                        get_feedback_store(),
                        cleanup=drop,
                        source=str(stored) if stored else None,
                        resumable=True,
                        resumed=True,
                        # A restore of chosen documents resumes as one.
                        only=_only(record.get("only")),
                    ).import_id
                    or ""
                )
            else:
                raise ValueError(f"unknown kind {kind!r}")
        except Exception as exc:  # noqa: BLE001 - one bad leftover must not stop the rest
            logger.warning("Could not resume the %s import %r: %s", kind, name, exc)
            drop()
            continue
        logger.info("Resumed the %s import %r after a restart.", kind, name)
        started.append(status_id)
    return started


def _only(value: object) -> frozenset[str] | None:
    """A resumed restore's chosen documents (its record's ``only``), if any."""

    return frozenset(str(n) for n in value) if isinstance(value, list) and value else None


def check_for_lost_index(settings: AppSettings, reported: set[str] | None = None) -> list[str]:
    """``find_lost_indexes``, never raising: it runs on its own thread."""

    try:
        return find_lost_indexes(settings, reported)
    except Exception:  # noqa: BLE001 - advice only; readiness reports what is down
        logger.warning("Could not check the indexes against their originals.", exc_info=True)
        return []


# How often the lost-index check runs again after startup: an index wiped while the
# server runs (or a Qdrant not up yet at boot) used to go unnoticed until a restart.
LOST_INDEX_CHECK_SECONDS = 15 * 60
# How long the first lost-index check waits for cut-off imports to resume.
RESUME_WAIT_SECONDS = 300.0


def watch_for_lost_index(
    settings: AppSettings, stop: threading.Event, resumed: threading.Event | None = None
) -> None:
    """``check_for_lost_index`` at startup, then every LOST_INDEX_CHECK_SECONDS until
    ``stop``; each loss is reported (and rebuilt) once, until the index is back.
    ``resumed`` is set once the imports a restart cut off are running again: a restore
    resumed is filling the index, and a rebuild started beside it raced it."""

    if resumed is not None:
        resumed.wait(RESUME_WAIT_SECONDS)
    reported: set[str] = set()
    check_for_lost_index(settings, reported)
    while not stop.wait(LOST_INDEX_CHECK_SECONDS):
        check_for_lost_index(settings, reported)


def find_lost_indexes(settings: AppSettings, reported: set[str] | None = None) -> list[str]:
    """Each workspace whose index holds no document while its originals are kept (a
    wiped Qdrant volume, a collection rebuilt by hand) is reported (``index_lost``),
    and with AUTO_REBUILD_LOST_INDEX rebuilt from those originals in the background.
    The workspaces found. Only the listing and the originals store are read: the
    embedding model and an indexing queue are built for a rebuild alone.

    ``reported`` (the periodic check's memory): a workspace already reported isn't
    reported or rebuilt again until its index is seen holding documents; one with an
    import (a rebuild, a restore) still running is left to it."""

    from api.imports import importing_into
    from api.rebuild import originals_to_rebuild, rebuild

    client = get_qdrant_client()
    repository = VectorRepository(client)
    candidates = [settings]
    if settings.workspaces_enabled:
        candidates += [
            settings_for(settings, name)
            for name in list_workspaces(client, settings.qdrant_collection)
        ]
    lost: list[str] = []
    for workspace_settings in candidates:
        label = workspace_of(workspace_settings) or "default"
        # The whole check and any rebuild per workspace: one's error (a store that
        # can't be read, a collection that can't be written) leaves the rest checked.
        try:
            found = _lost_index(
                workspace_settings,
                repository,
                label,
                reported,
                importing_into,
                originals_to_rebuild,
                rebuild,
            )
        except Exception:  # noqa: BLE001 - one workspace's check; the rest go on
            logger.warning("Could not check workspace %r for a lost index.", label, exc_info=True)
            continue
        if found:
            lost.append(label)
    return lost


def _lost_index(
    workspace_settings: AppSettings,
    repository: VectorRepository,
    label: str,
    reported: set[str] | None,
    importing_into: Any,
    originals_to_rebuild: Any,
    rebuild: Any,
) -> bool:
    store = raw_store_for(workspace_settings)
    if store is None or not store.stored_names():
        return False
    notified = f"{label}\x1fnotified"
    if repository.filename_metadata(workspace_settings):
        if reported is not None:
            reported.discard(label)
            reported.discard(notified)
        return False
    if importing_into(str(workspace_settings.qdrant_collection)):
        return False  # a rebuild or restore is filling it
    names = originals_to_rebuild(repository, workspace_settings, store)
    if not names:
        return False
    # Reported once (``notified``), but done with (``label``) only once its rebuild has
    # started: one that failed to start (Qdrant away for a moment) used to be marked
    # done first, and was never tried again until a restart.
    if reported is not None and label in reported:
        return True
    quiet = reported is not None and notified in reported
    if not quiet:
        logger.warning(
            "Workspace %r has %d stored originals and no indexed document: its index looks lost.",
            label,
            len(names),
        )
    rebuilding = bool(workspace_settings.auto_rebuild_lost_index)
    action = (
        "it is being rebuilt from them now (AUTO_REBUILD_LOST_INDEX)"
        if rebuilding
        else "rebuild it with `docrag rebuild` or POST /admin/rebuild"
    )
    # Told first: a rebuild that fails to start must not keep the loss quiet.
    if not quiet:
        notify(
            "index_lost",
            f"Workspace '{label}' has {len(names)} stored originals and no indexed "
            f"document: its index looks lost; {action}.",
        )
        if reported is not None:
            reported.add(notified)
    if rebuilding:
        try:
            target = workspace_target(workspace_settings, repository, get_token_counter())
            queue = target.queue
            assert queue is not None
            repository.ensure_writable(workspace_settings)
            start_import(
                "rebuild",
                f"{len(names)} originals",
                workspace_of(workspace_settings),
                str(workspace_settings.qdrant_collection),
                lambda status, w=workspace_settings, n=names, q=queue, r=store: rebuild(  # type: ignore[misc]
                    n, repository, w, r, q, status
                ),
            )
        except Exception:  # noqa: BLE001 - tried again at the next check
            logger.warning(
                "Could not start rebuilding workspace %r; trying again at the next check.",
                label,
                exc_info=True,
            )
            return True
    if reported is not None:
        reported.add(label)
    return True


def notify_config_warnings(settings: AppSettings) -> None:
    """Tell the operator (NOTIFY_URL) what the settings check warns about at startup."""

    try:
        warnings = [f for f in diagnose(settings) if f.level == "warning"]
    except Exception:  # noqa: BLE001 - advice only
        logger.warning("The startup settings check failed.", exc_info=True)
        return
    if warnings:
        notify(
            "config_warning",
            "; ".join(f"{finding.subject}: {finding.message}" for finding in warnings[:5]),
        )


def _index_size(settings: AppSettings) -> tuple[int, int] | None:
    size = VectorRepository(get_qdrant_client()).index_size(settings)
    return (size.points, size.vector_bytes) if size is not None else None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Optionally warm up lazy-loaded models once at startup (see AppSettings.warmup_models)."""

    settings = get_app_settings()
    check_writable_storage(settings)
    # Free disk per store, read whenever /metrics is scraped.
    watch_storage(
        [(name, str(directory)) for name, directory in enabled_store_directories(settings)]
    )
    watch_index(lambda: _index_size(settings))
    set_notifier(Notifier(settings) if settings.notify_url.strip() else None)
    # Audit events into AUDIT_LOG_PATH from now on (every action, the watcher's too).
    set_audit_store(app.dependency_overrides.get(get_audit_log, get_audit_log)())
    if settings.warmup_models:
        health = ModelHealth(lambda names: warm_models(settings, names))
        health.record(await run_in_threadpool(warm_models, settings))
        # For /health/ready: a model that never loaded makes the app "degraded".
        app.state.model_health = health
    watchers: list[FolderWatcher] = []
    lost_index_stop = threading.Event()
    reindexer = None
    refresher = None
    backup = None
    try:
        watchers = [
            start_folder_watcher(settings.model_copy(update=folder.settings_update()))
            for folder in watch_folders(settings)
        ]
        watcher = watchers[0] if watchers else None
        # For GET /watch and POST /watch/scan.
        app.state.watcher = watcher
        app.state.watchers = watchers
        reindexer = start_stale_reindexer(settings) if settings.auto_reindex_stale else None
        if settings.requeue_interrupted_jobs:
            # On its own thread: a URL fetch can take its whole timeout, and startup
            # must not wait for it.
            threading.Thread(
                target=requeue_interrupted_jobs,
                args=(settings,),
                name="docrag-requeue",
                daemon=True,
            ).start()
        if settings.url_refresh_interval_minutes and settings.url_ingest_enabled:
            refresher = start_url_refresher(settings)
        imports_resumed = threading.Event()
        if settings.import_spool_dir:

            def resume_then_say_so() -> None:
                try:
                    resume_imports(settings)
                finally:
                    imports_resumed.set()

            threading.Thread(target=resume_then_say_so, name="docrag-resume", daemon=True).start()
        else:
            imports_resumed.set()
        if settings.notify_url.strip():
            threading.Thread(
                target=notify_config_warnings, args=(settings,), name="docrag-doctor", daemon=True
            ).start()
        if settings.raw_document_dir:
            threading.Thread(
                target=watch_for_lost_index,
                args=(settings, lost_index_stop, imports_resumed),
                name="docrag-lost",
                daemon=True,
            ).start()
        backup = start_scheduled_backup(settings) if settings.backup_dir else None
        # For GET /health/ready's last_backup.
        app.state.backup = backup
        yield
    finally:
        lost_index_stop.set()
        # Their threads would go on calling the model while the executors shut down.
        batch_job_store().abort()
        if backup is not None:
            backup.stop()
            backup_enabled.set(0)
        if refresher is not None:
            refresher.stop()
        if reindexer is not None:
            reindexer.stop()
        for each in watchers:
            each.stop()
        shutdown_imports()
        shutdown_background_imports()
        set_audit_store(None)
        set_notifier(None)
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
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
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
    watch_routes.register(app)


app = create_app()
