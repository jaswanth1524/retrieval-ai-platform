"""AUTO_REINDEX_STALE (api/auto_reindex.py): stale documents brought up to date in the
background, a few jobs at a time."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from api.auto_reindex import ReindexTarget, StaleReindexer
from api.chunking import HeuristicTokenCounter
from api.ingest_jobs import IngestQueue
from api.ingest_service import IngestService
from api.jobs import IngestBacklog, IngestJobStore
from api.qdrant_schema import CollectionSchemaError
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.settings import AppSettings
from tests.factories import wait_until
from tests.routes_http.support import FakeEmbeddingProvider, make_settings


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(max_workers=1)
    yield pool
    pool.shutdown(wait=True)


def _setup(
    tmp_path: Path, executor: ThreadPoolExecutor, **overrides: object
) -> tuple[VectorRepository, AppSettings, RawDocumentStore, IngestJobStore, list[str]]:
    settings = make_settings(raw_document_dir=str(tmp_path), auto_reindex_stale=True, **overrides)
    repository = VectorRepository(QdrantClient(":memory:"))
    store = RawDocumentStore(str(tmp_path), 0)
    counter = HeuristicTokenCounter()
    service = IngestService(repository, FakeEmbeddingProvider(), settings, counter)
    names = ["a.txt", "b.txt", "c.txt"]
    for name in names:
        content = f"Intro\n{name} alpha beta gamma".encode()
        service.ingest(name, content)
        store.save(name, content)
    return repository, settings, store, IngestJobStore(100), names


def _reindexer(
    repository: VectorRepository,
    current: AppSettings,
    store: RawDocumentStore,
    jobs: IngestJobStore,
    executor: ThreadPoolExecutor,
) -> StaleReindexer:
    counter = HeuristicTokenCounter()
    service = IngestService(repository, FakeEmbeddingProvider(), current, counter)
    queue = IngestQueue(jobs, executor, IngestBacklog(10_000_000), service)
    return StaleReindexer(
        current,
        repository,
        lambda: [ReindexTarget(current, store, queue)],
        jobs,
        counter,
    )


def _stale(repository: VectorRepository, settings: AppSettings) -> list[str]:
    from api.ingest_jobs import staleness

    is_stale = staleness(settings, HeuristicTokenCounter())
    return sorted(
        name for name, meta in repository.filename_metadata(settings).items() if is_stale(meta)
    )


def test_stale_documents_are_reindexed_a_batch_at_a_time(
    tmp_path: Path, executor: ThreadPoolExecutor
) -> None:
    repository, settings, store, jobs, names = _setup(tmp_path, executor)
    # A chunk-setting change makes every document stale.
    current = settings.model_copy(update={"chunk_size_tokens": 30, "auto_reindex_batch": 2})
    assert _stale(repository, current) == names
    reindexer = _reindexer(repository, current, store, jobs, executor)

    assert reindexer.run_once() == 2
    # Its own two jobs are still counted until they finish: nothing more is queued.
    wait_until(lambda: len(_stale(repository, current)) == 1)
    assert reindexer.run_once() == 1
    wait_until(lambda: _stale(repository, current) == [])
    assert reindexer.run_once() == 0


def test_nothing_is_queued_before_the_token_counter_has_loaded(
    tmp_path: Path, executor: ThreadPoolExecutor
) -> None:
    repository, settings, store, jobs, _ = _setup(tmp_path, executor)
    current = settings.model_copy(update={"chunk_size_tokens": 30})
    reindexer = _reindexer(repository, current, store, jobs, executor)

    class Loading(HeuristicTokenCounter):
        mode = "unknown"  # type: ignore[assignment]

    reindexer._token_counter = Loading()  # noqa: SLF001
    assert reindexer.run_once() == 0


def test_an_incompatible_workspace_is_skipped_and_named_once(
    tmp_path: Path, executor: ThreadPoolExecutor, caplog: pytest.LogCaptureFixture
) -> None:
    repository, settings, store, jobs, _ = _setup(tmp_path, executor)
    current = settings.model_copy(update={"chunk_size_tokens": 30})
    reindexer = _reindexer(repository, current, store, jobs, executor)

    def refuse(*args: object, **kwargs: object) -> None:
        raise CollectionSchemaError("built for another embedding setup")

    repository.ensure_writable = refuse  # type: ignore[method-assign]
    caplog.set_level(logging.WARNING, logger="api.auto_reindex")

    assert reindexer.run_once() == 0
    assert reindexer.run_once() == 0
    assert sum("Not re-indexing" in record.getMessage() for record in caplog.records) == 1


def test_the_background_loop_runs_a_pass_and_stops(
    tmp_path: Path, executor: ThreadPoolExecutor
) -> None:
    repository, settings, store, jobs, _ = _setup(tmp_path, executor)
    current = settings.model_copy(update={"chunk_size_tokens": 30})
    reindexer = _reindexer(repository, current, store, jobs, executor)
    reindexer.start()
    try:
        wait_until(lambda: _stale(repository, current) == [])
    finally:
        reindexer.stop()


def test_an_incomplete_document_is_repaired_from_its_original(
    tmp_path: Path, executor: ThreadPoolExecutor
) -> None:
    from qdrant_client import models

    from api.corpus import bump_corpus_generation

    repository, settings, store, jobs, _ = _setup(tmp_path, executor)
    client = repository._client  # noqa: SLF001
    content = b"Intro\n" + " ".join(f"Sentence number {n} is here." for n in range(40)).encode()
    service = IngestService(repository, FakeEmbeddingProvider(), settings, HeuristicTokenCounter())
    service.ingest("long.txt", content)
    store.save("long.txt", content)
    whole = repository.filename_metadata(settings)["long.txt"].chunk_count
    points, _ = client.scroll(
        settings.qdrant_collection,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="filename", match=models.MatchValue(value="long.txt"))]
        ),
        limit=1,
    )
    client.delete(
        settings.qdrant_collection, points_selector=models.PointIdsList(points=[points[0].id])
    )
    bump_corpus_generation(["long.txt"], collection=settings.qdrant_collection)
    assert repository.filename_metadata(settings)["long.txt"].incomplete

    reindexer = _reindexer(repository, settings, store, jobs, executor)
    assert reindexer.run_once() == 1
    wait_until(lambda: not repository.filename_metadata(settings)["long.txt"].incomplete)
    assert repository.filename_metadata(settings)["long.txt"].chunk_count == whole
