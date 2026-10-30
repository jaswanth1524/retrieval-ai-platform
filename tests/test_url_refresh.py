"""URL_REFRESH_INTERVAL_MINUTES (api/url_refresh.py): fetched documents checked again in
the background, conditionally, a few per pass."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from qdrant_client import QdrantClient

from api.chunking import HeuristicTokenCounter
from api.corpus import bump_corpus_generation
from api.documents import DocumentSource
from api.ingest_jobs import IngestQueue
from api.ingest_service import IngestService
from api.jobs import IngestBacklog, IngestJobStore
from api.repository import VectorRepository
from api.settings import AppSettings
from api.url_fetch import FetchedDocument, UrlFetchError, fetch_slots
from api.url_refresh import RefreshTarget, UrlRefresher
from tests.factories import wait_until
from tests.routes_http.support import FakeEmbeddingProvider, make_settings


class Remote:
    """Fake fetch_document: a page per URL, with an ETag, and what was asked."""

    def __init__(self) -> None:
        self.pages: dict[str, tuple[bytes, str]] = {}
        self.asked: list[tuple[str, str | None]] = []
        self.fail: set[str] = set()

    def __call__(self, url: str, settings: AppSettings, **kwargs: Any) -> FetchedDocument:
        self.asked.append((url, kwargs.get("etag")))
        if url in self.fail:
            raise UrlFetchError("The server answered 500.")
        body, etag = self.pages[url]
        if kwargs.get("etag") == etag:
            return FetchedDocument(kwargs["filename"], b"", url, etag=etag, not_modified=True)
        return FetchedDocument(kwargs["filename"], body, url, etag=etag)


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(max_workers=1)
    yield pool
    pool.shutdown(wait=True)


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _setup(
    executor: ThreadPoolExecutor, **overrides: object
) -> tuple[VectorRepository, AppSettings, Remote, UrlRefresher, Clock, IngestJobStore]:
    settings = make_settings(url_ingest_enabled=True, url_refresh_interval_minutes=60, **overrides)
    repository = VectorRepository(QdrantClient(":memory:"))
    service = IngestService(repository, FakeEmbeddingProvider(), settings, HeuristicTokenCounter())
    remote = Remote()
    for name in ("a.txt", "b.txt", "c.txt"):
        url = f"https://example.com/{name}"
        remote.pages[url] = (f"Intro\n{name} first".encode(), '"v1"')
        service.ingest(
            name,
            remote.pages[url][0],
            uploaded_at=1000.0,
            source=DocumentSource(url, '"v1"'),
        )
    service.ingest("uploaded.txt", b"Intro\nnot fetched", uploaded_at=1000.0)
    jobs = IngestJobStore(100)
    queue = IngestQueue(jobs, executor, IngestBacklog(10_000_000), service)
    clock = Clock(1000.0)
    refresher = UrlRefresher(
        settings,
        repository,
        lambda: [RefreshTarget(settings, None, queue)],
        fetch=remote,
        clock=clock,
    )
    return repository, settings, remote, refresher, clock, jobs


def _text(repository: VectorRepository, settings: AppSettings, name: str) -> str:
    return " ".join(str(chunk["text"]) for chunk in repository.chunks_for_filename(settings, name))


def test_nothing_is_fetched_before_the_interval_has_passed(executor: ThreadPoolExecutor) -> None:
    _, _, remote, refresher, clock, _ = _setup(executor)
    clock.now = 1000.0 + 59 * 60
    assert sum(refresher.run_once().values()) == 0
    assert remote.asked == []


def test_due_documents_are_asked_conditionally_a_batch_at_a_time(
    executor: ThreadPoolExecutor,
) -> None:
    repository, settings, remote, refresher, clock, _ = _setup(executor, url_refresh_batch=2)
    remote.pages["https://example.com/b.txt"] = (b"Intro\nb.txt edited", '"v2"')
    clock.now = 1000.0 + 61 * 60

    first = refresher.run_once()
    assert first == {"unchanged": 1, "queued": 1, "failed": 0, "deferred": 0}
    # The etag stored with each document went with the request; the upload wasn't asked.
    assert remote.asked == [
        ("https://example.com/a.txt", '"v1"'),
        ("https://example.com/b.txt", '"v1"'),
    ]
    wait_until(lambda: "edited" in _text(repository, settings, "b.txt"))

    second = refresher.run_once()
    assert second["unchanged"] == 1 and len(remote.asked) == 3
    # Everything was checked this hour: nothing more until the next.
    assert sum(refresher.run_once().values()) == 0
    clock.now += 61 * 60
    assert sum(refresher.run_once().values()) == 2


def test_a_failed_fetch_waits_for_the_next_interval(
    executor: ThreadPoolExecutor, caplog: pytest.LogCaptureFixture
) -> None:
    _, _, remote, refresher, clock, _ = _setup(executor, url_refresh_batch=10)
    remote.fail.add("https://example.com/a.txt")
    clock.now = 1000.0 + 61 * 60
    caplog.set_level(logging.WARNING, logger="api.url_refresh")

    assert refresher.run_once()["failed"] == 1
    assert "Scheduled refresh of a.txt failed" in caplog.text
    assert sum(refresher.run_once().values()) == 0


def test_busy_fetch_slots_defer_the_check_to_the_next_pass(executor: ThreadPoolExecutor) -> None:
    _, _, remote, refresher, clock, _ = _setup(executor, url_refresh_batch=10)
    clock.now = 1000.0 + 61 * 60
    held = 0
    while fetch_slots.acquire(blocking=False):
        held += 1
    try:
        # The first deferral ends the pass: the others would wait for the same slot.
        assert refresher.run_once() == {"unchanged": 0, "queued": 0, "failed": 0, "deferred": 1}
    finally:
        for _ in range(held):
            fetch_slots.release()
    assert remote.asked == []
    assert refresher.run_once()["unchanged"] == 3


def test_the_loop_starts_and_stops(executor: ThreadPoolExecutor) -> None:
    _, settings, remote, _, _, _ = _setup(executor)
    repository = VectorRepository(QdrantClient(":memory:"))
    refresher = UrlRefresher(settings, repository, lambda: [], fetch=remote, interval=0.01)
    refresher.start()
    refresher.stop()
    assert remote.asked == []


def test_the_app_starts_the_refresher_only_when_url_ingest_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    import api.main
    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app

    started: list[AppSettings] = []

    class Recorded:
        def stop(self) -> None:
            started.append(None)  # type: ignore[arg-type]

    def fake_start(settings: AppSettings) -> Recorded:
        started.append(settings)
        return Recorded()

    monkeypatch.setattr(api.main, "start_url_refresher", fake_start)
    for enabled, expected in ((False, 0), (True, 2)):
        clear_dependency_caches()
        settings = make_settings(url_ingest_enabled=enabled, url_refresh_interval_minutes=5)
        monkeypatch.setattr(api.main, "get_app_settings", lambda s=settings: s)
        started.clear()
        with TestClient(hermetic_app(settings, QdrantClient(":memory:"))):
            pass
        assert len(started) == expected
    clear_dependency_caches()


def test_start_url_refresher_covers_every_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.main
    from api.dependencies import clear_dependency_caches

    clear_dependency_caches()
    settings = make_settings(url_ingest_enabled=True, url_refresh_interval_minutes=5)
    monkeypatch.setattr(api.main, "get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    monkeypatch.setattr(UrlRefresher, "start", lambda self: None)
    monkeypatch.setattr(api.main, "get_qdrant_client", lambda: QdrantClient(":memory:"))
    monkeypatch.setattr(api.main, "get_token_counter", HeuristicTokenCounter)
    monkeypatch.setattr(api.main, "get_embedding_provider", FakeEmbeddingProvider)
    refresher = api.main.start_url_refresher(settings)
    targets = refresher._targets()  # noqa: SLF001
    assert [target.settings.qdrant_collection for target in targets] == [settings.qdrant_collection]
    monkeypatch.undo()
    clear_dependency_caches()


def _finished(jobs: IngestJobStore, count: int) -> list[Any]:
    wait_until(lambda: len(jobs.list_finished("")) >= count)
    return jobs.list_finished("")


def test_a_document_deleted_while_its_refresh_was_fetched_stays_deleted(
    executor: ThreadPoolExecutor,
) -> None:
    repository, settings, remote, refresher, clock, jobs = _setup(executor, url_refresh_batch=1)
    remote.pages["https://example.com/a.txt"] = (b"Intro\na.txt edited", '"v2"')

    def fetch_then_delete(url: str, given: AppSettings, **kwargs: Any) -> FetchedDocument:
        fetched = remote(url, given, **kwargs)
        repository.delete_filename(settings, "a.txt")  # the user's DELETE lands meanwhile
        return fetched

    refresher._fetch = fetch_then_delete
    clock.now = 1000.0 + 61 * 60
    assert refresher.run_once()["queued"] == 1
    (job,) = _finished(jobs, 1)
    assert job.state == "failed"
    assert "changed or deleted while its refresh was being fetched" in str(job.error)
    assert repository.document_metadata(settings, "a.txt") is None


def test_an_upload_landing_during_the_fetch_is_not_overwritten(
    executor: ThreadPoolExecutor,
) -> None:
    repository, settings, remote, refresher, clock, jobs = _setup(executor, url_refresh_batch=1)
    remote.pages["https://example.com/a.txt"] = (b"Intro\na.txt from the url", '"v2"')
    uploader = IngestService(repository, FakeEmbeddingProvider(), settings, HeuristicTokenCounter())

    def fetch_then_upload(url: str, given: AppSettings, **kwargs: Any) -> FetchedDocument:
        fetched = remote(url, given, **kwargs)
        uploader.ingest("a.txt", b"Intro\na.txt uploaded by hand")
        return fetched

    refresher._fetch = fetch_then_upload
    clock.now = 1000.0 + 61 * 60
    refresher.run_once()
    (job,) = _finished(jobs, 1)
    assert job.state == "failed"
    assert "uploaded by hand" in _text(repository, settings, "a.txt")


def test_new_validators_on_the_same_bytes_are_stored_without_reindexing(
    executor: ThreadPoolExecutor,
) -> None:
    repository, settings, remote, refresher, clock, jobs = _setup(executor, url_refresh_batch=1)
    url = "https://example.com/a.txt"
    remote.pages[url] = (remote.pages[url][0], '"v2"')  # a CDN re-stamping the same page
    clock.now = 1000.0 + 61 * 60

    assert refresher.run_once()["queued"] == 1
    (job,) = _finished(jobs, 1)
    assert job.state == "done" and job.result is not None and job.result["unchanged"] is True
    meta = repository.document_metadata(settings, "a.txt")
    assert meta is not None and meta.source_etag == '"v2"' and meta.source_url == url
    # The next check sends the new ETag, so it is a 304.
    clock.now += 61 * 60
    assert refresher.run_once()["unchanged"] == 1


def test_no_fetch_while_a_batch_of_its_own_jobs_is_still_indexing(
    executor: ThreadPoolExecutor,
) -> None:
    import threading

    _, _, remote, refresher, clock, _ = _setup(executor, url_refresh_batch=1)
    for name in ("a.txt", "b.txt"):
        remote.pages[f"https://example.com/{name}"] = (f"Intro\n{name} edited".encode(), '"v2"')
    release = threading.Event()
    executor.submit(release.wait, 10)  # the one worker is busy: the job stays queued
    clock.now = 1000.0 + 61 * 60
    try:
        assert refresher.run_once()["queued"] == 1
        assert sum(refresher.run_once().values()) == 0
        assert len(remote.asked) == 1
    finally:
        release.set()


def _metadata(repository: VectorRepository, settings: AppSettings, name: str) -> Any:
    meta = repository.document_metadata(settings, name)
    assert meta is not None
    return meta


def test_a_failing_document_waits_twice_as_long_after_each_failure(
    executor: ThreadPoolExecutor,
) -> None:
    from api.url_refresh_state import MemoryRefreshStateStore

    _, settings, remote, _, clock, jobs = _setup(executor, url_refresh_batch=10)
    repository = VectorRepository(QdrantClient(":memory:"))
    service = IngestService(repository, FakeEmbeddingProvider(), settings, HeuristicTokenCounter())
    url = "https://example.com/flaky.txt"
    remote.pages[url] = (b"Intro\nflaky", '"v1"')
    service.ingest("flaky.txt", b"Intro\nflaky", uploaded_at=1000.0, source=DocumentSource(url))
    remote.fail.add(url)
    state = MemoryRefreshStateStore()
    queue = IngestQueue(jobs, executor, IngestBacklog(10_000_000), service)
    refresher = UrlRefresher(
        settings,
        repository,
        lambda: [RefreshTarget(settings, None, queue)],
        fetch=remote,
        clock=clock,
        state=state,
    )
    hour = 60 * 60.0
    clock.now = 1000.0 + hour
    assert refresher.run_once()["failed"] == 1
    # One failure: twice the interval before the next check.
    clock.now += hour
    assert sum(refresher.run_once().values()) == 0
    clock.now += hour
    assert refresher.run_once()["failed"] == 1
    assert state.get_all(settings.qdrant_collection)["flaky.txt"].failures == 2
    # Two in a row: four times. A success resets it.
    clock.now += 3 * hour
    assert sum(refresher.run_once().values()) == 0
    remote.fail.clear()
    clock.now += hour
    assert refresher.run_once()["queued"] == 1
    assert state.get_all(settings.qdrant_collection)["flaky.txt"].failures == 0


def test_a_documents_own_interval_and_never(executor: ThreadPoolExecutor) -> None:
    repository, settings, remote, refresher, clock, _ = _setup(executor, url_refresh_batch=10)
    a = _metadata(repository, settings, "a.txt")
    b = _metadata(repository, settings, "b.txt")
    repository.set_source_details(settings, "a.txt", a.source_etag, None, 0)
    repository.set_source_details(settings, "b.txt", b.source_etag, None, 10)
    # The listing is cached; the route bumps the generation after a change like this.
    bump_corpus_generation(["a.txt", "b.txt"], collection=settings.qdrant_collection)
    # Eleven minutes on: b's own 10 minutes have passed, the server's hour hasn't.
    clock.now = 1000.0 + 11 * 60
    assert refresher.run_once()["unchanged"] == 1
    assert [url for url, _ in remote.asked] == ["https://example.com/b.txt"]
    # A day on: c is due on the server's interval; a (0) never is.
    clock.now += 24 * 60 * 60
    refresher.run_once()
    assert "https://example.com/a.txt" not in [url for url, _ in remote.asked]
    assert "https://example.com/c.txt" in [url for url, _ in remote.asked]


def test_the_check_times_survive_a_restart_in_the_state_file(
    executor: ThreadPoolExecutor, tmp_path: Any
) -> None:
    from api.url_refresh_state import SqliteRefreshStateStore

    repository, settings, remote, _, clock, jobs = _setup(executor, url_refresh_batch=10)
    path = str(tmp_path / "state.db")
    service = IngestService(repository, FakeEmbeddingProvider(), settings, HeuristicTokenCounter())
    queue = IngestQueue(jobs, executor, IngestBacklog(10_000_000), service)

    def refresher() -> UrlRefresher:
        return UrlRefresher(
            settings,
            repository,
            lambda: [RefreshTarget(settings, None, queue)],
            fetch=remote,
            clock=clock,
            state=SqliteRefreshStateStore(path),
        )

    clock.now = 1000.0 + 61 * 60
    assert refresher().run_once()["unchanged"] == 3
    # Restarted ten minutes later: checked ten minutes ago, not an hour past upload.
    clock.now += 10 * 60
    assert sum(refresher().run_once().values()) == 0
    # A deleted document's row goes, and a workspace no longer listed loses all of its.
    repository.delete_filename(settings, "a.txt")
    bump_corpus_generation(["a.txt"], collection=settings.qdrant_collection)
    refresher().run_once()
    assert set(SqliteRefreshStateStore(path).get_all(settings.qdrant_collection)) == {
        "b.txt",
        "c.txt",
    }
    UrlRefresher(settings, repository, lambda: [], state=SqliteRefreshStateStore(path)).run_once()
    assert SqliteRefreshStateStore(path).get_all(settings.qdrant_collection) == {}
