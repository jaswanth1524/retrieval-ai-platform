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
        assert refresher.run_once()["deferred"] == 3
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
