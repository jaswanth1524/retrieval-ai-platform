"""Pass 15 ingest/ops fixes: retries that would overwrite, versions kept on a failed
save, parse timeouts as transient, validator-only refreshes, backup leftovers and
status, gzip completeness, evicted jobs, re-import, stale documents without originals."""

from __future__ import annotations

import functools
import os
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.auto_reindex import StaleReindexer
from api.background import OwnedJobs
from api.chunking import HeuristicTokenCounter
from api.corpus import corpus_generation
from api.dependencies import clear_dependency_caches
from api.document_ops import reindex_stale, retry_refusal
from api.ingest_jobs import IngestQueue, ingest_error_type
from api.ingest_service import IngestService
from api.jobs import IngestBacklog, IngestJob, IngestJobStore
from api.parse_isolation import ParseInterruptedError
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.scheduled_backup import BackupStatus, ScheduledBackup
from api.url_fetch import UrlFetchError, fetch_document
from api.workspace_target import WorkspaceTarget
from tests.factories import wait_until
from tests.routes_http.support import (
    FakeEmbeddingProvider,
    hermetic_app,
    make_settings,
    upload_and_wait,
)
from tests.test_url_fetch import _coded, _gzipped, fetch, resolver, transport


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(max_workers=1)
    yield pool
    pool.shutdown(wait=True)


def _finished(client: TestClient, job_id: str) -> dict[str, Any]:
    wait_until(
        lambda: client.get(f"/documents/jobs/{job_id}").json()["state"] in ("done", "failed")
    )
    return dict(client.get(f"/documents/jobs/{job_id}").json())


# --- URL retry ---------------------------------------------------------------------


def test_a_retry_never_replaces_an_upload_made_since_the_job_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_dependency_caches()
    app = hermetic_app(make_settings(url_ingest_enabled=True), QdrantClient(":memory:"))
    serving = {"body": b""}  # empty first: the job fails

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=serving["body"], headers={"content-type": "text/plain"})

    fake = functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler))
    monkeypatch.setattr("api.routes.documents_write.fetch_document", fake)
    monkeypatch.setattr("api.routes.documents_jobs.fetch_document", fake)
    with TestClient(app) as client:
        accepted = client.post(
            "/documents/from-url", json={"url": "https://example.com/page.txt"}
        ).json()
        assert _finished(client, accepted["job_id"])["state"] == "failed"
        # Someone uploads their own page.txt; then the old job's Retry is clicked.
        assert upload_and_wait(client, "page.txt", b"Intro\nmine")["state"] == "done"
        serving["body"] = b"Intro\nthe URL's"
        retried = client.post(f"/documents/jobs/{accepted['job_id']}/retry")

        assert retried.status_code == 409 and "replaced since" in retried.json()["detail"]
        text = client.get("/documents/page.txt/content").json()["chunks"][0]["text"]
        assert "mine" in text
    clear_dependency_caches()


# --- versions survive a failed save -------------------------------------------------


def test_a_failed_save_removes_the_current_original_but_keeps_its_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_dependency_caches()
    settings = make_settings(raw_document_dir=str(tmp_path), raw_document_versions=3)
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as client:
        assert upload_and_wait(client, "p.txt", b"Intro\none")["state"] == "done"
        assert upload_and_wait(client, "p.txt", b"Intro\ntwo")["state"] == "done"
        assert len(client.get("/documents/p.txt/versions").json()["versions"]) == 1

        def disk_full(self: RawDocumentStore, filename: str, content: bytes) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(RawDocumentStore, "save", disk_full)
        assert upload_and_wait(client, "p.txt", b"Intro\nthree")["state"] == "done"

        assert client.get("/documents/p.txt/original").status_code == 404
        assert len(client.get("/documents/p.txt/versions").json()["versions"]) == 1
    monkeypatch.undo()
    clear_dependency_caches()


def test_remove_current_leaves_the_history(tmp_path: Path) -> None:
    store = RawDocumentStore(str(tmp_path), keep_versions=2)
    store.save("a.txt", b"one")
    store.save("a.txt", b"two")

    assert store.remove_current("a.txt") and not store.remove_current("a.txt")
    assert store.read("a.txt") is None
    assert [store.read_version("a.txt", v.version) for v in store.versions("a.txt")] == [b"one"]


# --- parse timeouts are transient ---------------------------------------------------


def test_a_stopped_parse_is_its_own_transient_error_type() -> None:
    assert ingest_error_type(ParseInterruptedError("took too long")) == "parse_timeout"
    job = IngestJob(
        id="j", filename="a.txt", state="failed", kind="reindex", error_type="parse_timeout"
    )
    assert retry_refusal(job) is None


def _out_of_memory(connection: Any, *args: object) -> None:
    connection.send(("memory", None))
    connection.close()


def test_a_parse_that_runs_out_of_memory_names_the_setting() -> None:
    from api.parse_isolation import parse_with_limit

    settings = make_settings(parse_timeout_seconds=30, parse_memory_limit_mb=64)
    from api.documents import DocumentParseError

    with pytest.raises(DocumentParseError, match="PARSE_MEMORY_LIMIT_MB") as caught:
        parse_with_limit("a.txt", b"x", settings, _target=_out_of_memory)
    # A limit the operator can raise: transient (retried), not a bad file.
    from api.ingest_jobs import ingest_error_type
    from api.jobs import IngestJob

    assert ingest_error_type(caught.value) == "parse_memory"
    job = IngestJob(
        id="j", filename="a.txt", state="failed", kind="reindex", error_type="parse_memory"
    )
    assert retry_refusal(job) is None


# --- validator-only refresh ---------------------------------------------------------


def test_the_same_bytes_under_a_new_etag_keep_the_answer_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_dependency_caches()
    settings = make_settings(url_ingest_enabled=True)
    app = hermetic_app(settings, QdrantClient(":memory:"))
    etag = {"value": '"v1"'}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"Intro\nthe same page",
            headers={"content-type": "text/plain", "etag": etag["value"]},
        )

    fake = functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler))
    monkeypatch.setattr("api.routes.documents_write.fetch_document", fake)
    with TestClient(app) as client:
        url = {"url": "https://example.com/page.txt"}
        first = client.post("/documents/from-url", json=url).json()
        assert _finished(client, first["job_id"])["state"] == "done"
        client.get("/documents")  # the listing is cached now
        before = corpus_generation(settings.qdrant_collection)

        etag["value"] = '"v2"'
        refreshed = client.post("/documents/page.txt/refresh")
        assert refreshed.status_code == 202, refreshed.text
        assert _finished(client, refreshed.json()["job_id"])["unchanged"] is True
        assert corpus_generation(settings.qdrant_collection) == before
        settings_change = client.patch(
            "/documents/page.txt/refresh-settings", json={"refresh_minutes": 90}
        )
        assert settings_change.status_code == 200
        assert corpus_generation(settings.qdrant_collection) == before
        # The listing shows the change all the same.
        assert client.get("/documents").json()["url_refresh"]["page.txt"]["refresh_minutes"] == 90
    clear_dependency_caches()


# --- scheduled backups --------------------------------------------------------------


def _backup(tmp_path: Path, clock: list[float]) -> ScheduledBackup:
    settings = make_settings(
        raw_document_dir=str(tmp_path / "raw"),
        backup_dir=str(tmp_path / "backups"),
        backup_interval_hours=1,
    )
    repository = VectorRepository(QdrantClient(":memory:"))
    IngestService(repository, FakeEmbeddingProvider(), settings).ingest("g.txt", b"Intro\nalpha")
    store = RawDocumentStore(str(tmp_path / "raw"), 0)
    return ScheduledBackup(
        settings,
        repository,
        lambda: [WorkspaceTarget(settings, store)],
        None,
        threading.Lock(),
        clock=lambda: clock[0],
    )


def test_a_partial_backup_left_by_a_killed_process_is_removed_later(tmp_path: Path) -> None:
    clock = [time.time()]
    backup = _backup(tmp_path, clock)
    directory = tmp_path / "backups" / "default"
    directory.mkdir(parents=True)
    old = directory / ".docrag-20200101-000000.zip.part"
    young = directory / ".docrag-20200101-000001.zip.part"
    other = directory / "notes.part"
    for path in (old, young, other):
        path.write_bytes(b"x")
    os.utime(old, (clock[0] - 7200, clock[0] - 7200))

    backup.run_once()

    assert not old.exists() and young.exists() and other.exists()


def test_each_workspace_keeps_its_own_status_and_failing_ones_are_named(tmp_path: Path) -> None:
    backup = _backup(tmp_path, [time.time()])
    backup._note(BackupStatus(1.0, "legal", False, error="disk full"))  # noqa: SLF001
    backup._note(BackupStatus(2.0, "default", True, file="x.zip"))  # noqa: SLF001

    assert backup.last is not None and backup.last.ok
    assert backup.failing() == ["legal"]


def test_readiness_hides_backup_details_when_a_key_is_set() -> None:
    from fastapi import FastAPI

    from api.routes.ops import _backups_failing, _last_backup

    app = FastAPI()

    class Backup:
        last = BackupStatus(5.0, "legal", False, error="/srv/backups: permission denied")

        def failing(self) -> list[str]:
            return ["legal"]

    app.state.backup = Backup()
    open_settings = make_settings()
    keyed = make_settings(api_key="secret")

    shown = _last_backup(app, open_settings)
    assert shown is not None and shown.workspace == "legal" and "permission" in str(shown.error)
    hidden = _last_backup(app, keyed)
    assert hidden is not None and hidden.workspace == "" and hidden.file is None
    assert hidden.error == "The backup failed; see the server log."
    assert _backups_failing(app, open_settings) == ["legal"]
    assert _backups_failing(app, keyed) == ["(hidden)"]


# --- gzip completeness --------------------------------------------------------------


def test_a_gzip_body_cut_off_is_refused() -> None:
    whole = _gzipped(b"Intro\n" + b"word " * 5000)
    with pytest.raises(UrlFetchError, match="cut off"):
        fetch("https://example.com/a.txt", _coded(whole[: len(whole) // 2], "gzip"))


def test_every_member_of_a_multi_member_gzip_body_is_read() -> None:
    body = _gzipped(b"Intro\nfirst ") + _gzipped(b"second") + b"\0\0\0"
    assert fetch("https://example.com/a.txt", _coded(body, "gzip")).content == (
        b"Intro\nfirst second"
    )


def test_the_limit_holds_across_members() -> None:
    settings = make_settings(url_ingest_enabled=True, max_upload_bytes=1000)
    body = _gzipped(b"a" * 600) + _gzipped(b"b" * 600)
    with pytest.raises(UrlFetchError, match="larger than 1000"):
        fetch("https://example.com/a.txt", _coded(body, "gzip"), settings=settings)


# --- evicted jobs -------------------------------------------------------------------


def test_a_job_the_store_let_go_is_reported_as_forgotten() -> None:
    store = IngestJobStore(100)
    job = store.create("a.txt")
    owned: OwnedJobs[str] = OwnedJobs()
    owned.add(store, job.id, "a")
    forgotten: list[str] = []
    store._jobs.clear()  # noqa: SLF001 - evicted

    assert owned.running(on_forgotten=forgotten.append) == 0
    assert forgotten == ["a"]


def test_the_reindexer_backs_off_a_document_whose_job_was_evicted(
    tmp_path: Path, executor: ThreadPoolExecutor
) -> None:
    settings = make_settings(raw_document_dir=str(tmp_path), auto_reindex_stale=True)
    repository = VectorRepository(QdrantClient(":memory:"))
    store = RawDocumentStore(str(tmp_path), 0)
    counter = HeuristicTokenCounter()
    service = IngestService(repository, FakeEmbeddingProvider(), settings, counter)
    service.ingest("a.txt", b"Intro\nalpha beta")
    store.save("a.txt", b"Intro\nalpha beta")
    current = settings.model_copy(update={"chunk_size_tokens": 30})
    jobs = IngestJobStore(100)
    pool = IngestQueue(
        jobs,
        executor,
        IngestBacklog(10_000_000),
        IngestService(repository, FakeEmbeddingProvider(), current, counter),
    )
    reindexer = StaleReindexer(
        current, repository, lambda: [WorkspaceTarget(current, store, pool)], jobs, counter
    )
    # Never runs: the queue's executor is busy, and its job is evicted before it ends.
    blocker = threading.Event()
    executor.submit(blocker.wait)
    try:
        assert reindexer.run_once() == 1
        jobs._jobs.clear()  # noqa: SLF001
        assert reindexer.run_once() == 0  # held back, not queued again at once
    finally:
        blocker.set()


# --- re-import and stale documents without originals --------------------------------


def test_a_stale_document_with_no_original_is_named(
    tmp_path: Path, executor: ThreadPoolExecutor
) -> None:
    settings = make_settings(raw_document_dir=str(tmp_path))
    repository = VectorRepository(QdrantClient(":memory:"))
    counter = HeuristicTokenCounter()
    IngestService(repository, FakeEmbeddingProvider(), settings, counter).ingest(
        "a.txt", b"Intro\nalpha"
    )
    current = settings.model_copy(update={"chunk_size_tokens": 30})
    queue = IngestQueue(
        IngestJobStore(100),
        executor,
        IngestBacklog(10_000_000),
        IngestService(repository, FakeEmbeddingProvider(), current, counter),
    )

    outcome = reindex_stale(repository, current, queue, RawDocumentStore(str(tmp_path), 0), counter)

    assert outcome.jobs == [] and outcome.no_original == ["a.txt"]


def test_importing_while_a_document_is_being_indexed_skips_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import json
    import zipfile

    from api.ingestion import ingest_sequencer

    manifest = {
        "manifest_version": 1,
        "documents": [{"filename": "a.txt", "original": "originals/a.txt"}],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("originals/a.txt", b"Intro\nalpha")
    clear_dependency_caches()
    settings = make_settings()
    with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as client:
        sequence = ingest_sequencer.admit("a.txt")  # a first import's job, still running
        try:
            response = client.post(
                "/import", files={"file": ("b.zip", buffer.getvalue(), "application/zip")}
            )
        finally:
            ingest_sequencer.release("a.txt")
        assert response.is_success, response.text
        assert response.json()["jobs"] == [] and response.json()["skipped"] == ["a.txt"]
        assert sequence is not None
    clear_dependency_caches()
