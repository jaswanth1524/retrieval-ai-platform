"""Document routes: upload, list, delete, content, original, re-index, tags, jobs."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

import api.ingestion as ingestion
from api.dependencies import (
    clear_dependency_caches,
    get_embedding_provider,
    get_ingest_backlog,
    get_ingest_job_store,
)
from api.embeddings import EmbeddedText
from api.ingestion import filename_write_lock
from api.jobs import IngestBacklog, SqliteIngestJobStore
from api.raw_documents import RawDocumentStore
from tests.factories import wait_until
from tests.routes_http.support import (
    ApiTestContext,
    hermetic_app,
    make_settings,
    raw_storage_client,
    upload_and_wait,
    wait_for_job,
)


def test_document_upload_ingests_chunks(api_context: ApiTestContext) -> None:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")

    assert status["state"] == "done"
    assert status["result"] == {
        "filename": "guide.txt",
        "sections_parsed": 1,
        "chunks_ingested": 1,
        "collection_name": "api_documents",
    }
    records, _ = api_context.qdrant.scroll(
        collection_name="api_documents",
        with_payload=True,
        with_vectors=False,
    )
    assert len(records) == 1
    assert records[0].payload is not None
    assert records[0].payload["filename"] == "guide.txt"
    assert records[0].payload["chunk_id"]


def test_document_upload_rejects_file_over_size_limit() -> None:
    """A single oversized upload must be rejected with 413 before it can OOM the
    process — proven by capping the limit far below the test payload's size."""

    clear_dependency_caches()
    settings = make_settings(max_upload_bytes=10)
    qdrant = QdrantClient(":memory:")
    app = hermetic_app(settings, qdrant)
    with TestClient(app) as client:
        response = client.post(
            "/documents",
            files={
                "file": ("guide.txt", b"this content is definitely over ten bytes", "text/plain")
            },
        )
    clear_dependency_caches()

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_document_upload_rejects_unsupported_file(api_context: ApiTestContext) -> None:
    """Rejected at upload time with a 400 — it used to be accepted (202) and only fail
    once the client polled the background job."""

    response = api_context.client.post(
        "/documents", files={"file": ("archive.zip", b"not supported", "application/zip")}
    )

    assert response.status_code == 400
    assert "Unsupported document type" in response.json()["detail"]


@pytest.mark.parametrize("filename", ["..", "."])
def test_document_upload_rejects_unusable_filenames(
    api_context: ApiTestContext, filename: str
) -> None:
    response = api_context.client.post(
        "/documents", files={"file": (filename, b"alpha", "text/plain")}
    )

    assert response.status_code == 400


def test_document_upload_rejects_a_filename_longer_than_the_filesystem_allows(
    api_context: ApiTestContext,
) -> None:
    long_name = "é" * 130 + ".txt"  # 264 UTF-8 bytes, 134 characters

    response = api_context.client.post(
        "/documents", files={"file": (long_name, b"alpha", "text/plain")}
    )

    assert response.status_code == 400
    assert "255 bytes" in response.json()["detail"]


def test_a_document_indexed_under_an_overlong_name_does_not_break_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_name = "a" * 300 + ".txt"
    with raw_storage_client(tmp_path, monkeypatch) as client:
        # An upload from before the cap: indexed, but its original could never be saved.
        monkeypatch.setattr("api.documents.MAX_FILENAME_BYTES", 10_000)
        assert upload_and_wait(client, long_name, b"alpha beta")["state"] == "done"
        monkeypatch.setattr("api.documents.MAX_FILENAME_BYTES", 255)

        listing = client.get("/documents")
        assert listing.status_code == 200
        assert long_name in listing.json()["filenames"]
        assert client.delete(f"/documents/{quote(long_name)}").status_code == 200


def test_document_upload_releases_its_backlog_bytes_when_the_job_finishes(
    api_context: ApiTestContext,
) -> None:
    backlog = IngestBacklog(max_bytes=1024)
    api_context.app.dependency_overrides[get_ingest_backlog] = lambda: backlog

    assert upload_and_wait(api_context.client, "guide.txt", b"alpha beta")["state"] == "done"

    wait_until(lambda: backlog.pending_bytes == 0, message="backlog bytes never released")


def test_document_upload_during_shutdown_fails_the_job_and_frees_its_bytes(
    api_context: ApiTestContext,
) -> None:
    """An executor that is already shut down used to leave the created job "queued"
    forever and answer with an unhandled 500."""

    from concurrent.futures import ThreadPoolExecutor

    from api.dependencies import get_ingest_executor
    from api.jobs import IngestJobStore

    stopped = ThreadPoolExecutor(max_workers=1)
    stopped.shutdown()
    backlog = IngestBacklog(max_bytes=1024)
    job_store = IngestJobStore(max_retained=10)
    api_context.app.dependency_overrides[get_ingest_executor] = lambda: stopped
    api_context.app.dependency_overrides[get_ingest_backlog] = lambda: backlog
    api_context.app.dependency_overrides[get_ingest_job_store] = lambda: job_store

    response = api_context.client.post(
        "/documents", files={"file": ("guide.txt", b"alpha beta", "text/plain")}
    )

    assert response.status_code == 503
    assert response.headers["retry-after"]
    assert backlog.pending_bytes == 0
    jobs = list(job_store._jobs.values())
    assert [job.state for job in jobs] == ["failed"]
    assert "shutting down" in (jobs[0].error or "")


def test_shutdown_lets_a_running_ingest_finish(api_context: ApiTestContext) -> None:
    """Shutdown cancels only *queued* jobs. One already embedding must still complete —
    a redeploy mid-upload would otherwise leave half a new version next to the old."""

    import threading

    from api.dependencies import shutdown_executors

    embedding_started = threading.Event()
    release = threading.Event()
    inner = api_context.embeddings

    class BlockingEmbeddings:
        def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
            embedding_started.set()
            assert release.wait(5)
            return inner.embed_texts(texts)

    api_context.app.dependency_overrides[get_embedding_provider] = lambda: BlockingEmbeddings()
    response = api_context.client.post(
        "/documents", files={"file": ("slow.txt", b"alpha beta", "text/plain")}
    )
    job_id = response.json()["job_id"]
    assert embedding_started.wait(5)

    shutdown_executors()
    release.set()

    assert wait_for_job(api_context.client, job_id)["state"] == "done"


def test_a_queued_job_cancelled_by_shutdown_is_failed_and_frees_its_bytes() -> None:
    from concurrent.futures import Future

    from api.ingest_jobs import finish_ingest_future
    from api.jobs import INTERRUPTED_JOB_ERROR, IngestJobStore

    backlog = IngestBacklog(max_bytes=100)
    assert backlog.try_reserve(40)
    job_store = IngestJobStore(max_retained=10)
    job = job_store.create("queued.txt")
    future: Future[None] = Future()
    future.add_done_callback(
        lambda done: finish_ingest_future(done, lambda: backlog.release(40), job_store, job.id)
    )

    assert future.cancel()

    assert backlog.pending_bytes == 0
    finished = job_store.get(job.id)
    assert finished is not None
    assert (finished.state, finished.error) == ("failed", INTERRUPTED_JOB_ERROR)


def test_document_job_status_returns_404_for_unknown_job(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/documents/jobs/does-not-exist")

    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_sqlite_job_store_backend_survives_a_real_upload_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end proof the job_store_backend="sqlite" DI wiring actually works, not
    just SqliteIngestJobStore in isolation: upload through the real API, then open a
    brand-new store instance pointed at the same file — simulating the process
    restarting between the ingest completing and the client's next poll — and confirm
    the job is still readable.

    get_ingest_job_store() (api/dependencies.py) reads settings via a direct
    get_app_settings() call, not FastAPI's Depends — same as the warmup tests above,
    that call has to be patched at the name each module actually holds.
    """

    clear_dependency_caches()
    db_path = str(tmp_path / "jobs.db")
    settings = make_settings(job_store_backend="sqlite", job_store_path=db_path)
    qdrant = QdrantClient(":memory:")
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)

    app = hermetic_app(settings, qdrant)

    try:
        with TestClient(app) as client:
            status = upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")
        assert status["state"] == "done"
        job_id = status["job_id"]
    finally:
        # Undo the monkeypatches now (not at test end) so clear_dependency_caches()
        # below calls the real, cache_clear-bearing get_app_settings — it resolves
        # that name through this module's own dependencies.py, which is still patched
        # to a bare lambda until this point.
        monkeypatch.undo()
        clear_dependency_caches()

    # A fresh store instance, as a new process would create — proves persistence,
    # not just that the same in-memory object answered its own question.
    reloaded_store = SqliteIngestJobStore(path=db_path, max_retained=50)
    reloaded = reloaded_store.get(job_id)
    assert reloaded is not None
    assert reloaded.state == "done"
    assert reloaded.result == status["result"]


def test_list_documents_returns_indexed_filenames(api_context: ApiTestContext) -> None:
    assert upload_and_wait(api_context.client, "a.txt", b"alpha")["state"] == "done"
    assert upload_and_wait(api_context.client, "b.txt", b"beta")["state"] == "done"

    response = api_context.client.get("/documents")

    assert response.status_code == 200
    assert response.json()["filenames"] == ["a.txt", "b.txt"]


def test_list_documents_includes_page_count_and_upload_metadata(
    api_context: ApiTestContext,
) -> None:
    content = b"alpha beta gamma"
    assert upload_and_wait(api_context.client, "a.txt", content)["state"] == "done"

    body = api_context.client.get("/documents").json()

    assert body["page_counts"]["a.txt"] == 1
    assert body["byte_sizes"]["a.txt"] == len(content)
    assert body["uploaded_ats"]["a.txt"] > 0


def test_list_documents_empty_for_no_uploads(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/documents")

    assert response.status_code == 200
    assert response.json()["filenames"] == []


def test_list_documents_returns_chunk_counts_per_filename(api_context: ApiTestContext) -> None:
    assert upload_and_wait(api_context.client, "a.txt", b"alpha")["state"] == "done"
    assert upload_and_wait(api_context.client, "b.txt", b"beta")["state"] == "done"

    response = api_context.client.get("/documents").json()

    assert response["chunk_counts"] == {"a.txt": 1, "b.txt": 1}


def test_list_documents_chunk_counts_replaced_not_accumulated_on_reupload(
    api_context: ApiTestContext,
) -> None:
    assert upload_and_wait(api_context.client, "a.txt", b"alpha")["state"] == "done"
    assert upload_and_wait(api_context.client, "a.txt", b"alpha again")["state"] == "done"

    response = api_context.client.get("/documents").json()

    assert response["chunk_counts"] == {"a.txt": 1}


def test_list_documents_chunk_counts_drop_deleted_filename(api_context: ApiTestContext) -> None:
    assert upload_and_wait(api_context.client, "a.txt", b"alpha")["state"] == "done"
    assert upload_and_wait(api_context.client, "b.txt", b"beta")["state"] == "done"

    api_context.client.delete("/documents/a.txt")

    response = api_context.client.get("/documents").json()

    assert response["chunk_counts"] == {"b.txt": 1}


def test_delete_document_removes_indexed_chunks(api_context: ApiTestContext) -> None:
    assert upload_and_wait(api_context.client, "a.txt", b"alpha")["state"] == "done"

    response = api_context.client.delete("/documents/a.txt")

    assert response.status_code == 200
    assert response.json() == {"filename": "a.txt", "points_deleted": 1}
    assert api_context.client.get("/documents").json()["filenames"] == []


def test_delete_document_returns_404_for_unknown_filename(api_context: ApiTestContext) -> None:
    response = api_context.client.delete("/documents/does-not-exist.txt")

    assert response.status_code == 404
    assert "does-not-exist.txt" in response.json()["detail"]


def test_delete_document_leaves_other_filenames_searchable(api_context: ApiTestContext) -> None:
    assert upload_and_wait(api_context.client, "a.txt", b"alpha")["state"] == "done"
    assert upload_and_wait(api_context.client, "b.txt", b"beta")["state"] == "done"

    response = api_context.client.delete("/documents/a.txt")

    assert response.status_code == 200
    assert response.json()["points_deleted"] == 1
    assert api_context.client.get("/documents").json()["filenames"] == ["b.txt"]

    question_response = api_context.client.post("/questions", json={"question": "beta"})
    assert question_response.status_code == 200


def test_document_content_returns_chunks_in_ordinal_order(api_context: ApiTestContext) -> None:
    status = upload_and_wait(api_context.client, "a.md", b"Intro\nalpha beta gamma")
    assert status["state"] == "done"

    response = api_context.client.get("/documents/a.md/content")

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "a.md"
    assert len(body["chunks"]) >= 1
    ordinals = [chunk["chunk_ordinal"] for chunk in body["chunks"]]
    assert ordinals == sorted(o for o in ordinals if o is not None)
    assert all({"chunk_id", "page", "section", "text"} <= set(chunk) for chunk in body["chunks"])


def test_document_content_returns_404_for_unknown_filename(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/documents/missing.md/content")

    assert response.status_code == 404


def test_document_original_returns_404_when_raw_storage_is_off(
    api_context: ApiTestContext,
) -> None:
    """raw_document_dir is unset by default (make_settings() doesn't set it) — the
    same 404 a client would get for a genuinely unknown filename, not a distinct
    "feature unavailable" response."""

    assert upload_and_wait(api_context.client, "guide.txt", b"alpha")["state"] == "done"

    response = api_context.client.get("/documents/guide.txt/original")

    assert response.status_code == 404


def test_document_original_returns_the_exact_uploaded_bytes_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: raw_document_dir set -> upload -> GET .../original returns the
    identical bytes, and deleting the document removes the stored original too.

    get_raw_document_store() (api/dependencies.py) reads settings via a direct
    get_app_settings() call, not FastAPI's Depends — same pattern as the sqlite job
    store test above, so the patch has to target the name each module actually holds.
    """

    clear_dependency_caches()
    settings = make_settings(raw_document_dir=str(tmp_path))
    qdrant = QdrantClient(":memory:")
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)

    app = hermetic_app(settings, qdrant)

    content = b"the exact original bytes\nwith a newline"
    try:
        with TestClient(app) as client:
            assert upload_and_wait(client, "guide.txt", content)["state"] == "done"

            response = client.get("/documents/guide.txt/original")
            assert response.status_code == 200
            assert response.content == content
            assert 'filename="guide.txt"' in response.headers["content-disposition"]

            delete_response = client.delete("/documents/guide.txt")
            assert delete_response.status_code == 200

            after_delete = client.get("/documents/guide.txt/original")
            assert after_delete.status_code == 404
    finally:
        monkeypatch.undo()
        clear_dependency_caches()


def test_document_original_serves_a_non_latin1_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-built latin-1 Content-Disposition header 500'd on names like these."""

    filename = "日本 notes.txt"
    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert upload_and_wait(client, filename, b"alpha beta")["state"] == "done"

        response = client.get(f"/documents/{quote(filename)}/original")

        assert response.status_code == 200
        assert response.content == b"alpha beta"
        assert "filename*=utf-8''" in response.headers["content-disposition"]
        assert response.headers["x-content-type-options"] == "nosniff"


def test_failed_ingest_does_not_store_or_replace_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert upload_and_wait(client, "guide.txt", b"version one")["state"] == "done"
        # Whitespace-only: parses to nothing, so this ingest fails.
        assert upload_and_wait(client, "guide.txt", b"   \n  ")["state"] == "failed"
        assert upload_and_wait(client, "empty.txt", b"   ")["state"] == "failed"

        # The still-indexed version keeps its original; the failed upload left none.
        assert client.get("/documents/guide.txt/original").content == b"version one"
        assert client.get("/documents/empty.txt/original").status_code == 404
    assert sorted(path.name for path in tmp_path.iterdir()) == ["guide.txt"]


def test_failed_original_save_drops_the_previous_original_instead_of_serving_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-upload indexes fine but storing its bytes fails (disk full): the previous
    version's original must not keep being served as if it matched the index."""

    from api.raw_documents import RawDocumentStore

    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert upload_and_wait(client, "guide.txt", b"version one")["state"] == "done"

        def disk_full(self: RawDocumentStore, filename: str, content: bytes) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(RawDocumentStore, "save", disk_full)
        status = upload_and_wait(client, "guide.txt", b"version two, indexed")

        # The index did update, so the job still reports success...
        assert status["state"] == "done"
        # ...and the stale original is gone rather than silently mismatched.
        assert client.get("/documents/guide.txt/original").status_code == 404
    assert not (tmp_path / "guide.txt").exists()


def test_delete_removes_an_orphaned_original_with_no_indexed_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An original left behind by an older release's failed ingest was undeletable:
    DELETE 404'd on "no points" before it reached the stored file."""

    (tmp_path / "orphan.txt").write_bytes(b"leftover")
    with raw_storage_client(tmp_path, monkeypatch) as client:
        response = client.delete("/documents/orphan.txt")

        assert response.status_code == 200
        assert response.json()["points_deleted"] == 0
        assert client.delete("/documents/orphan.txt").status_code == 404
    assert not (tmp_path / "orphan.txt").exists()


def test_delete_document_waits_for_the_filename_write_lock(api_context: ApiTestContext) -> None:
    """DELETE must take the same per-filename lock ingest_chunks does.

    Snapshot-then-delete is a read-modify-write, so a delete landing between a
    concurrent ingest's upsert and its stale-cleanup would remove the points that
    ingest just wrote while the ingest still reported success. Holding the lock here
    and asserting the request can't finish proves the route participates in it —
    guarding only the ingest path left this half of the race open.
    """

    upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    statuses: list[int] = []

    def delete_in_background() -> None:
        statuses.append(api_context.client.delete("/documents/guide.txt").status_code)

    worker = threading.Thread(target=delete_in_background, daemon=True)
    with filename_write_lock("guide.txt"):
        worker.start()
        worker.join(timeout=0.75)
        assert worker.is_alive(), "DELETE finished while the filename write lock was held"
        assert statuses == []

    worker.join(timeout=15)
    assert not worker.is_alive(), "DELETE never completed after the lock was released"
    assert statuses == [200]


def test_a_delete_overtaken_by_a_newer_upload_at_the_lock_keeps_that_upload(
    api_context: ApiTestContext,
) -> None:
    """The filename lock isn't fair: an upload admitted after a DELETE can reach it
    first. The delete used to run anyway and remove the document just re-uploaded."""

    from api.ingestion import ingest_sequencer

    client = api_context.client
    upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")
    statuses: list[tuple[int, str]] = []

    def delete_in_background() -> None:
        response = client.delete("/documents/guide.txt")
        statuses.append((response.status_code, response.json()["detail"]))

    worker = threading.Thread(target=delete_in_background, daemon=True)
    with filename_write_lock("guide.txt"):
        worker.start()
        wait_until(
            lambda: (
                (entry := ingestion._filename_locks.get("guide.txt")) is not None
                and entry.users >= 2
            ),
            message="the DELETE never reached the filename lock",
        )
        # What a newer upload does when it wins the lock: indexed and marked applied.
        newer = ingest_sequencer.admit("guide.txt")
        ingest_sequencer.mark_applied("guide.txt", newer)
        ingest_sequencer.release("guide.txt")

    worker.join(timeout=15)
    assert statuses and statuses[0][0] == 409
    assert "newer upload" in statuses[0][1]
    assert client.get("/documents").json()["filenames"] == ["guide.txt"]


def test_reindex_rechunks_a_stale_document_from_its_original_and_keeps_its_upload_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.pipeline as pipeline_module
    from api.documents import CHUNKER_VERSION

    with raw_storage_client(tmp_path, monkeypatch) as client:
        # Indexed by an older chunker, as a corpus from before a CHUNKER_VERSION bump is.
        with pytest.MonkeyPatch.context() as older_chunker:
            older_chunker.setattr(pipeline_module, "CHUNKER_VERSION", CHUNKER_VERSION - 1)
            uploaded = upload_and_wait(client, "guide.txt", b"Intro\nalpha beta gamma")
            assert uploaded["state"] == "done"
        listing = client.get("/documents").json()
        assert listing["stale_filenames"] == ["guide.txt"]
        assert listing["reindexable_filenames"] == ["guide.txt"]
        uploaded_at = listing["uploaded_ats"]["guide.txt"]

        response = client.post("/documents/guide.txt/reindex")
        assert response.status_code == 202
        assert response.json()["filename"] == "guide.txt"
        assert wait_for_job(client, response.json()["job_id"])["state"] == "done"

        listing = client.get("/documents").json()
        assert listing["stale_filenames"] == []
        # A re-index is not a new upload: the document keeps its original timestamp.
        assert listing["uploaded_ats"]["guide.txt"] == uploaded_at
        # The original is untouched and still served.
        assert client.get("/documents/guide.txt/original").content == b"Intro\nalpha beta gamma"


def test_reindex_all_requeues_only_stale_documents_with_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.pipeline as pipeline_module
    from api.documents import CHUNKER_VERSION

    with raw_storage_client(tmp_path, monkeypatch) as client:
        with pytest.MonkeyPatch.context() as older_chunker:
            older_chunker.setattr(pipeline_module, "CHUNKER_VERSION", CHUNKER_VERSION - 1)
            assert upload_and_wait(client, "old.txt", b"Intro\nalpha")["state"] == "done"
        assert upload_and_wait(client, "current.txt", b"Intro\nbeta")["state"] == "done"

        response = client.post("/documents/reindex")

        assert response.status_code == 202
        body = response.json()
        assert [job["filename"] for job in body["jobs"]] == ["old.txt"]
        assert body["deferred"] == []
        assert wait_for_job(client, body["jobs"][0]["job_id"])["state"] == "done"
        assert client.get("/documents").json()["stale_filenames"] == []


def test_reindex_all_defers_what_the_backlog_cannot_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.pipeline as pipeline_module
    from api.documents import CHUNKER_VERSION

    with raw_storage_client(tmp_path, monkeypatch) as client:
        with pytest.MonkeyPatch.context() as older_chunker:
            older_chunker.setattr(pipeline_module, "CHUNKER_VERSION", CHUNKER_VERSION - 1)
            assert upload_and_wait(client, "old.txt", b"Intro\nalpha")["state"] == "done"
        full = IngestBacklog(max_bytes=4)
        assert full.try_reserve(4)
        client.app.dependency_overrides[get_ingest_backlog] = lambda: full  # type: ignore[attr-defined]
        reads: list[str] = []
        real_read = RawDocumentStore.read

        def counting_read(self: RawDocumentStore, filename: str) -> bytes | None:
            reads.append(filename)
            return real_read(self, filename)

        monkeypatch.setattr(RawDocumentStore, "read", counting_read)

        response = client.post("/documents/reindex")

        assert response.status_code == 202
        assert response.json() == {"jobs": [], "deferred": ["old.txt"]}
        assert reads == []  # deferred without reading the original into memory


def test_reindex_is_a_409_without_raw_storage(api_context: ApiTestContext) -> None:
    off = api_context.client.post("/documents/guide.txt/reindex")
    assert off.status_code == 409
    assert "RAW_DOCUMENT_DIR" in off.json()["detail"]
    assert api_context.client.post("/documents/reindex").status_code == 409
    assert api_context.client.get("/config").json()["raw_documents_enabled"] is False


def test_reindex_is_a_404_without_a_stored_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert client.get("/config").json()["raw_documents_enabled"] is True
        assert client.post("/documents/never-uploaded.txt/reindex").status_code == 404


def test_a_delete_while_a_reindex_is_queued_stays_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The re-index read the original at enqueue time; without the precondition a DELETE
    landing while it waited in the queue was silently undone when the job ran."""

    import threading
    from concurrent.futures import ThreadPoolExecutor

    from api.dependencies import get_ingest_executor

    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        single = ThreadPoolExecutor(max_workers=1)
        busy = threading.Event()
        single.submit(busy.wait, 5)  # the one worker is busy: the re-index queues
        client.app.dependency_overrides[get_ingest_executor] = lambda: single  # type: ignore[attr-defined]
        try:
            job_id = client.post("/documents/guide.txt/reindex").json()["job_id"]
            assert client.delete("/documents/guide.txt").status_code == 200
            busy.set()

            job = wait_for_job(client, job_id)
        finally:
            busy.set()
            single.shutdown(wait=True)

        assert job["state"] == "failed"
        assert "deleted before its re-index ran" in job["error"]
        assert "guide.txt" not in client.get("/documents").json()["filenames"]


def test_a_delete_while_an_upload_is_queued_stays_deleted(api_context: ApiTestContext) -> None:
    """An upload admitted before a DELETE used to index the document straight back once
    it reached the front of the queue."""

    import threading
    from concurrent.futures import ThreadPoolExecutor

    from api.dependencies import get_ingest_executor

    client = api_context.client
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
    single = ThreadPoolExecutor(max_workers=1)
    busy = threading.Event()
    single.submit(busy.wait, 5)  # the one worker is busy: the re-upload queues
    api_context.app.dependency_overrides[get_ingest_executor] = lambda: single
    try:
        response = client.post(
            "/documents", files={"file": ("guide.txt", b"Intro\nalpha gamma", "text/plain")}
        )
        job_id = response.json()["job_id"]
        assert client.delete("/documents/guide.txt").status_code == 200
        busy.set()

        job = wait_for_job(client, job_id)
    finally:
        busy.set()
        single.shutdown(wait=True)

    assert job["state"] == "failed"
    assert "was deleted after this upload was queued" in job["error"]
    assert "guide.txt" not in client.get("/documents").json()["filenames"]


def test_an_upload_after_a_delete_is_indexed(api_context: ApiTestContext) -> None:
    client = api_context.client
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
    assert client.delete("/documents/guide.txt").status_code == 200

    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
    assert "guide.txt" in client.get("/documents").json()["filenames"]


def test_a_delete_succeeds_and_invalidates_cached_answers_when_removing_the_original_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The points are gone by then: a 500 told the user a delete that happened failed."""

    from api.corpus import corpus_generation

    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        before = corpus_generation()

        def broken_delete(self: RawDocumentStore, filename: str) -> bool:
            raise PermissionError("read-only volume")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(RawDocumentStore, "delete", broken_delete)
            response = client.delete("/documents/guide.txt")

        assert response.status_code == 200
        assert response.json()["points_deleted"] > 0
        assert corpus_generation() > before
        assert client.get("/documents").json()["filenames"] == []
        # The orphaned original goes with a second DELETE, through the no-points path.
        assert client.delete("/documents/guide.txt").status_code == 200
    assert not (tmp_path / "guide.txt").exists()


def test_a_queued_reindex_does_not_overwrite_a_newer_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The re-index captured v1's bytes at enqueue; a v2 upload indexed while it waited
    in the queue must win, or the index silently goes back to v1 while /original serves
    v2 and nothing flags it (the chunker version is current either way)."""

    from concurrent.futures import ThreadPoolExecutor

    from api.dependencies import get_ingest_executor

    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nversion one")["state"] == "done"
        single = ThreadPoolExecutor(max_workers=1)
        busy = threading.Event()
        single.submit(busy.wait, 5)
        overrides = client.app.dependency_overrides  # type: ignore[attr-defined]
        overrides[get_ingest_executor] = lambda: single
        try:
            reindex_id = client.post("/documents/guide.txt/reindex").json()["job_id"]
            # The upload runs on a free executor, so it is applied first.
            overrides.pop(get_ingest_executor)
            assert upload_and_wait(client, "guide.txt", b"Intro\nversion two")["state"] == "done"
            busy.set()
            job = wait_for_job(client, reindex_id)
        finally:
            busy.set()
            single.shutdown(wait=True)

        assert job["state"] == "failed"
        assert "newer version" in job["error"]
        chunks = client.get("/documents/guide.txt/content").json()["chunks"]
        assert "version two" in " ".join(chunk["text"] for chunk in chunks)
        assert client.get("/documents/guide.txt/original").content == b"Intro\nversion two"


def test_the_backlog_caps_in_flight_jobs_at_the_job_store_retention() -> None:
    backlog = IngestBacklog(max_bytes=1000, max_jobs=2)

    assert backlog.try_reserve(1)
    assert backlog.try_reserve(1)
    assert not backlog.try_reserve(1)
    backlog.release(1)
    assert backlog.try_reserve(1)


def test_document_routes_do_not_normalize_a_name_into_another_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raw store normalizes names itself, so DELETE /documents/%20a.txt used to
    delete a.txt's original — outside a.txt's lock — while leaving its points."""

    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert upload_and_wait(client, "a.txt", b"Intro\nalpha")["state"] == "done"

        for name in (" a.txt", "x%5Ca.txt"):
            assert client.delete(f"/documents/{quote(name)}").status_code == 404
            assert client.get(f"/documents/{quote(name)}/content").status_code == 404
            assert client.get(f"/documents/{quote(name)}/original").status_code == 404
            assert client.post(f"/documents/{quote(name)}/reindex").status_code == 404

        assert client.get("/documents/a.txt/original").content == b"Intro\nalpha"
        assert client.get("/documents").json()["filenames"] == ["a.txt"]


def test_tag_errors_are_4xx(api_context: ApiTestContext) -> None:
    client = api_context.client
    assert upload_and_wait(client, "a.txt", b"Intro\nalpha")["state"] == "done"

    assert client.patch("/documents/missing.txt/tags", json={"tags": ["x"]}).status_code == 404
    assert client.patch("/documents/a.txt/tags", json={"tags": ["   "]}).status_code == 400
    assert client.patch("/documents/a.txt/tags", json={"tags": ["x" * 41]}).status_code == 400
    too_many = {"tags": [f"t{i}" for i in range(21)]}
    assert client.patch("/documents/a.txt/tags", json=too_many).status_code == 422


def test_a_tag_change_made_while_a_re_upload_waits_for_the_lock_survives_it(
    api_context: ApiTestContext,
) -> None:
    """Tags used to be read before the ingest took the filename lock, so a PATCH that
    got the lock first was overwritten when the re-upload's points replaced the old ones."""

    import api.ingestion as ingestion
    from api.repository import VectorRepository

    client = api_context.client
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha")["state"] == "done"
    assert client.patch("/documents/guide.txt/tags", json={"tags": ["old"]}).status_code == 200

    with filename_write_lock("guide.txt"):
        job_id = client.post(
            "/documents", files={"file": ("guide.txt", b"Intro\nalpha v2", "text/plain")}
        ).json()["job_id"]
        # The ingest has parsed and is now queued on the lock this test holds.
        wait_until(
            lambda: (
                (entry := ingestion._filename_locks.get("guide.txt")) is not None
                and entry.users >= 2
            ),
            message="the re-upload never reached the filename lock",
        )
        # What PATCH does, as the holder of the lock.
        VectorRepository(api_context.qdrant).set_tags(api_context.settings, "guide.txt", ["new"])

    assert wait_for_job(client, job_id)["state"] == "done"
    assert client.get("/documents").json()["tags"] == {"guide.txt": ["new"]}


def test_document_content_pages_around_a_chunk_or_by_ordinal_range(
    api_context: ApiTestContext,
) -> None:
    client = api_context.client
    body = "\n\n".join(
        f"Section {i}\n" + " ".join(f"word{i}x{j}" for j in range(30)) for i in range(8)
    )
    assert upload_and_wait(client, "long.txt", body.encode())["state"] == "done"
    everything = client.get("/documents/long.txt/content").json()
    chunks = everything["chunks"]
    total = len(chunks)
    assert total >= 6 and everything["total_chunks"] == total
    middle = chunks[3]

    around = client.get(
        "/documents/long.txt/content", params={"around": middle["chunk_id"], "radius": 1}
    ).json()
    assert [c["chunk_ordinal"] for c in around["chunks"]] == [3, 4, 5]
    assert around["total_chunks"] == total and around["target_found"] is True

    missing = client.get(
        "/documents/long.txt/content", params={"around": "gone", "radius": 1}
    ).json()
    assert missing["target_found"] is False
    assert [c["chunk_ordinal"] for c in missing["chunks"]] == [1, 2]

    page = client.get("/documents/long.txt/content", params={"start": 2, "end": 3}).json()
    assert [c["chunk_ordinal"] for c in page["chunks"]] == [2, 3]

    bad = client.get("/documents/long.txt/content", params={"start": 5, "end": 2})
    assert bad.status_code == 422
    assert client.get("/documents/long.txt/content", params={"radius": 1000}).status_code == 422
    assert client.get("/documents/none.txt/content", params={"start": 1}).status_code == 404


def test_unpaged_document_content_is_capped_but_reports_the_real_total(
    api_context: ApiTestContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("api.routes.documents.CONTENT_UNPAGED_MAX_CHUNKS", 3)
    client = api_context.client
    body = "\n\n".join(
        f"Section {i}\n" + " ".join(f"word{i}x{j}" for j in range(30)) for i in range(8)
    )
    assert upload_and_wait(client, "long.txt", body.encode())["state"] == "done"

    content = client.get("/documents/long.txt/content").json()

    assert [c["chunk_ordinal"] for c in content["chunks"]] == [1, 2, 3]
    assert content["total_chunks"] > 3


def test_with_stored_originals_a_name_differing_only_by_case_or_accents_is_a_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a case-insensitive disk both names are one original file: the second upload
    replaced the first document's bytes, and deleting either removed both."""

    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert upload_and_wait(client, "Report.pdf.txt", b"Intro\nalpha")["state"] == "done"
        assert upload_and_wait(client, "R\u00e9sum\u00e9.txt", b"Intro\nbeta")["state"] == "done"

        for clash in ("report.PDF.txt", "Re\u0301sume\u0301.txt"):
            response = client.post(
                "/documents", files={"file": (clash, b"Intro\ngamma", "text/plain")}
            )
            assert response.status_code == 409, clash
            assert "only by letter case" in response.json()["detail"]
        # The exact same name is a re-upload, always allowed.
        assert upload_and_wait(client, "Report.pdf.txt", b"Intro\nv2")["state"] == "done"


def test_without_stored_originals_names_differing_by_case_are_separate_documents(
    api_context: ApiTestContext,
) -> None:
    client = api_context.client
    assert upload_and_wait(client, "Report.txt", b"Intro\nalpha")["state"] == "done"
    assert upload_and_wait(client, "report.txt", b"Intro\nbeta")["state"] == "done"
    assert client.get("/documents").json()["filenames"] == ["Report.txt", "report.txt"]


def test_changing_the_chunk_settings_lists_documents_as_stale_and_reindex_all_fixes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a CHUNKER_VERSION bump used to count; a new CHUNK_SIZE_TOKENS left every
    document chunked the old way with nothing to say so."""

    with raw_storage_client(tmp_path, monkeypatch) as client:
        import api.dependencies

        settings = api.dependencies.get_app_settings()  # patched to the client's settings
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        assert client.get("/documents").json()["stale_filenames"] == []

        settings.__dict__["chunk_size_tokens"] = settings.chunk_size_tokens - 1
        assert client.get("/documents").json()["stale_filenames"] == ["guide.txt"]

        accepted = client.post("/documents/reindex").json()
        assert [job["filename"] for job in accepted["jobs"]] == ["guide.txt"]
        assert wait_for_job(client, accepted["jobs"][0]["job_id"])["state"] == "done"
        assert client.get("/documents").json()["stale_filenames"] == []
