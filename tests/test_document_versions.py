"""Earlier versions of stored originals (RAW_DOCUMENT_VERSIONS)."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.raw_documents import VERSIONS_DIR, RawDocumentStore
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait, wait_for_job


def test_a_save_keeps_the_newest_earlier_versions(tmp_path: Path) -> None:
    store = RawDocumentStore(str(tmp_path), keep_versions=2)
    for n in range(4):
        store.save("a.txt", f"v{n}".encode())
    store.save("a.txt", b"v3")  # the same bytes again: no new version

    versions = store.versions("a.txt")
    assert [store.read_version("a.txt", v.version) for v in versions] == [b"v2", b"v1"]
    assert store.read("a.txt") == b"v3"
    # Versions are a directory: never listed as a stored original.
    assert store.stored_names() == {"a.txt"}
    assert store.read_version("a.txt", "../a.txt") is None
    # Deleting the document deletes its history.
    assert store.delete("a.txt")
    assert store.versions("a.txt") == []


def test_versions_are_off_by_default(tmp_path: Path) -> None:
    store = RawDocumentStore(str(tmp_path))
    store.save("a.txt", b"one")
    store.save("a.txt", b"two")
    assert store.versions("a.txt") == []
    assert not (tmp_path / VERSIONS_DIR).exists()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    clear_dependency_caches()
    settings = make_settings(
        raw_document_dir=str(tmp_path / "raw"), raw_document_versions=3, workspaces_enabled=True
    )
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as test_client:
        yield test_client
    monkeypatch.undo()
    clear_dependency_caches()


def test_a_re_upload_keeps_the_last_version_and_it_can_be_restored(client: TestClient) -> None:
    assert upload_and_wait(client, "p.txt", b"Intro\nfirst version")["state"] == "done"
    assert upload_and_wait(client, "p.txt", b"Intro\nsecond version")["state"] == "done"

    (version,) = client.get("/documents/p.txt/versions").json()["versions"]
    restored = client.post(f"/documents/p.txt/versions/{version['version']}/restore")
    assert restored.status_code == 202
    assert wait_for_job(client, restored.json()["job_id"])["state"] == "done"

    text = " ".join(c["text"] for c in client.get("/documents/p.txt/content").json()["chunks"])
    assert "first version" in text
    # The version it replaced is kept in its turn.
    assert len(client.get("/documents/p.txt/versions").json()["versions"]) == 2
    assert client.post("/documents/p.txt/versions/123/restore").status_code == 404


def test_a_workspace_whose_documents_are_gone_deletes_despite_their_history(
    client: TestClient,
) -> None:
    ws = {"X-Workspace": "legal"}
    assert client.post("/workspaces", json={"name": "legal"}).status_code == 201
    assert upload_and_wait(client, "p.txt", b"Intro\none", headers=ws)["state"] == "done"
    assert upload_and_wait(client, "p.txt", b"Intro\ntwo", headers=ws)["state"] == "done"
    assert client.delete("/documents/p.txt", headers=ws).status_code == 200
    assert client.delete("/workspaces/legal").status_code == 200


def test_a_version_is_not_restored_while_the_document_is_indexing(client: TestClient) -> None:
    from api.ingestion import ingest_sequencer

    assert upload_and_wait(client, "p.txt", b"Intro\nfirst version")["state"] == "done"
    assert upload_and_wait(client, "p.txt", b"Intro\nsecond version")["state"] == "done"
    (version,) = client.get("/documents/p.txt/versions").json()["versions"]

    ingest_sequencer.admit("p.txt")  # an upload of it is being indexed
    try:
        refused = client.post(f"/documents/p.txt/versions/{version['version']}/restore")
    finally:
        ingest_sequencer.release("p.txt")
    assert refused.status_code == 409
    assert "already being indexed" in refused.json()["detail"]


def test_a_restored_version_of_a_fetched_document_can_still_be_refreshed(
    client: TestClient,
) -> None:
    from api.documents import DocumentSource
    from api.repository import VectorRepository

    assert upload_and_wait(client, "p.txt", b"Intro\nfirst version")["state"] == "done"
    assert upload_and_wait(client, "p.txt", b"Intro\nsecond version")["state"] == "done"
    (version,) = client.get("/documents/p.txt/versions").json()["versions"]
    seen: list[DocumentSource | None] = []
    original = VectorRepository.document_metadata

    def with_source(self: VectorRepository, settings: object, name: str) -> object:
        meta = original(self, settings, name)  # type: ignore[arg-type]
        return meta.__class__(
            **{
                **meta.__dict__,
                "source_url": "https://example.com/p.txt",
                "source_etag": '"v2"',
                "source_last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            }
        )

    import api.ingest_jobs as ingest_jobs

    real_source_of = ingest_jobs.source_of

    def recording(meta: object, **options: bool) -> DocumentSource | None:
        found = real_source_of(meta, **options)  # type: ignore[arg-type]
        seen.append(found)
        return found

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(VectorRepository, "document_metadata", with_source)
        patch.setattr("api.routes.documents.source_of", recording)
        restored = client.post(f"/documents/p.txt/versions/{version['version']}/restore")
        assert restored.status_code == 202
        assert wait_for_job(client, restored.json()["job_id"])["state"] == "done"
    # The URL, not the validators: they describe the version being replaced, and a
    # refresh sending them would get a 304 and keep the restored version for good.
    assert seen == [DocumentSource("https://example.com/p.txt")]
    assert client.get("/documents").json()["source_urls"] == {"p.txt": "https://example.com/p.txt"}


def test_a_failed_reindex_is_run_again_on_retry_and_a_restore_is_not(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert upload_and_wait(client, "p.txt", b"Intro\nfirst version")["state"] == "done"
    assert upload_and_wait(client, "p.txt", b"Intro\nsecond version")["state"] == "done"

    import api.ingest_service

    real = api.ingest_service.parse_with_limit

    def broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("the parser process died")

    monkeypatch.setattr(api.ingest_service, "parse_with_limit", broken)
    job = client.post("/documents/p.txt/reindex").json()
    failed = wait_for_job(client, job["job_id"])
    assert failed["state"] == "failed" and failed["kind"] == "reindex" and failed["retryable"]

    monkeypatch.setattr(api.ingest_service, "parse_with_limit", real)
    retried = client.post(f"/documents/jobs/{job['job_id']}/retry")
    assert retried.status_code == 202, retried.text
    done = wait_for_job(client, retried.json()["job_id"])
    assert done["state"] == "done" and done["kind"] == "reindex"

    # A version restore that failed is restored again by hand, not retried.
    (version,) = client.get("/documents/p.txt/versions").json()["versions"]
    monkeypatch.setattr(api.ingest_service, "parse_with_limit", broken)
    restore = client.post(f"/documents/p.txt/versions/{version['version']}/restore").json()
    assert wait_for_job(client, restore["job_id"])["state"] == "failed"
    refused = client.post(f"/documents/jobs/{restore['job_id']}/retry")
    assert refused.status_code == 409 and "Restore that version again" in refused.json()["detail"]
    # The read key can't retry (full key only).
    from api.request_guard import needs_full_key

    assert needs_full_key("POST", f"/documents/jobs/{job['job_id']}/retry")
