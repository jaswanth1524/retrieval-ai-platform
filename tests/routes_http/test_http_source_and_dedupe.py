"""Identical re-sends are left alone; URL-ingested documents keep their source and refresh."""

from __future__ import annotations

import functools
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.repository import VectorRepository
from api.url_fetch import fetch_document
from tests.routes_http.support import (
    ApiTestContext,
    hermetic_app,
    make_settings,
    upload_and_wait,
    wait_for_job,
)
from tests.test_url_fetch import resolver, transport

GUIDE = b"Intro\nalpha beta"


def _points(context: ApiTestContext) -> list[dict[str, object]]:
    records, _ = context.qdrant.scroll(
        collection_name=context.settings.qdrant_collection, with_payload=True, limit=100
    )
    return [dict(record.payload or {}) for record in records]


def test_the_same_bytes_again_are_left_as_indexed(api_context: ApiTestContext) -> None:
    first = upload_and_wait(api_context.client, "guide.txt", GUIDE)
    assert first["state"] == "done" and first["result"]["unchanged"] is False
    embedded = len(api_context.embeddings.seen_text_batches)
    stamp = _points(api_context)[0]["uploaded_at"]

    again = upload_and_wait(api_context.client, "guide.txt", GUIDE)

    assert again["state"] == "done" and again["unchanged"] is True
    assert again["result"]["unchanged"] is True and again["result"]["chunks_ingested"] == 1
    assert len(api_context.embeddings.seen_text_batches) == embedded  # nothing re-embedded
    assert _points(api_context)[0]["uploaded_at"] == stamp  # left exactly as it was

    changed = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha gamma")
    assert changed["unchanged"] is False
    assert len(api_context.embeddings.seen_text_batches) > embedded


def test_a_stale_or_untagged_identical_document_is_still_reindexed(
    api_context: ApiTestContext,
) -> None:
    assert upload_and_wait(api_context.client, "guide.txt", GUIDE)["state"] == "done"
    # As if chunked by an older chunker: re-indexing would change it, so it isn't skipped.
    api_context.qdrant.set_payload(
        collection_name=api_context.settings.qdrant_collection,
        payload={"chunker_version": 1},
        points=[
            record.id
            for record in api_context.qdrant.scroll(
                collection_name=api_context.settings.qdrant_collection, limit=100
            )[0]
        ],
    )
    again = upload_and_wait(api_context.client, "guide.txt", GUIDE)
    assert again["unchanged"] is False


def test_an_unchanged_watched_file_only_takes_its_new_time() -> None:
    from api.ingest_service import IngestService
    from api.repository import VectorRepository
    from tests.routes_http.support import FakeEmbeddingProvider

    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    embeddings = FakeEmbeddingProvider()
    service = IngestService(repository, embeddings, settings)
    service.ingest("guide.txt", GUIDE, uploaded_at=100.0, dedupe=True)
    embedded = len(embeddings.seen_text_batches)

    # `touch` on a watched file: same bytes, a new mtime the watcher stamps.
    outcome = service.ingest("guide.txt", GUIDE, uploaded_at=200.0, dedupe=True)

    assert outcome.unchanged
    assert len(embeddings.seen_text_batches) == embedded
    meta = repository.document_metadata(settings, "guide.txt")
    assert meta is not None and meta.uploaded_at == 200.0  # current to the watcher now
    # A re-index never skips.
    assert not service.ingest("guide.txt", GUIDE, uploaded_at=200.0).unchanged


class Remote:
    """A web server whose document can change, answering conditional requests."""

    def __init__(self) -> None:
        self.body = b"Intro\nAlpha from the web."
        self.etag = '"v1"'
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("if-none-match") == self.etag:
            return httpx.Response(304, headers={"etag": self.etag})
        return httpx.Response(
            200, content=self.body, headers={"content-type": "text/plain", "etag": self.etag}
        )


@pytest.fixture
def url_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[tuple[TestClient, Remote]]:
    clear_dependency_caches()
    remote = Remote()
    settings = make_settings(url_ingest_enabled=True)
    monkeypatch.setattr(
        "api.routes.documents.fetch_document",
        functools.partial(
            fetch_document, resolve=resolver({}), transport=transport(remote.handler)
        ),
    )
    with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as client:
        yield client, remote
    clear_dependency_caches()


def test_a_url_document_keeps_its_source_and_refreshes_conditionally(
    url_client: tuple[TestClient, Remote],
) -> None:
    client, remote = url_client
    url = "https://example.com/notes/alpha.txt"
    accepted = client.post("/documents/from-url", json={"url": url})
    assert wait_for_job(client, accepted.json()["job_id"])["state"] == "done"
    assert client.get("/documents").json()["source_urls"] == {"alpha.txt": url}

    # Unchanged at the source: one conditional request, no job.
    unchanged = client.post("/documents/alpha.txt/refresh")
    assert unchanged.status_code == 200, unchanged.text
    assert unchanged.json() == {"filename": "alpha.txt", "unchanged": True}
    assert remote.requests[-1].headers["if-none-match"] == '"v1"'

    remote.body, remote.etag = b"Intro\nAlpha, edited.", '"v2"'
    refreshed = client.post("/documents/alpha.txt/refresh")
    assert refreshed.status_code == 202
    assert wait_for_job(client, refreshed.json()["job_id"])["state"] == "done"
    content = client.get("/documents/alpha.txt/content").json()
    assert "edited" in " ".join(chunk["text"] for chunk in content["chunks"])
    assert client.get("/documents").json()["source_urls"] == {"alpha.txt": url}

    # An upload under the same name is a different document: no source any more.
    assert upload_and_wait(client, "alpha.txt", b"Intro\nUploaded.")["state"] == "done"
    assert client.get("/documents").json()["source_urls"] == {}
    assert client.post("/documents/alpha.txt/refresh").status_code == 404
    assert client.post("/documents/missing.txt/refresh").status_code == 404


def test_refresh_is_404_when_url_ingest_is_off(api_context: ApiTestContext) -> None:
    assert upload_and_wait(api_context.client, "guide.txt", GUIDE)["state"] == "done"
    assert api_context.client.post("/documents/guide.txt/refresh").status_code == 404


class FailingSecondBatch(VectorRepository):
    """Writes the first upsert batch, then fails (Qdrant dropping mid-ingest)."""

    def __init__(self, client: QdrantClient) -> None:
        super().__init__(client)
        self.fail = True
        self.batches = 0

    def upsert(self, settings: Any, points: Any) -> None:
        self.batches += 1
        if self.fail and self.batches == 2:
            raise RuntimeError("Qdrant went away")
        super().upsert(settings, points)


def test_a_half_written_document_is_finished_by_the_same_bytes_again() -> None:
    """Every point a failed ingest wrote carries the new hash: "same bytes, so leave it"
    would have kept the document half-indexed until its bytes changed — and a watcher
    retry, or the user retrying the upload, is exactly the same bytes."""

    from api.ingest_service import IngestService
    from tests.routes_http.support import FakeEmbeddingProvider

    settings = make_settings(ingest_upsert_batch_size=1, ingest_batch_size=1)
    repository = FailingSecondBatch(QdrantClient(":memory:"))
    service = IngestService(repository, FakeEmbeddingProvider(), settings)
    body = b"".join(f"# Part {n}\n{'word ' * 30}section {n} text.\n\n".encode() for n in range(4))
    with pytest.raises(RuntimeError):
        service.ingest("long.md", body, dedupe=True)
    partial = repository.document_metadata(settings, "long.md")
    assert partial is not None and partial.chunk_count < (partial.document_chunks or 0)

    repository.fail = False
    outcome = service.ingest("long.md", body, dedupe=True)

    assert not outcome.unchanged
    complete = repository.document_metadata(settings, "long.md")
    assert complete is not None and complete.chunk_count == complete.document_chunks
    # Now complete, the same bytes are left alone.
    assert service.ingest("long.md", body, dedupe=True).unchanged


def test_a_source_url_is_listed_without_its_query_string(
    url_client: tuple[TestClient, Remote],
) -> None:
    client, remote = url_client
    url = "https://example.com/notes/alpha.txt?X-Amz-Signature=secret#top"
    accepted = client.post("/documents/from-url", json={"url": url})
    assert wait_for_job(client, accepted.json()["job_id"])["state"] == "done"

    assert client.get("/documents").json()["source_urls"] == {
        "alpha.txt": "https://example.com/notes/alpha.txt"
    }
    # The refresh still asks for the URL as given.
    assert client.post("/documents/alpha.txt/refresh").status_code == 200
    assert "X-Amz-Signature=secret" in str(remote.requests[-1].url)


def _url_backup_client(
    tmp_path: Path, name: str, remote: Remote, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    from api.dependencies import get_raw_document_store
    from api.raw_documents import RawDocumentStore

    clear_dependency_caches()
    raw = RawDocumentStore(str(tmp_path / name))
    settings = make_settings(url_ingest_enabled=True)
    monkeypatch.setattr(
        "api.routes.documents.fetch_document",
        functools.partial(
            fetch_document, resolve=resolver({}), transport=transport(remote.handler)
        ),
    )
    app = hermetic_app(settings, QdrantClient(":memory:"))
    app.dependency_overrides[get_raw_document_store] = lambda: raw
    return TestClient(app)


def test_a_restored_url_document_can_still_be_refreshed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="api.audit")
    remote = Remote()
    url = "https://example.com/notes/alpha.txt"
    with _url_backup_client(tmp_path, "old", remote, monkeypatch) as old:
        accepted = old.post("/documents/from-url", json={"url": url})
        assert wait_for_job(old, accepted.json()["job_id"])["state"] == "done"
        backup = old.get("/export").content

    with _url_backup_client(tmp_path, "new", remote, monkeypatch) as new:
        imported = new.post(
            "/import", files={"file": ("backup.zip", backup, "application/zip")}
        ).json()
        (job,) = imported["jobs"]
        assert wait_for_job(new, job["job_id"])["state"] == "done"
        assert new.get("/documents").json()["source_urls"] == {"alpha.txt": url}
        # Unchanged at the source: the restored etag made it a conditional request.
        assert new.post("/documents/alpha.txt/refresh").json() == {
            "filename": "alpha.txt",
            "unchanged": True,
        }
    clear_dependency_caches()

    events = [record.getMessage().split()[0] for record in caplog.records]
    assert "document.upload_from_url" in events
    assert "corpus.export" in events


def test_a_url_document_can_be_tagged(url_client: tuple[TestClient, Remote]) -> None:
    client, _ = url_client
    accepted = client.post(
        "/documents/from-url",
        json={"url": "https://example.com/notes/alpha.txt", "tags": ["web"]},
    )
    assert wait_for_job(client, accepted.json()["job_id"])["state"] == "done"
    assert client.get("/documents").json()["tags"] == {"alpha.txt": ["web"]}
    too_many = client.post(
        "/documents/from-url",
        json={"url": "https://example.com/notes/beta.txt", "tags": [f"t{n}" for n in range(21)]},
    )
    assert too_many.status_code == 422
