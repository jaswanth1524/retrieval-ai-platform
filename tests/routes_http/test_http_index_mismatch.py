"""An index built for another embedding setup: what still works, and the way back."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.qdrant_schema import EMBEDDING_MODEL_TAG_KEY, read_collection_metadata
from api.settings import AppSettings
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait, wait_for_job

RECOVERY = "delete every document"


@pytest.fixture
def mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, AppSettings, QdrantClient]]:
    clear_dependency_caches()
    settings = make_settings(raw_document_dir=str(tmp_path / "raw"))
    qdrant = QdrantClient(":memory:")
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    try:
        with TestClient(hermetic_app(settings, qdrant)) as client:
            assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
            assert (
                client.patch("/documents/guide.txt/tags", json={"tags": ["Finance"]}).status_code
                == 200
            )
            # What an operator does by changing EMBEDDING_MODEL_TAG (a new model).
            settings.__dict__["embedding_model_tag"] = "other-model:v2"
            yield client, settings, qdrant
    finally:
        monkeypatch.undo()
        clear_dependency_caches()


def _stored_tag(qdrant: QdrantClient, settings: AppSettings) -> str | None:
    info = qdrant.get_collection(settings.qdrant_collection)
    return read_collection_metadata(info).get(EMBEDDING_MODEL_TAG_KEY)


def test_listing_export_tags_and_readiness_keep_working(
    mismatch: tuple[TestClient, AppSettings, QdrantClient],
) -> None:
    """Every route used to 409, so the operator couldn't even see what to delete."""

    client, _, _ = mismatch

    assert client.get("/documents").json()["filenames"] == ["guide.txt"]
    assert client.get("/documents/guide.txt/content").status_code == 200
    assert client.get("/export").status_code == 200
    assert client.patch("/documents/guide.txt/tags", json={"tags": ["Ops"]}).status_code == 200
    ready = client.get("/health/ready").json()
    assert ready["qdrant"] is True
    assert ready["index_compatible"] is False
    assert RECOVERY in ready["index_detail"]


def test_questions_search_and_writes_are_refused_with_the_way_out(
    mismatch: tuple[TestClient, AppSettings, QdrantClient],
) -> None:
    client, _, _ = mismatch

    # An upload is refused up front too, like /reindex and /import, rather than
    # accepted with a 202 for a job that could only fail.
    for response in (
        client.post("/questions", json={"question": "alpha"}),
        client.post("/search", json={"query": "alpha"}),
        client.post("/documents/guide.txt/reindex"),
        client.post("/documents/reindex"),
        client.post("/documents", files={"file": ("other.txt", b"Intro\ngamma", "text/plain")}),
        # Summaries read stored text, not vectors, but follow the same rule (owner's
        # decision): a mismatched index is rebuilt before it answers anything.
        client.post("/questions/summary", json={"mode": "summarize", "filenames": ["guide.txt"]}),
        client.post(
            "/questions/summary/stream", json={"mode": "summarize", "filenames": ["guide.txt"]}
        ),
    ):
        assert response.status_code == 409
        assert RECOVERY in response.json()["detail"]
    assert client.get("/documents").json()["filenames"] == ["guide.txt"]


def test_export_delete_all_import_rebuilds_the_index_for_the_new_setup(
    mismatch: tuple[TestClient, AppSettings, QdrantClient],
) -> None:
    client, settings, qdrant = mismatch
    before = client.get("/documents").json()
    backup = client.get("/export").content

    assert client.delete("/documents/guide.txt").status_code == 200
    response = client.post("/import", files={"file": ("backup.zip", backup, "application/zip")})

    assert response.status_code == 202
    (job,) = response.json()["jobs"]
    assert wait_for_job(client, job["job_id"])["state"] == "done"
    after = client.get("/documents").json()
    assert after["filenames"] == ["guide.txt"]
    assert after["tags"] == {"guide.txt": ["Finance"]}
    assert after["uploaded_ats"] == before["uploaded_ats"]
    assert _stored_tag(qdrant, settings) == "other-model:v2"
    assert client.post("/questions", json={"question": "alpha"}).status_code == 200
    assert client.get("/health/ready").json()["index_compatible"] is True


def test_delete_all_then_upload_rebuilds_the_index(
    mismatch: tuple[TestClient, AppSettings, QdrantClient],
) -> None:
    client, settings, qdrant = mismatch

    assert client.delete("/documents/guide.txt").status_code == 200
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"

    assert _stored_tag(qdrant, settings) == "other-model:v2"
    assert client.get("/documents").json()["filenames"] == ["guide.txt"]
