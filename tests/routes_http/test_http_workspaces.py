"""Workspaces: separate corpora chosen per request with X-Workspace."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches, get_generator
from api.settings import AppSettings
from api.workspaces import document_key, filename_of, settings_for
from tests.factories import make_test_settings
from tests.routes_http.support import FakeGenerator, hermetic_app, make_settings, upload_and_wait

FULL = {"X-API-Key": "full-key"}
LEGAL = {**FULL, "X-Workspace": "legal"}


@dataclass
class Workspaces:
    client: TestClient
    generator: FakeGenerator
    raw_dir: Path


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Workspaces]:
    clear_dependency_caches()
    settings = make_settings(
        api_key="full-key",
        api_read_key="read-key",
        workspaces_enabled=True,
        raw_document_dir=str(tmp_path / "raw"),
        feedback_enabled=True,
        feedback_store_path=str(tmp_path / "feedback.db"),
    )
    # The originals store reads the settings directly (see get_raw_document_store).
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    app = hermetic_app(settings, QdrantClient(":memory:"))
    generator = FakeGenerator()
    app.dependency_overrides[get_generator] = lambda: generator
    try:
        with TestClient(app) as client:
            yield Workspaces(client, generator, tmp_path / "raw")
    finally:
        monkeypatch.undo()
        clear_dependency_caches()


def listing(client: TestClient, headers: dict[str, str]) -> list[str]:
    response = client.get("/documents", headers=headers)
    assert response.status_code == 200, response.text
    return sorted(response.json()["filenames"])


def test_a_workspace_is_created_listed_and_named_in_config(ws: Workspaces) -> None:
    client = ws.client
    assert client.get("/workspaces", headers=FULL).json() == {"enabled": True, "workspaces": []}
    assert client.post("/workspaces", json={"name": "legal"}, headers=FULL).status_code == 201
    assert client.post("/workspaces", json={"name": "legal"}, headers=FULL).status_code == 409
    assert client.post("/workspaces", json={"name": "Bad Name"}, headers=FULL).status_code == 400
    read_only = client.post("/workspaces", json={"name": "x"}, headers={"X-API-Key": "read-key"})
    assert read_only.status_code == 403
    assert client.get("/workspaces", headers=FULL).json()["workspaces"] == ["legal"]
    config = client.get("/config", headers={"X-Workspace": "deleted-since"})
    # Boot never fails on a stale workspace: /config reports the app's own settings.
    assert config.status_code == 200 and config.json()["workspaces_enabled"] is True


def test_an_unknown_or_malformed_workspace_is_refused_not_created(ws: Workspaces) -> None:
    client = ws.client
    missing = client.get("/documents", headers={**FULL, "X-Workspace": "nope"})
    assert missing.status_code == 404
    assert client.get("/documents", headers={**FULL, "X-Workspace": "../x"}).status_code == 400
    assert client.get("/workspaces", headers=FULL).json()["workspaces"] == []


def test_documents_answers_and_cache_stay_in_their_workspace(ws: Workspaces) -> None:
    client = ws.client
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    default_doc = b"Intro\nThe default handbook says alpha."
    legal_doc = b"Intro\nThe legal contract says alpha differently."
    # The same filename in both: neither upload blocks or replaces the other.
    assert upload_and_wait(client, "guide.txt", default_doc, headers=FULL)["state"] == "done"
    assert upload_and_wait(client, "guide.txt", legal_doc, headers=LEGAL)["state"] == "done"
    assert (
        upload_and_wait(client, "only-legal.txt", b"Intro\nLegal only.", headers=LEGAL)["state"]
        == "done"
    )

    assert listing(client, FULL) == ["guide.txt"]
    assert listing(client, LEGAL) == ["guide.txt", "only-legal.txt"]

    first = client.post("/questions", json={"question": "alpha"}, headers=FULL).json()
    assert "default handbook" in ws.generator.messages[-1]["content"]
    legal = client.post("/questions", json={"question": "alpha"}, headers=LEGAL).json()
    # Not the default workspace's cached answer: it was generated from the legal corpus.
    assert legal["cached"] is False and first["cached"] is False
    assert "legal contract" in ws.generator.messages[-1]["content"]
    again = client.post("/questions", json={"question": "alpha"}, headers=LEGAL).json()
    assert again["cached"] is True

    content = client.get("/documents/guide.txt/content", headers=LEGAL).json()
    assert "legal contract" in content["chunks"][0]["text"]


def test_originals_and_deletes_are_per_workspace(ws: Workspaces) -> None:
    client = ws.client
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    upload_and_wait(client, "guide.txt", b"Intro\nDefault copy.", headers=FULL)
    upload_and_wait(client, "guide.txt", b"Intro\nLegal copy.", headers=LEGAL)

    assert (ws.raw_dir / "workspaces" / "legal").is_dir()
    original = client.get("/documents/guide.txt/original", headers=LEGAL)
    assert original.content == b"Intro\nLegal copy."

    assert client.delete("/documents/guide.txt", headers=LEGAL).status_code == 200
    assert listing(client, LEGAL) == []
    assert listing(client, FULL) == ["guide.txt"]
    assert client.get("/documents/guide.txt/original", headers=FULL).content == (
        b"Intro\nDefault copy."
    )


def test_only_an_empty_workspace_can_be_deleted(ws: Workspaces) -> None:
    client = ws.client
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    upload_and_wait(client, "guide.txt", b"Intro\nLegal copy.", headers=LEGAL)

    assert client.delete("/workspaces/legal", headers=FULL).status_code == 409
    client.delete("/documents/guide.txt", headers=LEGAL)
    assert client.delete("/workspaces/legal", headers=FULL).status_code == 200
    assert client.get("/workspaces", headers=FULL).json()["workspaces"] == []
    assert client.get("/documents", headers=LEGAL).status_code == 404
    assert client.delete("/workspaces/legal", headers=FULL).status_code == 404


def test_workspaces_are_refused_when_the_feature_is_off(api_context) -> None:  # type: ignore[no-untyped-def]
    client = api_context.client
    assert client.get("/workspaces").json() == {"enabled": False, "workspaces": []}
    assert client.post("/workspaces", json={"name": "legal"}).status_code == 404
    off = client.get("/documents", headers={"X-Workspace": "legal"})
    assert off.status_code == 400 and "not enabled" in off.json()["detail"]
    # No header (or an empty one) is the default workspace, as before.
    assert client.get("/documents", headers={"X-Workspace": ""}).status_code == 200


def test_keys_are_scoped_outside_the_default_workspace() -> None:
    base: AppSettings = make_test_settings(qdrant_collection="docs", raw_document_dir="/r")
    legal = settings_for(base, "legal")

    assert legal.qdrant_collection == "docs__ws_legal"
    assert legal.raw_document_dir == "/r/workspaces/legal"
    assert document_key(base, "a.pdf") == "a.pdf"
    assert document_key(legal, "a.pdf") == "legal\x1fa.pdf"
    assert filename_of(legal, "legal\x1fa.pdf") == "a.pdf"
    assert filename_of(base, "legal\x1fa.pdf") is None
    assert filename_of(legal, "a.pdf") is None
    assert filename_of(base, "a.pdf") == "a.pdf"
    assert settings_for(base, "") is base


def test_feedback_lists_backs_up_and_restores_per_workspace(ws: Workspaces) -> None:
    import io
    import json
    import zipfile

    client = ws.client
    assert client.post("/workspaces", json={"name": "legal"}, headers=FULL).status_code == 201
    assert client.post("/workspaces", json={"name": "archive"}, headers=FULL).status_code == 201
    rating = {"question": "q", "answer_excerpt": "a", "cited_filenames": [], "rating": "up"}
    assert (
        client.post("/feedback", json={**rating, "question": "default q"}, headers=FULL).status_code
        == 201
    )
    assert (
        client.post("/feedback", json={**rating, "question": "legal q"}, headers=LEGAL).status_code
        == 201
    )

    def questions(headers: dict[str, str]) -> list[str]:
        response = client.get("/feedback", headers=headers)
        assert response.status_code == 200, response.text
        return [item["question"] for item in response.json()["feedback"]]

    assert questions(FULL) == ["default q"]
    assert questions(LEGAL) == ["legal q"]

    # A workspace's backup carries its own ratings only, in the usual row format.
    backup = client.get("/export", headers=LEGAL)
    assert backup.status_code == 200
    with zipfile.ZipFile(io.BytesIO(backup.content)) as archive:
        rows = [json.loads(line) for line in archive.read("feedback.jsonl").splitlines()]
    assert [row["question"] for row in rows] == ["legal q"]
    assert "workspace" not in rows[0]

    # Restored elsewhere, the rows join that workspace.
    archive_headers = {**FULL, "X-Workspace": "archive"}
    restored = client.post(
        "/import",
        files={"file": ("backup.zip", backup.content, "application/zip")},
        headers=archive_headers,
    )
    assert restored.status_code == 202, restored.text
    assert restored.json()["feedback_imported"] == 1
    assert questions(archive_headers) == ["legal q"]
    assert questions(FULL) == ["default q"]
