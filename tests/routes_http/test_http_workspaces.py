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
from tests.routes_http.support import (
    FakeGenerator,
    hermetic_app,
    make_settings,
    upload_and_wait,
    wait_for_job,
)

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


def test_a_workspace_with_work_in_flight_is_not_deleted(ws: Workspaces) -> None:
    from api.ingestion import ingest_sequencer

    client = ws.client
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    legal_key = document_key(settings_for(make_test_settings(), "legal"), "queued.txt")
    ingest_sequencer.admit(legal_key)  # what a queued upload holds until its job ends
    try:
        busy = client.delete("/workspaces/legal", headers=FULL)
        assert busy.status_code == 409 and "being indexed" in busy.json()["detail"]
    finally:
        ingest_sequencer.release(legal_key)
    # Another workspace's work in flight doesn't hold this one.
    ingest_sequencer.admit("queued.txt")
    try:
        assert client.delete("/workspaces/legal", headers=FULL).status_code == 200
    finally:
        ingest_sequencer.release("queued.txt")


def test_nothing_is_admitted_into_a_workspace_while_it_is_being_deleted(
    ws: Workspaces, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.workspaces as routes
    from api.dependencies import get_ingest_backlog

    client = ws.client
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    backlog = get_ingest_backlog()
    real = routes._delete_if_empty
    during: list[tuple[int, str]] = []

    def delete_with_an_upload_racing_it(*args: object) -> None:
        # Between the in-flight check and the collection's removal: the window an
        # upload used to slip through, its job then failing or recreating the workspace.
        racing = client.post(
            "/documents", files={"file": ("late.txt", b"Intro\nLate.")}, headers=LEGAL
        )
        during.append((racing.status_code, racing.json()["detail"]))
        real(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(routes, "_delete_if_empty", delete_with_an_upload_racing_it)
    assert client.delete("/workspaces/legal", headers=FULL).status_code == 200
    assert during == [
        (
            409,
            "Workspace 'legal' is being deleted; nothing can be added to or removed from it now.",
        )
    ]
    assert backlog.pending_bytes == 0  # the refused upload's reservation was given back
    assert client.get("/workspaces", headers=FULL).json()["workspaces"] == []

    # Reopened however the delete ended: a refused delete leaves it usable.
    monkeypatch.setattr(routes, "_delete_if_empty", real)
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    upload_and_wait(client, "guide.txt", b"Intro\nLegal copy.", headers=LEGAL)
    assert client.delete("/workspaces/legal", headers=FULL).status_code == 409
    upload_and_wait(client, "more.txt", b"Intro\nMore legal copy.", headers=LEGAL)
    assert listing(client, LEGAL) == ["guide.txt", "more.txt"]


def test_a_deleted_workspace_is_never_brought_back_by_a_late_request(
    ws: Workspaces, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request that found the workspace before it was deleted used to recreate its
    collection on the way to writing: the deleted workspace came back, empty."""

    from api.dependencies import get_qdrant_client
    from api.qdrant_schema import WorkspaceGoneError
    from api.repository import VectorRepository

    client = ws.client
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    assert client.delete("/workspaces/legal", headers=FULL).status_code == 200

    # As if the workspace check ran before the delete and the write after it.
    monkeypatch.setattr("api.dependencies.workspace_exists", lambda *args: True)
    late = client.post("/documents", files={"file": ("late.txt", b"Intro\nLate.")}, headers=LEGAL)
    assert late.status_code == 404 and late.json()["detail"] == "No workspace named 'legal'."
    monkeypatch.undo()
    assert client.get("/workspaces", headers=FULL).json()["workspaces"] == []

    # A job already past every check fails instead of recreating it on its write.
    overrides = client.app.dependency_overrides  # type: ignore[attr-defined]
    repository = VectorRepository(overrides[get_qdrant_client]())
    legal = settings_for(make_test_settings(), "legal")
    with pytest.raises(WorkspaceGoneError):
        repository.ensure_writable(legal)
    with pytest.raises(WorkspaceGoneError):
        repository.point_ids_for_filename(legal, "late.txt")
    assert client.get("/workspaces", headers=FULL).json()["workspaces"] == []


def test_an_import_into_a_workspace_being_deleted_defers_its_documents(
    ws: Workspaces,
) -> None:
    """It used to abort the whole import with that 409."""

    from api.ingestion import ingest_sequencer

    client = ws.client
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    upload_and_wait(client, "a.txt", b"Intro\nAlpha.", headers=LEGAL)
    upload_and_wait(client, "b.txt", b"Intro\nBeta.", headers=LEGAL)
    backup = client.get("/export", headers=LEGAL).content
    for name in ("a.txt", "b.txt"):
        assert client.delete(f"/documents/{name}", headers=LEGAL).status_code == 200

    def restore() -> dict[str, list[str]]:
        response = client.post(
            "/import", files={"file": ("backup.zip", backup, "application/zip")}, headers=LEGAL
        )
        assert response.status_code == 202, response.text
        return response.json()  # type: ignore[no-any-return]

    legal = settings_for(make_test_settings(), "legal")
    assert ingest_sequencer.close_if_idle("legal", lambda key: filename_of(legal, key) is not None)
    try:
        report = restore()
    finally:
        ingest_sequencer.reopen("legal")
    assert report["jobs"] == [] and sorted(report["deferred"]) == ["a.txt", "b.txt"]

    report = restore()  # once the delete is over (here: it never happened)
    for job in report["jobs"]:
        wait_for_job(client, job["job_id"], headers=LEGAL)
    assert listing(client, LEGAL) == ["a.txt", "b.txt"]


def test_the_watched_folders_workspace_is_not_deleted(
    ws: Workspaces, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The original function, the overrides' key (the fixture patches the module's).
    from tests.routes_http.support import get_app_settings

    client = ws.client
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    app = client.app  # type: ignore[attr-defined]
    base = app.dependency_overrides[get_app_settings]()
    watching = base.model_copy(update={"watch_dir": str(tmp_path), "watch_workspace": "legal"})
    app.dependency_overrides[get_app_settings] = lambda: watching
    try:
        refused = client.delete("/workspaces/legal", headers=FULL)
        assert refused.status_code == 409 and "WATCH_WORKSPACE" in refused.json()["detail"]
    finally:
        app.dependency_overrides[get_app_settings] = lambda: base
    assert client.delete("/workspaces/legal", headers=FULL).status_code == 200


def test_the_sequencer_closes_a_group_only_when_idle() -> None:
    from api.ingestion import ClosedForWorkError, IngestSequencer

    sequencer = IngestSequencer()
    in_legal = lambda key: key.startswith("legal:")  # noqa: E731
    sequencer.admit("legal:a.txt")
    assert sequencer.close_if_idle("legal", in_legal) is False
    sequencer.release("legal:a.txt")
    assert sequencer.close_if_idle("legal", in_legal) is True
    assert sequencer.close_if_idle("legal", in_legal) is False  # already closing
    with pytest.raises(ClosedForWorkError):
        sequencer.admit("legal:b.txt")
    sequencer.admit("other.txt")  # other groups are untouched
    sequencer.reopen("legal")
    sequencer.admit("legal:b.txt")


def test_a_leftover_temp_file_does_not_keep_a_workspace_and_rmdir_failure_is_a_409(
    ws: Workspaces, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = ws.client
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    upload_and_wait(client, "guide.txt", b"Intro\nLegal copy.", headers=LEGAL)
    client.delete("/documents/guide.txt", headers=LEGAL)
    directory = ws.raw_dir / "workspaces" / "legal"
    (directory / ".upload-abcd1234").write_bytes(b"half a save")

    def refuse(path: object) -> None:
        raise PermissionError(13, "Permission denied")

    # Its own context: undoing the whole monkeypatch would also undo the fixture's.
    with monkeypatch.context() as patch:
        patch.setattr("api.routes.workspaces.os.rmdir", refuse)
        refused = client.delete("/workspaces/legal", headers=FULL)
    assert refused.status_code == 409 and "could not be removed" in refused.json()["detail"]
    assert client.get("/workspaces", headers=FULL).json()["workspaces"] == ["legal"]

    assert client.delete("/workspaces/legal", headers=FULL).status_code == 200
    assert not directory.exists()


def test_a_workspace_created_again_stores_originals_again(ws: Workspaces) -> None:
    client = ws.client
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    upload_and_wait(client, "guide.txt", b"Intro\nFirst life.", headers=LEGAL)
    client.delete("/documents/guide.txt", headers=LEGAL)
    assert client.delete("/workspaces/legal", headers=FULL).status_code == 200

    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    assert (
        upload_and_wait(client, "guide.txt", b"Intro\nSecond life.", headers=LEGAL)["state"]
        == "done"
    )
    original = client.get("/documents/guide.txt/original", headers=LEGAL)
    assert original.status_code == 200 and original.content == b"Intro\nSecond life."


def test_traces_and_jobs_are_seen_only_from_their_own_workspace(ws: Workspaces) -> None:
    client = ws.client
    client.post("/workspaces", json={"name": "legal"}, headers=FULL)
    upload_and_wait(client, "guide.txt", b"Intro\nDefault alpha.", headers=FULL)
    accepted = client.post(
        "/documents", files={"file": ("guide.txt", b"Intro\nLegal alpha.")}, headers=LEGAL
    )
    legal_job = accepted.json()["job_id"]
    assert wait_for_job(client, legal_job, headers=LEGAL)["state"] == "done"
    # Polled or cancelled from another workspace, the job does not exist.
    assert client.get(f"/documents/jobs/{legal_job}", headers=FULL).status_code == 404
    assert client.delete(f"/documents/jobs/{legal_job}", headers=FULL).status_code == 404
    assert client.get(f"/documents/jobs/{legal_job}", headers=LEGAL).status_code == 200

    default_trace = client.post("/questions", json={"question": "alpha"}, headers=FULL).json()[
        "trace_id"
    ]
    legal_trace = client.post("/questions", json={"question": "alpha"}, headers=LEGAL).json()[
        "trace_id"
    ]
    listed = client.get("/traces", headers=LEGAL).json()["traces"]
    assert [trace["trace_id"] for trace in listed] == [legal_trace]
    assert [t["trace_id"] for t in client.get("/traces", headers=FULL).json()["traces"]] == [
        default_trace
    ]
    assert client.get(f"/traces/{default_trace}", headers=LEGAL).status_code == 404
    assert client.get(f"/traces/{legal_trace}", headers=FULL).status_code == 404
    assert client.get(f"/traces/{legal_trace}", headers=LEGAL).status_code == 200


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
