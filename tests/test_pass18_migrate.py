"""Pass 18: the whole server as one bundle (docrag migrate pack / unpack)."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.migrate import MIGRATE_VERSION, can_use_snapshot, embedding_setup
from tests.factories import make_test_settings
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait, wait_until


@contextmanager
def _server(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Iterator[TestClient]:
    clear_dependency_caches()
    settings = make_settings(**overrides)
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    try:
        with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as client:
            yield client
    finally:
        monkeypatch.undo()
        clear_dependency_caches()


def _packed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bytes:
    with _server(
        monkeypatch,
        raw_document_dir=str(tmp_path / "old-raw"),
        workspaces_enabled=True,
        feedback_enabled=True,
        feedback_store_path=str(tmp_path / "old-feedback.db"),
    ) as old:
        upload_and_wait(old, "a.txt", b"Intro\nalpha in the default workspace")
        assert old.patch("/documents/a.txt/tags", json={"tags": ["hr"]}).status_code == 200
        assert old.post("/workspaces", json={"name": "legal"}).status_code in (200, 201)
        legal = {"X-Workspace": "legal"}
        upload_and_wait(old, "contract.txt", b"Intro\nthe contract", headers=legal)
        put = old.put("/workspaces/legal/settings", json={"rerank_min_score": 0.3})
        assert put.status_code == 200, put.text
        packed = old.get("/admin/migrate/export")
        assert packed.status_code == 200, packed.text
        return packed.content


def test_a_bundle_holds_every_workspace_and_unpacks_on_a_fresh_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _packed(tmp_path, monkeypatch)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["migrate_version"] == MIGRATE_VERSION
        assert [w["name"] for w in manifest["workspaces"]] == ["default", "legal"]
        # The embedded Qdrant of tests makes no snapshots: originals only.
        assert [w["snapshot"] for w in manifest["workspaces"]] == [None, None]
        assert manifest["workspaces"][1]["settings"] == {"rerank_min_score": 0.3}
        assert "workspaces/legal/backup.zip" in archive.namelist()

    with _server(
        monkeypatch, raw_document_dir=str(tmp_path / "new-raw"), workspaces_enabled=True
    ) as new:
        started = new.post("/admin/migrate/import", files={"file": ("bundle.zip", bundle)})
        assert started.status_code == 202, started.text
        assert started.json()["workspaces"] == ["default", "legal"]
        import_id = started.json()["import_id"]
        wait_until(lambda: new.get(f"/documents/imports/{import_id}").json()["finished_at"])
        status = new.get(f"/documents/imports/{import_id}").json()
        assert (status["kind"], status["state"], status["queued"]) == ("migrate", "done", 2)
        assert any("indexed from its originals" in note for note in status["problems"])

        wait_until(lambda: new.get("/documents").json()["filenames"] == ["a.txt"])
        assert new.get("/documents").json()["tags"] == {"a.txt": ["hr"]}
        legal = {"X-Workspace": "legal"}
        assert new.get("/workspaces").json()["workspaces"] == ["legal"]
        wait_until(
            lambda: new.get("/documents", headers=legal).json()["filenames"] == ["contract.txt"]
        )
        own = new.get("/workspaces/legal/settings").json()
        assert own["settings"]["rerank_min_score"] == 0.3
        assert new.get("/documents/contract.txt/original", headers=legal).content == (
            b"Intro\nthe contract"
        )


def test_a_workspace_is_left_out_where_workspaces_are_off_and_a_bad_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _packed(tmp_path, monkeypatch)
    with _server(monkeypatch) as new:
        started = new.post("/admin/migrate/import", files={"file": ("bundle.zip", bundle)})
        import_id = started.json()["import_id"]
        wait_until(lambda: new.get(f"/documents/imports/{import_id}").json()["finished_at"])
        status = new.get(f"/documents/imports/{import_id}").json()
        assert status["failed"] == 1
        assert any("workspaces are off here" in note for note in status["problems"])
        wait_until(lambda: new.get("/documents").json()["filenames"] == ["a.txt"])

        not_zip = new.post("/admin/migrate/import", files={"file": ("x.zip", b"nope")})
        assert not_zip.status_code == 400
        plain = io.BytesIO()
        with zipfile.ZipFile(plain, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"manifest_version": 1, "documents": []}))
        backup = new.post("/admin/migrate/import", files={"file": ("b.zip", plain.getvalue())})
        assert backup.status_code == 400 and "migration manifest" in backup.json()["detail"]


def test_a_snapshot_is_used_only_for_the_same_embedding_setup_into_an_empty_workspace() -> None:
    settings = make_test_settings()
    manifest = {"embedding": embedding_setup(settings)}
    assert can_use_snapshot(manifest, settings, 0) is None
    assert "already holds documents" in str(can_use_snapshot(manifest, settings, 3))
    other = settings.model_copy(update={"embedding_model_tag": "another-model-v2"})
    assert "embedding setup" in str(can_use_snapshot(manifest, other, 0))
