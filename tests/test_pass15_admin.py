"""Pass 15 ops admin: BACKUP_DIR's backups listed, run, verified and restored over HTTP
and the CLI, and the consistency report."""

from __future__ import annotations

import io
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.raw_documents import VERSIONS_DIR, RawDocumentStore
from api.repository import VectorRepository
from api.restore import verify_backup
from api.scheduled_backup import ScheduledBackup
from api.workspace_target import WorkspaceTarget
from cli.__main__ import main
from cli.client import DocragClient
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait, wait_for_job


class Clock:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def admin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, ScheduledBackup, Clock, Path]]:
    clear_dependency_caches()
    raw = tmp_path / "raw"
    settings = make_settings(
        raw_document_dir=str(raw),
        raw_document_versions=2,
        backup_dir=str(tmp_path / "backups"),
        backup_interval_hours=24,
        backup_keep=3,
    )
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    # The test drives its own loop (below), one pass at a time.
    monkeypatch.setattr("api.main.start_scheduled_backup", lambda _settings: None)
    qdrant = QdrantClient(":memory:")
    app = hermetic_app(settings, qdrant)
    clock = Clock()
    with TestClient(app) as client:
        backup = ScheduledBackup(
            settings,
            VectorRepository(qdrant),
            lambda: [WorkspaceTarget(settings, RawDocumentStore(str(raw), 2))],
            None,
            threading.Lock(),
            clock=clock,
        )
        app.state.backup = backup
        yield client, backup, clock, raw
    monkeypatch.undo()
    clear_dependency_caches()


def test_backups_are_listed_verified_run_on_demand_and_restored(
    admin: tuple[TestClient, ScheduledBackup, Clock, Path], caplog: pytest.LogCaptureFixture
) -> None:
    client, backup, clock, _ = admin
    assert upload_and_wait(client, "guide.txt", b"Intro\none")["state"] == "done"
    assert upload_and_wait(client, "guide.txt", b"Intro\ntwo")["state"] == "done"
    (first,) = backup.run_once()

    listed = client.get("/backups").json()
    assert [b["name"] for b in listed["backups"]] == [first.name]
    assert listed["backups"][0]["verified"] is True and listed["keep"] == 3
    assert backup.last is not None and backup.last.ok

    # Not due for a day, but asked for: written on the next pass.
    clock.now += 60
    assert backup.run_once() == []
    assert client.post("/backups/run").status_code == 202
    (second,) = backup.run_once()
    assert len(client.get("/backups").json()["backups"]) == 2

    check = client.post(f"/backups/{second.name}/verify").json()
    assert check["ok"] and check["documents"] == 1 and check["versions"] == 1

    assert client.delete("/documents/guide.txt").status_code == 200
    import logging

    caplog.set_level(logging.INFO, logger="api.audit")
    restored = client.post(f"/backups/{second.name}/restore")
    assert restored.status_code == 202, restored.text
    (job,) = restored.json()["jobs"]
    assert wait_for_job(client, job["job_id"])["state"] == "done"
    assert "guide.txt" in client.get("/documents").json()["filenames"]
    assert len(client.get("/documents/guide.txt/versions").json()["versions"]) == 1
    audited = [r.audit for r in caplog.records if r.name == "api.audit"]  # type: ignore[attr-defined]
    (imported,) = [event for event in audited if event["event"] == "backup.import"]
    assert imported["backup"] == second.name
    # POST /backups/run is audited as asked for; the backup itself, as an export.
    assert any(event["event"] == "corpus.export_requested" for event in audited)
    assert not any(event.get("requested") for event in audited)

    for bad in ("secrets.zip", "docrag-1.zip", "docrag-20000101-000000.zip"):
        assert client.post(f"/backups/{bad}/verify").status_code == 404


def test_a_damaged_backup_is_named_by_verify(
    admin: tuple[TestClient, ScheduledBackup, Clock, Path],
) -> None:
    client, backup, _, _ = admin
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
    (written,) = backup.run_once()
    data = bytearray(written.read_bytes())
    # Flip the first byte of the stored original's data (past its local header).
    with zipfile.ZipFile(written) as archive:
        info = archive.getinfo("originals/guide.txt")
    name_length = int.from_bytes(data[info.header_offset + 26 : info.header_offset + 28], "little")
    extra_length = int.from_bytes(data[info.header_offset + 28 : info.header_offset + 30], "little")
    data[info.header_offset + 30 + name_length + extra_length] ^= 0xFF
    damaged = client.post(
        "/backups/verify", files={"file": ("x.zip", bytes(data), "application/zip")}
    ).json()
    assert damaged["ok"] is False and damaged["problems"]

    missing = io.BytesIO()
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr(
            "manifest.json",
            '{"manifest_version": 1, "documents": [{"filename": "a.txt",'
            ' "original": "originals/a.txt"}]}',
        )
    report = verify_backup(io.BytesIO(missing.getvalue()))
    assert report.problems == ["originals/a.txt is listed but not in the archive."]
    assert verify_backup(io.BytesIO(b"not a zip")).problems == ["It isn't a zip archive."]


def test_a_backups_check_is_still_listed_after_a_restart(
    admin: tuple[TestClient, ScheduledBackup, Clock, Path],
) -> None:
    import os

    from api.scheduled_backup import check_path

    client, backup, clock, raw = admin
    assert upload_and_wait(client, "guide.txt", b"Intro\none")["state"] == "done"
    (first,) = backup.run_once()
    assert check_path(first).is_file()

    # A new process: nothing in memory, the check read from beside the file.
    from api.dependencies import get_app_settings

    restarted = ScheduledBackup(
        get_app_settings(),
        backup._repository,
        backup._targets,
        None,
        threading.Lock(),
        clock=clock,
    )
    client.app.state.backup = restarted  # type: ignore[attr-defined]
    (listed,) = client.get("/backups").json()["backups"]
    assert (listed["name"], listed["verified"]) == (first.name, True)
    assert check_path(first).name.startswith(".") and first.name in check_path(first).name

    # Damaged after it was checked: no longer listed as checked; checking it again
    # records what is found now.
    data = bytearray(first.read_bytes())
    data[len(data) // 2] ^= 0xFF
    first.write_bytes(bytes(data))
    os.utime(first, ns=(1, 1))
    restarted.checks.clear()
    assert client.get("/backups").json()["backups"][0]["verified"] is None
    assert client.post(f"/backups/{first.name}/verify").json()["ok"] is False
    restarted.checks.clear()
    assert client.get("/backups").json()["backups"][0]["verified"] is False

    # Pruning takes the kept check with its backup.
    for _ in range(3):
        clock.now += 60
        restarted.request_now(get_app_settings())
        restarted.run_once()
    assert not first.exists() and not check_path(first).exists()
    assert len(client.get("/backups").json()["backups"]) == 3


def test_a_backup_name_reaches_only_this_workspaces_directory(tmp_path: Path) -> None:
    from fastapi import HTTPException

    from api.routes.backup import _stored_backup
    from api.workspaces import settings_for

    settings = make_settings(backup_dir=str(tmp_path), workspaces_enabled=True)
    name = "docrag-20261101-000000.zip"
    (tmp_path / "legal").mkdir()
    (tmp_path / "legal" / name).write_bytes(b"x")
    (tmp_path / "default").mkdir()
    (tmp_path / "secret.zip").write_bytes(b"x")

    assert _stored_backup(settings_for(settings, "legal"), name) == tmp_path / "legal" / name
    # The same name, asked for from the default workspace: not its file.
    for bad in (name, "..", "../legal/" + name, "../secret.zip", f"{name}/"):
        with pytest.raises(HTTPException) as refused:
            _stored_backup(settings, bad)
        assert refused.value.status_code == 404


def test_the_routes_need_backup_dir(tmp_path: Path) -> None:
    clear_dependency_caches()
    app = hermetic_app(make_settings(), QdrantClient(":memory:"))
    with TestClient(app) as client:
        assert client.get("/backups").status_code == 404
        assert client.post("/backups/run").status_code == 404
    clear_dependency_caches()


def test_the_consistency_report_names_what_disagrees(
    admin: tuple[TestClient, ScheduledBackup, Clock, Path],
) -> None:
    client, _, _, raw = admin
    assert upload_and_wait(client, "kept.txt", b"Intro\nkept")["state"] == "done"
    assert upload_and_wait(client, "same.txt", b"Intro\nkept")["state"] == "done"
    assert upload_and_wait(client, "lost.txt", b"Intro\nlost")["state"] == "done"
    (raw / "lost.txt").unlink()
    (raw / "orphan.txt").write_bytes(b"left behind")
    (raw / VERSIONS_DIR / "gone.txt").mkdir(parents=True)

    report = client.get("/admin/consistency").json()

    assert report["documents"] == 3 and report["originals_kept"] is True
    assert report["originals_without_document"] == ["orphan.txt"]
    assert report["documents_without_original"] == ["lost.txt"]
    assert report["versions_without_document"] == ["gone.txt"]
    assert report["duplicate_groups"] == [["kept.txt", "same.txt"]]
    assert report["problems"] == 3


def test_the_cli_lists_verifies_and_checks(
    admin: tuple[TestClient, ScheduledBackup, Clock, Path], tmp_path: Path
) -> None:
    client, backup, _, raw = admin
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha")["state"] == "done"
    (written,) = backup.run_once()
    docrag = DocragClient("http://testserver", None, http=client, sleep=lambda _: None)

    out = io.StringIO()
    assert main(["backup", "list"], client=docrag, out=out) == 0
    assert written.name in out.getvalue() and "verified" in out.getvalue()
    out = io.StringIO()
    assert main(["backup", "verify", str(written)], client=docrag, out=out) == 0
    assert out.getvalue().startswith("ok: 1 documents")
    out = io.StringIO()
    assert main(["backup", "verify", written.name], client=docrag, out=out) == 0

    out = io.StringIO()
    assert main(["check"], client=docrag, out=out) == 0
    assert "0 problem(s)" in out.getvalue()
    (raw / "orphan.txt").write_bytes(b"x")
    out = io.StringIO()
    assert main(["check"], client=docrag, out=out) == 1
    assert "original, no document: orphan.txt" in out.getvalue()


def test_a_stored_backup_is_downloaded_and_deleted(
    admin: tuple[TestClient, ScheduledBackup, Clock, Path],
) -> None:
    from api.scheduled_backup import check_path

    client, backup, _, _ = admin
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha")["state"] == "done"
    (written,) = backup.run_once()

    downloaded = client.get(f"/backups/{written.name}")
    assert downloaded.status_code == 200
    assert downloaded.content == written.read_bytes()
    assert downloaded.headers["content-type"] == "application/zip"

    assert client.delete(f"/backups/{written.name}").json() == {"name": written.name}
    assert not written.exists() and not check_path(written).exists()
    assert client.get("/backups").json()["backups"] == []
    assert client.get(f"/backups/{written.name}").status_code == 404
    assert client.delete("/backups/secret.zip").status_code == 404


def test_the_cli_downloads_and_deletes_a_backup(
    admin: tuple[TestClient, ScheduledBackup, Clock, Path], tmp_path: Path
) -> None:
    client, backup, _, _ = admin
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha")["state"] == "done"
    (written,) = backup.run_once()
    docrag = DocragClient("http://testserver", None, http=client, sleep=lambda _: None)
    target = tmp_path / "kept.zip"
    assert main(["backup", "download", written.name, "-o", str(target)], client=docrag) == 0
    assert target.read_bytes() == written.read_bytes()
    out = io.StringIO()
    assert main(["backup", "delete", written.name], client=docrag, out=out) == 0
    assert out.getvalue().strip() == f"deleted {written.name}"


def test_repair_puts_right_only_what_is_asked(
    admin: tuple[TestClient, ScheduledBackup, Clock, Path],
) -> None:
    client, _, _, raw = admin
    assert upload_and_wait(client, "kept.txt", b"Intro\nkept")["state"] == "done"
    (raw / "orphan.txt").write_bytes(b"left behind")
    (raw / VERSIONS_DIR / "gone.txt").mkdir(parents=True)
    (raw / VERSIONS_DIR / "gone.txt" / "v1").write_bytes(b"old")

    nothing = client.post("/admin/consistency/repair", json={}).json()
    assert nothing["originals_deleted"] == [] and (raw / "orphan.txt").exists()

    fixed = client.post(
        "/admin/consistency/repair",
        json={"delete_orphan_originals": True, "delete_orphan_versions": True},
    ).json()
    assert fixed["originals_deleted"] == ["orphan.txt"]
    assert fixed["versions_deleted"] == ["gone.txt"]
    assert not (raw / "orphan.txt").exists() and not (raw / VERSIONS_DIR / "gone.txt").exists()
    # The indexed document and its original are untouched.
    assert (raw / "kept.txt").exists()
    assert client.get("/admin/consistency").json()["problems"] == 0


def test_the_cli_repairs_on_request(
    admin: tuple[TestClient, ScheduledBackup, Clock, Path],
) -> None:
    client, _, _, raw = admin
    assert upload_and_wait(client, "kept.txt", b"Intro\nkept")["state"] == "done"
    (raw / "orphan.txt").write_bytes(b"x")
    docrag = DocragClient("http://testserver", None, http=client, sleep=lambda _: None)
    out = io.StringIO()
    assert main(["check", "--repair", "originals"], client=docrag, out=out) == 0
    assert "deleted original: orphan.txt" in out.getvalue()
    assert "0 problem(s)" in out.getvalue()
