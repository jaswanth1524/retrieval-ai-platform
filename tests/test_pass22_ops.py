"""Pass 22: ingest, ops, CLI and MCP fixes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.test_pass15_admin import admin  # noqa: F401 - the fixture


def test_a_bad_range_leaves_the_bundle_to_download(tmp_path: Path) -> None:
    from fastapi import FastAPI

    import api.routes.migrate as migrate_routes

    body = bytes(range(256)) * 40
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(body)
    migrate_routes._keep_export("r22", bundle, None)
    app = FastAPI()

    @app.get("/b")
    def get_bundle() -> object:
        return migrate_routes._KeptUntilSent("r22", bundle, "b.zip", None)

    with TestClient(app) as client:
        for bad in ("bytes=abc", f"bytes={len(body) + 10}-", "bytes=5-2"):
            response = client.get("/b", headers={"Range": bad})
            assert response.status_code in (400, 416), bad
            assert bundle.exists(), bad
        whole = client.get("/b")
        assert whole.status_code == 200 and whole.content == body
        assert not bundle.exists()


def test_an_export_that_cant_open_its_scratch_file_lets_go_of_the_lock(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import api.routes.backup as backup_routes
    import api.routes.migrate as migrate_routes
    from tests.test_pass18_migrate import _server

    def unwritable(directory: Path | None) -> Path:
        raise OSError(28, "No space left on device")

    with _server(monkeypatch, raw_document_dir=str(tmp_path / "raw")) as client:
        monkeypatch.setattr(migrate_routes, "_scratch_file", unwritable)
        with pytest.raises(OSError):
            client.get("/admin/migrate/export")
        assert not backup_routes._export_running.locked()


def test_duplicate_groups_leave_out_incomplete_copies_and_put_the_best_one_first() -> None:
    from api.repository import DocumentMetadata
    from api.routes.documents_read import duplicate_groups

    def meta(**values: Any) -> DocumentMetadata:
        return DocumentMetadata(
            chunk_count=values.pop("chunk_count", 3),
            page_count=1,
            byte_size=10,
            uploaded_at=None,
            content_sha256="same",
            document_chunks=3,
            **values,
        )

    metadata = {
        "a.txt": meta(),
        "b.txt": meta(tags=("hr",)),
        "c.txt": meta(source_url="https://example.com/c"),
        # A write cut off part way: same hash, fewer points than it meant to write.
        "0-partial.txt": meta(chunk_count=1, tags=("hr", "x"), incomplete=True),
    }
    assert duplicate_groups(metadata) == [["b.txt", "c.txt", "a.txt"]]


def test_dedupe_goes_on_past_a_copy_it_cant_delete() -> None:
    import argparse
    import io
    import json

    from cli.client import DocragError
    from cli.document_commands import cmd_dedupe

    class _Client:
        deleted: list[str] = []

        def duplicates(self) -> dict[str, Any]:
            return {"groups": [{"filenames": ["a", "b", "c"]}]}

        def delete(self, name: str) -> None:
            if name == "b":
                raise DocragError("409: busy")
            self.deleted.append(name)

    out = io.StringIO()
    code = cmd_dedupe(_Client(), argparse.Namespace(delete=True, json=True), out)  # type: ignore[arg-type]
    report = json.loads(out.getvalue())
    assert code == 1 and report["deleted"] == ["c"] and list(report["failed"]) == ["b"]


def test_a_background_archive_whose_record_cant_be_written_leaves_no_spool_file(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import io
    import zipfile

    from qdrant_client import QdrantClient

    import api.routes.documents_write as documents_write
    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app, make_settings

    spool = tmp_path / "spool"
    spool.mkdir()

    def full_disk(*args: Any, **kwargs: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(documents_write, "remember", full_disk)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.txt", "Intro\nalpha")
    clear_dependency_caches()
    app = hermetic_app(make_settings(import_spool_dir=str(spool)), QdrantClient(":memory:"))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/documents/archive",
            data={"background": "true"},
            files={"file": ("d.zip", buffer.getvalue(), "application/zip")},
        )
        assert response.status_code == 500
    assert not [p for p in spool.rglob("*") if p.is_file()]
    clear_dependency_caches()


def test_restore_says_which_names_it_didnt_restore_and_fails() -> None:
    import argparse
    import io

    from cli.admin_commands import cmd_backup

    class _Client:
        def restore_stored_backup(self, name: str, only: Any = None, **_: Any) -> dict[str, Any]:
            return {
                "jobs": [],
                "skipped": [],
                "deferred": [],
                "rejected": ["typo.pdf (not in this backup)"],
            }

    out = io.StringIO()
    args = argparse.Namespace(action="restore", target="b.zip", only=["typo.pdf"], json=False)
    assert cmd_backup(_Client(), args, out) == 1  # type: ignore[arg-type]
    assert "not restored: typo.pdf (not in this backup)" in out.getvalue()


def test_a_drill_stops_at_its_time_budget() -> None:
    import io
    import itertools
    import json
    import zipfile

    from api.restore import drill_backup

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        names = [f"d{i}.txt" for i in range(5)]
        archive.writestr(
            "manifest.json",
            json.dumps({"manifest_version": 1, "documents": [{"filename": n} for n in names]}),
        )
        for name in names:
            archive.writestr(f"originals/{name}", "Intro\nalpha")
    ticks = itertools.count()
    report = drill_backup(
        io.BytesIO(buffer.getvalue()),
        lambda name, content: 1,
        sample=None,
        max_original_bytes=1 << 20,
        stored=False,
        budget_seconds=2.5,
        clock=lambda: float(next(ticks)),
    )
    assert report.listed == 5 and len(report.documents) == 2
    assert report.problems and "Stopped after" in report.problems[0]


def test_a_background_bundle_download_counts_against_the_rate_limit() -> None:
    from api.request_guard import is_rate_limited

    assert is_rate_limited("GET", "/admin/migrate/export/abc123")
    assert not is_rate_limited("GET", "/admin/migrate/exports")


def test_originals_from_a_bundle_are_stored_under_their_write_lock(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import contextlib
    import json
    import zipfile

    import api.migrate as migrate
    from api.raw_documents import RawDocumentStore

    held: list[str] = []

    @contextlib.contextmanager
    def lock(key: str) -> Any:
        held.append(key)
        yield

    monkeypatch.setattr(migrate, "filename_write_lock", lock)
    backup = tmp_path / "b.zip"
    with zipfile.ZipFile(backup, "w") as archive:
        document = {"filename": "a.txt", "original": "originals/a.txt"}
        archive.writestr(
            "manifest.json", json.dumps({"manifest_version": 1, "documents": [document]})
        )
        archive.writestr("originals/a.txt", "Intro\nalpha")
    store = RawDocumentStore(str(tmp_path / "raw"))
    stored = migrate.store_originals(
        backup, store, 1 << 20, None, "team", lock_key=lambda name: f"team\x1f{name}"
    )
    assert stored == 1 and held == ["team\x1fa.txt"]
    assert store.read("a.txt") == b"Intro\nalpha"


def test_a_scheduled_backup_whose_drill_fails_is_reported(admin: Any) -> None:  # noqa: F811
    from api.restore import DrillDocument, DrillReport
    from tests.routes_http.support import upload_and_wait

    client, backup, _, _ = admin
    assert upload_and_wait(client, "a.txt", b"Intro\nalpha")["state"] == "done"
    drilled: list[str] = []

    def drill(path: Path, settings: Any) -> DrillReport:
        drilled.append(path.name)
        return DrillReport(
            documents=[DrillDocument("a.txt", False, error="it makes no passages")], listed=1
        )

    backup._drill = drill
    (written,) = backup.run_once()
    assert drilled == [written.name]
    status = backup.by_workspace["default"]
    assert not status.ok and "restore drill: 1 of 1" in (status.error or "")
    assert "a.txt: it makes no passages" in (status.error or "")


def test_a_drill_of_every_original_that_parses_keeps_the_backup_ok(admin: Any) -> None:  # noqa: F811
    from api.chunking import HeuristicTokenCounter
    from api.routes.backup import drill_file
    from tests.routes_http.support import upload_and_wait

    client, backup, _, _ = admin
    assert upload_and_wait(client, "a.txt", b"Intro\nalpha")["state"] == "done"
    counter = HeuristicTokenCounter()
    backup._drill = lambda path, settings: drill_file(path, settings, counter, 3)
    backup.run_once()
    assert backup.by_workspace["default"].ok


def test_a_restore_dry_run_lists_what_it_would_do_and_changes_nothing(admin: Any) -> None:  # noqa: F811
    from tests.routes_http.support import upload_and_wait

    client, backup, _, _ = admin
    for name in ("a.txt", "b.txt"):
        assert upload_and_wait(client, name, f"Intro\n{name} body".encode())["state"] == "done"
    (written,) = backup.run_once()
    assert client.delete("/documents/a.txt").status_code == 200
    dry = client.post(f"/backups/{written.name}/restore", params={"dry_run": "true"}).json()
    assert dry["dry_run"] is True and dry["jobs"] == []
    assert dry["would_restore"] == ["a.txt"] and dry["skipped"] == ["b.txt"]
    assert client.get("/documents").json()["filenames"] == ["b.txt"]


def test_dedupe_merge_gives_the_kept_copy_every_tag_first() -> None:
    import argparse
    import io

    from cli.document_commands import cmd_dedupe

    calls: list[Any] = []

    class _Client:
        def duplicates(self) -> dict[str, Any]:
            return {"groups": [{"filenames": ["a", "b"]}]}

        def documents(self) -> dict[str, Any]:
            return {"tags": {"a": ["hr"], "b": ["legal", "hr"]}}

        def set_tags(self, name: str, tags: list[str]) -> list[str]:
            calls.append(("tags", name, tags))
            return tags

        def delete(self, name: str) -> None:
            calls.append(("delete", name))

    out = io.StringIO()
    args = argparse.Namespace(delete=True, merge=True, json=False)
    assert cmd_dedupe(_Client(), args, out) == 0  # type: ignore[arg-type]
    assert calls == [("tags", "a", ["hr", "legal"]), ("delete", "b")]


def test_migrate_verify_reads_a_bundle_through_and_finds_damage(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import argparse
    import io
    import json

    from cli.admin_commands import verify_bundle
    from tests.routes_http.support import upload_and_wait
    from tests.test_pass18_migrate import _server

    with _server(monkeypatch, raw_document_dir=str(tmp_path / "raw")) as client:
        upload_and_wait(client, "a.txt", b"Intro\nalpha beta gamma")
        bundle = tmp_path / "server.zip"
        bundle.write_bytes(client.get("/admin/migrate/export").content)
    out = io.StringIO()
    args = argparse.Namespace(json=True)
    assert verify_bundle(bundle, args, out) == 0, out.getvalue()
    report = json.loads(out.getvalue())
    assert report["ok"] and report["workspaces"][0]["originals"] == 1

    # One byte of the stored original flipped: the backup's checksum no longer holds.
    data = bytearray(bundle.read_bytes())
    at = bytes(data).index(b"alpha beta gamma")
    data[at] ^= 0x20
    damaged = tmp_path / "damaged.zip"
    damaged.write_bytes(bytes(data))
    out = io.StringIO()
    assert verify_bundle(damaged, args, out) == 1
    assert json.loads(out.getvalue())["problems"]

    out = io.StringIO()
    (tmp_path / "not.zip").write_bytes(b"nope")
    assert verify_bundle(tmp_path / "not.zip", argparse.Namespace(json=False), out) == 1


def test_the_scheduled_drill_leaves_out_documents_the_backup_has_no_original_of() -> None:
    import io
    import json
    import zipfile

    from api.restore import drill_backup

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        documents = [
            {"filename": "kept.txt", "original": "originals/kept.txt"},
            {"filename": "old.txt", "original": None},
        ]
        archive.writestr(
            "manifest.json", json.dumps({"manifest_version": 1, "documents": documents})
        )
        archive.writestr("originals/kept.txt", "Intro\nalpha")
    data = buffer.getvalue()
    every = drill_backup(io.BytesIO(data), lambda n, c: 1, sample=None, max_original_bytes=1 << 20)
    assert [d.ok for d in every.documents] == [True, False]  # the route still says so
    scheduled = drill_backup(
        io.BytesIO(data),
        lambda n, c: 1,
        sample=None,
        max_original_bytes=1 << 20,
        originals_only=True,
    )
    assert [(d.filename, d.ok) for d in scheduled.documents] == [("kept.txt", True)]
    assert scheduled.listed == 2
