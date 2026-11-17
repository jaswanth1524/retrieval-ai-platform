"""Pass 18: background imports that survive a restart (IMPORT_SPOOL_DIR)."""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api import imports
from api.chunking import HeuristicTokenCounter
from api.dependencies import clear_dependency_caches
from tests.routes_http.support import (
    FakeEmbeddingProvider,
    hermetic_app,
    make_settings,
    wait_until,
)


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_an_import_stopped_by_shutdown_keeps_its_file_and_one_cancelled_does_not(
    tmp_path: Path,
) -> None:
    kept, removed = tmp_path / "kept.zip", tmp_path / "removed.zip"
    kept.write_bytes(b"x")
    removed.write_bytes(b"x")

    def until_stopped(status: imports.BackgroundImport) -> None:
        while not status.cancel.is_set():
            time.sleep(0.01)

    first = imports.start(
        "archive", "kept.zip", "", "c", until_stopped, cleanup=kept.unlink, resumable=True
    )
    wait_until(lambda: first.state == "running")
    imports.shutdown()
    wait_until(lambda: first.finished_at is not None)
    assert first.interrupted and kept.exists()

    second = imports.start(
        "archive", "removed.zip", "", "c", until_stopped, cleanup=removed.unlink, resumable=True
    )
    second.cancel.set()  # someone asked: this one is over
    wait_until(lambda: second.finished_at is not None)
    assert not removed.exists()
    imports.shutdown()


def test_leftovers_are_listed_and_strays_swept(tmp_path: Path) -> None:
    good = imports.spool_upload(io.BytesIO(b"zip bytes"), "docrag-import-", tmp_path)
    record = imports.remember(tmp_path, {"kind": "archive", "name": "a.zip"}, good)
    lone = imports.remember(tmp_path, {"kind": "restore", "stored": "docrag-1.zip"})
    orphan = tmp_path / "docrag-import-gone.zip.import.json"
    orphan.write_text('{"kind": "archive", "spool": "docrag-import-gone.zip"}')
    broken = tmp_path / "x.import.json"
    broken.write_text("not json")
    stale = tmp_path / "docrag-restore-old.zip"
    stale.write_bytes(b"old")
    os.utime(stale, (time.time() - 7200, time.time() - 7200))
    fresh = tmp_path / "docrag-restore-new.zip"
    fresh.write_bytes(b"still being copied")

    found = imports.leftover(tmp_path)

    assert sorted(path.name for path, _, _ in found) == sorted([record.name, lone.name])
    assert {spool.name if spool else None for _, _, spool in found} == {good.name, None}
    assert not orphan.exists() and not broken.exists() and not stale.exists()
    assert fresh.exists() and good.exists()


def test_an_import_a_restart_cut_off_runs_again_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api import main

    clear_dependency_caches()
    spool_dir = tmp_path / "imports"
    settings = make_settings(import_spool_dir=str(spool_dir))
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    qdrant = QdrantClient(":memory:")
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: qdrant)
    monkeypatch.setattr("api.main.get_embedding_provider", lambda: FakeEmbeddingProvider())
    monkeypatch.setattr("api.main.get_token_counter", lambda: HeuristicTokenCounter())
    try:
        with TestClient(hermetic_app(settings, qdrant)) as client:
            # Sent in the background: kept under IMPORT_SPOOL_DIR while it runs, gone after.
            sent = client.post(
                "/documents/archive",
                files={"file": ("first.zip", _zip({"a.txt": b"Intro\nalpha"}))},
                data={"background": "true"},
            )
            assert sent.status_code == 202, sent.text
            first_id = sent.json()["import_id"]
            wait_until(lambda: client.get(f"/documents/imports/{first_id}").json()["finished_at"])
            assert list(spool_dir.iterdir()) == []

            # What a restart mid-import leaves: the spooled zip and its record.
            spool = imports.spool_upload(
                io.BytesIO(_zip({"b.txt": b"Intro\nbeta", "c.txt": b"Intro\ngamma"})),
                "docrag-import-",
                spool_dir,
            )
            imports.remember(
                spool_dir,
                {"kind": "archive", "name": "second.zip", "workspace": "", "tags": ["hr"]},
                spool,
            )
            (resumed_id,) = main.resume_imports(settings)
            status = lambda: client.get(f"/documents/imports/{resumed_id}").json()  # noqa: E731
            wait_until(lambda: status()["finished_at"])
            assert status()["resumed"] is True and status()["queued"] == 2
            wait_until(lambda: len(client.get("/documents").json()["filenames"]) == 3)
            assert client.get("/documents").json()["tags"] == {"b.txt": ["hr"], "c.txt": ["hr"]}
            assert list(spool_dir.iterdir()) == []

            # A leftover that isn't a zip is dropped, not retried at every start.
            bad = imports.spool_upload(io.BytesIO(b"not a zip"), "docrag-import-", spool_dir)
            imports.remember(spool_dir, {"kind": "archive", "name": "bad.zip"}, bad)
            assert main.resume_imports(settings) == []
            assert list(spool_dir.iterdir()) == []
    finally:
        monkeypatch.undo()
        clear_dependency_caches()
