"""BACKUP_DIR: the export zip written on a timer, per workspace, the newest few kept."""

from __future__ import annotations

import io
import json
import os
import threading
import zipfile
from pathlib import Path

from qdrant_client import QdrantClient

from api.ingest_service import IngestService
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.scheduled_backup import ScheduledBackup, backups_in
from api.workspace_target import WorkspaceTarget
from tests.routes_http.support import FakeEmbeddingProvider, make_settings


class Clock:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now


def _setup(tmp_path: Path) -> tuple[ScheduledBackup, Clock, threading.Lock]:
    settings = make_settings(
        raw_document_dir=str(tmp_path / "raw"),
        backup_dir=str(tmp_path / "backups"),
        backup_interval_hours=1,
        backup_keep=2,
    )
    repository = VectorRepository(QdrantClient(":memory:"))
    store = RawDocumentStore(str(tmp_path / "raw"), 0)
    IngestService(repository, FakeEmbeddingProvider(), settings).ingest(
        "guide.txt",
        b"Intro\nalpha beta",
        on_indexed=lambda: store.save("guide.txt", b"Intro\nalpha beta"),
    )
    clock = Clock()
    lock = threading.Lock()
    backup = ScheduledBackup(
        settings,
        repository,
        lambda: [WorkspaceTarget(settings, store)],
        None,
        lock,
        clock=clock,
    )
    return backup, clock, lock


def test_a_backup_is_written_restorable_once_per_interval_and_pruned(tmp_path: Path) -> None:
    backup, clock, _ = _setup(tmp_path)
    directory = tmp_path / "backups" / "default"

    (first,) = backup.run_once()
    with zipfile.ZipFile(io.BytesIO(first.read_bytes())) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert [doc["filename"] for doc in manifest["documents"]] == ["guide.txt"]
        assert archive.read("originals/guide.txt") == b"Intro\nalpha beta"
    assert backup.last is not None and backup.last.ok and backup.last.file == first.name
    # Within the interval: nothing new (by the file's own time, so a restart adds none).
    assert backup.run_once() == []

    for _ in range(3):
        clock.now += 3601
        # The newest file's mtime is what's compared: make it look an interval old.
        newest = backups_in(directory)[-1]
        os.utime(newest, (clock.now - 3601, clock.now - 3601))
        assert len(backup.run_once()) == 1
    assert len(backups_in(directory)) == 2  # BACKUP_KEEP
    assert not list(directory.glob(".*.part"))


def test_a_running_export_is_not_raced(tmp_path: Path) -> None:
    backup, _, lock = _setup(tmp_path)
    with lock:  # GET /export streaming
        assert backup.run_once() == []
    assert len(backup.run_once()) == 1


def test_the_last_backup_says_how_many_documents_it_cannot_restore(tmp_path: Path) -> None:
    backup, _, _ = _setup(tmp_path)
    (tmp_path / "raw" / "guide.txt").unlink()  # indexed, its original gone

    backup.run_once()

    assert backup.last is not None and backup.last.originals_missing == 1
