from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from qdrant_client import QdrantClient

from api.chunking import HeuristicTokenCounter
from api.dependencies import clear_dependency_caches
from api.repository import VectorRepository
from api.watch import FolderWatcher, scan_folder
from tests.factories import wait_until
from tests.routes_http.support import FakeEmbeddingProvider, make_settings


def write(path: Path, text: str, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class Recorder:
    def __init__(self, indexed: dict[str, float | None] | None = None) -> None:
        self.indexed: dict[str, float | None] = dict(indexed or {})
        self.enqueued: list[tuple[str, bytes]] = []
        self.deleted: list[str] = []
        self.full = False

    def enqueue(self, name: str, content: bytes) -> None:
        if self.full:
            raise HTTPException(status_code=503, detail="full")
        self.enqueued.append((name, content))
        self.indexed[name] = time.time()

    def delete(self, name: str) -> None:
        self.deleted.append(name)
        self.indexed.pop(name, None)


def watcher(tmp_path: Path, recorder: Recorder, **overrides: object) -> FolderWatcher:
    settings = make_settings(watch_dir=str(tmp_path), **overrides)
    return FolderWatcher(
        settings, lambda: dict(recorder.indexed), recorder.enqueue, recorder.delete
    )


def test_scan_finds_supported_files_and_reports_the_rest(tmp_path: Path) -> None:
    write(tmp_path / "a.md", "# A")
    write(tmp_path / "sub" / "b.txt", "B")
    write(tmp_path / "other" / "a.md", "duplicate name")
    write(tmp_path / ".hidden" / "c.txt", "hidden")
    write(tmp_path / "image.png", "not a document")
    write(tmp_path / "big.txt", "x" * 50)

    files, problems = scan_folder(tmp_path, max_bytes=20)

    assert sorted(files) == ["a.md", "b.txt"]
    assert any("already indexed as 'a.md'" in problem for problem in problems)
    assert any("big.txt" in problem and "MAX_UPLOAD_BYTES" in problem for problem in problems)
    assert not any("c.txt" in problem or "image.png" in problem for problem in problems)


def test_new_and_changed_files_are_indexed_once(tmp_path: Path) -> None:
    old = time.time() - 3600
    write(tmp_path / "current.txt", "unchanged", mtime=old)
    write(tmp_path / "new.txt", "new")
    recorder = Recorder({"current.txt": old + 10})
    folder = watcher(tmp_path, recorder)

    assert folder.scan_once()["indexed"] == ["new.txt"]
    assert folder.scan_once()["indexed"] == []  # nothing changed since

    write(tmp_path / "current.txt", "edited", mtime=time.time() + 5)
    assert folder.scan_once()["indexed"] == ["current.txt"]
    assert recorder.enqueued[-1] == ("current.txt", b"edited")


def test_a_full_backlog_defers_to_the_next_pass(tmp_path: Path) -> None:
    write(tmp_path / "a.txt", "A")
    recorder = Recorder()
    recorder.full = True
    folder = watcher(tmp_path, recorder)

    assert folder.scan_once()["deferred"] == ["a.txt"]
    recorder.full = False
    assert folder.scan_once()["indexed"] == ["a.txt"]


def test_removed_files_are_deleted_only_when_asked_and_only_if_seen(tmp_path: Path) -> None:
    write(tmp_path / "a.txt", "A")
    recorder = Recorder({"uploaded-elsewhere.pdf": 1.0})
    keep = watcher(tmp_path, recorder)
    keep.scan_once()
    (tmp_path / "a.txt").unlink()
    assert keep.scan_once()["deleted"] == []  # WATCH_DELETE_MISSING is off

    write(tmp_path / "b.txt", "B")
    recorder = Recorder({"uploaded-elsewhere.pdf": 1.0})
    prune = watcher(tmp_path, recorder, watch_delete_missing=True)
    prune.scan_once()
    (tmp_path / "b.txt").unlink()
    assert prune.scan_once()["deleted"] == ["b.txt"]
    # A document the watcher never saw in the folder is never deleted.
    assert "uploaded-elsewhere.pdf" in recorder.indexed


def test_a_missing_folder_is_reported_not_fatal(tmp_path: Path) -> None:
    folder = watcher(tmp_path / "nope", Recorder())
    assert folder.scan_once() == {"indexed": [], "deleted": [], "deferred": []}


def test_the_app_watcher_indexes_reindexes_and_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.main as main

    clear_dependency_caches()
    folder = tmp_path / "watched"
    write(folder / "guide.txt", "Intro\nAlpha is first.", mtime=time.time() - 60)
    settings = make_settings(
        watch_dir=str(folder), watch_interval_seconds=2, watch_delete_missing=True
    )
    qdrant = QdrantClient(":memory:")
    monkeypatch.setattr(main, "get_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(main, "get_embedding_provider", lambda: FakeEmbeddingProvider())
    monkeypatch.setattr(main, "get_token_counter", lambda: HeuristicTokenCounter())
    monkeypatch.setattr(main, "get_raw_document_store", lambda: None)
    repository = VectorRepository(qdrant)

    def text(name: str) -> str:
        return " ".join(str(p.get("text")) for p in repository.chunks_for_filename(settings, name))

    running = main.start_folder_watcher(settings)
    try:
        wait_until(lambda: "Alpha is first." in text("guide.txt"), timeout=20)
        write(folder / "guide.txt", "Intro\nAlpha is second now.", mtime=time.time() + 5)
        wait_until(lambda: "Alpha is second now." in text("guide.txt"), timeout=20)
        (folder / "guide.txt").unlink()
        wait_until(lambda: text("guide.txt") == "", timeout=20)
    finally:
        running.stop()
        clear_dependency_caches()


def test_watch_target_names_the_setting_when_it_cannot_be_used() -> None:
    import api.main as main

    assert main.watch_target(make_settings()).qdrant_collection == make_settings().qdrant_collection
    with pytest.raises(RuntimeError, match="WATCH_WORKSPACE needs WORKSPACES_ENABLED"):
        main.watch_target(make_settings(watch_workspace="legal"))
    with pytest.raises(RuntimeError, match="WATCH_WORKSPACE:"):
        main.watch_target(make_settings(watch_workspace="Not OK", workspaces_enabled=True))
    target = main.watch_target(make_settings(watch_workspace="legal", workspaces_enabled=True))
    assert target.qdrant_collection.endswith("__ws_legal")


def test_the_app_watcher_indexes_into_its_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.main as main
    from api.workspaces import list_workspaces, settings_for

    clear_dependency_caches()
    folder = tmp_path / "watched"
    write(folder / "contract.txt", "Intro\nKappa is defined here.", mtime=time.time() - 60)
    settings = make_settings(
        watch_dir=str(folder),
        watch_interval_seconds=2,
        watch_workspace="legal",
        workspaces_enabled=True,
        raw_document_dir=str(tmp_path / "raw"),
    )
    qdrant = QdrantClient(":memory:")
    monkeypatch.setattr(main, "get_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(main, "get_embedding_provider", lambda: FakeEmbeddingProvider())
    monkeypatch.setattr(main, "get_token_counter", lambda: HeuristicTokenCounter())
    from api.raw_documents import RawDocumentStore

    monkeypatch.setattr(
        main, "get_raw_document_store", lambda: RawDocumentStore(str(tmp_path / "raw"))
    )
    repository = VectorRepository(qdrant)
    legal = settings_for(settings, "legal")

    running = main.start_folder_watcher(settings)
    try:
        # Created at startup, then filled from the folder; the default workspace stays empty.
        assert list_workspaces(qdrant, settings.qdrant_collection) == ["legal"]
        wait_until(lambda: "contract.txt" in repository.filename_metadata(legal), timeout=20)
        assert "contract.txt" not in repository.filename_metadata(settings)
        assert (tmp_path / "raw" / "workspaces" / "legal" / "contract.txt").exists()
    finally:
        running.stop()
        clear_dependency_caches()
