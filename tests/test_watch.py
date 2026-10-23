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
from api.watch import FolderWatcher, UnreadableFileError, WatchedJob, scan_folder
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

    def enqueue(self, name: str, content: bytes, mtime: float) -> None:
        if self.full:
            raise HTTPException(status_code=503, detail="full")
        self.enqueued.append((name, content))
        # What the app does: the document is stamped with the file's mtime.
        self.indexed[name] = mtime

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

    files, problems, incomplete = scan_folder(tmp_path, max_bytes=20)

    assert sorted(files) == ["a.md", "b.txt"] and not incomplete
    assert any("already indexed as 'a.md'" in problem for problem in problems)
    assert any("big.txt" in problem and "MAX_UPLOAD_BYTES" in problem for problem in problems)
    assert not any("c.txt" in problem or "image.png" in problem for problem in problems)


def test_new_and_changed_files_are_indexed_once(tmp_path: Path) -> None:
    old = time.time() - 3600
    write(tmp_path / "current.txt", "unchanged", mtime=old)
    write(tmp_path / "new.txt", "new")
    recorder = Recorder({"current.txt": old})
    folder = watcher(tmp_path, recorder)

    assert folder.scan_once()["indexed"] == []  # new.txt waits one pass to settle
    assert folder.scan_once()["indexed"] == ["new.txt"]
    assert folder.scan_once()["indexed"] == []  # nothing changed since

    write(tmp_path / "current.txt", "edited", mtime=time.time() + 5)
    assert folder.scan_once()["indexed"] == []
    assert folder.scan_once()["indexed"] == ["current.txt"]
    assert recorder.enqueued[-1] == ("current.txt", b"edited")


def test_a_full_backlog_defers_to_the_next_pass(tmp_path: Path) -> None:
    write(tmp_path / "a.txt", "A")
    recorder = Recorder()
    recorder.full = True
    folder = watcher(tmp_path, recorder)

    folder.scan_once()
    assert folder.scan_once()["deferred"] == ["a.txt"]
    recorder.full = False
    assert folder.scan_once()["indexed"] == ["a.txt"]


def test_removed_files_are_deleted_only_when_asked_and_only_if_seen(tmp_path: Path) -> None:
    write(tmp_path / "a.txt", "A")
    write(tmp_path / "stays.txt", "S")
    recorder = Recorder({"uploaded-elsewhere.pdf": 1.0})
    keep = watcher(tmp_path, recorder)
    keep.scan_once()
    keep.scan_once()
    (tmp_path / "a.txt").unlink()
    assert keep.scan_once()["deleted"] == []  # WATCH_DELETE_MISSING is off

    write(tmp_path / "b.txt", "B")
    recorder = Recorder({"uploaded-elsewhere.pdf": 1.0})
    prune = watcher(tmp_path, recorder, watch_delete_missing=True)
    prune.scan_once()
    prune.scan_once()
    (tmp_path / "b.txt").unlink()
    assert prune.scan_once()["deleted"] == ["b.txt"]
    # A document the watcher never saw in the folder is never deleted.
    assert "uploaded-elsewhere.pdf" in recorder.indexed


def test_a_missing_folder_is_reported_not_fatal(tmp_path: Path) -> None:
    folder = watcher(tmp_path / "nope", Recorder())
    assert not any(folder.scan_once().values())
    assert "not a directory" in folder.status().problems[0]


def test_the_app_watcher_indexes_reindexes_and_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.main as main

    clear_dependency_caches()
    folder = tmp_path / "watched"
    write(folder / "guide.txt", "Intro\nAlpha is first.", mtime=time.time() - 60)
    # Stays: a folder that lists no files at all deletes nothing (a dropped mount).
    write(folder / "keep.txt", "Intro\nKeep me.", mtime=time.time() - 60)
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
        # The watcher thread writes to the same in-process client this reads from, and
        # qdrant-client's local mode isn't thread-safe: a read racing a write can hit
        # an IndexError inside it (seen once in `make check`). Read again.
        for _ in range(20):
            try:
                points = repository.chunks_for_filename(settings, name)
            except IndexError:
                time.sleep(0.05)
                continue
            return " ".join(str(p.get("text")) for p in points)
        raise AssertionError("the local Qdrant read kept racing the watcher")

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


@pytest.mark.timeout(20)
def test_links_and_special_files_are_never_read(tmp_path: Path) -> None:
    outside = write(tmp_path.parent / f"{tmp_path.name}-secret.txt", "server secret")
    folder = tmp_path / "watched"
    write(folder / "real.txt", "Real.")
    (folder / "linked.txt").symlink_to(outside)
    (folder / "zero.txt").symlink_to("/dev/zero")
    os.mkfifo(folder / "pipe.txt")

    files, problems, incomplete = scan_folder(folder, max_bytes=1000)

    assert sorted(files) == ["real.txt"] and not incomplete
    assert any("linked.txt" in problem and "symbolic link" in problem for problem in problems)
    assert any("zero.txt" in problem and "symbolic link" in problem for problem in problems)
    assert any("pipe.txt" in problem and "not a regular file" in problem for problem in problems)

    recorder = Recorder()
    folder_watcher = watcher(folder, recorder)
    folder_watcher.scan_once()
    folder_watcher.scan_once()
    assert recorder.enqueued == [("real.txt", b"Real.")]


@pytest.mark.timeout(20)
def test_the_read_refuses_a_file_swapped_or_grown_after_the_listing(tmp_path: Path) -> None:
    from api.watch import UnreadableFileError, read_regular_file

    os.mkfifo(tmp_path / "pipe.txt")
    with pytest.raises(UnreadableFileError, match="not a regular file"):
        read_regular_file(tmp_path / "pipe.txt", 100)
    (tmp_path / "zero.txt").symlink_to("/dev/zero")
    with pytest.raises(OSError):  # O_NOFOLLOW: the link is not followed
        read_regular_file(tmp_path / "zero.txt", 100)
    write(tmp_path / "grown.txt", "x" * 101)
    with pytest.raises(UnreadableFileError, match="larger than"):
        read_regular_file(tmp_path / "grown.txt", 100)
    assert read_regular_file(write(tmp_path / "ok.txt", "x" * 100), 100) == b"x" * 100


def test_nothing_is_deleted_in_a_pass_that_could_not_read_the_whole_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(tmp_path / "a.txt", "A")
    write(tmp_path / "sub" / "b.txt", "B")
    recorder = Recorder()
    folder = watcher(tmp_path, recorder, watch_delete_missing=True)
    folder.scan_once()
    folder.scan_once()
    assert sorted(recorder.indexed) == ["a.txt", "b.txt"]

    real_walk = os.walk

    def walk_without_sub(top: object, onerror: object = None, **kwargs: object) -> object:
        for directory, subdirectories, filenames in real_walk(top, **kwargs):  # type: ignore[call-overload]
            if "sub" in subdirectories:
                subdirectories.remove("sub")
                error = PermissionError(13, "Permission denied")
                error.filename = str(tmp_path / "sub")
                onerror(error)  # type: ignore[operator]
            yield directory, subdirectories, filenames

    monkeypatch.setattr("api.watch.os.walk", walk_without_sub)
    (tmp_path / "a.txt").unlink()
    assert folder.scan_once()["deleted"] == []  # b.txt merely unseen; a.txt waits too
    assert sorted(recorder.indexed) == ["a.txt", "b.txt"]

    monkeypatch.setattr("api.watch.os.walk", real_walk)
    assert folder.scan_once()["deleted"] == ["a.txt"]  # the next complete pass
    assert sorted(recorder.indexed) == ["b.txt"]


def test_an_emptied_folder_deletes_nothing(tmp_path: Path) -> None:
    write(tmp_path / "a.txt", "A")
    write(tmp_path / "b.txt", "B")
    recorder = Recorder()
    folder = watcher(tmp_path, recorder, watch_delete_missing=True)
    folder.scan_once()
    folder.scan_once()

    (tmp_path / "a.txt").unlink()
    (tmp_path / "b.txt").unlink()  # as a dropped mount looks
    assert folder.scan_once()["deleted"] == []
    assert folder.scan_once()["deleted"] == []  # not just for one pass
    assert sorted(recorder.indexed) == ["a.txt", "b.txt"]

    write(tmp_path / "b.txt", "B")  # back: what is really gone goes now
    assert folder.scan_once()["deleted"] == ["a.txt"]


def test_a_file_still_being_written_is_read_once_it_has_settled(tmp_path: Path) -> None:
    path = write(tmp_path / "copy.txt", "half", mtime=time.time() - 30)
    recorder = Recorder()
    folder = watcher(tmp_path, recorder)

    assert folder.scan_once()["indexed"] == []
    write(path, "half and the rest", mtime=time.time() - 20)  # grew since the last pass
    assert folder.scan_once()["indexed"] == []
    assert folder.scan_once()["indexed"] == ["copy.txt"]
    assert recorder.enqueued == [("copy.txt", b"half and the rest")]


def test_a_replacement_with_an_older_mtime_is_reindexed(tmp_path: Path) -> None:
    now = time.time()
    write(tmp_path / "a.txt", "version two", mtime=now - 10)
    recorder = Recorder()
    folder = watcher(tmp_path, recorder)
    folder.scan_once()
    assert folder.scan_once()["indexed"] == ["a.txt"]
    assert recorder.indexed["a.txt"] == os.stat(tmp_path / "a.txt").st_mtime

    # `cp -p` of an older copy: the mtime goes back, and it is still a change.
    write(tmp_path / "a.txt", "version one", mtime=now - 3600)
    folder.scan_once()
    assert folder.scan_once()["indexed"] == ["a.txt"]
    assert recorder.enqueued[-1] == ("a.txt", b"version one")

    # A document stamped when its job ran (before this fix) is re-indexed once.
    stale = Recorder({"b.txt": now})
    write(tmp_path / "b.txt", "B", mtime=now - 100)
    other = watcher(tmp_path, stale)
    other.scan_once()
    assert "b.txt" in other.scan_once()["indexed"]


def test_the_app_watcher_refuses_a_near_duplicate_name_with_originals_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import api.main as main
    from api.raw_documents import RawDocumentStore

    clear_dependency_caches()
    folder = tmp_path / "watched"
    write(folder / "one" / "Report.txt", "Intro\nUpper case.", mtime=time.time() - 60)
    write(folder / "two" / "report.txt", "Intro\nLower case.", mtime=time.time() - 60)
    settings = make_settings(
        watch_dir=str(folder), watch_interval_seconds=2, raw_document_dir=str(tmp_path / "raw")
    )
    qdrant = QdrantClient(":memory:")
    monkeypatch.setattr(main, "get_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(main, "get_embedding_provider", lambda: FakeEmbeddingProvider())
    monkeypatch.setattr(main, "get_token_counter", lambda: HeuristicTokenCounter())
    monkeypatch.setattr(
        main, "get_raw_document_store", lambda: RawDocumentStore(str(tmp_path / "raw"))
    )
    repository = VectorRepository(qdrant)

    running = main.start_folder_watcher(settings)
    try:
        wait_until(lambda: "only by letter case" in caplog.text, timeout=20)
        wait_until(lambda: bool(repository.filename_metadata(settings)), timeout=20)
        assert list(repository.filename_metadata(settings)) == ["Report.txt"]
        stamp = repository.filename_metadata(settings)["Report.txt"].uploaded_at
        assert stamp == os.stat(folder / "one" / "Report.txt").st_mtime
    finally:
        running.stop()
        clear_dependency_caches()


def test_a_stand_down_is_logged_once_not_every_pass(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write(tmp_path / "a.txt", "Intro\nAlpha.", mtime=time.time() - 60)
    recorder = Recorder()
    folder = watcher(tmp_path, recorder, watch_delete_missing=True)
    folder.scan_once()
    folder.scan_once()
    (tmp_path / "a.txt").unlink()

    caplog.set_level("WARNING", logger="api.watch")
    for _ in range(4):
        assert folder.scan_once()["deleted"] == []
    assert caplog.text.count("lists no files at all") == 1

    # Back to normal, then empty again: said again, once.
    write(tmp_path / "b.txt", "Intro\nBeta.", mtime=time.time() - 60)
    folder.scan_once()
    (tmp_path / "b.txt").unlink()
    folder.scan_once()
    folder.scan_once()
    assert caplog.text.count("lists no files at all") == 2


class JobRecorder(Recorder):
    """Each enqueue makes a job the test settles by hand, as the ingest pool would."""

    def __init__(self) -> None:
        super().__init__()
        self.jobs: dict[str, WatchedJob] = {}
        self.refuse: Exception | None = None

    def enqueue(self, name: str, content: bytes, mtime: float) -> str:  # type: ignore[override]
        if self.refuse is not None:
            raise self.refuse
        self.enqueued.append((name, content))
        job_id = f"job-{len(self.enqueued)}"
        self.jobs[job_id] = WatchedJob("queued")
        return job_id

    def finish(self, job_id: str, state: str, error: str | None = None, **kw: bool) -> None:
        self.jobs[job_id] = WatchedJob(state, error, **kw)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def retrying_watcher(
    tmp_path: Path, recorder: JobRecorder, clock: Clock, **overrides: object
) -> FolderWatcher:
    settings = make_settings(
        watch_dir=str(tmp_path), watch_interval_seconds=10, watch_max_retries=2, **overrides
    )
    return FolderWatcher(
        settings,
        lambda: dict(recorder.indexed),
        recorder.enqueue,
        recorder.delete,
        recorder.jobs.get,
        clock=clock,
    )


def test_a_failed_job_is_retried_with_backoff_then_given_up(tmp_path: Path) -> None:
    """A file whose job failed (Qdrant down, an embedding error) used to stay "sent"
    until it changed or the process restarted."""

    write(tmp_path / "a.txt", "A")
    recorder, clock = JobRecorder(), Clock()
    folder = retrying_watcher(tmp_path, recorder, clock)

    folder.scan_once()
    assert folder.scan_once()["indexed"] == ["a.txt"]
    recorder.finish("job-1", "failed", "Qdrant is unavailable")

    assert folder.scan_once()["retrying"] == ["a.txt"]  # failed: waits 10s
    status = folder.status()
    assert [(r.filename, r.attempts, r.error) for r in status.retrying] == [
        ("a.txt", 1, "Qdrant is unavailable")
    ]
    clock.now += 10
    assert folder.scan_once()["indexed"] == ["a.txt"]  # second attempt
    recorder.finish("job-2", "failed", "still down")
    folder.scan_once()
    clock.now += 19
    assert folder.scan_once()["retrying"] == ["a.txt"]  # 20s now, not 10
    clock.now += 1
    assert folder.scan_once()["indexed"] == ["a.txt"]
    recorder.finish("job-3", "failed", "still down")

    assert folder.scan_once()["gave_up"] == ["a.txt"]  # past WATCH_MAX_RETRIES=2
    clock.now += 10_000
    assert not any(folder.scan_once().values())
    assert folder.status().gave_up == ["a.txt"] and folder.status().retrying == []

    # Changing the file starts over.
    write(tmp_path / "a.txt", "A, edited", mtime=time.time() + 5)
    folder.scan_once()
    assert folder.scan_once()["indexed"] == ["a.txt"]
    assert folder.status().gave_up == []


def test_a_finished_or_cancelled_job_is_not_retried(tmp_path: Path) -> None:
    write(tmp_path / "done.txt", "D")
    write(tmp_path / "cancelled.txt", "C")
    recorder, clock = JobRecorder(), Clock()
    folder = retrying_watcher(tmp_path, recorder, clock)

    folder.scan_once()
    assert sorted(folder.scan_once()["indexed"]) == ["cancelled.txt", "done.txt"]
    for job_id, (name, _) in zip(recorder.jobs, recorder.enqueued, strict=True):
        if name == "done.txt":
            recorder.finish(job_id, "done")
        else:
            recorder.finish(job_id, "failed", "Cancelled", cancelled=True)

    clock.now += 10_000
    assert not any(folder.scan_once().values())
    assert len(recorder.enqueued) == 2


def test_a_job_the_store_forgot_falls_back_to_the_index(tmp_path: Path) -> None:
    write(tmp_path / "a.txt", "A")
    recorder, clock = JobRecorder(), Clock()
    folder = retrying_watcher(tmp_path, recorder, clock)
    folder.scan_once()
    folder.scan_once()
    recorder.jobs.clear()  # evicted, and the document never made it into the index

    assert folder.scan_once()["indexed"] == ["a.txt"]


def test_a_refused_file_waits_for_a_change_and_an_unreadable_one_is_retried(
    tmp_path: Path,
) -> None:
    write(tmp_path / "a.txt", "A")
    recorder, clock = JobRecorder(), Clock()
    folder = retrying_watcher(tmp_path, recorder, clock)
    folder.scan_once()

    recorder.refuse = HTTPException(status_code=409, detail="conflict")
    done = folder.scan_once()
    assert done["gave_up"] == ["a.txt"] and done["indexed"] == []  # not counted as sent
    recorder.refuse = None
    clock.now += 10_000
    assert not any(folder.scan_once().values())

    recorder.refuse = UnreadableFileError("gone mid-read")
    write(tmp_path / "a.txt", "A2", mtime=time.time() + 5)
    folder.scan_once()
    assert folder.scan_once()["retrying"] == ["a.txt"]
    recorder.refuse = None
    clock.now += 10
    assert folder.scan_once()["indexed"] == ["a.txt"]


def test_scan_now_runs_the_next_pass_without_waiting_for_the_interval(tmp_path: Path) -> None:
    recorder = Recorder()
    settings = make_settings(watch_dir=str(tmp_path), watch_interval_seconds=3600)
    folder = FolderWatcher(
        settings, lambda: dict(recorder.indexed), recorder.enqueue, recorder.delete
    )
    folder.start()
    try:
        wait_until(lambda: folder.status().last_scan_at is not None)
        first = folder.status().last_scan_at
        folder.scan_now()
        wait_until(lambda: folder.status().last_scan_at != first)
    finally:
        folder.stop()


def test_a_file_that_cannot_be_parsed_is_not_retried(tmp_path: Path) -> None:
    """Retrying a corrupt PDF only parsed it again to the same error, WATCH_MAX_RETRIES
    times; it now waits for the file to change."""

    write(tmp_path / "a.txt", "A")
    recorder, clock = JobRecorder(), Clock()
    folder = retrying_watcher(tmp_path, recorder, clock)
    folder.scan_once()
    folder.scan_once()
    recorder.finish("job-1", "failed", "Could not parse 'a.txt'.", permanent=True)

    assert folder.scan_once()["gave_up"] == ["a.txt"]
    clock.now += 10_000
    assert not any(folder.scan_once().values())
    assert len(recorder.enqueued) == 1


def test_a_delete_that_fails_is_retried_and_the_others_still_happen(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        write(tmp_path / name, name)
    recorder = Recorder()
    folder = watcher(tmp_path, recorder, watch_delete_missing=True)
    folder.scan_once()
    folder.scan_once()
    real_delete = recorder.delete
    down = {"a.txt"}

    def flaky(name: str) -> None:
        if name in down:
            raise ConnectionError("qdrant unreachable")
        real_delete(name)

    folder._delete = flaky
    (tmp_path / "a.txt").unlink()
    (tmp_path / "b.txt").unlink()
    assert folder.scan_once()["deleted"] == ["b.txt"]
    down.clear()
    assert folder.scan_once()["deleted"] == ["a.txt"]
