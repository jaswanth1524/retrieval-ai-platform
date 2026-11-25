"""Pass 21: backup and recovery fixes."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

from qdrant_client import QdrantClient

from tests.factories import make_test_settings


def _backup_with_copy(tmp_path: Any, command: str) -> Any:
    from api.raw_documents import RawDocumentStore
    from api.repository import VectorRepository
    from api.scheduled_backup import ScheduledBackup
    from api.workspace_target import WorkspaceTarget

    settings = make_test_settings(
        raw_document_dir=str(tmp_path / "raw"),
        backup_dir=str(tmp_path / "backups"),
        backup_copy_command=command,
    )
    store = RawDocumentStore(str(tmp_path / "raw"))
    store.save("a.txt", b"Intro\nalpha")
    return ScheduledBackup(
        settings,
        VectorRepository(QdrantClient(":memory:")),
        lambda: [WorkspaceTarget(settings, store)],
        None,
        threading.Lock(),
        clock=time.time,
    )


def test_a_failed_backup_copy_is_retried_on_the_next_pass(tmp_path: Any) -> None:
    flag = tmp_path / "failed-once"
    script = tmp_path / "copy.py"
    script.write_text(
        "import pathlib, sys\n"
        f"flag = pathlib.Path({str(flag)!r})\n"
        "if not flag.exists():\n"
        "    flag.write_text('x')\n"
        "    sys.exit(5)\n",
        encoding="utf-8",
    )
    backup = _backup_with_copy(tmp_path, f"{sys.executable} {script}")
    assert backup.run_once()
    assert backup.last is not None and not backup.last.ok and "exit 5" in (backup.last.error or "")
    # Not due for another backup: the pass only retries the copy.
    assert backup.run_once() == []
    assert backup.last.ok and backup.last.error is None
    assert backup.run_once() == []  # copied: nothing more to retry


def test_a_copy_past_its_timeout_is_stopped_with_its_children(tmp_path: Any) -> None:
    from api.scheduled_backup import copy_backup

    child_pid = tmp_path / "child.pid"
    script = tmp_path / "copy.py"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    backup = tmp_path / "docrag-1.zip"
    backup.write_bytes(b"zip")
    settings = make_test_settings(
        backup_copy_command=f"{sys.executable} {script}", backup_copy_timeout_seconds=2
    )
    began = time.monotonic()
    assert "ran past" in (copy_backup(settings, backup) or "")
    assert time.monotonic() - began < 20
    pid = int(child_pid.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, 9)
        raise AssertionError("the copy command's child was left running")


def test_a_lost_index_is_reported_once_and_before_a_rebuild_that_fails_to_start(
    tmp_path: Any, monkeypatch: Any
) -> None:
    import api.main as main
    from api.chunking import HeuristicTokenCounter
    from api.dependencies import clear_dependency_caches
    from api.raw_documents import RawDocumentStore

    client = QdrantClient(":memory:")
    monkeypatch.setattr(main, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(main, "get_token_counter", lambda: HeuristicTokenCounter())
    sent: list[str] = []
    monkeypatch.setattr(main, "notify", lambda event, text: sent.append(event))

    def refuses(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the import thread is gone")

    monkeypatch.setattr(main, "start_import", refuses)
    RawDocumentStore(str(tmp_path)).save("a.txt", b"Intro\nalpha")
    settings = make_test_settings(raw_document_dir=str(tmp_path), auto_rebuild_lost_index=True)
    clear_dependency_caches()
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    reported: set[str] = set()
    # The rebuild couldn't start, but the loss was told first.
    main.check_for_lost_index(settings, reported)
    assert sent == ["index_lost"]
    # The periodic check doesn't tell it again while it lasts.
    assert main.check_for_lost_index(settings, reported) == ["default"]
    assert sent == ["index_lost"]
    monkeypatch.undo()
    clear_dependency_caches()


def test_the_lost_index_watch_checks_again_until_stopped(monkeypatch: Any) -> None:
    import api.main as main

    checks: list[int] = []
    monkeypatch.setattr(main, "LOST_INDEX_CHECK_SECONDS", 0.01)
    monkeypatch.setattr(main, "check_for_lost_index", lambda s, r: checks.append(1) or [])
    stop = threading.Event()
    thread = threading.Thread(target=main.watch_for_lost_index, args=(None, stop))
    thread.start()
    deadline = time.monotonic() + 5
    while len(checks) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(5)
    assert len(checks) >= 3 and not thread.is_alive()
