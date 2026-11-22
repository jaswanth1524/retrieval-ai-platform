"""Pass 20: features."""

from __future__ import annotations

from typing import Any

import pytest
from qdrant_client import QdrantClient

from tests.factories import make_test_settings


def test_a_lost_index_is_reported_at_startup_and_rebuilt_when_asked(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.main as main
    from api.chunking import HeuristicTokenCounter
    from api.raw_documents import RawDocumentStore

    client = QdrantClient(":memory:")
    monkeypatch.setattr(main, "get_qdrant_client", lambda: client)

    def no_model() -> Any:
        raise AssertionError("detecting a lost index must not load the tokenizer")

    monkeypatch.setattr(main, "get_token_counter", no_model)
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(main, "notify", lambda event, text: sent.append((event, text)))
    started: list[str] = []
    monkeypatch.setattr(main, "start_import", lambda kind, *a, **k: started.append(kind))
    RawDocumentStore(str(tmp_path)).save("a.txt", b"Intro\nalpha")

    from api.dependencies import clear_dependency_caches

    settings = make_test_settings(raw_document_dir=str(tmp_path))
    clear_dependency_caches()
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    assert main.check_for_lost_index(settings) == ["default"]
    assert sent and sent[0][0] == "index_lost" and "docrag rebuild" in sent[0][1]
    assert started == []

    sent.clear()
    monkeypatch.setattr(main, "get_token_counter", lambda: HeuristicTokenCounter())
    on = make_test_settings(raw_document_dir=str(tmp_path), auto_rebuild_lost_index=True)
    assert main.check_for_lost_index(on) == ["default"]
    assert started == ["rebuild"] and "being rebuilt" in sent[0][1]
    monkeypatch.undo()
    clear_dependency_caches()


def test_a_background_import_says_how_it_ended(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    import api.imports as imports

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(imports, "notify", lambda event, text: sent.append((event, text)))
    done = threading.Event()

    def run(status: Any) -> None:
        status.queued = 3

    def failing(status: Any) -> None:
        status.failed = 1
        status.note("b.txt: not a document")

    imports.start("archive", "docs.zip", "", "c", run, cleanup=done.set)
    assert done.wait(5)
    imports.start("archive", "bad.zip", "legal", "c", failing)
    for _ in range(100):
        if len(sent) == 2:
            break
        threading.Event().wait(0.05)
    assert sent[0][0] == "import_finished" and "3 queued" in sent[0][1]
    assert sent[1][0] == "import_failed" and "workspace 'legal'" in sent[1][1]
    assert "b.txt: not a document" in sent[1][1]


def test_each_backup_is_handed_to_the_copy_command_and_a_failed_copy_fails_it(
    tmp_path: Any,
) -> None:
    import sys

    from api.scheduled_backup import copy_backup

    backup = tmp_path / "docrag-1.zip"
    backup.write_bytes(b"zip")
    copies = tmp_path / "offsite"
    copies.mkdir()
    script = tmp_path / "copy.py"
    script.write_text(
        "import shutil, sys\nshutil.copy(sys.argv[2], sys.argv[1])\n", encoding="utf-8"
    )
    ok = make_test_settings(backup_copy_command=f"{sys.executable} {script} {copies}")
    assert copy_backup(ok, backup) is None
    assert (copies / "docrag-1.zip").read_bytes() == b"zip"

    failing = make_test_settings(
        backup_copy_command=f"{sys.executable} -c 'import sys; sys.exit(3)'"
    )
    assert "exit 3" in (copy_backup(failing, backup) or "")
    missing = make_test_settings(backup_copy_command="/no/such/command")
    assert "could not run" in (copy_backup(missing, backup) or "")
    assert copy_backup(make_test_settings(), backup) is None


def test_wrong_keys_are_limited_even_with_the_rate_limit_off() -> None:
    from fastapi.testclient import TestClient

    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app
    from tests.routes_http.support import make_settings as route_settings

    clear_dependency_caches()
    settings = route_settings(api_key="right-key", failed_key_attempts_per_minute=3)
    with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as client:
        codes = [
            client.get("/documents", headers={"X-API-Key": "guess"}).status_code for _ in range(5)
        ]
        assert codes == [401, 401, 401, 429, 429]
        # The right key is never held back.
        assert client.get("/documents", headers={"X-API-Key": "right-key"}).status_code == 200
    clear_dependency_caches()


def test_a_scope_that_names_nothing_indexed_says_so_and_a_follow_up_says_what_it_searched() -> None:
    import json

    from fastapi.testclient import TestClient

    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings

    clear_dependency_caches()
    with TestClient(hermetic_app(route_settings(), QdrantClient(":memory:"))) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        for body in ({"filenames": ["guid.txt"]}, {"tags": ["nothing-has-this"]}):
            sync = client.post("/questions", json={"question": "alpha?", **body}).json()
            assert sync["answer_status"] == "empty_scope", body
            streamed = client.post("/questions/stream", json={"question": "alpha?", **body})
            done = [
                json.loads(line[6:])
                for line in streamed.text.splitlines()
                if line.startswith("data: ") and '"done"' in line
            ][0]
            assert done["answer_status"] == "empty_scope", body
        # A named document that is indexed is a scope like any other.
        plain = client.post("/questions", json={"question": "alpha?", "filenames": ["guide.txt"]})
        assert plain.json()["answer_status"] != "empty_scope"
        follow = client.post(
            "/questions",
            json={
                "question": "and beta?",
                "history": [
                    {"role": "user", "content": "alpha?"},
                    {"role": "assistant", "content": "Alpha is documented [1]."},
                ],
            },
        ).json()
        assert "retrieval_query" in follow
    clear_dependency_caches()


def test_a_pasted_text_is_checked_sentence_by_sentence_against_the_documents() -> None:
    from fastapi.testclient import TestClient

    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings

    clear_dependency_caches()
    with TestClient(hermetic_app(route_settings(), QdrantClient(":memory:"))) as client:
        assert (
            upload_and_wait(
                client, "leave.txt", b"Leave\nStaff get 28 days of annual leave each year."
            )["state"]
            == "done"
        )
        text = (
            "Staff get 28 days of annual leave. Staff get 30 days of annual leave [1]. "
            "The moon is made of green cheese. Hi."
        )
        response = client.post("/questions/verify", json={"text": text})
        assert response.status_code == 200
        checks = response.json()["sentences"]
        assert [c["status"] for c in checks] == [
            "supported",
            "numbers_differ",
            "unsupported",
            "no_claim",
        ]
        assert checks[0]["filename"] == "leave.txt" and "28 days" in checks[0]["quote"]
        assert checks[1]["missing_numbers"] == ["30"]
        assert client.post("/questions/verify", json={"text": "  "}).status_code in (400, 422)
    clear_dependency_caches()


def test_docrag_verify_prints_each_sentence_and_fails_on_an_unsupported_one(
    tmp_path: Any,
) -> None:
    import argparse
    import io

    from cli.question_commands import cmd_verify

    class _Client:
        def verify(self, text: str, filenames: Any, tags: Any) -> dict[str, Any]:
            return {
                "sentences": [
                    {
                        "sentence": "Leave is 28 days.",
                        "status": "supported",
                        "filename": "a.md",
                        "page": 1,
                        "section": "Leave",
                        "quote": "28 days of leave",
                    },
                    {
                        "sentence": "Pay is 9 euros.",
                        "status": "numbers_differ",
                        "filename": "a.md",
                        "page": 2,
                        "section": "Pay",
                        "quote": "Pay is 10 euros",
                        "missing_numbers": ["9"],
                    },
                ],
                "more": False,
            }

    draft = tmp_path / "draft.txt"
    draft.write_text("Leave is 28 days. Pay is 9 euros.", encoding="utf-8")
    out = io.StringIO()
    code = cmd_verify(
        _Client(),  # type: ignore[arg-type]
        argparse.Namespace(path=str(draft), file=None, tag=None, json=False),
        out,
    )
    assert code == 1
    text = out.getvalue()
    assert "ok  Leave is 28 days." in text and "not in it: 9" in text


def test_a_page_added_from_a_url_can_be_kept_under_a_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    import api.routes.documents_write as documents_write
    from api.dependencies import clear_dependency_caches
    from api.url_fetch import FetchedDocument
    from tests.routes_http.support import hermetic_app, wait_until
    from tests.routes_http.support import make_settings as route_settings

    def fetched(url: str, settings: Any, **kwargs: Any) -> FetchedDocument:
        return FetchedDocument("policy.txt", b"Intro\nalpha", url)

    monkeypatch.setattr(documents_write, "fetch_document", fetched)
    clear_dependency_caches()
    app = hermetic_app(route_settings(url_ingest_enabled=True), QdrantClient(":memory:"))
    with TestClient(app) as client:
        accepted = client.post(
            "/documents/from-url",
            json={"url": "https://example.com/policy", "folder": "/handbook/2024/"},
        )
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["filename"] == "handbook/2024/policy.txt"
        wait_until(
            lambda: "handbook/2024/policy.txt" in client.get("/documents").json()["filenames"]
        )
        bad = client.post(
            "/documents/from-url", json={"url": "https://example.com/p", "folder": "../up"}
        )
        assert bad.status_code == 400
    clear_dependency_caches()


def test_a_failed_copy_fails_the_backup_after_the_export_lock_is_released(
    tmp_path: Any,
) -> None:
    import sys
    import threading
    import time

    from api.raw_documents import RawDocumentStore
    from api.repository import VectorRepository
    from api.scheduled_backup import ScheduledBackup, WorkspaceTarget

    lock = threading.Lock()
    held: list[bool] = []
    script = tmp_path / "copy.py"
    script.write_text("import sys\nsys.exit(4)\n", encoding="utf-8")
    settings = make_test_settings(
        raw_document_dir=str(tmp_path / "raw"),
        backup_dir=str(tmp_path / "backups"),
        backup_copy_command=f"{sys.executable} {script}",
    )
    store = RawDocumentStore(str(tmp_path / "raw"))
    store.save("a.txt", b"Intro\nalpha")
    backup = ScheduledBackup(
        settings,
        VectorRepository(QdrantClient(":memory:")),
        lambda: [WorkspaceTarget(settings, store)],
        None,
        lock,
        clock=time.time,
    )
    import api.scheduled_backup as scheduled

    real = scheduled.copy_backup

    def watching(s: Any, path: Any) -> Any:
        held.append(lock.locked())
        return real(s, path)

    scheduled.copy_backup = watching  # type: ignore[assignment]
    try:
        assert backup.run_once()
    finally:
        scheduled.copy_backup = real  # type: ignore[assignment]
    assert held == [False]
    assert backup.last is not None and not backup.last.ok
    assert "exit 4" in (backup.last.error or "")


def test_a_verify_whose_client_left_stops_before_the_next_sentence() -> None:
    from api.verify import VerifyAbandoned, verify_text

    searched: list[str] = []
    asked = iter([False, True])

    def search(sentence: str) -> list[Any]:
        searched.append(sentence)
        return []

    with pytest.raises(VerifyAbandoned):
        verify_text(
            "Staff get 28 days of leave. Pay is monthly on the 25th.",
            search,
            lambda: next(asked),
        )
    assert searched == ["Staff get 28 days of leave."]
