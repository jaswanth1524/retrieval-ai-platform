"""Pass 23: migration, backup and restore fixes."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import httpx
import pytest


def test_a_download_leaves_the_bundle_and_ranges_resume_it(tmp_path: Path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import api.routes.migrate as migrate_routes

    body = bytes(range(256)) * 400
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(body)
    migrate_routes._keep_export("r23", bundle, hashlib.sha256(body).hexdigest())
    app = FastAPI()
    app.get("/b")(lambda: migrate_routes.migrate_export_download("r23"))
    try:
        with TestClient(app) as client:
            # A tail read (a zip tool reading the directory first) ends at the last byte.
            tail = client.get("/b", headers={"Range": "bytes=-100"})
            assert tail.status_code == 206 and tail.content == body[-100:]
            whole = client.get("/b")
            assert whole.content == body and "X-Content-SHA256" in whole.headers
            assert client.get("/b", headers={"Range": "bytes=100-"}).content == body[100:]
        assert bundle.exists()
    finally:
        migrate_routes._forget_export("r23")


def test_the_client_removes_the_servers_bundle_only_once_it_arrived_whole(
    tmp_path: Path,
) -> None:
    from cli.client import DocragClient, DocragError

    body = b"bundle" * 1000
    sha = hashlib.sha256(body).hexdigest()

    def client_for(served: bytes, calls: list[str]) -> DocragClient:
        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.method)
            if request.method == "POST":
                return httpx.Response(202, json={"import_id": "x"})
            if request.method == "DELETE":
                return httpx.Response(204)
            if "/documents/imports/" in request.url.path:
                return httpx.Response(200, json={"state": "done", "finished_at": 1.0})
            return httpx.Response(200, headers={"X-Content-SHA256": sha}, content=served)

        http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
        return DocragClient("http://x", None, http=http, sleep=lambda _: None)

    calls: list[str] = []
    client_for(body, calls).migrate_pack_in_background(tmp_path / "ok.zip")
    assert calls[-1] == "DELETE"

    calls = []
    with pytest.raises(DocragError, match=r"checksum.*kept in"):
        client_for(body[:-1], calls).migrate_pack_in_background(tmp_path / "bad.zip")
    assert "DELETE" not in calls
    assert (tmp_path / ".bad.zip.part").read_bytes() == body[:-1]


def test_an_older_server_is_refused_before_anything_is_restored() -> None:
    from cli.client import DocragClient, DocragError

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"status": "ok", "version": "0.2.0"})

    http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    client = DocragClient("http://x", None, http=http)
    with pytest.raises(DocragError, match="Nothing was sent"):
        client.require_feature("restore_dry_run", "a restore dry run")
    assert calls == ["/health"]


def test_a_bundles_originals_dont_replace_a_newer_upload(tmp_path: Path) -> None:
    from api.migrate import _newer_here
    from api.raw_documents import RawDocumentStore

    store = RawDocumentStore(tmp_path)
    store.save("a.txt", b"version two")
    indexed = hashlib.sha256(b"version two").hexdigest()
    assert _newer_here(store, "a.txt", b"version one", lambda _: indexed)
    # The bundle's own content is what is indexed: it is stored.
    assert not _newer_here(
        store, "a.txt", b"version one", lambda _: hashlib.sha256(b"version one").hexdigest()
    )
    # Nothing indexed, or a stored original that isn't what is indexed.
    assert not _newer_here(store, "a.txt", b"version one", lambda _: None)
    assert not _newer_here(store, "a.txt", b"version one", lambda _: "f" * 64)


def test_a_snapshot_is_not_recovered_over_a_workspace_with_work_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import api.routes.migrate as migrate_routes
    from api.ingestion import ClosedForWorkError, IngestSequencer
    from tests.factories import make_test_settings

    # A sequencer of its own: other tests' queued jobs must not count here.
    ingest_sequencer = IngestSequencer()
    monkeypatch.setattr(migrate_routes, "ingest_sequencer", ingest_sequencer)
    settings = make_test_settings()
    target: Any = SimpleNamespace(settings=settings)
    seen: list[str] = []

    class Client:
        def count(self, **_: Any) -> Any:
            # While the workspace is closed, nothing new is admitted.
            with pytest.raises(ClosedForWorkError):
                ingest_sequencer.admit("during.txt")
            seen.append("counted")
            return SimpleNamespace(count=3)

    server: Any = SimpleNamespace(client=Client(), settings=settings)
    sequence = ingest_sequencer.admit("queued.txt")
    try:
        reason = migrate_routes._recover_closed(None, {}, server, target, "", 0)  # type: ignore[arg-type]
        assert reason == "documents were being indexed or deleted here" and not seen
    finally:
        ingest_sequencer.release("queued.txt")
    assert sequence
    reason = migrate_routes._recover_closed(None, {}, server, target, "", 0)  # type: ignore[arg-type]
    assert reason == "3 passage(s) were indexed here meanwhile" and seen == ["counted"]
    # Reopened on every path.
    ingest_sequencer.admit("after.txt")
    ingest_sequencer.release("after.txt")


def _kept(tmp_path: Path, name: str, drill_ok: bool | None) -> Path:
    from api.restore import BackupCheck
    from api.scheduled_backup import save_check

    backup = tmp_path / name
    backup.write_bytes(b"zip")
    save_check(backup, BackupCheck(documents=1, originals=1, drill_ok=drill_ok))
    return backup


def test_pruning_keeps_the_last_backup_that_passed_its_drill(tmp_path: Path) -> None:
    from api.scheduled_backup import ScheduledBackup, load_check

    good = _kept(tmp_path, "docrag-20260101-000000.zip", True)
    failed = [_kept(tmp_path, f"docrag-2026010{n}-000000.zip", False) for n in (2, 3, 4)]
    loop = ScheduledBackup.__new__(ScheduledBackup)
    loop._keep, loop.checks = 2, {}
    loop._prune(tmp_path)
    assert good.exists() and not failed[0].exists() and failed[-1].exists()
    check = load_check(good)
    assert check is not None and check.restorable


def test_a_check_kept_before_drills_were_still_counts_as_good(tmp_path: Path) -> None:
    import json

    from api.scheduled_backup import ScheduledBackup, check_path

    old = _kept(tmp_path, "docrag-20260101-000000.zip", None)
    record = json.loads(check_path(old).read_text())
    del record["check"]["drill_ok"]
    check_path(old).write_text(json.dumps(record))
    newer = [_kept(tmp_path, f"docrag-2026010{n}-000000.zip", False) for n in (2, 3)]
    loop = ScheduledBackup.__new__(ScheduledBackup)
    loop._keep, loop.checks = 1, {}
    loop._prune(tmp_path)
    assert old.exists() and newer[-1].exists() and not newer[0].exists()


def test_an_ods_sheets_hidden_columns_are_left_out() -> None:
    import zipfile

    from api.parsers.ods import parse_ods_document
    from tests.factories import make_test_settings

    office = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    table = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    text = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"

    def row(*values: str) -> str:
        cells = "".join(
            f"<table:table-cell><text:p>{v}</text:p></table:table-cell>" for v in values
        )
        return f"<table:table-row>{cells}</table:table-row>"

    content = (
        f'<office:document-content xmlns:office="{office}" xmlns:table="{table}" '
        f'xmlns:text="{text}"><office:body><office:spreadsheet>'
        '<table:table table:name="Pay"><table:table-column/>'
        '<table:table-column table:visibility="collapse" '
        'table:number-columns-repeated="2"/><table:table-column/>'
        + row("Name", "Salary", "Bonus", "Team")
        + row("Ada", "90000", "5000", "Core")
        + "</table:table></office:spreadsheet></office:body></office:document-content>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("content.xml", content)
    found = " ".join(s.text for s in parse_ods_document("pay.ods", buffer.getvalue()))
    assert "Ada" in found and "Core" in found
    assert "90000" not in found and "Salary" not in found and "5000" not in found
    shown = parse_ods_document(
        "pay.ods", buffer.getvalue(), make_test_settings(index_hidden_content=True)
    )
    assert "90000" in " ".join(s.text for s in shown)


def test_a_dry_run_rejects_the_name_a_restore_would(tmp_path: Path) -> None:
    import json
    import zipfile
    from types import SimpleNamespace

    import api.routes.backup as backup_routes
    from api.raw_documents import RawDocumentStore
    from tests.factories import make_test_settings

    backup = io.BytesIO()
    with zipfile.ZipFile(backup, "w") as archive:
        documents = [
            {"filename": name, "original": f"originals/{name}"} for name in ("A.txt", "new.txt")
        ]
        archive.writestr(
            "manifest.json", json.dumps({"manifest_version": 1, "documents": documents})
        )
        for name in ("A.txt", "new.txt"):
            archive.writestr(f"originals/{name}", b"Intro\nalpha")
    repository: Any = SimpleNamespace(filename_metadata=lambda settings: {"a.txt": None})
    report = backup_routes._restore(
        backup,
        repository,
        make_test_settings(),
        None,  # type: ignore[arg-type]
        RawDocumentStore(tmp_path),
        None,
        dry_run=True,
    )
    assert report.rejected == ["A.txt"] and report.would_restore == ["new.txt"]
    assert not backup_routes._verify_running.locked()


def test_verifying_a_bundle_counts_the_originals_it_holds(tmp_path: Path) -> None:
    import argparse
    import json
    import zipfile

    from cli.admin_commands import MIGRATE_VERSION, verify_bundle

    def bundle(manifest: object) -> Path:
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as backup:
            backup.writestr("manifest.json", json.dumps(manifest))
            backup.writestr("originals/a.txt", b"a")
        path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(path, "w") as outer:
            outer.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "migrate_version": MIGRATE_VERSION,
                        "workspaces": [{"name": "default", "backup": "default.zip"}],
                    }
                ),
            )
            outer.writestr("default.zip", inner.getvalue())
        return path

    def verify(path: Path) -> dict[str, Any]:
        out = io.StringIO()
        verify_bundle(path, argparse.Namespace(json=True), out)
        result: dict[str, Any] = json.loads(out.getvalue())
        return result

    listed = [{"original": "originals/a.txt"}, {"original": "originals/b.txt"}, {}]
    found = verify(bundle({"documents": listed}))
    assert found["workspaces"][0]["originals"] == 1
    assert not found["ok"] and "originals/b.txt" in found["problems"][0]
    assert not verify(bundle(["not an object"]))["ok"]


def test_eval_noise_and_categories_measure_what_the_gate_does() -> None:
    from eval.stats import answer_noise

    def row(status: str, labelled: bool) -> dict[str, Any]:
        return {"question": "and then?", "answer_status": status, "labelled": labelled}

    baseline = [row("answered", True), row("insufficient_context", True), row("answered", False)]
    current = [row("answered", True), row("insufficient_context", True), row("answered", False)]
    assert answer_noise(current, baseline)["answered_rate"] == (0.0, 0.0, 0.0)


def test_the_decomposed_arms_baseline_records_the_model() -> None:
    from eval.harness import settings_fingerprint
    from tests.factories import make_test_settings

    settings = make_test_settings()
    assert "llm_model" in settings_fingerprint(settings, "decomposed")
    assert "llm_model" not in settings_fingerprint(settings, "reranked")
