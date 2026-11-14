"""Pass 17: ingest and ops fixes."""

from __future__ import annotations

import functools

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.url_fetch import fetch_document
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait, wait_until
from tests.test_url_fetch import resolver, transport


def test_a_url_named_like_an_uploaded_document_is_refused_unless_replacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_dependency_caches()
    app = hermetic_app(make_settings(url_ingest_enabled=True), QdrantClient(":memory:"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"Intro\nFrom the web.", headers={"content-type": "text/plain"}
        )

    monkeypatch.setattr(
        "api.routes.documents_write.fetch_document",
        functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler)),
    )
    with TestClient(app) as client:
        upload_and_wait(client, "handbook.txt", b"Intro\nUploaded by hand.")
        refused = client.post(
            "/documents/from-url", json={"url": "https://other.example/handbook.txt"}
        )
        assert refused.status_code == 409
        assert "another source" in refused.json()["detail"]

        accepted = client.post(
            "/documents/from-url",
            json={"url": "https://other.example/handbook.txt", "replace": True},
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]
        wait_until(lambda: client.get(f"/documents/jobs/{job_id}").json()["state"] == "done")
        # Now it comes from that URL: adding it again updates it in place.
        again = client.post(
            "/documents/from-url", json={"url": "https://other.example/handbook.txt"}
        )
        assert again.status_code == 202
    clear_dependency_caches()


def test_a_sitemap_page_does_not_replace_a_document_uploaded_after_the_import_began() -> None:
    from api.document_ops import free_or_from
    from api.ingestion import IngestionError
    from api.pipeline import IngestService
    from api.repository import VectorRepository
    from tests.test_pipeline import StaticEmbeddingProvider, make_embedding
    from tests.test_pipeline import make_settings as pipeline_settings

    settings = pipeline_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    check = free_or_from(repository, settings, "intro.html", "https://site.example/intro")
    check()  # free: fine
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "intro.html", b"<p>Uploaded by hand.</p>"
    )
    with pytest.raises(IngestionError, match="from elsewhere"):
        check()


def test_verify_does_not_decompress_a_member_bigger_than_restore_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import json
    import zipfile

    from api import restore

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"manifest_version": 1, "documents": []}),
        )
        archive.writestr("feedback.jsonl", b"\0" * 4096)
        archive.writestr("originals/big.txt", b"\0" * 512)
    buffer.seek(0)
    monkeypatch.setattr(restore, "VERIFY_MAX_MEMBER_BYTES", 1024)
    read: list[str] = []
    real_open = zipfile.ZipFile.open

    def spy(self: zipfile.ZipFile, name: object, *args: object, **kwargs: object) -> object:
        read.append(getattr(name, "filename", str(name)))
        return real_open(self, name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(zipfile.ZipFile, "open", spy)
    check = restore.verify_backup(buffer, 256)
    # Past the check's own cap: not decompressed, and said so.
    assert "feedback.jsonl" not in read
    assert any("more than a check reads" in problem for problem in check.problems)
    # Past the upload cap: restore refuses it by size; not decompressed, not "damage".
    assert "originals/big.txt" not in read
    assert not any("big.txt" in problem for problem in check.problems)


def test_pruning_keeps_the_newest_backup_that_passed_its_check(tmp_path: object) -> None:
    import threading
    from pathlib import Path

    from api.restore import BackupCheck
    from api.scheduled_backup import ScheduledBackup
    from tests.factories import make_test_settings

    root = Path(str(tmp_path))
    settings = make_test_settings(backup_dir=str(root), backup_keep=1)
    loop = ScheduledBackup(settings, None, lambda: [], None, threading.Lock())  # type: ignore[arg-type]
    directory = root / "default"
    directory.mkdir()
    names = [
        "docrag-20260101-000000.zip",
        "docrag-20260102-000000.zip",
        "docrag-20260103-000000.zip",
    ]
    for name in names:
        (directory / name).write_bytes(b"zip")
    loop.record_check(directory / names[0], BackupCheck())
    loop.record_check(directory / names[1], BackupCheck(problems=["damaged"]))
    loop.record_check(directory / names[2], BackupCheck(problems=["damaged"]))
    loop._prune(directory)
    assert sorted(p.name for p in directory.glob("docrag-*.zip")) == [names[0], names[2]]

    loop.by_workspace["gone"] = object()  # type: ignore[assignment]
    loop.forget_workspace("gone")
    assert "gone" not in loop.by_workspace


def test_one_long_csv_cell_does_not_fail_the_file() -> None:
    from api.parsers.csv import parse_csv_document

    sections = parse_csv_document("long.csv", "name,notes\nx," + "y" * 200_000 + "\n")
    assert sections and "y" * 1000 in sections[0].text


def test_a_backup_download_counts_against_the_rate_limit() -> None:
    from api.request_guard import is_rate_limited

    assert is_rate_limited("GET", "/backups/docrag-20260101-000000.zip")
    assert not is_rate_limited("GET", "/backups")


def test_the_cli_preview_raises_a_docrag_error_for_an_error_response(tmp_path: object) -> None:
    from pathlib import Path

    from cli.client import DocragClient, DocragError

    file = Path(str(tmp_path)) / "notes.xyz"
    file.write_bytes(b"x")
    http = httpx.Client(
        base_url="http://docrag.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(400, json={"detail": "Unsupported file type."})
        ),
    )
    with pytest.raises(DocragError, match="Unsupported"):
        DocragClient("http://docrag.test", http=http).preview(file)


def test_a_restored_document_keeps_its_watched_folder_only_when_its_file_is_there(
    tmp_path: object,
) -> None:
    from pathlib import Path

    from api.restore import _watched_from

    folder = Path(str(tmp_path))
    (folder / "here.txt").write_text("x")
    assert _watched_from({"watched_from": str(folder)}, "here.txt") == str(folder)
    assert _watched_from({"watched_from": str(folder)}, "elsewhere.txt") is None
    assert _watched_from({"watched_from": "relative/path"}, "here.txt") is None


def test_the_ingest_pool_has_ingest_workers_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    from api import dependencies

    monkeypatch.setenv("INGEST_WORKERS", "5")
    dependencies.clear_dependency_caches()
    try:
        assert dependencies.get_ingest_executor()._max_workers == 5
    finally:
        dependencies.get_ingest_executor().shutdown(wait=False)
        dependencies.clear_dependency_caches()


BACKSLASH = chr(92)


def test_rtf_keeps_body_text_and_refuses_what_isnt_rtf() -> None:
    from api.parsers.common import DocumentParseError, EmptyDocumentError
    from api.parsers.rtf import parse_rtf_document, rtf_to_text

    text = rtf_to_text(
        r"{\rtf1\ansi{\info{\title Secret title}}{\colortbl;\red0;}"
        r"Price: 5\{\}\\ euros\line next\tab col\par {\*\bkmkstart x}Kept\uc0"
        + BACKSLASH
        + "u8364 .}"
    )
    assert "Secret title" not in text
    assert "Price: 5{}\\ euros\nnext\tcol" in text and "Kept€." in text
    with pytest.raises(DocumentParseError):
        rtf_to_text("plain text")
    with pytest.raises(EmptyDocumentError):
        parse_rtf_document("e.rtf", rb"{\rtf1{\fonttbl{\f0 Arial;}}}")
    with pytest.raises(DocumentParseError, match="nests"):
        rtf_to_text("{\\rtf1" + "{" * 300 + "}" * 300 + "}")


def test_subtitles_group_cues_by_minute_and_drop_timings_and_tags() -> None:
    from api.parsers.common import EmptyDocumentError
    from api.parsers.subtitles import parse_subtitles_document

    sections = parse_subtitles_document(
        "t.srt",
        "1\r\n01:02:03,000 --> 01:02:05,000\r\n<b>Late</b> {\\an8}remark\r\n\r\n"
        "2\r\n00:00:10,000 --> 00:00:12,000\r\nFirst line\r\nsecond line\r\n",
    )
    assert [(s.page, s.section, s.text) for s in sections] == [
        (1, "0:00", "First line second line"),
        (63, "1:02:00", "Late remark"),
    ]
    with pytest.raises(EmptyDocumentError):
        parse_subtitles_document("t.vtt", "WEBVTT\n\nNOTE nothing else\n")


def test_doctor_finds_risky_settings(tmp_path: object) -> None:
    from collections import namedtuple
    from pathlib import Path

    from api.doctor import diagnose
    from tests.factories import make_test_settings

    Usage = namedtuple("Usage", "total used free")
    base = Path(str(tmp_path))
    settings = make_test_settings(
        backup_dir=str(base / "backups"),
        raw_document_dir="",
        ingest_workers=8,
    )
    found = diagnose(
        settings,
        chunk_counts={"huge.csv": 50_000, "small.md": 3},
        memory_limit=lambda: 1024**3,
        disk_usage=lambda path: Usage(100 * 1024**3, 0, 10 * 1024**2),
        environ={"WEB_CONCURRENCY": "4"},
        cpu_count=4,
    )
    subjects = {(f.level, f.subject) for f in found}
    assert ("warning", "BACKUP_DIR") in subjects  # backups without originals
    assert ("warning", "INGEST_MAX_PENDING_BYTES") in subjects  # 1 GB isn't enough
    assert ("warning", "WEB_CONCURRENCY") in subjects
    assert ("note", "INGEST_WORKERS") in subjects
    assert ("note", "huge.csv") in subjects and ("note", "small.md") not in subjects
    assert ("note", "API_KEY") in subjects
    assert any(f.subject == "BACKUP_DIR" and "MB free" in f.message for f in found)

    calm = diagnose(
        make_test_settings(api_key="k", rate_limit_per_minute=60, parse_memory_limit_mb=1500),
        memory_limit=lambda: 64 * 1024**3,
        disk_usage=lambda path: Usage(100 * 1024**3, 0, 90 * 1024**3),
        environ={},
        cpu_count=8,
    )
    assert calm == []


def test_the_doctor_route_needs_the_full_key() -> None:
    from fastapi.testclient import TestClient

    clear_dependency_caches()
    app = hermetic_app(make_settings(api_key="full", api_read_key="read"), QdrantClient(":memory:"))
    with TestClient(app) as client:
        assert client.get("/admin/doctor", headers={"X-API-Key": "read"}).status_code == 403
        report = client.get("/admin/doctor", headers={"X-API-Key": "full"})
        assert report.status_code == 200 and isinstance(report.json()["findings"], list)
    clear_dependency_caches()


def test_audit_events_are_kept_and_listed_when_audit_log_path_is_set(tmp_path: object) -> None:
    from pathlib import Path

    from fastapi.testclient import TestClient

    from api.audit_log import AuditLog
    from api.dependencies import get_audit_log

    store = AuditLog(str(Path(str(tmp_path)) / "audit.db"), 1000)
    clear_dependency_caches()
    app = hermetic_app(make_settings(), QdrantClient(":memory:"))
    app.dependency_overrides[get_audit_log] = lambda: store
    with TestClient(app) as client:
        upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")
        assert client.delete("/documents/guide.txt").status_code in (200, 204)
        listed = client.get("/admin/audit").json()["entries"]
        events = [entry["event"] for entry in listed]
        assert events[0] == "document.delete" and "document.upload" in events
        assert listed[0]["fields"]["filename"] == "guide.txt"
        only = client.get("/admin/audit", params={"event": "document.delete"}).json()["entries"]
        assert [entry["event"] for entry in only] == ["document.delete"]
        prefix = client.get("/admin/audit", params={"event": "document."}).json()["entries"]
        assert len(prefix) == len(events)
    clear_dependency_caches()

    off = hermetic_app(make_settings(), QdrantClient(":memory:"))
    with TestClient(off) as client:
        assert client.get("/admin/audit").status_code == 404
    clear_dependency_caches()


def test_backup_and_storage_gauges_report_what_the_alerts_watch(tmp_path: object) -> None:
    import threading

    from prometheus_client import REGISTRY

    from api.metrics import watch_storage
    from api.scheduled_backup import BackupStatus, ScheduledBackup
    from tests.factories import make_test_settings

    loop = ScheduledBackup(
        make_test_settings(backup_dir=str(tmp_path)),
        None,
        lambda: [],
        None,
        threading.Lock(),  # type: ignore[arg-type]
    )
    loop._note(BackupStatus(at=1234.0, workspace="w1", ok=True))
    loop._note(BackupStatus(at=1300.0, workspace="w1", ok=False, error="disk"))
    assert REGISTRY.get_sample_value("docrag_backup_last_ok", {"workspace": "w1"}) == 0
    assert (
        REGISTRY.get_sample_value(
            "docrag_backup_last_success_timestamp_seconds", {"workspace": "w1"}
        )
        == 1234.0
    )
    watch_storage([("BACKUP_DIR", str(tmp_path))])
    free = REGISTRY.get_sample_value("docrag_storage_free_bytes", {"setting": "BACKUP_DIR"})
    assert free is not None and free > 0


def _zip_of(files: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_a_zip_bigger_than_the_queue_is_imported_whole_in_the_background() -> None:
    from fastapi.testclient import TestClient

    clear_dependency_caches()
    # Room for two jobs at a time: the request alone would defer the rest.
    app = hermetic_app(make_settings(ingest_jobs_max_retained=2), QdrantClient(":memory:"))
    files = {f"doc{i}.txt": f"Intro\nalpha number {i}".encode() for i in range(6)}
    with TestClient(app) as client:
        started = client.post(
            "/documents/archive",
            files={"file": ("a.zip", _zip_of(files), "application/zip")},
            data={"background": "true"},
        )
        assert started.status_code == 202
        import_id = started.json()["import_id"]
        assert import_id and started.json()["deferred"] == []
        wait_until(
            lambda: client.get(f"/documents/imports/{import_id}").json()["state"] == "done",
            timeout=30,
        )
        status = client.get(f"/documents/imports/{import_id}").json()
        assert status["queued"] == 6 and status["kind"] == "archive"
        wait_until(lambda: len(client.get("/documents").json()["filenames"]) == 6, timeout=30)
        assert [i["id"] for i in client.get("/documents/imports").json()["imports"]] == [import_id]
        assert client.delete(f"/documents/imports/{import_id}").status_code == 409
        assert client.get("/documents/imports/nope").status_code == 404
    clear_dependency_caches()


def test_an_import_waits_for_room_and_stops_when_cancelled() -> None:
    from api.imports import BackgroundImport, wait_for_room

    status = BackgroundImport("i", "archive", "a.zip", "", "c", 0.0)
    answers = iter([False, False, True])
    naps: list[float] = []
    assert wait_for_room(lambda size: next(answers), 10, status, sleep=naps.append)
    assert len(naps) == 2
    status.cancel.set()
    assert not wait_for_room(lambda size: False, 10, status, sleep=naps.append)
    assert status.state == "cancelling"


def test_a_backup_is_restored_in_the_background(tmp_path: object) -> None:
    import json
    from pathlib import Path

    from fastapi.testclient import TestClient

    backup = _zip_of(
        {
            "manifest.json": json.dumps(
                {
                    "manifest_version": 1,
                    "documents": [
                        {"filename": name, "original": f"originals/{name}", "tags": []}
                        for name in ("a.txt", "b.txt")
                    ],
                }
            ).encode(),
            "originals/a.txt": b"Intro\nalpha",
            "originals/b.txt": b"Intro\nbeta",
        }
    )
    clear_dependency_caches()
    target = hermetic_app(
        make_settings(raw_document_dir=str(Path(str(tmp_path)) / "raw2")), QdrantClient(":memory:")
    )
    with TestClient(target) as client:
        refused = client.post(
            "/import",
            files={"file": ("x.zip", _zip_of({"a.txt": b"x"}), "application/zip")},
            data={"background": "true"},
        )
        assert refused.status_code == 400  # no manifest: said now, not in the import
        started = client.post(
            "/import",
            files={"file": ("backup.zip", backup, "application/zip")},
            data={"background": "true"},
        ).json()
        import_id = started["import_id"]
        wait_until(
            lambda: client.get(f"/documents/imports/{import_id}").json()["state"] == "done",
            timeout=30,
        )
        final = client.get(f"/documents/imports/{import_id}").json()
        assert final["queued"] == 2, final
        wait_until(lambda: len(client.get("/documents").json()["filenames"]) == 2, timeout=30)
    clear_dependency_caches()


def test_the_outline_lists_sections_and_page_starts_without_text() -> None:
    from fastapi.testclient import TestClient

    clear_dependency_caches()
    app = hermetic_app(make_settings(), QdrantClient(":memory:"))
    with TestClient(app) as client:
        upload_and_wait(
            client,
            "guide.md",
            b"# Leave\n\n"
            + b"Employees get twenty days of paid leave every year. " * 8
            + b"\n\n# Travel\n\n"
            + b"Book flights fourteen days ahead through the approved agency. " * 8
            + b"\n",
        )
        outline = client.get("/documents/guide.md/outline").json()
        assert [s["section"] for s in outline["sections"]] == ["Leave", "Travel"]
        assert outline["pages"][0]["page"] == 1 and outline["total_chunks"] >= 2
        assert all("text" not in s for s in outline["sections"])
        assert client.get("/documents/missing.md/outline").status_code == 404
    clear_dependency_caches()


def test_a_failed_job_is_counted_by_what_started_it() -> None:
    from prometheus_client import REGISTRY

    def failures(kind: str) -> float:
        value = REGISTRY.get_sample_value("docrag_ingest_job_failures_total", {"kind": kind})
        return value or 0.0

    before = failures("upload")
    clear_dependency_caches()
    with TestClient(hermetic_app(make_settings(), QdrantClient(":memory:"))) as client:
        assert upload_and_wait(client, "empty.txt", b"   ")["state"] == "failed"
    assert failures("upload") == before + 1
