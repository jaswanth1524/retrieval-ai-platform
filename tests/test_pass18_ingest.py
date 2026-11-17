"""Pass 18: ingest, backup and ops fixes found in review."""

from __future__ import annotations

import functools
import io
import json
import os
import zipfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.url_fetch import fetch_document
from tests.factories import make_test_settings
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait, wait_until
from tests.test_url_fetch import resolver, transport


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _backup(names: list[str]) -> bytes:
    manifest = {
        "manifest_version": 1,
        "documents": [{"filename": n, "original": f"originals/{n}", "tags": []} for n in names],
    }
    files = {"manifest.json": json.dumps(manifest).encode()}
    files.update({f"originals/{n}": f"Intro\ntext of {n}".encode() for n in names})
    return _zip(files)


def test_a_zips_directory_is_sized_before_it_is_read() -> None:
    from api.zip_safety import central_directory, too_many_entries

    small = io.BytesIO(_zip({f"f{i}.txt": b"" for i in range(40)}))
    entries, size = central_directory(small) or (0, 0)
    assert entries == 40 and size > 40 * 46 and small.tell() == 0
    assert not too_many_entries(small, 40) and too_many_entries(small, 39)
    # A count that lies is caught by the directory's size.
    lying = bytearray(small.getvalue())
    at = lying.rfind(b"PK\x05\x06")
    lying[at + 8 : at + 12] = (1).to_bytes(2, "little") * 2
    assert too_many_entries(io.BytesIO(bytes(lying)), 5, name_bytes=8)
    assert central_directory(io.BytesIO(b"not a zip")) is None


def test_a_backup_this_server_wrote_restores_past_the_upload_entry_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.restore as restore

    monkeypatch.setattr(restore, "MAX_MEMBERS", 3)
    backup = _backup([f"d{i}.txt" for i in range(6)])

    uploaded = restore.verify_backup(io.BytesIO(backup), 1 << 20)
    assert not uploaded.ok and "more than 3 entries" in uploaded.problems[0]
    stored = restore.verify_backup(io.BytesIO(backup), 1 << 20, stored=True)
    assert stored.ok and stored.documents == 6 and stored.would_restore == 6

    queued: list[str] = []

    def enqueue(name: str, *args: object, **kwargs: object) -> object:
        queued.append(name)
        return {"job_id": name}

    with pytest.raises(restore.BackupImportError):
        restore.restore_backup(
            io.BytesIO(backup),
            existing_filenames=set(),
            max_original_bytes=1 << 20,
            has_room=lambda size: True,
            enqueue=enqueue,  # type: ignore[arg-type]
            feedback_store=None,
        )
    restore.restore_backup(
        io.BytesIO(backup),
        existing_filenames=set(),
        max_original_bytes=1 << 20,
        has_room=lambda size: True,
        enqueue=enqueue,  # type: ignore[arg-type]
        feedback_store=None,
        stored=True,
    )
    assert len(queued) == 6


def test_a_cancelled_restore_queues_nothing_more_and_leaves_the_feedback() -> None:
    import api.restore as restore

    queued: list[str] = []

    def enqueue(name: str, *args: object, **kwargs: object) -> object:
        queued.append(name)
        return {"job_id": name}

    outcome = restore.restore_backup(
        io.BytesIO(_backup(["a.txt", "b.txt", "c.txt"])),
        existing_filenames=set(),
        max_original_bytes=1 << 20,
        has_room=lambda size: True,
        enqueue=enqueue,  # type: ignore[arg-type]
        feedback_store=None,
        stop=lambda: len(queued) >= 1,
    )
    assert queued == ["a.txt"] and outcome.deferred == []


def test_an_archive_listing_far_more_entries_than_documents_is_refused_unread() -> None:
    clear_dependency_caches()
    app = hermetic_app(make_settings(max_archive_members=2), QdrantClient(":memory:"))
    with TestClient(app) as client:
        huge = _zip({f"__MACOSX/f{i}": b"" for i in range(60)})
        refused = client.post(
            "/documents/archive", files={"file": ("huge.zip", huge, "application/zip")}
        )
        assert refused.status_code == 400 and "MAX_ARCHIVE_MEMBERS" in refused.json()["detail"]
        # A Finder zip (an entry of metadata per file) is still taken.
        finder = _zip({"a.txt": b"Intro\nalpha", "__MACOSX/._a.txt": b"x", "docs/": b""})
        taken = client.post(
            "/documents/archive", files={"file": ("ok.zip", finder, "application/zip")}
        )
        assert taken.status_code == 202 and len(taken.json()["jobs"]) == 1
    clear_dependency_caches()


def test_an_original_dated_before_1980_is_exported(tmp_path: Path) -> None:
    from api.export import _write_member

    old = tmp_path / "old.txt"
    old.write_bytes(b"from 1970")
    os.utime(old, (0, 0))
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive, old.open("rb") as source:
        _write_member(archive, "originals/old.txt", source)
    with zipfile.ZipFile(io.BytesIO(target.getvalue())) as archive:
        assert archive.read("originals/old.txt") == b"from 1970"
        assert archive.getinfo("originals/old.txt").date_time[0] == 1980


def test_caption_lines_starting_with_note_or_style_are_speech() -> None:
    from api.parsers.subtitles import parse_subtitles_document

    captions = (
        "WEBVTT\n\nNOTE this block is a comment\nstill the comment\n\n"
        "00:00:01.000 --> 00:00:04.000\nNOTE THAT THE DEADLINE IS FRIDAY\n\n"
        "00:00:05.000 --> 00:00:08.000\nSTYLE IS EVERYTHING\nREGION ONE REPORTS\n"
    )
    (section,) = parse_subtitles_document("talk.vtt", captions)
    assert section.text == (
        "NOTE THAT THE DEADLINE IS FRIDAY STYLE IS EVERYTHING REGION ONE REPORTS"
    )
    assert "comment" not in section.text


def test_rtf_emoji_and_binary_pictures_are_read_and_lone_surrogates_replaced() -> None:
    from api.chunking import HeuristicTokenCounter
    from api.documents import chunk_sections, parse_document_bytes
    from api.parsers.common import clean_text
    from api.parsers.rtf import rtf_to_text

    backslash = chr(92)
    emoji = f"{backslash}u-10179?{backslash}u-8704?"  # U+1F600 as its surrogate pair
    assert rtf_to_text("{" + backslash + "rtf1 Hello " + emoji + " world}") == "Hello 😀 world"
    # Half a pair alone is not a character.
    assert rtf_to_text("{" + backslash + "rtf1 A " + backslash + "u-10179? B}") == "A � B"
    # Raw bytes after \binN: a brace among them closes nothing.
    picture = "{" + backslash + "pict" + backslash + "bin4 a}b!}"
    text = rtf_to_text("{" + backslash + "rtf1 before " + picture + " after}")
    assert "before" in text and "after" in text

    lone = "ok " + chr(0xD83D) + " end"
    assert clean_text(lone) == "ok � end"
    settings = make_test_settings()
    source = (
        "{" + backslash + "rtf1 Intro" + backslash + "par Smile " + emoji + " today}"
    ).encode()
    sections = parse_document_bytes("memo.rtf", source, settings)
    counter = HeuristicTokenCounter()
    assert chunk_sections(sections, settings, counter)  # used to fail hashing the chunk id
    notes = json.dumps({"note": "half " + chr(0xD83D)}).encode()
    assert chunk_sections(parse_document_bytes("n.json", notes, settings), settings, counter)


def report_a_parser_fault(connection: object, *args: object) -> None:
    connection.send(("parser_fault", "IndexError: list index out of range"))  # type: ignore[attr-defined]


def test_a_parser_breaking_on_a_files_content_is_the_documents_fault() -> None:
    from api.documents import DocumentParseError
    from api.ingest_jobs import ingest_error_type
    from api.parse_isolation import ParseEnvironmentError, parse_with_limit

    with pytest.raises(DocumentParseError) as raised:
        parse_with_limit(
            "odd.pdf",
            b"%PDF",
            make_test_settings(parse_timeout_seconds=60),
            _target=report_a_parser_fault,
        )
    assert not isinstance(raised.value, ParseEnvironmentError)
    # Permanent: the watcher and the re-indexer give it up instead of retrying for ever.
    assert ingest_error_type(raised.value) == "document"


def test_replace_overrides_a_name_the_remote_server_chose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_dependency_caches()
    app = hermetic_app(make_settings(url_ingest_enabled=True), QdrantClient(":memory:"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"Intro\nFrom the web.",
            headers={
                "content-type": "text/plain",
                "content-disposition": 'attachment; filename="handbook.txt"',
            },
        )

    monkeypatch.setattr(
        "api.routes.documents_write.fetch_document",
        functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler)),
    )
    with TestClient(app) as client:
        upload_and_wait(client, "handbook.txt", b"Intro\nUploaded by hand.")
        url = {"url": "https://other.example/download?id=7"}
        assert client.post("/documents/from-url", json=url).status_code == 409
        replaced = client.post("/documents/from-url", json={**url, "replace": True})
        assert replaced.status_code == 202
        job = replaced.json()["job_id"]
        wait_until(lambda: client.get(f"/documents/jobs/{job}").json()["state"] == "done")
        # The same URL again: its own document, updated in place.
        assert client.post("/documents/from-url", json=url).status_code == 202
    clear_dependency_caches()


def test_a_background_restore_the_index_refuses_leaves_no_spooled_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    import api.imports as imports

    monkeypatch.setattr(imports.tempfile, "tempdir", str(tmp_path))
    clear_dependency_caches()
    app = hermetic_app(make_settings(), QdrantClient(":memory:"))

    def refuse(self: object, settings: object, **kwargs: object) -> None:
        raise HTTPException(status_code=409, detail="built for another embedding setup")

    with TestClient(app) as client:
        monkeypatch.setattr("api.repository.VectorRepository.ensure_writable", refuse)
        refused = client.post(
            "/import",
            files={"file": ("backup.zip", _backup(["a.txt"]), "application/zip")},
            data={"background": "true"},
        )
        assert refused.status_code == 409
    assert list(tmp_path.glob("docrag-restore-*")) == []
    clear_dependency_caches()


def test_an_import_that_ends_on_an_error_says_failed_and_one_never_run_cleans_up() -> None:
    import api.imports as imports

    def broken(status: imports.BackgroundImport) -> None:
        raise RuntimeError("the index can't be written")

    cleaned: list[str] = []
    failed = imports.start("restore", "b.zip", "", "c", broken, lambda: cleaned.append("ran"))
    wait_until(lambda: failed.finished_at is not None)
    assert failed.state == "failed" and "can't be written" in failed.problems[0]
    assert cleaned == ["ran"]

    import threading

    gate = threading.Event()
    first = imports.start("archive", "1.zip", "", "c", lambda status: gate.wait(5))
    queued = imports.start(
        "archive", "2.zip", "", "c", lambda status: None, lambda: cleaned.append("queued")
    )
    imports.shutdown()
    gate.set()
    # Never started: ended at shutdown, its spooled file removed.
    assert queued.finished_at is not None and "queued" in cleaned
    wait_until(lambda: first.finished_at is not None)


def test_a_deleted_workspaces_backups_are_set_aside(tmp_path: Path) -> None:
    import threading

    from api.repository import VectorRepository
    from api.scheduled_backup import DELETED_DIRECTORY, ScheduledBackup

    root = tmp_path / "backups"
    (root / "team").mkdir(parents=True)
    (root / "team" / "docrag-20260101-000000.zip").write_bytes(b"old workspace's backup")
    settings = make_settings(backup_dir=str(root), raw_document_dir=str(tmp_path / "raw"))
    backup = ScheduledBackup(
        settings, VectorRepository(QdrantClient(":memory:")), lambda: [], None, threading.Lock()
    )
    backup.forget_workspace("team")
    assert not (root / "team").exists()
    (kept,) = (root / DELETED_DIRECTORY).iterdir()
    assert kept.name.startswith("team-")
    assert [p.name for p in kept.iterdir()] == ["docrag-20260101-000000.zip"]


def test_repair_will_not_delete_the_originals_of_a_lost_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_dependency_caches()
    raw = tmp_path / "raw"
    qdrant = QdrantClient(":memory:")
    settings = make_settings(raw_document_dir=str(raw))
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    app = hermetic_app(settings, qdrant)
    try:
        _lost_index_repair(app, qdrant, raw)
    finally:
        monkeypatch.undo()
        clear_dependency_caches()


def _lost_index_repair(app: object, qdrant: QdrantClient, raw: Path) -> None:
    with TestClient(app) as client:  # type: ignore[arg-type]
        for index in range(6):
            upload_and_wait(client, f"d{index}.txt", f"Intro\ndocument {index}".encode())
        # The index is lost; the originals are all that is left.
        for collection in qdrant.get_collections().collections:
            qdrant.delete_collection(collection.name)
        from api.repository import clear_metadata_cache

        clear_metadata_cache()
        report = client.get("/admin/consistency").json()
        assert len(report["originals_without_document"]) == 6
        refused = client.post("/admin/consistency/repair", json={"delete_orphan_originals": True})
        assert refused.status_code == 409 and "index looks lost" in refused.json()["detail"]
        assert len(list(raw.glob("d*.txt"))) == 6
        forced = client.post(
            "/admin/consistency/repair",
            json={"delete_orphan_originals": True, "delete_despite_lost_index": True},
        )
        assert forced.status_code == 200 and len(forced.json()["originals_deleted"]) == 6


def test_the_settings_check_does_not_warn_on_the_stock_container() -> None:
    from api.doctor import diagnose

    # The compose file's limit (4g) with every default.
    found = diagnose(
        make_test_settings(api_key="k", rate_limit_per_minute=60, parse_memory_limit_mb=1500),
        memory_limit=lambda: 4 * 1024**3,
        environ={},
        cpu_count=4,
    )
    assert [f for f in found if f.level == "warning"] == []
    # A backlog the limit can't hold is still called out.
    tight = diagnose(
        make_test_settings(ingest_max_pending_bytes=3 * 1024**3),
        memory_limit=lambda: 4 * 1024**3,
        environ={},
        cpu_count=4,
    )
    assert ("warning", "INGEST_MAX_PENDING_BYTES") in {(f.level, f.subject) for f in tight}


def test_a_sitemap_index_cannot_send_the_server_to_other_hosts() -> None:
    from api.sitemap import read_sitemap
    from api.url_fetch import FetchedDocument

    index = (
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>https://site.example/pages.xml</loc></sitemap>"
        b"<sitemap><loc>https://internal.example/secret.xml</loc></sitemap>"
        b"</sitemapindex>"
    )
    pages = (
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>https://site.example/a.html</loc></url></urlset>"
    )
    fetched: list[str] = []

    def fetch(url: str) -> FetchedDocument:
        fetched.append(url)
        content = index if url.endswith("sitemap.xml") else pages
        return FetchedDocument("s.xml", content, url)

    found = read_sitemap("https://site.example/sitemap.xml", fetch, same_host=True, limit=10)
    assert found.urls == ["https://site.example/a.html"]
    assert "https://internal.example/secret.xml" not in fetched


def test_a_background_import_leaves_room_for_interactive_work() -> None:
    from api.jobs import IngestBacklog

    backlog = IngestBacklog(max_bytes=1000, max_jobs=4)
    assert backlog.has_spare_room(900)  # an empty queue takes anything
    for _ in range(3):
        assert backlog.try_reserve(100)
    # Three of four places taken: an upload still fits, a background import waits.
    assert backlog.has_room(100) and not backlog.has_spare_room(100)
    backlog.release(100)
    assert backlog.has_spare_room(100) and not backlog.has_spare_room(600)


def test_the_settings_check_says_when_ollama_lacks_the_model() -> None:
    from api.doctor import diagnose

    settings = make_test_settings(llm_provider="ollama", llm_model="qwen2.5")

    def subjects(pulled: list[str] | None) -> set[str]:
        found = diagnose(
            settings,
            memory_limit=lambda: None,
            environ={},
            cpu_count=4,
            ollama_models=lambda _: pulled,
        )
        return {f.subject for f in found if f.level == "warning"}

    assert "LLM_MODEL" in subjects(["llama3.1:8b"])
    assert "LLM_MODEL" not in subjects(["qwen2.5:latest"])  # no tag means :latest
    assert "LLM_MODEL" not in subjects(None)  # unreachable: readiness says that


def test_a_backup_pass_that_cannot_list_workspaces_is_recorded_as_failed(tmp_path: Path) -> None:
    import threading

    from api.repository import VectorRepository
    from api.scheduled_backup import ScheduledBackup

    def unreachable() -> list:  # type: ignore[type-arg]
        raise ConnectionError("Qdrant is down")

    settings = make_settings(backup_dir=str(tmp_path / "b"), raw_document_dir=str(tmp_path / "r"))
    backup = ScheduledBackup(
        settings, VectorRepository(QdrantClient(":memory:")), unreachable, None, threading.Lock()
    )
    assert backup.run_once() == []
    assert backup.last is not None and not backup.last.ok and backup.failing() == ["default"]


def test_a_listing_past_the_protocol_limit_is_read_as_a_stream_and_capped(monkeypatch):
    from api import sitemap

    monkeypatch.setattr(sitemap, "MAX_LISTED_ENTRIES", 3)
    xml = (
        b"<urlset>" + b"".join(b"<url><loc>/p%d</loc></url>" % n for n in range(10)) + b"</urlset>"
    )
    pages, children = sitemap.listed_urls(xml, "https://example.com/")
    assert pages == ["https://example.com/p0", "https://example.com/p1", "https://example.com/p2"]
    assert children == []
    # A <url> nested inside another element isn't a top-level entry.
    nested = b"<urlset><x><url><loc>/deep</loc></url></x><url><loc>/top</loc></url></urlset>"
    assert sitemap.listed_urls(nested, "https://example.com/")[0] == ["https://example.com/top"]
