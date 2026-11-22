"""Pass 20: ingest, backup and ops fixes."""

from __future__ import annotations

import io
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import QdrantClient

from api.repository import VectorRepository
from tests.test_pass18_ingest import _backup
from tests.test_pipeline import StaticEmbeddingProvider, make_embedding, make_settings


class _Each(StaticEmbeddingProvider):
    def embed_texts(self, texts: Sequence[str]) -> list[Any]:
        return [make_embedding(1.0) for _ in texts]


def test_a_restore_never_overwrites_a_document_indexed_while_it_waited() -> None:
    from api.ingestion import IngestionError
    from api.pipeline import IngestService
    from api.routes.backup import _restore

    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    queued: dict[str, Any] = {}

    def enqueue(**kwargs: Any) -> Any:
        queued[kwargs["filename"]] = kwargs
        from api.schemas import DocumentJobAcceptedResponse

        return DocumentJobAcceptedResponse(
            job_id=kwargs["filename"], filename=kwargs["filename"], state="queued"
        )

    queue = SimpleNamespace(enqueue=enqueue, backlog=SimpleNamespace(has_room=lambda size: True))
    _restore(io.BytesIO(_backup(["a.txt"])), repository, settings, queue, None, None)  # type: ignore[arg-type]
    precondition = queued["a.txt"]["precondition"]
    precondition()  # still absent: the restore goes ahead
    IngestService(repository, _Each([]), settings).ingest("a.txt", b"Intro\nnewer upload")
    with pytest.raises(IngestionError, match="newer document was kept"):
        precondition()


def test_a_re_index_that_read_an_original_since_replaced_is_refused(tmp_path: Any) -> None:
    from api.document_ops import original_still_stored
    from api.ingestion import IngestionError
    from api.raw_documents import RawDocumentStore

    store = RawDocumentStore(tmp_path)
    store.save("a.txt", b"old text")
    check = original_still_stored(store, "a.txt", store.identity("a.txt"))
    check()  # unchanged: the re-index goes ahead
    store.save("a.txt", b"the newer upload's text")
    with pytest.raises(IngestionError, match="newer version was kept"):
        check()
    store.remove_current("a.txt")
    with pytest.raises(IngestionError, match="deleted"):
        check()


def test_a_backup_after_the_index_is_lost_keeps_the_originals_and_the_last_full_backup(
    tmp_path: Any,
) -> None:
    import json
    import threading
    import time
    import zipfile

    from api.pipeline import IngestService
    from api.raw_documents import RawDocumentStore
    from api.scheduled_backup import ScheduledBackup, WorkspaceTarget, backups_in

    settings = make_settings(
        raw_document_dir=str(tmp_path / "raw"),
        backup_dir=str(tmp_path / "backups"),
        backup_interval_hours=1,
        backup_keep=1,
    )
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    store = RawDocumentStore(str(tmp_path / "raw"), 0)
    for name in ("a.txt", "b.txt"):
        IngestService(repository, _Each([]), settings).ingest(
            name, b"Intro\nalpha " + name.encode()
        )
        store.save(name, b"Intro\nalpha " + name.encode())
        store.save_details(name, {"tags": ["kept"], "uploaded_at": 5.0})
    clock = [time.time()]
    backup = ScheduledBackup(
        settings,
        repository,
        lambda: [WorkspaceTarget(settings, store)],
        None,
        threading.Lock(),
        clock=lambda: clock[0],
    )
    written = backup.run_once()
    assert written, backup.last
    first = written[0]

    # The index is lost (a wiped volume): the originals are still on disk.
    client.delete_collection(settings.qdrant_collection)
    repository.forget_listing(settings)
    clock[0] += 7200
    second = backup.run_once()[0]
    with zipfile.ZipFile(second) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    entries = {doc["filename"]: doc for doc in manifest["documents"]}
    assert set(entries) == {"a.txt", "b.txt"}
    assert entries["a.txt"]["not_indexed"] is True and entries["a.txt"]["tags"] == ["kept"]
    assert backup.last is not None and backup.last.ok

    # The originals go too: a backup holding nothing is reported, and the last backup
    # that held documents outlives BACKUP_KEEP=1.
    for name in ("a.txt", "b.txt"):
        store.remove_current(name)
    clock[0] += 7200
    third = backup.run_once()[0]
    assert backup.last is not None and not backup.last.ok
    assert "holds 0 documents with their originals" in (backup.last.error or "")
    assert second in backups_in(tmp_path / "backups" / "default")
    assert third.exists() and not first.exists()


def test_a_zip64_record_claiming_more_entries_than_the_end_record_is_caught() -> None:
    import struct
    import zipfile

    from api.zip_safety import central_directory, too_many_entries
    from tests.test_pass18_ingest import _zip

    plain = _zip({"a.txt": b"a", "b.txt": b"b"})
    at = plain.rfind(b"PK\x05\x06")
    directory_size, directory_offset = struct.unpack_from("<LL", plain, at + 12)
    claimed = 5_000_000
    record = struct.pack(
        "<4sQHHLLQQQQ",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        claimed,
        claimed,
        directory_size,
        directory_offset,
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, at, 1)
    # The end record still says 2 entries; zipfile believes the zip64 record.
    crafted = plain[:at] + record + locator + plain[at:]
    # zipfile takes the zip64 record's directory, as the guard now does.
    assert len(zipfile.ZipFile(io.BytesIO(crafted)).infolist()) == 2
    entries, _ = central_directory(io.BytesIO(crafted)) or (0, 0)
    assert entries == claimed
    assert too_many_entries(io.BytesIO(crafted), 1000)
    assert not too_many_entries(io.BytesIO(plain), 1000)


def test_a_migration_bundle_with_too_many_entries_is_refused_before_it_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.routes.migrate as migrate_routes
    from tests.test_pass18_ingest import _zip
    from tests.test_pass18_migrate import _server

    with _server(monkeypatch) as client:
        monkeypatch.setattr(migrate_routes, "BUNDLE_MAX_MEMBERS", 10)
        bundle = _zip({f"f{i}": b"" for i in range(20)})
        response = client.post(
            "/admin/migrate/import", files={"file": ("bundle.zip", bundle, "application/zip")}
        )
        assert response.status_code == 400 and "more than 10 entries" in response.text


def test_an_upload_bigger_than_the_room_where_it_is_received_is_refused_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    import api.request_guard as guard
    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app
    from tests.routes_http.support import make_settings as route_settings

    clear_dependency_caches()
    monkeypatch.setattr(guard, "free_bytes", lambda directory: 2 * 1024 * 1024)
    with TestClient(hermetic_app(route_settings(), QdrantClient(":memory:"))) as client:
        big = client.post("/import", files={"file": ("b.zip", b"x" * (3 * 1024 * 1024))})
        assert big.status_code == 507 and "API_TMP_SIZE" in big.json()["detail"]
        small = client.post("/documents", files={"file": ("a.txt", b"Intro\nalpha")})
        assert small.status_code == 202
    clear_dependency_caches()


def test_an_index_with_no_document_left_is_a_lost_one_however_small_the_corpus() -> None:
    from api.consistency import ConsistencyReport, looks_like_a_lost_index

    lost = ConsistencyReport(
        documents=0, originals_kept=True, originals_without_document=["a", "b"]
    )
    assert looks_like_a_lost_index(lost, 2)
    leftovers = ConsistencyReport(
        documents=3, originals_kept=True, originals_without_document=["a", "b"]
    )
    assert not looks_like_a_lost_index(leftovers, 5)


def _office_files() -> dict[str, bytes]:
    from docx import Document
    from openpyxl import Workbook
    from pptx import Presentation

    workbook = Workbook()
    shown = workbook.active
    shown.title = "Rates"
    shown.append(["Item", "Price"])
    shown.append(["Tea", 2])
    secret = workbook.create_sheet("Salaries")
    secret.append(["Name", "Salary"])
    secret.append(["Ann", 99000])
    secret.sheet_state = "hidden"
    xlsx = io.BytesIO()
    workbook.save(xlsx)

    document = Document()
    paragraph = document.add_paragraph("Visible policy text. ")
    paragraph.add_run("Secret bonus pool.").font.hidden = True
    docx = io.BytesIO()
    document.save(docx)

    deck = Presentation()
    for title in ("Roadmap", "Layoffs plan"):
        slide = deck.slides.add_slide(deck.slide_layouts[5])
        slide.shapes.title.text = title
    deck.slides[1]._element.set("show", "0")  # noqa: SLF001
    pptx = io.BytesIO()
    deck.save(pptx)
    return {"r.xlsx": xlsx.getvalue(), "p.docx": docx.getvalue(), "d.pptx": pptx.getvalue()}


@pytest.mark.parametrize(
    ("name", "shown", "secret"),
    [
        ("r.xlsx", "Tea", "Ann"),
        ("p.docx", "Visible policy", "bonus pool"),
        ("d.pptx", "Roadmap", "Layoffs"),
    ],
)
def test_hidden_office_content_is_left_out_unless_asked_for(
    name: str, shown: str, secret: str
) -> None:
    from api.documents import parse_document_bytes
    from tests.factories import make_test_settings

    content = _office_files()[name]

    def text(settings: Any) -> str:
        sections = parse_document_bytes(name, content, settings)
        return " ".join(f"{s.section} {s.text}" for s in sections)

    default = text(make_test_settings())
    assert shown in default and secret not in default
    assert secret in text(make_test_settings(index_hidden_content=True))


def test_temp_files_left_anywhere_in_the_originals_store_are_swept(tmp_path: Any) -> None:
    import os
    import time

    from api.raw_documents import RawDocumentStore

    store = RawDocumentStore(str(tmp_path))
    store.save("a.txt", b"a")
    stale = [
        tmp_path / ".versions" / "a.txt" / ".restore-abcd1234",
        tmp_path / ".meta" / "docs" / ".upload-abcd1234",
        tmp_path / ".upload-zyxw9876",
    ]
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial")
        os.utime(path, (time.time() - 7200, time.time() - 7200))
    kept = tmp_path / "docs" / ".restore-notes.txt"  # an original's own name
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(b"x")
    os.utime(kept, (time.time() - 7200, time.time() - 7200))

    RawDocumentStore(str(tmp_path))
    assert not any(path.exists() for path in stale)
    assert kept.exists() and (tmp_path / "a.txt").exists()


def test_a_second_archive_download_waits_for_the_first(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    import api.routes.documents_write as documents_write
    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app
    from tests.routes_http.support import make_settings as route_settings

    clear_dependency_caches()
    app = hermetic_app(route_settings(url_ingest_enabled=True), QdrantClient(":memory:"))
    with TestClient(app) as client:
        assert documents_write._archive_fetches.acquire(blocking=False)  # noqa: SLF001
        try:
            busy = client.post(
                "/documents/archive-from-url", json={"url": "https://example.com/a.zip"}
            )
        finally:
            documents_write._archive_fetches.release()  # noqa: SLF001
        assert busy.status_code == 503 and busy.headers["Retry-After"] == "30"
    clear_dependency_caches()


def test_a_re_index_applied_first_never_makes_an_earlier_admitted_upload_fail() -> None:
    from api.ingest_jobs import IngestRequest, run_ingest_job
    from api.ingest_service import DocumentIngestOutcome
    from api.ingestion import ingest_sequencer
    from api.jobs import IngestJobStore

    settings = make_settings()

    class _Service:
        def __init__(self) -> None:
            self.settings = settings
            self.repository = None

        def ingest(
            self, filename: str, content: bytes, on_progress: Any, on_indexed: Any, **kwargs: Any
        ) -> Any:
            kwargs["precondition"]()
            on_indexed()
            return DocumentIngestOutcome(
                filename=filename, sections_parsed=1, chunks_ingested=1, collection_name="c"
            )

    store = IngestJobStore(max_retained=10)
    name = "race-pass20.txt"
    upload_seq = ingest_sequencer.admit(name)
    reindex_seq = ingest_sequencer.admit(name)
    for request, sequence in (
        (IngestRequest(name, b"old", kind="reindex", ordered=False), reindex_seq),
        (IngestRequest(name, b"new", kind="upload"), upload_seq),
    ):
        job = store.create(name, "")
        run_ingest_job(store, job.id, _Service(), request, sequence)  # type: ignore[arg-type]
        done = store.get(job.id)
        assert done is not None and done.state == "done", done


def test_the_settings_check_says_when_the_temp_directory_is_too_small_for_an_import() -> None:
    from types import SimpleNamespace

    from api.doctor import diagnose
    from tests.factories import make_test_settings

    settings = make_test_settings(max_migrate_bytes=4 * 1024**3)
    small = diagnose(
        settings,
        disk_usage=lambda path: SimpleNamespace(free=2 * 1024**3, total=4 * 1024**3, used=0),
    )
    notes = [f for f in small if f.subject == "TMPDIR"]
    assert notes and "two copies" in notes[0].message and notes[0].level == "note"
    roomy = diagnose(
        settings,
        disk_usage=lambda path: SimpleNamespace(free=100 * 1024**3, total=200 * 1024**3, used=0),
    )
    assert not [f for f in roomy if f.subject == "TMPDIR"]


def test_a_migration_bundle_can_be_written_in_the_background_and_downloaded_once(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import zipfile

    from tests.routes_http.support import upload_and_wait, wait_until
    from tests.test_pass18_migrate import _server

    with _server(monkeypatch, raw_document_dir=str(tmp_path / "raw")) as client:
        upload_and_wait(client, "a.txt", b"Intro\nalpha")
        started = client.post("/admin/migrate/export")
        assert started.status_code == 202, started.text
        import_id = started.json()["import_id"]
        wait_until(lambda: client.get(f"/documents/imports/{import_id}").json()["finished_at"])
        status = client.get(f"/documents/imports/{import_id}").json()
        assert (status["kind"], status["state"]) == ("migrate-export", "done"), status
        assert any("ready to download" in note for note in status["problems"])
        bundle = client.get(f"/admin/migrate/export/{import_id}")
        assert bundle.status_code == 200
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            names = [w["name"] for w in json.loads(archive.read("manifest.json"))["workspaces"]]
        assert names == ["default"]
        assert client.get(f"/admin/migrate/export/{import_id}").status_code == 404


def test_docrag_migrate_pack_background_downloads_the_finished_bundle(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile

    from cli.__main__ import main
    from cli.client import DocragClient
    from tests.routes_http.support import upload_and_wait
    from tests.test_pass18_migrate import _server

    with _server(monkeypatch, raw_document_dir=str(tmp_path / "raw")) as client:
        upload_and_wait(client, "a.txt", b"Intro\nalpha")
        target = tmp_path / "server.zip"
        out = io.StringIO()
        code = main(
            ["migrate", "pack", str(target), "--background"],
            client=DocragClient("http://testserver", None, http=client),
            out=out,
        )
        assert code == 0, out.getvalue()
        assert "manifest.json" in zipfile.ZipFile(target).namelist()


def test_a_migration_bundle_is_not_copied_again_without_a_spool_directory(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.migrate as migrate_routes
    from tests.routes_http.support import wait_until
    from tests.test_pass18_migrate import _packed, _server

    bundle = _packed(tmp_path, monkeypatch)
    with _server(monkeypatch, raw_document_dir=str(tmp_path / "new")) as client:

        def no_copy(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("copied the upload a second time")

        monkeypatch.setattr(migrate_routes, "spool_upload", no_copy)
        started = client.post("/admin/migrate/import", files={"file": ("b.zip", bundle)})
        assert started.status_code == 202, started.text
        import_id = started.json()["import_id"]
        wait_until(lambda: client.get(f"/documents/imports/{import_id}").json()["finished_at"])
        assert client.get(f"/documents/imports/{import_id}").json()["state"] == "done"
        wait_until(lambda: "a.txt" in client.get("/documents").json()["filenames"])


def test_a_backup_inside_a_migration_bundle_takes_the_bundles_own_entry_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.restore as restore
    from api.restore import BackupImportError, restore_backup
    from tests.test_pass18_ingest import _backup

    monkeypatch.setattr(restore, "MAX_MEMBERS", 3)
    backup = _backup([f"d{i}.txt" for i in range(5)])
    queued: list[str] = []

    def run(max_members: int | None) -> None:
        restore_backup(
            io.BytesIO(backup),
            existing_filenames=set(),
            max_original_bytes=1 << 20,
            has_room=lambda size: True,
            enqueue=lambda name, *a, **k: queued.append(name) or {"job_id": name},  # type: ignore[func-returns-value]
            feedback_store=None,
            max_members=max_members,
        )

    with pytest.raises(BackupImportError, match="more than 3 entries"):
        run(None)  # an upload's cap
    run(100)  # a migration bundle's (MAX_MIGRATE_MEMBERS)
    assert len(queued) == 5


def test_a_document_uploaded_while_a_background_restore_waits_is_kept(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import api.routes.backup as backup_routes
    from tests.routes_http.support import upload_and_wait, wait_until
    from tests.test_pass18_migrate import _server

    room = threading.Event()
    waiting = threading.Event()

    def held_until_released(has_room: Any, size: int, status: Any) -> bool:
        waiting.set()
        return room.wait(10)

    with _server(monkeypatch, raw_document_dir=str(tmp_path / "raw")) as client:
        monkeypatch.setattr(backup_routes, "wait_for_room", held_until_released)
        started = client.post(
            "/import",
            data={"background": "true"},
            files={"file": ("b.zip", _backup(["a.txt"]))},
        )
        assert started.status_code in (200, 202), started.text
        assert waiting.wait(10)
        # Uploaded while the restore waits for room: newer than the backup's copy.
        upload_and_wait(client, "a.txt", b"Intro\nthe newer upload")
        room.set()
        import_id = started.json()["import_id"]
        wait_until(lambda: client.get(f"/documents/imports/{import_id}").json()["finished_at"])

        def failed_jobs() -> list[Any]:
            return client.get("/documents/jobs", params={"state": "failed"}).json().get("jobs", [])

        wait_until(lambda: any(j["filename"] == "a.txt" for j in failed_jobs()))
        text = " ".join(c["text"] for c in client.get("/documents/a.txt/content").json()["chunks"])
        assert "newer upload" in text


def test_the_settings_check_warns_when_the_temp_path_is_too_long_for_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile

    from api.doctor import diagnose
    from tests.factories import make_test_settings

    settings = make_test_settings(parse_timeout_seconds=60)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/private/tmp/" + "x" * 100)
    warned = [f for f in diagnose(settings) if f.subject == "TMPDIR" and f.level == "warning"]
    assert warned and "AF_UNIX" in warned[0].message
    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/tmp")
    assert not [f for f in diagnose(settings) if f.level == "warning" and f.subject == "TMPDIR"]
