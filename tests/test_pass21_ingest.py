"""Pass 21: ingest, backup and ops fixes."""

from __future__ import annotations

import io
import struct
import zipfile
from typing import Any

import pytest


def test_a_zip64_locator_pointing_at_junk_does_not_hide_the_record_zipfile_reads() -> None:
    # Python 3.12's zipfile reads the zip64 record right before its locator, whatever
    # offset the locator gives: a guard that read only at that offset kept the end
    # record's small count while zipfile built every entry the record claimed.
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
    # The locator's offset points at the first local header, not at the record.
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, 0, 1)
    crafted = plain[:at] + record + locator + plain[at:]
    zipfile.ZipFile(io.BytesIO(crafted))  # zipfile accepts it (and believes the record)
    entries, _ = central_directory(io.BytesIO(crafted)) or (0, 0)
    assert entries == claimed
    assert too_many_entries(io.BytesIO(crafted), 1000)
    assert not too_many_entries(io.BytesIO(plain), 1000)


def test_docrag_migrate_pack_background_works_from_another_workspace(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cli.__main__ import main
    from cli.client import DocragClient
    from tests.routes_http.support import upload_and_wait
    from tests.test_pass18_migrate import _server

    with _server(
        monkeypatch, raw_document_dir=str(tmp_path / "raw"), workspaces_enabled=True
    ) as client:
        upload_and_wait(client, "a.txt", b"Intro\nalpha")
        assert client.post("/workspaces", json={"name": "team"}).status_code == 201
        target = tmp_path / "server.zip"
        out = io.StringIO()
        code = main(
            ["migrate", "pack", str(target), "--background"],
            client=DocragClient("http://testserver", None, http=client, workspace="team"),
            out=out,
        )
        assert code == 0, out.getvalue()
        assert "manifest.json" in zipfile.ZipFile(target).namelist()


def test_a_background_export_survives_a_cut_download_and_only_the_newest_is_kept(
    tmp_path: Any,
) -> None:
    import asyncio

    import api.routes.migrate as migrate_routes

    first = tmp_path / "first.zip"
    first.write_bytes(b"x" * 300_000)
    migrate_routes._keep_export("one", first)
    response = migrate_routes._KeptUntilSent("one", first, "bundle.zip", None)
    scope = {"type": "http", "method": "GET", "headers": [], "path": "/"}

    async def receive() -> Any:
        await asyncio.sleep(3600)

    async def cut(message: Any) -> None:
        if message["type"] == "http.response.body" and message.get("more_body"):
            raise OSError("connection reset")

    with pytest.raises(OSError):
        asyncio.run(response(scope, receive, cut))
    assert first.exists() and "one" in migrate_routes._background_exports

    sent: list[Any] = []

    async def whole(message: Any) -> None:
        sent.append(message)

    asyncio.run(
        migrate_routes._KeptUntilSent("one", first, "bundle.zip", None)(scope, receive, whole)
    )
    assert not first.exists() and "one" not in migrate_routes._background_exports

    older, newer = tmp_path / "older.zip", tmp_path / "newer.zip"
    older.write_bytes(b"a")
    newer.write_bytes(b"b")
    migrate_routes._keep_export("older", older)
    migrate_routes._keep_export("newer", newer)
    assert not older.exists() and newer.exists()
    assert list(migrate_routes._background_exports) == ["newer"]
    migrate_routes._forget_export("newer")


def test_an_import_waiting_behind_another_says_so_and_can_be_cancelled() -> None:
    import threading

    from api import imports
    from tests.routes_http.support import wait_until

    release = threading.Event()
    ran: list[str] = []
    first = imports.start("archive", "a.zip", "", "c", lambda s: release.wait(10))
    second = imports.start("archive", "b.zip", "", "c", lambda s: ran.append("b"))
    wait_until(lambda: first.state == "running")
    assert second.state == "waiting"
    second.cancel.set()
    release.set()
    wait_until(lambda: second.finished_at is not None)
    assert ran == [] and second.state == "cancelled"


def test_a_background_import_keeps_the_received_file_instead_of_copying_it(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.backup as backup_routes
    from tests.routes_http.support import wait_until
    from tests.test_pass18_ingest import _backup
    from tests.test_pass18_migrate import _server

    with _server(monkeypatch, raw_document_dir=str(tmp_path / "raw")) as client:

        def no_copy(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("copied the upload a second time")

        monkeypatch.setattr(backup_routes, "spool_upload", no_copy)
        started = client.post(
            "/import",
            files={"file": ("b.zip", _backup(["a.txt", "b.txt"]))},
            data={"background": "true"},
        )
        assert started.status_code == 202, started.text
        import_id = started.json()["import_id"]
        wait_until(lambda: client.get(f"/documents/imports/{import_id}").json()["finished_at"])
        status = client.get(f"/documents/imports/{import_id}").json()
        assert status["state"] == "done" and status["queued"] == 2, status


def test_a_zip_at_a_url_is_accepted_while_the_indexing_queue_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import functools

    import httpx
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches
    from api.url_fetch import fetch_document
    from tests.routes_http.support import hermetic_app, wait_until
    from tests.routes_http.support import make_settings as route_settings
    from tests.test_url_fetch import resolver, transport

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", b"Intro\nfirst")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=buffer.getvalue(), headers={"content-type": "application/zip"}
        )

    monkeypatch.setattr(
        "api.routes.documents_write.fetch_document",
        functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler)),
    )
    clear_dependency_caches()
    app = hermetic_app(route_settings(url_ingest_enabled=True), QdrantClient(":memory:"))
    with TestClient(app) as client:
        with monkeypatch.context() as full:
            full.setattr("api.jobs.IngestBacklog.has_room", lambda self, size: False)
            started = client.post(
                "/documents/archive-from-url", json={"url": "https://x.example/d.zip"}
            )
        assert started.status_code == 202, started.text
        import_id = started.json()["import_id"]
        wait_until(lambda: client.get(f"/documents/imports/{import_id}").json()["finished_at"])
    clear_dependency_caches()
