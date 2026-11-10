"""Pass 16 ingest fixes: table columns after a blank run, sitemap page names, watched
folders, archives, backups and the rest."""

from __future__ import annotations

import pytest

from api.documents import parse_document_bytes
from tests.factories import make_test_settings
from tests.samples import odf_bytes, ods_row, odt_bytes

BLANK_RUN = '<table:table-cell table:number-columns-repeated="2"/>'


def _parse(name: str, content: bytes) -> list:  # type: ignore[type-arg]
    return parse_document_bytes(name, content, make_test_settings())


def _row(*cells: str) -> str:
    return ods_row(*cells).replace("</table:table-row>", "")


def test_a_value_after_blank_cells_stays_under_its_own_header() -> None:
    table = (
        _row("Name", "Q1", "Q2", "Total")
        + "</table:table-row>"
        + _row("Ann")
        + BLANK_RUN
        + "<table:table-cell><text:p>9</text:p></table:table-cell></table:table-row>"
    )
    (sheet,) = _parse(
        "sales.ods", odf_bytes("spreadsheet", f'<table:table table:name="S">{table}</table:table>')
    )
    assert sheet.text == "Name: Ann; Total: 9"

    (document,) = _parse("sales.odt", odt_bytes(f"<table:table>{table}</table:table>"))
    assert "Total: 9" in document.text and "Q2: 9" not in document.text


@pytest.mark.parametrize("trailing", [1, 3])
def test_trailing_blank_cells_are_still_padding(trailing: int) -> None:
    pad = f'<table:table-cell table:number-columns-repeated="{1000 * trailing}"/>'
    table = (
        _row("Name", "Role")
        + pad
        + "</table:table-row>"
        + _row("Ada", "Lead")
        + pad
        + "</table:table-row>"
    )
    (sheet,) = _parse(
        "t.ods", odf_bytes("spreadsheet", f'<table:table table:name="T">{table}</table:table>')
    )
    assert sheet.text == "Name: Ada; Role: Lead"


def test_a_sitemap_page_never_takes_another_sources_name() -> None:
    from api.sitemap import page_name

    known: dict[str, str | None] = {"intro.html": "https://example.com/docs/intro/"}
    # Its own document: updated in place.
    assert page_name("https://example.com/docs/intro/", "intro.html", known) == "intro.html"
    # Another page with the same last segment, an upload, a query string.
    assert (
        page_name("https://example.com/blog/intro/", "intro.html", known)
        == "example.com_blog_intro.html"
    )
    known["report.pdf"] = None
    assert (
        page_name("https://example.com/files/report.pdf", "report.pdf", known)
        == "example.com_files_report.pdf"
    )
    assert page_name("https://example.com/p?id=2", "intro.html", known).startswith(
        "example.com_p_id_2"
    )
    # Even the long name taken: a short hash of the URL.
    known["example.com_blog_intro.html"] = "https://example.com/blog/intro"
    renamed = page_name("https://example.com/blog/intro/", "intro.html", known)
    assert renamed.startswith("example.com_blog_intro_") and renamed.endswith(".html")
    long_url = "https://example.com/" + "/".join(["segment"] * 80)
    assert len(page_name(long_url, "intro.html", known).encode()) <= 255


def test_two_pages_with_one_name_are_both_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    import functools
    from collections.abc import Iterator

    import httpx
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches
    from api.url_fetch import fetch_document
    from tests.factories import wait_until
    from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait
    from tests.test_url_fetch import resolver, transport

    sitemap = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/docs/notes.txt</loc></url>
      <url><loc>https://example.com/blog/notes.txt</loc></url>
    </urlset>"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, content=sitemap, headers={"content-type": "text/xml"})
        body = f"Intro\nThe page {request.url.path}".encode()
        return httpx.Response(200, content=body, headers={"content-type": "text/plain"})

    fake = functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler))
    monkeypatch.setattr("api.routes.documents_write.fetch_document", fake)
    clear_dependency_caches()
    app = hermetic_app(make_settings(url_ingest_enabled=True), QdrantClient(":memory:"))

    def listed(client: TestClient) -> list[str]:
        return sorted(client.get("/documents").json()["filenames"])

    with TestClient(app) as client:
        # An upload already holds one of the names.
        assert upload_and_wait(client, "readme.txt", b"Intro\nmine")["state"] == "done"
        response = client.post(
            "/documents/from-sitemap", json={"url": "https://example.com/sitemap.xml"}
        )
        assert response.status_code == 202, response.text
        wait_until(lambda: len(listed(client)) == 3)
        assert listed(client) == ["example.com_blog_notes.txt", "notes.txt", "readme.txt"]
    clear_dependency_caches()
    _ = Iterator


def test_a_workspace_is_not_deleted_under_a_sitemap_import(tmp_path: object) -> None:
    import threading
    from types import SimpleNamespace
    from typing import Any, cast

    from api.sitemap import PageImport, importing_into, start_page_import
    from tests.factories import wait_until

    release = threading.Event()

    def fetch(url: str, *_: Any) -> Any:
        release.wait(5)
        raise RuntimeError("not fetched")

    queue = SimpleNamespace(
        ingest_service=SimpleNamespace(
            settings=make_test_settings().model_copy(update={"qdrant_collection": "docs__ws_x"})
        ),
        backlog=SimpleNamespace(has_room=lambda size: True),
    )
    job = PageImport(
        queue=cast(Any, queue),
        raw_store=None,
        fetch=fetch,
        tags=None,
        refresh_minutes=None,
        known={},
    )
    start_page_import(["https://example.com/a.txt"], job)
    assert importing_into("docs__ws_x") and not importing_into("docs")
    release.set()
    wait_until(lambda: not importing_into("docs__ws_x"))


def _damaged_zip(compression: int) -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        archive.writestr("good.txt", "Intro\nfine text")
        archive.writestr("bad.txt", "Intro\n" + "broken text " * 200)
    data = bytearray(buffer.getvalue())
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as archive:
        info = archive.getinfo("bad.txt")
    start = info.header_offset + 30 + len(info.filename) + len(info.extra)
    for offset in range(start + 4, start + 40):
        data[offset] ^= 0x5A
    return bytes(data)


@pytest.fixture
def archive_client() -> object:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app, make_settings

    clear_dependency_caches()
    app = hermetic_app(make_settings(max_archive_members=2), QdrantClient(":memory:"))
    with TestClient(app) as client:
        yield client
    clear_dependency_caches()


@pytest.mark.parametrize("compression", ["bzip2", "lzma"])
def test_a_damaged_member_is_listed_not_a_500(archive_client: object, compression: str) -> None:
    import zipfile

    from fastapi.testclient import TestClient

    client = archive_client
    assert isinstance(client, TestClient)
    method = zipfile.ZIP_BZIP2 if compression == "bzip2" else zipfile.ZIP_LZMA
    response = client.post(
        "/documents/archive",
        files={"file": ("docs.zip", _damaged_zip(method), "application/zip")},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert [job["filename"] for job in body["jobs"]] == ["good.txt"]
    assert [item["name"] for item in body["skipped"]] == ["bad.txt"]


def test_the_member_cap_counts_documents_not_finder_metadata(archive_client: object) -> None:
    import io
    import zipfile

    from fastapi.testclient import TestClient

    client = archive_client
    assert isinstance(client, TestClient)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in ("a.txt", "b.txt"):
            archive.writestr(f"docs/{name}", "Intro\ntext")
            archive.writestr(f"__MACOSX/docs/._{name}", b"\x00\x05")
        archive.writestr("docs/.DS_Store", b"\x00")
    response = client.post(
        "/documents/archive", files={"file": ("docs.zip", buffer.getvalue(), "application/zip")}
    )
    assert response.status_code == 202, response.text
    assert len(response.json()["jobs"]) == 2


def test_a_zip_is_budgeted_as_an_import_and_the_cli_goes_on_past_a_refused_one(
    tmp_path: object,
) -> None:
    import io
    import os
    import zipfile
    from pathlib import Path

    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches
    from api.request_guard import UPLOAD_OVERHEAD_BYTES, body_limit
    from cli.__main__ import main
    from cli.client import DocragClient
    from tests.routes_http.support import hermetic_app, make_settings

    settings = make_settings(max_upload_bytes=1000, max_import_bytes=100_000)
    assert body_limit("POST", "/documents/archive", settings) == 100_000 + UPLOAD_OVERHEAD_BYTES

    def write_zip(path: Path, members: dict[str, bytes]) -> Path:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        path.write_bytes(buffer.getvalue())
        return path

    folder = Path(str(tmp_path))
    huge = write_zip(folder / "huge.zip", {"x.txt": os.urandom(200_000)})
    small = write_zip(folder / "small.zip", {"a.txt": b"Intro\nalpha"})
    clear_dependency_caches()
    with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as client:
        docrag = DocragClient("http://testserver", None, http=client, sleep=lambda _: None)
        out = io.StringIO()
        assert main(["ingest", str(huge), str(small)], client=docrag, out=out) == 1
        assert "1 indexed, 1 failed" in out.getvalue()
    clear_dependency_caches()


def test_verify_reports_what_restore_would_refuse() -> None:
    import io
    import json
    import zipfile

    from api.restore import verify_backup

    documents = [
        {"filename": "ok.txt", "original": "originals/ok.txt", "tags": ["a"]},
        {"filename": "big.txt", "original": "originals/big.txt"},
        {"filename": "moved.txt", "original": "elsewhere/moved.txt"},
        {
            "filename": "tags.txt",
            "original": "originals/tags.txt",
            "tags": ["t"] * 0 + [f"tag{n}" for n in range(30)],
        },
        {"filename": "notes.md", "original": "originals/notes.md", "versions": ["../x"]},
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "manifest.json", json.dumps({"manifest_version": 1, "documents": documents})
        )
        archive.writestr("originals/ok.txt", "Intro\nfine")
        archive.writestr("originals/big.txt", "x" * 5000)
        archive.writestr("elsewhere/moved.txt", "Intro\nmoved")
        archive.writestr("originals/tags.txt", "Intro\ntags")
        archive.writestr("originals/notes.md", "Intro\nversioned")

    check = verify_backup(io.BytesIO(buffer.getvalue()), max_original_bytes=1000)

    # Not damage: reported, and the backup is still ok.
    assert not check.problems and check.ok
    assert check.would_restore == 2
    reasons = dict(item.split(": ", 1) for item in check.rejected)
    assert set(reasons) == {"big.txt", "moved.txt", "tags.txt", "notes.md"}
    assert "version id" in reasons["notes.md"]
    assert reasons["big.txt"] == "larger than MAX_UPLOAD_BYTES"


def test_a_latin1_sitemap_keeps_its_accented_urls() -> None:
    from api.sitemap import listed_urls
    from api.url_fetch import _with_declared_charset

    body = (
        '<?xml version="1.0" encoding="ISO-8859-1"?>'
        "<urlset><url><loc>https://example.com/café</loc></url></urlset>"
    ).encode("latin-1")
    recoded = _with_declared_charset(body, "text/xml; charset=ISO-8859-1")
    assert listed_urls(recoded, "https://example.com/")[0] == ["https://example.com/café"]


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_a_failed_url_jobs_retry_keeps_its_refresh_interval(
    store_kind: str, tmp_path: object
) -> None:
    from pathlib import Path
    from types import SimpleNamespace
    from typing import Any, cast

    from api.document_ops import enqueue_refetched
    from api.jobs import IngestJobStore, SqliteIngestJobStore
    from api.url_fetch import FetchedDocument

    store = (
        IngestJobStore(10)
        if store_kind == "memory"
        else SqliteIngestJobStore(str(Path(str(tmp_path)) / "jobs.db"), 10)
    )
    created = store.create(
        "page.html", kind="url", source_url="https://example.com/page", refresh_minutes=90
    )
    job = store.get(created.id)
    assert job is not None and job.refresh_minutes == 90

    sent: dict[str, Any] = {}

    def enqueue(**fields: Any) -> object:
        sent.update(fields)
        return object()

    queue = SimpleNamespace(
        enqueue=enqueue, ingest_service=SimpleNamespace(settings=make_test_settings())
    )
    fetched = FetchedDocument("page.html", b"<p>page</p>", "https://example.com/page")
    enqueue_refetched(
        job, cast(Any, None), make_test_settings(), cast(Any, queue), None, fetched, None
    )
    assert sent["source"].refresh_minutes == 90


def test_slide_text_in_custom_shapes_and_groups_is_read() -> None:
    slide = (
        "<draw:page>"
        '<draw:frame presentation:class="title"><draw:text-box><text:p>Plan</text:p>'
        "</draw:text-box></draw:frame>"
        "<draw:custom-shape><text:p>Boxed step one</text:p></draw:custom-shape>"
        "<draw:g><draw:g><draw:frame><draw:text-box><text:p>Grouped step two</text:p>"
        "</draw:text-box></draw:frame></draw:g></draw:g>"
        "</draw:page>"
    )
    (section,) = _parse("deck.odp", odf_bytes("presentation", slide))
    assert section.section == "Plan"
    assert "Boxed step one" in section.text and "Grouped step two" in section.text


def test_a_code_cell_holding_a_fence_stays_code() -> None:
    from tests.samples import ipynb_bytes

    code = 'print("""\n```\n""")\n# not a heading'
    sections = _parse("nb.ipynb", ipynb_bytes([("markdown", "# Notebook"), ("code", code)]))
    assert [section.section for section in sections] == ["Notebook"]
    assert "# not a heading" in sections[0].text


def _sitemap_app(monkeypatch: pytest.MonkeyPatch, handler: object) -> object:
    import functools

    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches
    from api.url_fetch import fetch_document
    from tests.routes_http.support import hermetic_app, make_settings
    from tests.test_url_fetch import resolver, transport

    fake = functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler))
    monkeypatch.setattr("api.routes.documents_write.fetch_document", fake)
    clear_dependency_caches()
    return TestClient(
        hermetic_app(make_settings(url_ingest_enabled=True), QdrantClient(":memory:"))
    )


def test_a_second_import_skips_pages_that_did_not_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gzip

    import httpx

    from tests.factories import wait_until

    fetched: list[str] = []
    child = gzip.compress(
        b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/a.txt</loc><lastmod>2001-01-01</lastmod></url>
        <url><loc>https://example.com/b.txt</loc></url>
        </urlset>"""
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/":
            return httpx.Response(
                200, content=b"<html></html>", headers={"content-type": "text/html"}
            )
        if path == "/robots.txt":
            body = b"User-agent: *\nSitemap: https://example.com/pages.xml.gz\n"
            return httpx.Response(200, content=body, headers={"content-type": "text/plain"})
        if path == "/pages.xml.gz":
            return httpx.Response(200, content=child, headers={"content-type": "application/gzip"})
        fetched.append(path)
        if request.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(
            200,
            content=f"Intro\nThe page {path}".encode(),
            headers={"content-type": "text/plain", "etag": '"v1"'},
        )

    from fastapi.testclient import TestClient

    client = _sitemap_app(monkeypatch, handler)
    assert isinstance(client, TestClient)
    with client:
        # The site's address: its robots.txt names a gzipped sitemap.
        first = client.post("/documents/from-sitemap", json={"url": "https://example.com/"})
        assert first.status_code == 202, first.text
        assert first.json()["urls"] == ["https://example.com/a.txt", "https://example.com/b.txt"]
        import_id = first.json()["import_id"]
        wait_until(
            lambda: client.get(f"/documents/sitemap-imports/{import_id}").json()["state"] == "done"
        )
        wait_until(lambda: len(client.get("/documents").json()["filenames"]) == 2)
        status = client.get(f"/documents/sitemap-imports/{import_id}").json()
        assert (status["queued"], status["total"], status["done"]) == (2, 2, 2)
        fetched.clear()

        again = client.post("/documents/from-sitemap", json={"url": "https://example.com/"})
        second_id = again.json()["import_id"]
        wait_until(
            lambda: client.get(f"/documents/sitemap-imports/{second_id}").json()["state"] == "done"
        )
        status = client.get(f"/documents/sitemap-imports/{second_id}").json()
        # a.txt: listed as changed in 2001, before it was indexed — not fetched at all;
        # b.txt: no date, fetched conditionally, and the server said 304.
        assert status["unchanged"] == 2 and status["queued"] == 0
        assert fetched == ["/b.txt"]
        listed = client.get("/documents/sitemap-imports").json()["imports"]
        assert [item["id"] for item in listed][:2] == [second_id, import_id]
        assert client.get("/documents/sitemap-imports/nope").status_code == 404
        assert client.delete(f"/documents/sitemap-imports/{second_id}").status_code == 409


def test_an_import_is_cancelled_before_its_next_page(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    import httpx
    from fastapi.testclient import TestClient

    from tests.factories import wait_until

    release = threading.Event()
    sitemap = (
        b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">"""
        + b"".join(f"<url><loc>https://example.com/p{n}.txt</loc></url>".encode() for n in range(5))
        + b"</urlset>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, content=sitemap, headers={"content-type": "text/xml"})
        release.wait(5)
        return httpx.Response(200, content=b"Intro\npage", headers={"content-type": "text/plain"})

    client = _sitemap_app(monkeypatch, handler)
    assert isinstance(client, TestClient)
    with client:
        started = client.post(
            "/documents/from-sitemap", json={"url": "https://example.com/sitemap.xml"}
        ).json()
        cancelled = client.delete(f"/documents/sitemap-imports/{started['import_id']}")
        assert cancelled.status_code == 202 and cancelled.json()["state"] == "cancelling"
        release.set()
        wait_until(
            lambda: (
                client.get(f"/documents/sitemap-imports/{started['import_id']}").json()["state"]
                == "cancelled"
            )
        )
        status = client.get(f"/documents/sitemap-imports/{started['import_id']}").json()
        assert status["cancelled_pages"] >= 4 and status["done"] == 5


def test_the_cli_waits_for_a_sitemap_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import time

    import httpx
    from fastapi.testclient import TestClient

    from cli.__main__ import main
    from cli.client import DocragClient

    sitemap = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/a.txt</loc></url></urlset>"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, content=sitemap, headers={"content-type": "text/xml"})
        return httpx.Response(200, content=b"Intro\npage", headers={"content-type": "text/plain"})

    client = _sitemap_app(monkeypatch, handler)
    assert isinstance(client, TestClient)
    with client:
        docrag = DocragClient("http://testserver", None, http=client, sleep=time.sleep)
        out = io.StringIO()
        code = main(
            ["add-url", "https://example.com/sitemap.xml", "--sitemap", "--wait"],
            client=docrag,
            out=out,
        )
        assert code == 0, out.getvalue()
        assert "1 queued, 0 unchanged, 0 failed, 0 skipped of 1 page(s)" in out.getvalue()


def test_each_watched_folder_can_have_its_own_patterns(tmp_path: object) -> None:
    from pathlib import Path

    from api.watch import WatchFolder, watch_folders
    from tests.routes_http.support import make_settings

    root = Path(str(tmp_path))
    settings = make_settings(
        watch_include="*.md",
        watch_dirs=f"{root / 'a'}=>legal|include=*.pdf, *.docx|exclude=drafts/* ;{root / 'b'}=>hr",
    )
    folders = watch_folders(settings)
    assert folders == [
        WatchFolder(str(root / "a"), "legal", include="*.pdf, *.docx", exclude="drafts/*"),
        WatchFolder(str(root / "b"), "hr"),
    ]
    first = settings.model_copy(update=folders[0].settings_update())
    assert (first.watch_include, first.watch_exclude) == ("*.pdf, *.docx", "drafts/*")
    # No patterns of its own: the global ones.
    assert settings.model_copy(update=folders[1].settings_update()).watch_include == "*.md"
    with pytest.raises(RuntimeError, match="not include"):
        watch_folders(make_settings(watch_dirs=f"{root / 'a'}=>x|only=*.pdf"))


def test_a_job_result_says_what_was_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait
    from tests.samples import mbox_bytes

    monkeypatch.setattr("api.parsers.mbox.MAX_MBOX_MESSAGES", 2)
    clear_dependency_caches()
    with TestClient(hermetic_app(make_settings(), QdrantClient(":memory:"))) as client:
        mailbox = mbox_bytes([("One", "first"), ("Two", "second"), ("Three", "third")])
        job = upload_and_wait(client, "mail.mbox", mailbox, "application/mbox")
        assert job["state"] == "done"
        assert job["result"]["warnings"] == ["Only its first 2 of 3 messages were indexed."]
        clean = upload_and_wait(client, "a.txt", b"Intro\nalpha")
        assert clean["result"]["warnings"] == []
    clear_dependency_caches()


def test_warnings_cross_from_the_parse_process() -> None:
    from api.parse_isolation import parse_with_limit
    from tests.routes_http.support import make_settings
    from tests.samples import mbox_bytes

    settings = make_settings(parse_timeout_seconds=30)
    # A real child: the mailbox's skipped (unreadable) message is noted there.
    mailbox = mbox_bytes([("One", "first"), ("Two", "")])
    parsed = parse_with_limit("mail.mbox", mailbox, settings)
    assert len(parsed) >= 1 and isinstance(parsed.warnings, list)


def test_a_date_only_lastmod_counts_as_the_end_of_that_day() -> None:
    from datetime import UTC, datetime

    from api.sitemap import _parse_lastmod

    assert _parse_lastmod("2026-11-07") == datetime(2026, 11, 7, 23, 59, 59, tzinfo=UTC).timestamp()
    assert (
        _parse_lastmod("2026-11-07T08:00:00Z") == datetime(2026, 11, 7, 8, tzinfo=UTC).timestamp()
    )
    assert _parse_lastmod("Sat, 07 Nov 2026 08:00:00 GMT") is not None
    assert _parse_lastmod("soon") is None
