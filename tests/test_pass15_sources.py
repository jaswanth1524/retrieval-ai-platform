"""Pass 15 ingest sources: a zip of documents, and a site's pages from its sitemap or feed."""

from __future__ import annotations

import functools
import io
import zipfile
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.sitemap import SitemapError, listed_urls, select_pages
from api.url_fetch import fetch_document
from tests.factories import wait_until
from tests.routes_http.support import hermetic_app, make_settings
from tests.test_url_fetch import resolver, transport


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def client() -> Iterator[TestClient]:
    clear_dependency_caches()
    settings = make_settings(url_ingest_enabled=True, max_archive_members=10)
    with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as test_client:
        yield test_client
    clear_dependency_caches()


def _listed(client: TestClient) -> dict[str, Any]:
    return dict(client.get("/documents").json())


def test_a_zip_indexes_each_supported_member_as_its_own_document(client: TestClient) -> None:
    content = _zip(
        {
            "team/guide.md": b"# Guide\nalpha",
            "team/notes.txt": b"Intro\nbeta",
            "other/guide.md": b"# Guide\na second copy",
            "photo.bin": b"\0\1",
            ".DS_Store": b"x",
            "__MACOSX/team/._guide.md": b"x",
            "../escape.txt": b"Intro\nnamed by its base name only",
        }
    )

    response = client.post(
        "/documents/archive",
        files={"file": ("docs.zip", content, "application/zip")},
        data={"tags": ["team"]},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert sorted(job["filename"] for job in body["jobs"]) == [
        "escape.txt",
        "guide.md",
        "notes.txt",
    ]
    assert {item["name"]: item["reason"] for item in body["skipped"]} == {
        "other/guide.md": "Another file in the zip has the same name.",
        "photo.bin": "Not a supported document type.",
    }
    wait_until(lambda: len(_listed(client)["filenames"]) == 3)
    assert _listed(client)["tags"]["notes.txt"] == ["team"]


def test_a_zip_that_is_not_one_or_too_big_is_refused(client: TestClient) -> None:
    bad = client.post("/documents/archive", files={"file": ("x.zip", b"no", "application/zip")})
    assert bad.status_code == 400
    many = _zip({f"f{n}.txt": b"Intro\nx" for n in range(11)})
    refused = client.post("/documents/archive", files={"file": ("m.zip", many, "application/zip")})
    assert refused.status_code == 400 and "MAX_ARCHIVE_MEMBERS" in refused.text


def test_a_member_that_expands_past_the_limit_is_skipped() -> None:
    clear_dependency_caches()
    settings = make_settings(max_upload_bytes=2000)
    with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as client:
        bomb = _zip({"big.txt": b"a" * 100_000, "ok.txt": b"Intro\nsmall"})
        body = client.post(
            "/documents/archive", files={"file": ("b.zip", bomb, "application/zip")}
        ).json()
        assert [job["filename"] for job in body["jobs"]] == ["ok.txt"]
        assert body["skipped"][0]["name"] == "big.txt"
    clear_dependency_caches()


SITEMAP = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/a.txt</loc></url>
  <url><loc>https://example.com/b.txt</loc></url>
  <url><loc>https://elsewhere.com/c.txt</loc></url>
  <url><loc>https://example.com/a.txt</loc></url>
</urlset>"""


def test_feeds_and_sitemaps_list_their_pages() -> None:
    pages, children = listed_urls(SITEMAP, "https://example.com/sitemap.xml")
    assert len(pages) == 4 and children == []
    kept, left_out = select_pages(pages, "https://example.com/s.xml", same_host=True, limit=10)
    assert kept == ["https://example.com/a.txt", "https://example.com/b.txt"] and left_out == 2

    rss = b"<rss><channel><item><link>/post-1</link></item></channel></rss>"
    assert listed_urls(rss, "https://blog.example.com/feed")[0] == [
        "https://blog.example.com/post-1"
    ]
    atom = (
        b'<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        b'<link rel="enclosure" href="/x.mp3"/><link href="https://example.com/p"/>'
        b"</entry></feed>"
    )
    assert listed_urls(atom, "https://example.com/atom")[0] == ["https://example.com/p"]
    index = (
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>https://example.com/pages.xml</loc></sitemap></sitemapindex>"
    )
    assert listed_urls(index, "https://example.com/")[1] == ["https://example.com/pages.xml"]
    with pytest.raises(SitemapError, match="DTD"):
        listed_urls(b'<!DOCTYPE x [<!ENTITY a "b">]><urlset/>', "https://example.com/")
    with pytest.raises(SitemapError):
        listed_urls(b"<html/>", "https://example.com/")


def test_a_sitemaps_pages_are_fetched_and_queued_in_the_background(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, content=SITEMAP, headers={"content-type": "text/xml"})
        return httpx.Response(
            200,
            content=f"Intro\nThe page {request.url.path}".encode(),
            headers={"content-type": "text/plain"},
        )

    fake = functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler))
    monkeypatch.setattr("api.routes.documents_write.fetch_document", fake)

    response = client.post(
        "/documents/from-sitemap",
        json={"url": "https://example.com/sitemap.xml", "tags": ["site"], "refresh_minutes": 60},
    )

    assert response.status_code == 202, response.text
    assert response.json() == {
        "urls": ["https://example.com/a.txt", "https://example.com/b.txt"],
        "left_out": 2,
        "child_sitemaps": 0,
    }
    wait_until(lambda: sorted(_listed(client)["filenames"]) == ["a.txt", "b.txt"])
    listing = _listed(client)
    assert listing["tags"]["a.txt"] == ["site"]
    assert listing["source_urls"]["b.txt"] == "https://example.com/b.txt"


def test_a_page_that_is_not_a_sitemap_is_a_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html></html>", headers={"content-type": "text/html"})

    fake = functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler))
    monkeypatch.setattr("api.routes.documents_write.fetch_document", fake)
    response = client.post("/documents/from-sitemap", json={"url": "https://example.com/"})
    assert response.status_code == 400


def test_the_cli_ingests_a_zip_and_adds_a_sitemap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from cli.__main__ import main
    from cli.client import DocragClient

    archive = tmp_path / "docs.zip"
    archive.write_bytes(_zip({"a.md": b"# A\nalpha", "b.bin": b"x"}))
    docrag = DocragClient("http://testserver", None, http=client, sleep=lambda _: None)
    out = io.StringIO()
    assert main(["ingest", str(archive)], client=docrag, out=out) == 0
    assert "1 indexed, 0 failed, 1 skipped" in out.getvalue()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, content=SITEMAP, headers={"content-type": "text/xml"})
        return httpx.Response(200, content=b"Intro\npage", headers={"content-type": "text/plain"})

    fake = functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler))
    monkeypatch.setattr("api.routes.documents_write.fetch_document", fake)
    out = io.StringIO()
    assert (
        main(
            ["add-url", "https://example.com/sitemap.xml", "--sitemap", "--max-pages", "1"],
            client=docrag,
            out=out,
        )
        == 0
    )
    assert "1 page(s) being added in the background (3 left out)" in out.getvalue()


def test_watch_dirs_are_parsed_and_overlaps_refused(tmp_path: Any) -> None:
    from api.watch import WatchFolder, watch_folders, watched_workspaces

    a, b = tmp_path / "a", tmp_path / "b"
    settings = make_settings(
        watch_dir=str(a), watch_workspace="", watch_dirs=f" {b} => legal ; ;{tmp_path / 'c'}"
    )
    assert watch_folders(settings) == [
        WatchFolder(str(a), ""),
        WatchFolder(str(b), "legal"),
        WatchFolder(str(tmp_path / "c"), ""),
    ]
    assert watched_workspaces(settings) == {"", "legal"}
    for clash in (f"{a}=>x", f"{a / 'inner'}=>x", f"{tmp_path}=>x", "=>x"):
        with pytest.raises(RuntimeError, match="WATCH_DIRS"):
            watch_folders(make_settings(watch_dir=str(a), watch_dirs=clash))


def test_each_watched_folder_indexes_into_its_own_workspace(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import time

    import api.main as main
    from api.chunking import HeuristicTokenCounter
    from api.repository import VectorRepository
    from api.workspaces import settings_for
    from tests.routes_http.support import FakeEmbeddingProvider

    clear_dependency_caches()
    for name, text in (
        ("hr/policy.txt", "Intro\nLeave rules."),
        ("law/contract.txt", "Intro\nTerms."),
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        os.utime(path, (time.time() - 60, time.time() - 60))
    settings = make_settings(
        watch_dir=str(tmp_path / "hr"),
        watch_dirs=f"{tmp_path / 'law'}=>legal",
        watch_interval_seconds=2,
        workspaces_enabled=True,
    )
    qdrant = QdrantClient(":memory:")
    monkeypatch.setattr(main, "get_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(main, "get_embedding_provider", lambda: FakeEmbeddingProvider())
    monkeypatch.setattr(main, "get_token_counter", lambda: HeuristicTokenCounter())
    monkeypatch.setattr(main, "raw_store_for", lambda _settings: None)
    monkeypatch.setattr(main, "get_app_settings", lambda: settings)
    app = hermetic_app(settings, qdrant)
    repository = VectorRepository(qdrant)
    try:
        with TestClient(app) as client:
            wait_until(lambda: "policy.txt" in repository.filename_metadata(settings), timeout=20)
            legal = settings_for(settings, "legal")
            wait_until(lambda: "contract.txt" in repository.filename_metadata(legal), timeout=20)
            folders = client.get("/watch").json()["folders"]
            assert [(f["workspace"], f["files"]) for f in folders] == [("default", 1), ("legal", 1)]
            assert client.post("/watch/scan").status_code == 202
            assert client.delete("/workspaces/legal").status_code == 409
    finally:
        clear_dependency_caches()
