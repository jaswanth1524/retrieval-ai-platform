"""Pass 18: new text formats, front matter tags, and a zip fetched from a URL."""

from __future__ import annotations

import functools
import io
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.documents import SUPPORTED_EXTENSIONS, DocumentParseError, parse_document_bytes
from api.parsers.common import front_matter_tags
from api.url_fetch import UrlFetchError, choose_filename, fetch_document
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait, wait_until
from tests.test_url_fetch import resolver, transport


def _texts(name: str, content: bytes) -> list[str]:
    return [section.text for section in parse_document_bytes(name, content)]


def test_tab_separated_rows_keep_commas_inside_a_cell() -> None:
    assert _texts("r.tsv", b"city\tcap\nParis\t180\nOslo, Norway\t210\n") == [
        "city: Paris; cap: 180\ncity: Oslo, Norway; cap: 210"
    ]


def test_json_lines_are_one_section_per_record_and_a_bad_line_is_named() -> None:
    assert _texts("t.jsonl", b'{"id": 1, "t": "a"}\n\n{"id": 2}\n') == [
        "Record 1 › id: 1\nRecord 1 › t: a",
        "Record 2 › id: 2",
    ]
    with pytest.raises(DocumentParseError, match="line 2"):
        parse_document_bytes("bad.jsonl", b'{"a": 1}\nnope\n')


def test_xml_is_read_for_its_text_and_a_dtd_is_refused() -> None:
    assert _texts("f.xml", b"<a><b>Hello</b><c>World</c></a>") == ["Hello", "World"]
    with pytest.raises(DocumentParseError, match="DTD"):
        parse_document_bytes("x.xml", b'<!DOCTYPE a [<!ENTITY x "y">]><a>&x;</a>')


@pytest.mark.parametrize(
    "suffix",
    [".log", ".ini", ".conf", ".cfg", ".org", ".css", ".php", ".cs", ".kt", ".swift", ".lua"],
)
def test_more_text_and_code_types_are_read_as_text(suffix: str) -> None:
    assert suffix in SUPPORTED_EXTENSIONS
    assert _texts(f"a{suffix}", b"first block\n\nsecond block\n") == ["first block", "second block"]


def test_front_matter_tags_are_read_inline_or_as_a_block() -> None:
    assert front_matter_tags(b"---\ntitle: X\ntags: [hr, 'policy']\n---\n# Body") == [
        "hr",
        "policy",
    ]
    assert front_matter_tags(b"---\ntags:\n  - hr\n  - travel\nauthor: me\n---\n") == [
        "hr",
        "travel",
    ]
    assert front_matter_tags(b"---\ntags: a, b\n...\n") == ["a", "b"]
    assert front_matter_tags(b"# no front matter\ntags: x") == []
    assert front_matter_tags(b"---\ntags: [never, closed]\n") == []


def test_front_matter_tags_tag_a_markdown_document_only_when_turned_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"---\ntags: [hr, travel]\n---\n# Travel\nBook ahead."
    for enabled, expected in ((False, {}), (True, {"t.md": ["hr", "travel"]})):
        clear_dependency_caches()
        app = hermetic_app(make_settings(front_matter_tags=enabled), QdrantClient(":memory:"))
        with TestClient(app) as client:
            upload_and_wait(client, "t.md", content)
            assert client.get("/documents").json()["tags"] == expected
            if enabled:
                # Tags given with the upload win over the front matter.
                accepted = client.post(
                    "/documents", files={"file": ("t.md", content)}, data={"tags": ["mine"]}
                )
                status = f"/documents/jobs/{accepted.json()['job_id']}"
                wait_until(functools.partial(_done, client, status))
                assert client.get("/documents").json()["tags"] == {"t.md": ["mine"]}
        clear_dependency_caches()


def _done(client: TestClient, status: str) -> bool:
    return bool(client.get(status).json()["state"] == "done")


def test_a_zip_is_named_for_the_archive_route_only() -> None:
    url = "https://example.com/docs.zip"
    assert choose_filename(url, "application/zip", None, None, archive=True) == "docs.zip"
    with pytest.raises(UrlFetchError):
        choose_filename(url, "application/zip", None, None)
    with pytest.raises(UrlFetchError):
        choose_filename("https://example.com/a.pdf", "application/pdf", None, None, archive=True)


def test_a_zip_at_a_url_is_imported_in_the_background(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a/notes.txt", b"Intro\nfirst")
        archive.writestr("b/notes.txt", b"Intro\nsecond")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=buffer.getvalue(), headers={"content-type": "application/zip"}
        )

    monkeypatch.setattr(
        "api.routes.documents_write.fetch_document",
        functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler)),
    )
    clear_dependency_caches()
    with TestClient(hermetic_app(make_settings(), QdrantClient(":memory:"))) as client:
        off = client.post("/documents/archive-from-url", json={"url": "https://x.example/d.zip"})
        assert off.status_code == 404
    clear_dependency_caches()
    app = hermetic_app(make_settings(url_ingest_enabled=True), QdrantClient(":memory:"))
    with TestClient(app) as client:
        started = client.post(
            "/documents/archive-from-url",
            json={"url": "https://x.example/d.zip", "keep_paths": True, "tags": ["web"]},
        )
        assert started.status_code == 202, started.text
        import_id = started.json()["import_id"]
        wait_until(lambda: client.get(f"/documents/imports/{import_id}").json()["finished_at"])
        wait_until(lambda: len(client.get("/documents").json()["filenames"]) == 2)
        listing = client.get("/documents").json()
        assert listing["filenames"] == ["a/notes.txt", "b/notes.txt"]
        assert listing["tags"] == {"a/notes.txt": ["web"], "b/notes.txt": ["web"]}
    clear_dependency_caches()
