from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from api.url_fetch import MAX_REDIRECTS, UrlFetchError, choose_filename, fetch_document
from tests.factories import make_test_settings

PUBLIC = "93.184.215.14"


def resolver(table: dict[str, list[str]]) -> Callable[[str, int], list[str]]:
    return lambda host, port: table.get(host, [PUBLIC])


def transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def fetch(url: str, handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> Any:
    settings = kwargs.pop("settings", make_test_settings(url_ingest_enabled=True))
    return fetch_document(
        url,
        settings,
        resolve=kwargs.pop("resolve", resolver({})),
        transport=transport(handler),
        **kwargs,
    )


def ok(body: bytes = b"# Title\nHello.", content_type: str = "text/markdown") -> Any:
    return lambda request: httpx.Response(200, content=body, headers={"content-type": content_type})


def test_the_request_goes_to_the_checked_address_under_the_original_host() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"Hello.", headers={"content-type": "text/plain"})

    doc = fetch("https://docs.example.com/guides/setup.txt?x=1", handler)

    assert doc.filename == "setup.txt" and doc.content == b"Hello."
    request = seen[0]
    assert request.url.host == PUBLIC and request.url.port in (None, 443)  # 443 is implied
    assert request.url.query == b"x=1"
    assert request.headers["host"] == "docs.example.com"
    assert request.extensions["sni_hostname"] == "docs.example.com"


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.5", "192.168.1.2", "169.254.169.254", "::1", "fd00::1", "0.0.0.0"],
)
def test_private_and_reserved_addresses_are_refused(address: str) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200)

    with pytest.raises(UrlFetchError, match="URL_INGEST_ALLOW_PRIVATE"):
        fetch(
            "http://intranet.local/a.txt", handler, resolve=resolver({"intranet.local": [address]})
        )
    assert calls == []


def test_a_host_with_any_private_address_is_refused() -> None:
    mixed = resolver({"split.example": [PUBLIC, "10.1.2.3"]})
    with pytest.raises(UrlFetchError, match="private"):
        fetch("http://split.example/a.txt", ok(), resolve=mixed)


def test_private_addresses_are_allowed_when_configured() -> None:
    settings = make_test_settings(url_ingest_enabled=True, url_ingest_allow_private=True)
    doc = fetch(
        "http://wiki.lan:8080/page.md",
        ok(),
        settings=settings,
        resolve=resolver({"wiki.lan": ["192.168.1.9"]}),
    )
    assert doc.filename == "page.md"


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("ftp://example.com/a.txt", "Only http and https"),
        ("file:///etc/passwd", "Only http and https"),
        ("https://user:pw@example.com/a.txt", "credentials"),
        ("https:///a.txt", "no host"),
        ("https://example.com:99999/a.txt", "port"),
        ("https://example.com/" + "a" * 3000, "longer than"),
    ],
)
def test_unusable_urls_are_refused_before_any_request(url: str, message: str) -> None:
    with pytest.raises(UrlFetchError, match=message):
        fetch(url, ok())


def test_redirects_are_followed_and_each_hop_is_checked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["host"] == "short.example":
            return httpx.Response(302, headers={"location": "https://files.example/doc.txt"})
        return httpx.Response(200, content=b"Body", headers={"content-type": "text/plain"})

    doc = fetch("https://short.example/x", handler)
    assert doc.final_url == "https://files.example/doc.txt" and doc.filename == "doc.txt"

    def to_metadata(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://metadata.internal/latest"})

    with pytest.raises(UrlFetchError, match="private"):
        fetch(
            "https://short.example/x",
            to_metadata,
            resolve=resolver({"metadata.internal": ["169.254.169.254"]}),
        )


def test_redirect_loops_stop() -> None:
    hops: list[int] = []

    def loop(request: httpx.Request) -> httpx.Response:
        hops.append(1)
        return httpx.Response(301, headers={"location": "/again"})

    with pytest.raises(UrlFetchError, match="redirected more than"):
        fetch("https://example.com/start", loop)
    assert len(hops) == MAX_REDIRECTS + 1


def test_downloads_past_the_upload_limit_are_abandoned() -> None:
    settings = make_test_settings(url_ingest_enabled=True, max_upload_bytes=1000)
    declared = lambda request: httpx.Response(  # noqa: E731
        200, content=b"x" * 10, headers={"content-type": "text/plain", "content-length": "5000"}
    )
    with pytest.raises(UrlFetchError, match="larger than 1000"):
        fetch("https://example.com/a.txt", declared, settings=settings)

    def chunked(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, stream=httpx.ByteStream(b"x" * 5000), headers={"content-type": "text/plain"}
        )

    with pytest.raises(UrlFetchError, match="larger than 1000"):
        fetch("https://example.com/a.txt", chunked, settings=settings)


def test_upstream_errors_and_unreachable_hosts_are_named() -> None:
    with pytest.raises(UrlFetchError, match="answered 404"):
        fetch("https://example.com/a.txt", lambda request: httpx.Response(404))

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(UrlFetchError, match="Could not fetch"):
        fetch("https://example.com/a.txt", refuse)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(UrlFetchError, match="URL_INGEST_TIMEOUT_SECONDS"):
        fetch("https://example.com/a.txt", slow)


def test_filenames_come_from_the_request_the_server_or_the_url() -> None:
    assert choose_filename("https://e.com/x", "text/html", None, "Policy.html") == "Policy.html"
    assert (
        choose_filename("https://e.com/x", "application/pdf", 'attachment; filename="r.pdf"', None)
        == "r.pdf"
    )
    assert choose_filename(
        "https://e.com/wiki/Leave_policy", "text/html; charset=utf-8", None, None
    ) == ("Leave_policy.html")
    assert choose_filename("https://e.com/", "text/html", None, None) == "e.com.html"
    assert choose_filename("https://e.com/a%20b.txt", "text/plain", None, None) == "a b.txt"
    with pytest.raises(UrlFetchError, match="can't index"):
        choose_filename("https://e.com/pic", "image/png", None, None)


def test_the_resolver_names_an_unknown_host() -> None:
    from api.url_fetch import resolve_host

    with pytest.raises(UrlFetchError, match="Cannot resolve"):
        resolve_host("no-such-host.invalid", 443)


# The route, with the fetcher's network swapped for a mock.


def test_the_route_is_off_by_default_and_indexes_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches, get_app_settings
    from tests.routes_http.support import hermetic_app, make_settings, wait_until

    clear_dependency_caches()
    off = make_settings(api_key="full-key", api_read_key="read-key")
    app = hermetic_app(off, QdrantClient(":memory:"))
    monkeypatch.setattr(
        "api.routes.documents.fetch_document",
        functools.partial(
            fetch_document,
            resolve=resolver({}),
            transport=transport(ok(b"Intro\nAlpha from the web.", "text/plain")),
        ),
    )
    full = {"X-API-Key": "full-key"}
    body = {"url": "https://example.com/notes/alpha.txt"}
    with TestClient(app) as client:
        assert client.post("/documents/from-url", json=body, headers=full).status_code == 404
        assert client.get("/config").json()["url_ingest_enabled"] is False

        on = make_settings(api_key="full-key", api_read_key="read-key", url_ingest_enabled=True)
        app.dependency_overrides[get_app_settings] = lambda: on
        read_only = client.post("/documents/from-url", json=body, headers={"X-API-Key": "read-key"})
        assert read_only.status_code == 403
        accepted = client.post("/documents/from-url", json=body, headers=full)
        assert accepted.status_code == 202 and accepted.json()["filename"] == "alpha.txt"
        job_id = accepted.json()["job_id"]
        wait_until(
            lambda: client.get(f"/documents/jobs/{job_id}", headers=full).json()["state"] == "done"
        )
        assert client.get("/documents", headers=full).json()["filenames"] == ["alpha.txt"]

        bad = client.post("/documents/from-url", json={"url": "ftp://x/y.txt"}, headers=full)
        assert bad.status_code == 400 and "Only http and https" in bad.json()["detail"]
    clear_dependency_caches()
