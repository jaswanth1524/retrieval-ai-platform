from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from api.url_fetch import (
    MAX_REDIRECTS,
    UrlFetchError,
    choose_filename,
    fetch_document,
    usable_validator,
)
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


def test_a_full_backlog_is_asked_before_the_body_with_the_declared_size() -> None:
    from api.url_fetch import BacklogFullError

    asked: list[int] = []

    def no_room(size: int) -> bool:
        asked.append(size)
        return False

    declared = lambda request: httpx.Response(  # noqa: E731
        200, content=b"x" * 50, headers={"content-type": "text/plain", "content-length": "50"}
    )
    with pytest.raises(BacklogFullError):
        fetch("https://example.com/a.txt", declared, has_room=no_room)

    def undeclared(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, stream=httpx.ByteStream(b"x" * 50), headers={"content-type": "text/plain"}
        )

    settings = make_test_settings(url_ingest_enabled=True, max_upload_bytes=1000)
    with pytest.raises(BacklogFullError):
        fetch("https://example.com/a.txt", undeclared, has_room=no_room, settings=settings)
    assert asked == [50, 1000]


def test_the_whole_fetch_has_one_deadline() -> None:
    now = [0.0]

    def clock() -> float:
        now[0] += 4.0  # every look at the clock is four seconds later
        return now[0]

    class Dribbling(httpx.SyncByteStream):
        def __iter__(self):  # type: ignore[no-untyped-def]
            while True:
                yield b"x"

    settings = make_test_settings(url_ingest_enabled=True, url_ingest_timeout_seconds=10)
    with pytest.raises(UrlFetchError, match="did not finish within"):
        fetch(
            "https://example.com/a.txt",
            lambda request: httpx.Response(
                200, stream=Dribbling(), headers={"content-type": "text/plain"}
            ),
            settings=settings,
            clock=clock,
        )

    # Redirect hops count against the same deadline.
    now[0] = 0.0
    hop = lambda request: httpx.Response(302, headers={"location": "/next"})  # noqa: E731
    with pytest.raises(UrlFetchError, match="did not finish within"):
        fetch("https://example.com/a.txt", hop, settings=settings, clock=clock)


def test_broken_bodies_and_unparseable_urls_are_400s_not_500s() -> None:
    class Breaking(httpx.SyncByteStream):
        def __iter__(self):  # type: ignore[no-untyped-def]
            yield b"partial"
            raise httpx.ReadError("connection reset")

    with pytest.raises(UrlFetchError, match="broke off"):
        fetch(
            "https://example.com/a.txt",
            lambda request: httpx.Response(
                200, stream=Breaking(), headers={"content-type": "text/plain"}
            ),
        )
    with pytest.raises(UrlFetchError, match="not valid"):
        fetch("http://[::1/a.txt", ok())
    with pytest.raises(UrlFetchError, match="host name is not valid"):
        fetch("http://" + "ä" * 70 + ".example/a.txt", ok())


def test_a_non_ascii_host_is_sent_in_its_idna_form() -> None:
    seen: list[httpx.Request] = []
    resolved: list[str] = []

    def resolve(host: str, port: int) -> list[str]:
        resolved.append(host)
        return [PUBLIC]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"Hi.", headers={"content-type": "text/plain"})

    fetch("https://bücher.example/a.txt", handler, resolve=resolve)
    assert resolved == ["xn--bcher-kva.example"]
    assert seen[0].headers["host"] == "xn--bcher-kva.example"
    assert seen[0].extensions["sni_hostname"] == "xn--bcher-kva.example"


def test_ipv4_is_tried_first_and_an_unreachable_address_falls_back() -> None:
    v6 = "2606:4700:4700::1111"
    tried: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tried.append(request.url.host)
        if request.url.host == PUBLIC:
            raise httpx.ConnectError("no route", request=request)
        return httpx.Response(200, content=b"Hi.", headers={"content-type": "text/plain"})

    doc = fetch(
        "https://dual.example/a.txt", handler, resolve=resolver({"dual.example": [v6, PUBLIC]})
    )
    assert doc.content == b"Hi." and tried == [PUBLIC, v6]


def test_an_ipv6_literal_is_bracketed_in_the_host_header() -> None:
    v6 = "2606:4700:4700::1111"
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"Hi.", headers={"content-type": "text/plain"})

    fetch(f"http://[{v6}]:8080/a.txt", handler, resolve=resolver({v6: [v6]}))
    fetch(f"http://[{v6}]/a.txt", handler, resolve=resolver({v6: [v6]}))
    assert [request.headers["host"] for request in seen] == [f"[{v6}]:8080", f"[{v6}]"]


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("64:ff9b::7f00:1", False),  # NAT64 of 127.0.0.1
        ("64:ff9b::a9fe:a9fe", False),  # NAT64 of 169.254.169.254
        ("2002:7f00:1::", False),  # 6to4 of 127.0.0.1
        ("2002:a00:1::", False),  # 6to4 of 10.0.0.1
        ("::ffff:127.0.0.1", False),
        ("64:ff9b::5db8:d70e", True),  # NAT64 of a public address
        ("2606:4700:4700::1111", True),
        (PUBLIC, True),
    ],
)
def test_ipv4_carried_in_ipv6_is_checked_too(address: str, public: bool) -> None:
    from api.url_fetch import is_public_address

    assert is_public_address(address) is public
    if not public:
        with pytest.raises(UrlFetchError, match="private"):
            fetch(
                "http://sneaky.example/a.txt", ok(), resolve=resolver({"sneaky.example": [address]})
            )


def test_the_logged_url_drops_its_query_and_fragment() -> None:
    from api.url_fetch import loggable_url

    assert loggable_url("https://e.com/a.txt?X-Amz-Signature=secret#p2") == "https://e.com/a.txt"
    assert loggable_url("http://[::1") == "<unparseable URL>"


def test_the_route_limits_fetches_and_checks_the_backlog_first(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import threading

    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches, get_ingest_backlog
    from api.jobs import IngestBacklog
    from tests.routes_http.support import hermetic_app, make_settings, wait_until

    clear_dependency_caches()
    app = hermetic_app(make_settings(url_ingest_enabled=True), QdrantClient(":memory:"))
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched.append(str(request.url))
        return httpx.Response(200, content=b"Intro\nAlpha.", headers={"content-type": "text/plain"})

    monkeypatch.setattr(
        "api.routes.documents.fetch_document",
        functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler)),
    )
    body = {"url": "https://example.com/alpha.txt?token=secret"}
    with TestClient(app) as client:
        app.dependency_overrides[get_ingest_backlog] = lambda: IngestBacklog(1000, max_jobs=0)
        full = client.post("/documents/from-url", json=body)
        assert full.status_code == 503 and full.headers["retry-after"] == "15"
        assert fetched == []  # refused before anything was downloaded

        # Room for something, not for the 5000 bytes the server declares: refused once
        # the headers are in, before the body is read.
        nearly_full = IngestBacklog(1000)
        assert nearly_full.try_reserve(900)
        app.dependency_overrides[get_ingest_backlog] = lambda: nearly_full
        monkeypatch.setattr(
            "api.routes.documents.fetch_document",
            functools.partial(
                fetch_document,
                resolve=resolver({}),
                transport=transport(
                    lambda request: httpx.Response(
                        200,
                        content=b"x" * 5000,
                        headers={"content-type": "text/plain", "content-length": "5000"},
                    )
                ),
            ),
        )
        too_big = client.post("/documents/from-url", json=body)
        assert too_big.status_code == 503 and too_big.headers["retry-after"] == "15"
        assert nearly_full.pending_bytes == 900  # nothing was queued
        del app.dependency_overrides[get_ingest_backlog]
        monkeypatch.setattr(
            "api.routes.documents.fetch_document",
            functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler)),
        )

        busy = threading.BoundedSemaphore(1)
        busy.acquire()
        monkeypatch.setattr("api.routes.documents._url_fetch_slots", busy)
        limited = client.post("/documents/from-url", json=body)
        assert limited.status_code == 429 and "Too many URL fetches" in limited.json()["detail"]
        assert limited.headers["retry-after"] == "5" and fetched == []

        busy.release()
        caplog.set_level("INFO", logger="api.routes.documents")
        accepted = client.post("/documents/from-url", json=body)
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]
        wait_until(lambda: client.get(f"/documents/jobs/{job_id}").json()["state"] == "done")
        assert "https://example.com/alpha.txt" in caplog.text and "secret" not in caplog.text
        # The slot was given back: a second fetch is not refused.
        assert busy.acquire(blocking=False)
    clear_dependency_caches()


def test_the_name_is_marked_when_the_remote_server_chose_it() -> None:
    text = "text/plain"
    plain = fetch("https://example.com/notes/a.txt", ok(b"Hello.", text))
    assert (plain.filename, plain.remote_named) == ("a.txt", False)

    def disposition(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"Hello.",
            headers={"content-type": text, "content-disposition": 'attachment; filename="h.txt"'},
        )

    named = fetch("https://example.com/download?id=1", disposition)
    assert (named.filename, named.remote_named) == ("h.txt", True)
    # The caller's own name always wins, and is theirs to replace a document with.
    given = fetch("https://example.com/download?id=1", disposition, filename="mine.txt")
    assert (given.filename, given.remote_named) == ("mine.txt", False)

    def redirect(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/go":
            return httpx.Response(302, headers={"location": "/files/handbook.txt"})
        return httpx.Response(200, content=b"Hello.", headers={"content-type": text})

    moved = fetch("https://example.com/go", redirect)
    assert (moved.filename, moved.remote_named) == ("handbook.txt", True)

    # A file served with its own name (common for PDFs) is the URL's name, not the
    # server's: re-adding that URL replaces its document.
    def inline(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"Hello.",
            headers={"content-type": text, "content-disposition": 'inline; filename="a.txt"'},
        )

    same = fetch("https://example.com/notes/a.txt", inline)
    assert (same.filename, same.remote_named) == ("a.txt", False)


def test_a_name_lookup_that_hangs_is_given_up_on_within_the_deadline() -> None:
    import threading
    import time

    from api.url_fetch import resolve_within

    release = threading.Event()
    returned = threading.Event()

    def hanging(host: str, port: int) -> list[str]:
        release.wait(30)
        returned.set()
        return [PUBLIC]

    started = time.monotonic()
    try:
        with pytest.raises(UrlFetchError, match="could not be resolved within"):
            resolve_within(hanging, "slow.example", 443, 0.2)
        # Given up on while the lookup was still hanging, not when it came back: an
        # ordering, not a wall-clock bound a busy runner could miss.
        assert not returned.is_set()
    finally:
        release.set()
    assert time.monotonic() - started < 25  # only a backstop
    assert resolve_within(resolver({}), "fast.example", 443, 1.0) == [PUBLIC]


def _tls_contexts(tmp_path: Any) -> tuple[Any, Any]:
    """A server context with a throwaway certificate for local.test, and a client
    context trusting only it."""

    import datetime
    import ssl

    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "local.test")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("local.test")]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_file, key_file = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_file, key_file)
    client = ssl.create_default_context(cafile=str(cert_file))
    return server, client


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_the_deadline_holds_even_inside_one_slow_read(scheme: str, tmp_path: Any) -> None:
    """Over a real socket. httpcore takes the read timeout once, as the body starts: a
    body that stalled late got that whole timeout again past the deadline. Over TLS
    too, which takes the connection's socket over from the one first connected."""

    import socket
    import threading
    import time

    server_tls, client_tls = _tls_contexts(tmp_path) if scheme == "https" else (None, None)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    release = threading.Event()

    def serve() -> None:
        for _ in range(2):
            raw, _ = listener.accept()
            connection = server_tls.wrap_socket(raw, server_side=True) if server_tls else raw
            with connection:
                request = connection.recv(65536).decode()
                if "/stalls" in request:
                    time.sleep(1.5)  # the headers come late, but within the deadline
                head = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 12\r\n\r\n"
                if "/complete" in request:
                    connection.sendall(head + b"Hello there.")
                else:
                    connection.sendall(head + b"Hello")  # then nothing more
                    release.wait(10)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    settings = make_test_settings(
        url_ingest_enabled=True, url_ingest_allow_private=True, url_ingest_timeout_seconds=2
    )
    local = resolver({"local.test": ["127.0.0.1"]})

    def get(path: str) -> Any:
        transport = httpx.HTTPTransport(verify=client_tls) if client_tls else None
        url = f"{scheme}://local.test:{port}{path}"
        return fetch_document(url, settings, resolve=local, transport=transport)

    try:
        assert get("/complete.txt").content == b"Hello there."
        started = time.monotonic()
        with pytest.raises(UrlFetchError, match="did not finish within"):
            get("/stalls.txt")
        # At the deadline (2 s), not a full read timeout after the body began (3.5 s).
        # Midway between the two, so a busy runner has room either side.
        assert time.monotonic() - started < 3.0
    finally:
        release.set()
        server.join(5)
        listener.close()


def test_the_route_refuses_a_server_chosen_name_that_is_already_indexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app, make_settings, wait_until

    clear_dependency_caches()
    app = hermetic_app(make_settings(url_ingest_enabled=True), QdrantClient(":memory:"))

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": "text/plain"}
        if request.url.path == "/download":
            headers["content-disposition"] = 'attachment; filename="handbook.txt"'
        return httpx.Response(200, content=b"Intro\nFrom the web.", headers=headers)

    monkeypatch.setattr(
        "api.routes.documents.fetch_document",
        functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler)),
    )

    def index(body: dict[str, str]) -> None:
        accepted = client.post("/documents/from-url", json=body)
        assert accepted.status_code == 202, accepted.text
        job_id = accepted.json()["job_id"]
        wait_until(lambda: client.get(f"/documents/jobs/{job_id}").json()["state"] == "done")

    with TestClient(app) as client:
        index({"url": "https://example.com/handbook.txt"})
        # The same URL again replaces it: the name is the URL's own.
        index({"url": "https://example.com/handbook.txt"})

        refused = client.post("/documents/from-url", json={"url": "https://example.com/download"})
        assert refused.status_code == 409
        assert "'handbook.txt'" in refused.json()["detail"]

        # Naming it is how to replace (or keep both).
        index({"url": "https://example.com/download", "filename": "handbook.txt"})
        index({"url": "https://example.com/download", "filename": "other.txt"})
        listing = client.get("/documents").json()["filenames"]
        assert sorted(listing) == ["handbook.txt", "other.txt"]
    clear_dependency_caches()


def test_a_conditional_fetch_sends_the_validators_and_reads_a_304() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304, headers={"etag": '"v1"'})
        return httpx.Response(
            200,
            content=b"Hello.",
            headers={"content-type": "text/plain", "etag": '"v1"', "last-modified": "Mon"},
        )

    first = fetch("https://example.com/a.txt", handler)
    assert (first.etag, first.last_modified, first.not_modified) == ('"v1"', "Mon", False)

    again = fetch("https://example.com/a.txt", handler, etag='"v1"', last_modified="Mon")
    assert again.not_modified and again.content == b""
    assert seen[-1].headers["if-modified-since"] == "Mon"
    # Asked for unconditionally, a 304 is an error, never an empty document.
    with pytest.raises(UrlFetchError, match="304"):
        fetch("https://example.com/a.txt", lambda request: httpx.Response(304))


def test_validators_that_cannot_be_sent_back_are_not_kept_or_sent() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            content=b"Hello.",
            headers=[
                (b"content-type", b"text/plain"),
                (b"etag", '"caf\xe9"'.encode("latin-1")),
                (b"last-modified", b"Mon"),
            ],
        )

    # A Latin-1 ETag would be sent back as If-None-Match, which httpx can't encode.
    first = fetch("https://example.com/a.txt", handler)
    assert (first.etag, first.last_modified) == (None, "Mon")
    # One stored before this (or restored from a backup) is left out of the request.
    again = fetch("https://example.com/a.txt", handler, etag='"caf\xe9"', last_modified="Mon")
    assert "if-none-match" not in seen[-1].headers
    assert seen[-1].headers["if-modified-since"] == "Mon"
    assert again.content == b"Hello."

    assert usable_validator('W/"abc"') == 'W/"abc"'
    for unusable in ('"caf\xe9"', "a\r\nX-Injected: 1", "", "x" * 201, None, 7):
        assert usable_validator(unusable) is None


def test_broken_redirects_and_a_cut_off_body_are_clear_errors() -> None:
    with pytest.raises(UrlFetchError, match="redirected nowhere"):
        fetch("https://example.com/a.txt", lambda request: httpx.Response(302))
    # Refused as a fetch error (a 400 at the route), never an unhandled exception.
    with pytest.raises(UrlFetchError):
        fetch(
            "https://example.com/a.txt",
            lambda request: httpx.Response(302, headers={"location": "http://[::1"}),
        )

    def cut_off(request: httpx.Request) -> httpx.Response:
        def body() -> Any:
            yield b"Hello"
            raise httpx.ReadError("connection reset")

        return httpx.Response(200, content=body(), headers={"content-type": "text/plain"})

    with pytest.raises(UrlFetchError) as cut:
        fetch("https://example.com/a.txt", cut_off)
    assert "connection reset" not in str(cut.value)  # no transport detail to the client


def test_the_servers_charset_outranks_the_pages_meta_charset() -> None:
    from api.documents import parse_document_bytes

    body = '<meta charset="utf-8"><p>Отпуск 28 дней.</p>'.encode("cp1251")
    doc = fetch(
        "https://example.com/ru.html",
        lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/html; charset=windows-1251"}
        ),
    )
    (section,) = parse_document_bytes(doc.filename, doc.content)
    assert section.text == "Отпуск 28 дней."
    # Undeclared: left as sent.
    plain = fetch("https://example.com/a.txt", ok(b"Hello.", "text/plain"))
    assert plain.content == b"Hello."


def test_a_fetched_documents_refresh_interval_is_set_listed_and_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait, wait_until

    clear_dependency_caches()
    app = hermetic_app(
        make_settings(url_ingest_enabled=True, url_refresh_interval_minutes=60),
        QdrantClient(":memory:"),
    )
    handler = lambda request: httpx.Response(  # noqa: E731
        200, content=b"Intro\nFrom the web.", headers={"content-type": "text/plain"}
    )
    monkeypatch.setattr(
        "api.routes.documents.fetch_document",
        functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler)),
    )
    with TestClient(app) as client:
        accepted = client.post(
            "/documents/from-url",
            json={"url": "https://example.com/page.txt", "refresh_minutes": 30},
        )
        assert accepted.status_code == 202, accepted.text
        job = accepted.json()["job_id"]
        wait_until(lambda: client.get(f"/documents/jobs/{job}").json()["state"] == "done")
        listing = client.get("/documents").json()
        status = listing["url_refresh"]["page.txt"]
        assert status["refresh_minutes"] == 30 and status["failures"] == 0
        assert status["last_checked"] is None
        assert status["next_due"] == pytest.approx(listing["uploaded_ats"]["page.txt"] + 1800)

        never = client.patch("/documents/page.txt/refresh-settings", json={"refresh_minutes": 0})
        assert never.status_code == 200 and never.json()["refresh_minutes"] == 0
        status = client.get("/documents").json()["url_refresh"]["page.txt"]
        assert status["refresh_minutes"] == 0 and status["next_due"] is None
        default = client.patch("/documents/page.txt/refresh-settings", json={})
        assert default.json()["refresh_minutes"] is None
        assert client.get("/documents").json()["url_refresh"]["page.txt"]["refresh_minutes"] is None

        assert upload_and_wait(client, "upload.txt", b"Intro\nUploaded.")["state"] == "done"
        uploaded = client.patch("/documents/upload.txt/refresh-settings", json={})
        assert uploaded.status_code == 404
        assert client.patch("/documents/nope.txt/refresh-settings", json={}).status_code == 404
        too_long = client.patch(
            "/documents/page.txt/refresh-settings", json={"refresh_minutes": 10**7}
        )
        assert too_long.status_code == 422
        assert "upload.txt" not in client.get("/documents").json()["url_refresh"]
    clear_dependency_caches()


def test_a_failed_url_job_is_fetched_again_on_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait, wait_until

    clear_dependency_caches()
    app = hermetic_app(make_settings(url_ingest_enabled=True), QdrantClient(":memory:"))
    serving = {"body": b""}  # empty first: the job fails ("no extractable text")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=serving["body"], headers={"content-type": "text/plain"})

    monkeypatch.setattr(
        "api.routes.documents.fetch_document",
        functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler)),
    )

    def finished(client: TestClient, job_id: str) -> dict[str, Any]:
        wait_until(
            lambda: client.get(f"/documents/jobs/{job_id}").json()["state"] in ("done", "failed")
        )
        return dict(client.get(f"/documents/jobs/{job_id}").json())

    with TestClient(app) as client:
        accepted = client.post(
            "/documents/from-url", json={"url": "https://example.com/page.txt", "tags": ["hr"]}
        ).json()
        failed = finished(client, accepted["job_id"])
        assert failed["state"] == "failed"
        assert failed["kind"] == "url" and failed["retryable"] is True

        serving["body"] = b"Intro\nFixed at the source."
        retried = client.post(f"/documents/jobs/{accepted['job_id']}/retry")
        assert retried.status_code == 202, retried.text
        assert finished(client, retried.json()["job_id"])["state"] == "done"
        listing = client.get("/documents").json()
        assert listing["tags"]["page.txt"] == ["hr"]
        assert listing["source_urls"]["page.txt"] == "https://example.com/page.txt"

        # A job that didn't fail, an upload (its bytes weren't kept), an unknown job.
        again = client.post(f"/documents/jobs/{retried.json()['job_id']}/retry")
        assert again.status_code == 409 and "Only a failed job" in again.json()["detail"]
        upload = upload_and_wait(client, "empty.txt", b"")
        assert upload["state"] == "failed" and upload["retryable"] is False
        refused = client.post(f"/documents/jobs/{upload['job_id']}/retry")
        assert refused.status_code == 409 and "upload the file again" in refused.json()["detail"]
        assert client.post("/documents/jobs/nope/retry").status_code == 404
    clear_dependency_caches()
