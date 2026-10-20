"""Fetch a document from a URL for ``POST /documents/from-url`` (off by default).

The server makes the request, so the URL is a request *from inside* the deployment:
without a guard, ``http://169.254.169.254/`` or ``http://qdrant:6333/collections`` would
be fetched and indexed for anyone with the key to read back (SSRF). So:

- only ``http``/``https``, on any port, with no credentials in the URL;
- the host is resolved here and every address must be public (``ipaddress.is_global``)
  unless ``URL_INGEST_ALLOW_PRIVATE`` is set — including the IPv4 address inside a NAT64
  (``64:ff9b::/96``), 6to4 (``2002::/16``) or IPv4-mapped one, which reach that IPv4
  host while the IPv6 form itself can read as public;
- the connection goes to an exact address that was checked (IPv4 first, the next one
  when a connection fails) — the host name is kept for the ``Host`` header and TLS (SNI
  and certificate checks) — so a second DNS answer (rebinding) cannot swap in an
  internal one;
- redirects are followed by hand, each hop checked the same way, at most
  ``MAX_REDIRECTS``;
- the body is read in pieces and abandoned past ``MAX_UPLOAD_BYTES``;
- the whole fetch — name lookups, connects, redirects and the body — has one deadline,
  ``URL_INGEST_TIMEOUT_SECONDS``, so a slow resolver or a server dribbling bytes can't
  hold it open: when it passes, the connection is shut down under whatever read was
  waiting (``_Watchdog``).

The name a document is stored under can come from the remote server (its
``Content-Disposition``, or the path of a URL it redirected to); ``FetchedDocument``
says so, and the route refuses such a name when it is already indexed, so a page can't
quietly replace another document by naming itself ``handbook.pdf``.
"""

from __future__ import annotations

import codecs
import ipaddress
import re
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from email.message import Message
from pathlib import PurePosixPath
from urllib.parse import unquote, urljoin, urlsplit

import httpx

from api.documents import SUPPORTED_EXTENSIONS, DocumentError, validate_upload_filename
from api.settings import AppSettings

MAX_REDIRECTS = 3
MAX_URL_CHARS = 2048
_USER_AGENT = "DocRAG-url-ingest"

# What a response's type means when its URL names no supported file.
_TYPE_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
    "application/epub+zip": ".epub",
    "message/rfc822": ".eml",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}

Resolver = Callable[[str, int], list[str]]

# Name lookups run here so they can be waited on with a timeout: getaddrinfo takes
# none, and a resolver that doesn't answer blocks for as long as the system's resolver
# configuration allows. A lookup given up on still occupies its thread until then;
# the pool is small and bounded, so a resolver that hangs makes later lookups time out
# (waiting for a free thread) rather than piling up threads.
_RESOLVER_THREADS = 4
_resolver_pool = ThreadPoolExecutor(
    max_workers=_RESOLVER_THREADS, thread_name_prefix="docrag-resolve"
)

# NAT64's well-known prefix: the last 32 bits are the IPv4 host a NAT64 gateway reaches.
_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")


class UrlFetchError(DocumentError):
    """The URL can't be fetched as a document (a 400, with the reason)."""


@dataclass(frozen=True)
class FetchedDocument:
    filename: str
    content: bytes
    final_url: str
    # The name came from the remote server (Content-Disposition, or a redirect target's
    # path), not from the request's filename or the URL it asked for.
    remote_named: bool = False
    # The server's validators, kept for a later conditional refresh.
    etag: str | None = None
    last_modified: str | None = None
    # A conditional fetch the server answered 304: nothing changed, ``content`` is empty.
    not_modified: bool = False


class BacklogFullError(Exception):
    """The ingest backlog has no room for a document this size (the route's 503)."""


def resolve_host(host: str, port: int) -> list[str]:
    """Every address ``host`` resolves to (IPv4 and IPv6), in the resolver's order."""

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as exc:
        raise UrlFetchError(f"Cannot resolve '{host}'.") from exc
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 host an IPv6 address really reaches, when it carries one."""

    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip in _NAT64:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def is_public_address(address: str) -> bool:
    """Whether ``address`` — and any IPv4 address carried inside it — is public."""

    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if not ip.is_global:
        return False
    # NAT64 of 127.0.0.1 reads as global on every Python this runs on, and 6to4 on
    # some: the IPv6 form says nothing about the IPv4 host behind it.
    embedded = _embedded_ipv4(ip) if isinstance(ip, ipaddress.IPv6Address) else None
    return embedded is None or embedded.is_global


def resolve_within(resolve: Resolver, host: str, port: int, timeout: float) -> list[str]:
    """``resolve(host, port)``, given up after ``timeout`` seconds."""

    future = _resolver_pool.submit(resolve, host, port)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        future.cancel()  # still queued behind hung lookups: never runs
        raise UrlFetchError(
            f"'{host}' could not be resolved within URL_INGEST_TIMEOUT_SECONDS."
        ) from exc


def _checked_addresses(
    host: str, port: int, allow_private: bool, resolve: Resolver, timeout: float
) -> list[str]:
    """The addresses to try, IPv4 first, after checking every address the host has."""

    addresses = resolve_within(resolve, host, port, timeout)
    if not addresses:
        raise UrlFetchError(f"Cannot resolve '{host}'.")
    for address in addresses:
        try:
            public = is_public_address(address)
        except ValueError as exc:
            raise UrlFetchError(f"Cannot resolve '{host}'.") from exc
        if not allow_private and not public:
            raise UrlFetchError(
                f"'{host}' is a private, loopback or reserved address; set "
                "URL_INGEST_ALLOW_PRIVATE to fetch from the local network."
            )
    # IPv4 first (a stable sort keeps the resolver's order otherwise): plenty of hosts
    # publish an AAAA record a server without IPv6 routing can't reach.
    return sorted(addresses, key=lambda address: ":" in address)


def _validated(url: str) -> tuple[str, str, int]:
    """``(scheme, host, port)`` of a URL the fetcher will try, or a 400. The host is in
    its ASCII (IDNA) form, as DNS, the ``Host`` header and TLS SNI all need it."""

    if len(url) > MAX_URL_CHARS:
        raise UrlFetchError(f"The URL is longer than {MAX_URL_CHARS} characters.")
    try:
        parts = urlsplit(url.strip())
        hostname = parts.hostname
    except ValueError as exc:  # an unclosed "[", an invalid IPv6 literal
        raise UrlFetchError("The URL is not valid.") from exc
    try:
        port_value = parts.port
    except ValueError as exc:
        raise UrlFetchError("The URL's port is not valid.") from exc
    if parts.scheme not in ("http", "https"):
        raise UrlFetchError("Only http and https URLs can be fetched.")
    if parts.username or parts.password:
        raise UrlFetchError("A URL with credentials in it is not fetched.")
    if not hostname:
        raise UrlFetchError("The URL has no host.")
    try:
        host = hostname.encode("idna").decode("ascii") if not hostname.isascii() else hostname
    except UnicodeError as exc:
        raise UrlFetchError("The URL's host name is not valid.") from exc
    port = port_value or (443 if parts.scheme == "https" else 80)
    return parts.scheme, host, port


class _Watchdog:
    """Shuts the fetch's connections down when the deadline passes.

    httpx's timeouts are per operation — each read may wait the full read timeout —
    and httpcore takes the read timeout once when the body starts, so a server sending
    a byte every few seconds, or one slow read near the end, could run past the
    deadline by up to the whole timeout again. A timer fires at the deadline and shuts
    down every connection the fetch opened (httpcore hands each socket over through the
    ``trace`` request extension as it connects); the read waiting on it then fails at
    once. Shutting a socket down from another thread is safe — it is ``close`` that
    isn't, and that stays with httpcore. ``fired`` tells that failure from any other.

    The watchdog keeps a duplicate of each socket's descriptor, not the socket itself:
    TLS (``ssl_context.wrap_socket``) detaches the original, leaving it with no
    descriptor, and an https fetch was then never cut off. ``shutdown`` acts on the
    connection, not the descriptor, so the duplicate ends it whether TLS took it over
    or not, handshake included. ``cancel`` closes the duplicates.
    """

    def __init__(self, seconds: float) -> None:
        self._lock = threading.Lock()
        self._sockets: list[socket.socket] = []
        self._cancelled = False
        self.fired = False
        self._timer = threading.Timer(max(seconds, 0.0), self._fire)
        self._timer.daemon = True
        self._timer.start()

    def trace(self, event: str, info: dict[str, object]) -> None:
        if event != "connection.connect_tcp.complete":
            return
        stream = info.get("return_value")
        get_extra_info = getattr(stream, "get_extra_info", None)
        sock = get_extra_info("socket") if callable(get_extra_info) else None
        if not isinstance(sock, socket.socket):
            return
        try:
            duplicate = sock.dup()
        except OSError:
            return
        # Shut down and closed only under the lock (both are quick): a descriptor
        # closed by cancel while _fire still held it could be reused by another
        # connection in between, which _fire would then cut.
        with self._lock:
            if self._cancelled:
                duplicate.close()
                return
            self._sockets.append(duplicate)
            if self.fired:
                _shut_down(duplicate)  # connected just as the deadline passed

    def _fire(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self.fired = True
            for sock in self._sockets:
                _shut_down(sock)

    def cancel(self) -> None:
        self._timer.cancel()
        with self._lock:
            self._cancelled = True
            for sock in self._sockets:
                sock.close()
            self._sockets = []


def _shut_down(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # the connection is already gone


def _pinned_request(
    client: httpx.Client,
    url: str,
    address: str,
    host: str,
    port: int,
    timeout: float,
    trace: Callable[[str, dict[str, object]], None] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> httpx.Request:
    """A GET to ``address`` that still presents ``host`` (Host header, TLS SNI)."""

    parts = urlsplit(url)
    literal = f"[{address}]" if ":" in address else address
    target = parts._replace(netloc=f"{literal}:{port}").geturl()
    default_port = 443 if parts.scheme == "https" else 80
    # An IPv6 literal is bracketed in a Host header (RFC 3986), or "::1:8080" would
    # read as one address with no port.
    named = f"[{host}]" if ":" in host else host
    host_header = named if port == default_port else f"{named}:{port}"
    return client.build_request(
        "GET",
        target,
        headers={
            **(extra_headers or {}),
            "Host": host_header,
            "User-Agent": _USER_AGENT,
            "Accept": "*/*",
        },
        extensions={"sni_hostname": host, **({"trace": trace} if trace else {})},
        timeout=httpx.Timeout(timeout),
    )


# Validators longer than this aren't kept (they are stored on every chunk).
_MAX_VALIDATOR_CHARS = 200


def _header_value(value: str | None) -> str | None:
    if not value or len(value) > _MAX_VALIDATOR_CHARS:
        return None
    return value


def _conditional_headers(etag: str | None, last_modified: str | None) -> dict[str, str] | None:
    headers = {}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    return headers or None


def _with_declared_charset(body: bytes, content_type: str) -> bytes:
    """A text document re-encoded as UTF-8 (with a byte-order mark) when the server's
    Content-Type names its charset: that outranks a <meta charset> inside the page, as
    in a browser, and the parser trusts a BOM above both. Undeclared, or a charset
    Python doesn't know, or bytes that don't decode under it: left as they are."""

    media, _, params = content_type.partition(";")
    media = media.strip().lower()
    if not (media.startswith("text/") or media in ("application/xhtml+xml", "application/xml")):
        return body
    match = re.search(r"charset\s*=\s*\"?([A-Za-z0-9_.:-]+)", params, re.IGNORECASE)
    if match is None or body.startswith(codecs.BOM_UTF8):
        return body
    try:
        text = body.decode(match.group(1))
    except (LookupError, UnicodeDecodeError):
        return body
    return codecs.BOM_UTF8 + text.encode("utf-8")


def loggable_url(url: str) -> str:
    """``url`` without its query string or fragment, which can carry signed tokens."""

    try:
        return urlsplit(url)._replace(query="", fragment="").geturl()
    except ValueError:
        return "<unparseable URL>"


def _requested_name(url: str, content_type: str) -> str | None:
    """The name the requested URL itself gives (what re-adding it should replace), or
    None when it gives none DocRAG can use."""

    try:
        return choose_filename(url.strip(), content_type, None, None)
    except (DocumentError, ValueError):
        return None


def _name_from_disposition(value: str | None) -> str | None:
    if not value:
        return None
    message = Message()
    message["content-disposition"] = value
    name = message.get_filename()
    return name.strip() or None if name else None


def choose_filename(url: str, content_type: str, disposition: str | None, given: str | None) -> str:
    """The stored name: the one asked for, else the server's, else the URL's last path
    segment — with an extension from the content type when that names no supported one."""

    content_kind = content_type.split(";", 1)[0].strip().lower()
    candidate = (given or "").strip() or _name_from_disposition(disposition) or ""
    if not candidate:
        path = PurePosixPath(unquote(urlsplit(url).path))
        candidate = path.name or (urlsplit(url).hostname or "document")
    candidate = re.sub(r"[\x00-\x1f/\\]+", "_", candidate)
    if PurePosixPath(candidate).suffix.lower() not in SUPPORTED_EXTENSIONS:
        extension = _TYPE_EXTENSIONS.get(content_kind)
        if extension is None:
            raise UrlFetchError(
                f"The URL returned '{content_kind or 'no content type'}', which DocRAG "
                "can't index; name the file with a supported extension to override."
            )
        candidate = f"{candidate}{extension}"
    return validate_upload_filename(candidate)


def fetch_document(
    url: str,
    settings: AppSettings,
    *,
    filename: str | None = None,
    resolve: Resolver = resolve_host,
    transport: httpx.BaseTransport | None = None,
    has_room: Callable[[int], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
    etag: str | None = None,
    last_modified: str | None = None,
) -> FetchedDocument:
    """Download ``url`` within the guards above; blocking (run it off the event loop).

    ``has_room(size)`` is asked once the final response's headers are in, with its
    declared length (``MAX_UPLOAD_BYTES`` when it declares none), before any of the
    body is read: a full ingest backlog raises ``BacklogFullError`` then, rather than
    after downloading a document it can only refuse.

    ``etag``/``last_modified`` (from an earlier fetch) make it conditional: a 304
    returns a ``not_modified`` document with no content.
    """

    limit = int(settings.max_upload_bytes)
    allow_private = bool(settings.url_ingest_allow_private)
    deadline = clock() + float(settings.url_ingest_timeout_seconds)
    current = url.strip()

    def remaining(host: str) -> float:
        left = deadline - clock()
        if left <= 0:
            raise UrlFetchError(f"'{host}' did not finish within URL_INGEST_TIMEOUT_SECONDS.")
        return left

    # Each lookup, connect and read gets what is left of the deadline as its timeout,
    # the body loop checks it between pieces, and the watchdog ends whatever is still
    # waiting when it passes.
    watchdog = _Watchdog(deadline - clock())
    try:
        return _fetch(
            url,
            current,
            filename,
            limit,
            allow_private,
            resolve,
            transport,
            has_room,
            remaining,
            watchdog,
            _conditional_headers(etag, last_modified),
        )
    except (UrlFetchError, httpx.HTTPError) as exc:
        if not watchdog.fired:
            raise
        host = urlsplit(current).hostname or "the server"
        raise UrlFetchError(f"'{host}' did not finish within URL_INGEST_TIMEOUT_SECONDS.") from exc
    finally:
        watchdog.cancel()


def _fetch(
    url: str,
    current: str,
    filename: str | None,
    limit: int,
    allow_private: bool,
    resolve: Resolver,
    transport: httpx.BaseTransport | None,
    has_room: Callable[[int], bool] | None,
    remaining: Callable[[str], float],
    watchdog: _Watchdog,
    conditional: dict[str, str] | None = None,
) -> FetchedDocument:
    with httpx.Client(follow_redirects=False, transport=transport) as client:
        for _ in range(MAX_REDIRECTS + 1):
            _, host, port = _validated(current)
            response = _connect(
                client,
                current,
                host,
                port,
                allow_private,
                resolve,
                remaining,
                watchdog.trace,
                conditional,
            )
            try:
                if response.status_code == 304 and conditional:
                    # Before the redirect check: httpx counts every 3xx as a redirect, 304
                    # included, and a 304 has no Location.
                    return FetchedDocument(
                        filename or "",
                        b"",
                        current,
                        etag=response.headers.get("etag") or conditional.get("If-None-Match"),
                        last_modified=response.headers.get("last-modified")
                        or conditional.get("If-Modified-Since"),
                        not_modified=True,
                    )
                if response.is_redirect and response.status_code != 304:
                    location = response.headers.get("location")
                    if not location:
                        raise UrlFetchError("The URL redirected nowhere.")
                    try:
                        current = urljoin(current, location)
                    except ValueError as exc:
                        raise UrlFetchError("The URL redirected to an invalid URL.") from exc
                    continue
                if response.status_code == 304 or response.status_code >= 400:
                    raise UrlFetchError(
                        f"The URL answered {response.status_code}; nothing was fetched."
                    )
                declared = response.headers.get("content-length")
                size = int(declared) if declared and declared.isdigit() else None
                if size is not None and size > limit:
                    raise UrlFetchError(f"The document is larger than {limit} bytes.")
                if has_room is not None and not has_room(size if size is not None else limit):
                    raise BacklogFullError
                body = _with_declared_charset(
                    _read_body(response, limit, host, remaining),
                    response.headers.get("content-type", ""),
                )
                disposition = response.headers.get("content-disposition")
                name = choose_filename(
                    current, response.headers.get("content-type", ""), disposition, filename
                )
                remote_named = not (filename or "").strip() and name != _requested_name(
                    url, response.headers.get("content-type", "")
                )
                return FetchedDocument(
                    name,
                    body,
                    current,
                    remote_named=remote_named,
                    etag=_header_value(response.headers.get("etag")),
                    last_modified=_header_value(response.headers.get("last-modified")),
                )
            finally:
                response.close()
    raise UrlFetchError(f"The URL redirected more than {MAX_REDIRECTS} times.")


def _connect(
    client: httpx.Client,
    url: str,
    host: str,
    port: int,
    allow_private: bool,
    resolve: Resolver,
    remaining: Callable[[str], float],
    trace: Callable[[str, dict[str, object]], None] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> httpx.Response:
    """The response from the first checked address that accepts the connection."""

    addresses = _checked_addresses(host, port, allow_private, resolve, remaining(host))
    for index, address in enumerate(addresses):
        request = _pinned_request(
            client, url, address, host, port, remaining(host), trace, extra_headers
        )
        try:
            return client.send(request, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # A host with an unreachable address (often its IPv6 one) and a working
            # one: try the next, as a browser would.
            if index + 1 < len(addresses):
                continue
            if isinstance(exc, httpx.ConnectTimeout):
                raise UrlFetchError(
                    f"'{host}' did not answer within URL_INGEST_TIMEOUT_SECONDS."
                ) from exc
            raise UrlFetchError(f"Could not fetch from '{host}'.") from exc
        except httpx.TimeoutException as exc:
            raise UrlFetchError(
                f"'{host}' did not answer within URL_INGEST_TIMEOUT_SECONDS."
            ) from exc
        except httpx.HTTPError as exc:
            raise UrlFetchError(f"Could not fetch from '{host}'.") from exc
    raise UrlFetchError(f"Could not fetch from '{host}'.")  # pragma: no cover - never empty


def _read_body(
    response: httpx.Response, limit: int, host: str, remaining: Callable[[str], float]
) -> bytes:
    """The body, abandoned past ``limit`` bytes or the deadline. A connection that
    breaks or stalls mid-body is the URL's failure (a 400 naming it), not a 500."""

    body = bytearray()
    try:
        for piece in response.iter_bytes():
            body.extend(piece)
            if len(body) > limit:
                raise UrlFetchError(f"The document is larger than {limit} bytes.")
            remaining(host)
    except httpx.TimeoutException as exc:
        raise UrlFetchError(f"'{host}' did not finish within URL_INGEST_TIMEOUT_SECONDS.") from exc
    except httpx.HTTPError as exc:
        raise UrlFetchError(f"The download from '{host}' broke off.") from exc
    return bytes(body)
