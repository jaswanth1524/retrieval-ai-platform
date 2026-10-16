"""Fetch a document from a URL for ``POST /documents/from-url`` (off by default).

The server makes the request, so the URL is a request *from inside* the deployment:
without a guard, ``http://169.254.169.254/`` or ``http://qdrant:6333/collections`` would
be fetched and indexed for anyone with the key to read back (SSRF). So:

- only ``http``/``https``, on any port, with no credentials in the URL;
- the host is resolved here and every address must be public (``ipaddress.is_global``)
  unless ``URL_INGEST_ALLOW_PRIVATE`` is set;
- the connection goes to the exact address that was checked — the host name is kept for
  the ``Host`` header and TLS (SNI and certificate checks) — so a second DNS answer
  (rebinding) cannot swap in an internal one;
- redirects are followed by hand, each hop checked the same way, at most
  ``MAX_REDIRECTS``;
- the body is read in pieces and abandoned past ``MAX_UPLOAD_BYTES``.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
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


class UrlFetchError(DocumentError):
    """The URL can't be fetched as a document (a 400, with the reason)."""


@dataclass(frozen=True)
class FetchedDocument:
    filename: str
    content: bytes
    final_url: str


def resolve_host(host: str, port: int) -> list[str]:
    """Every address ``host`` resolves to (IPv4 and IPv6)."""

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlFetchError(f"Cannot resolve '{host}'.") from exc
    return sorted({str(info[4][0]) for info in infos})


def _checked_address(host: str, port: int, allow_private: bool, resolve: Resolver) -> str:
    """The address to connect to, after checking every address the host has."""

    addresses = resolve(host, port)
    if not addresses:
        raise UrlFetchError(f"Cannot resolve '{host}'.")
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if not allow_private and not ip.is_global:
            raise UrlFetchError(
                f"'{host}' is a private, loopback or reserved address; set "
                "URL_INGEST_ALLOW_PRIVATE to fetch from the local network."
            )
    return addresses[0]


def _validated(url: str) -> tuple[str, str, int]:
    """``(scheme, host, port)`` of a URL the fetcher will try, or a 400."""

    if len(url) > MAX_URL_CHARS:
        raise UrlFetchError(f"The URL is longer than {MAX_URL_CHARS} characters.")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise UrlFetchError("Only http and https URLs can be fetched.")
    if parts.username or parts.password:
        raise UrlFetchError("A URL with credentials in it is not fetched.")
    host = parts.hostname
    if not host:
        raise UrlFetchError("The URL has no host.")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise UrlFetchError("The URL's port is not valid.") from exc
    return parts.scheme, host, port


def _pinned_request(
    client: httpx.Client, url: str, address: str, host: str, port: int
) -> httpx.Request:
    """A GET to ``address`` that still presents ``host`` (Host header, TLS SNI)."""

    parts = urlsplit(url)
    literal = f"[{address}]" if ":" in address else address
    target = parts._replace(netloc=f"{literal}:{port}").geturl()
    default_port = 443 if parts.scheme == "https" else 80
    host_header = host if port == default_port else f"{host}:{port}"
    return client.build_request(
        "GET",
        target,
        headers={"Host": host_header, "User-Agent": _USER_AGENT, "Accept": "*/*"},
        extensions={"sni_hostname": host},
    )


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
) -> FetchedDocument:
    """Download ``url`` within the guards above; blocking (run it off the event loop)."""

    limit = int(settings.max_upload_bytes)
    allow_private = bool(settings.url_ingest_allow_private)
    timeout = httpx.Timeout(float(settings.url_ingest_timeout_seconds))
    current = url.strip()
    with httpx.Client(timeout=timeout, follow_redirects=False, transport=transport) as client:
        for _ in range(MAX_REDIRECTS + 1):
            _, host, port = _validated(current)
            address = _checked_address(host, port, allow_private, resolve)
            request = _pinned_request(client, current, address, host, port)
            try:
                response = client.send(request, stream=True)
            except httpx.TimeoutException as exc:
                raise UrlFetchError(
                    f"'{host}' did not answer within URL_INGEST_TIMEOUT_SECONDS."
                ) from exc
            except httpx.HTTPError as exc:
                raise UrlFetchError(f"Could not fetch from '{host}'.") from exc
            try:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise UrlFetchError("The URL redirected nowhere.")
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise UrlFetchError(
                        f"The URL answered {response.status_code}; nothing was fetched."
                    )
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > limit:
                    raise UrlFetchError(f"The document is larger than {limit} bytes.")
                body = bytearray()
                for piece in response.iter_bytes():
                    body.extend(piece)
                    if len(body) > limit:
                        raise UrlFetchError(f"The document is larger than {limit} bytes.")
                name = choose_filename(
                    current,
                    response.headers.get("content-type", ""),
                    response.headers.get("content-disposition"),
                    filename,
                )
                return FetchedDocument(name, bytes(body), current)
            finally:
                response.close()
    raise UrlFetchError(f"The URL redirected more than {MAX_REDIRECTS} times.")
