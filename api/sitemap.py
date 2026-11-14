"""Adding a site's pages from its sitemap or feed (``POST /documents/from-sitemap``).

The sitemap (or sitemap index, RSS or Atom feed) is read when the request arrives; the
pages it lists are then fetched one by one in the background, each through the same
guarded fetch as ``/documents/from-url`` (public addresses only, size and time caps,
the shared fetch slots) and queued as a ``url`` job, so they show in the job list as
they go. Only pages on the sitemap's own host by default, at most SITEMAP_MAX_PAGES.
"""

from __future__ import annotations

import email.utils
import hashlib
import logging
import re
import threading
import time
import uuid
import zlib
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePosixPath
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree

from api.audit import audit
from api.documents import DocumentError, DocumentSource
from api.ingest_jobs import IngestQueue
from api.parsers.common import declares_dtd
from api.raw_documents import RawDocumentStore
from api.url_fetch import BacklogFullError, FetchedDocument, fetch_slots, loggable_url
from api.workspaces import workspace_of

logger = logging.getLogger(__name__)

# Child sitemaps of a sitemap index read, at most.
MAX_CHILD_SITEMAPS = 10
# A full indexing queue is waited out this many times, this long each, per page.
_BACKLOG_RETRIES = 30
_BACKLOG_WAIT_SECONDS = 10.0

_pending_lock = threading.Lock()
# One import at a time, its pages one after another: a sitemap of a hundred pages must
# not take every fetch slot from the users adding single URLs.
_pool: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _pending_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docrag-sitemap")
        return _pool


class SitemapError(DocumentError):
    """Not a sitemap or feed this reads (a 400)."""


@dataclass(frozen=True)
class SitemapPages:
    urls: list[str]
    # Listed but left out: another host, not http(s), a duplicate, or past the cap.
    left_out: int
    # The listing was a sitemap index; these are its sitemaps, read for pages.
    child_sitemaps: list[str]
    # Each page's last change as the listing gives it (pages without one are absent).
    lastmods: dict[str, float] = field(default_factory=dict)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def listed_urls(xml: bytes, base_url: str) -> tuple[list[str], list[str]]:
    """The page URLs a sitemap, RSS or Atom feed lists, and (for a sitemap index) its
    child sitemaps, in document order."""

    entries, children = listed_entries(xml, base_url)
    return [url for url, _ in entries], children


_DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _parse_lastmod(text: str | None) -> float | None:
    """A sitemap ``<lastmod>`` / Atom ``<updated>`` (W3C datetime or a date) or an RSS
    ``<pubDate>`` (RFC 822) as epoch seconds; None when absent or unreadable."""

    value = (text or "").strip()
    if not value:
        return None
    # A date with no time ("2026-11-07", common in sitemaps) could be any time that day:
    # read as its end, so a page changed the afternoon after it was indexed that morning
    # isn't skipped as unchanged.
    if _DATE_ONLY_RE.fullmatch(value):
        value += "T23:59:59+00:00"
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        try:
            moment = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.timestamp()


def listed_entries(xml: bytes, base_url: str) -> tuple[list[tuple[str, float | None]], list[str]]:
    """``listed_urls`` with each page's last change as the listing gives it (sitemap
    ``<lastmod>``, RSS ``<pubDate>``, Atom ``<updated>``; None without one)."""

    if declares_dtd(xml):
        raise SitemapError("The sitemap declares a DTD or entities; it was not read.")
    try:
        root = ElementTree.fromstring(xml)
    except (ElementTree.ParseError, RecursionError) as exc:
        raise SitemapError("That URL is not a sitemap or feed (not XML).") from exc
    kind = _local(root.tag)
    pages: list[str] = []
    changed: list[float | None] = []
    children: list[str] = []

    def text_of(entry: ElementTree.Element, name: str) -> str | None:
        found = next((child for child in entry if _local(child.tag) == name), None)
        return found.text if found is not None else None

    if kind == "urlset":
        for entry in root:
            if _local(entry.tag) != "url":
                continue
            pages.append((text_of(entry, "loc") or "").strip())
            changed.append(_parse_lastmod(text_of(entry, "lastmod")))
    elif kind == "sitemapindex":
        children = [
            (loc.text or "").strip()
            for entry in root
            if _local(entry.tag) == "sitemap"
            for loc in entry
            if _local(loc.tag) == "loc"
        ]
    elif kind == "rss":
        for item in root.iter():
            if _local(item.tag) != "item":
                continue
            for link in item:
                if _local(link.tag) == "link":
                    pages.append((link.text or "").strip())
                    changed.append(_parse_lastmod(text_of(item, "pubDate")))
    elif kind == "feed":
        for entry in root:
            if _local(entry.tag) != "entry":
                continue
            links = [link for link in entry if _local(link.tag) == "link"]
            chosen = next((link for link in links if link.get("rel") in (None, "alternate")), None)
            if chosen is not None and chosen.get("href"):
                pages.append(chosen.get("href", "").strip())
                changed.append(_parse_lastmod(text_of(entry, "updated")))
    else:
        raise SitemapError("That URL is not a sitemap, sitemap index, RSS or Atom feed.")
    return (
        [(urljoin(base_url, url), when) for url, when in zip(pages, changed, strict=True) if url],
        [urljoin(base_url, url) for url in children if url],
    )


def select_pages(
    urls: list[str], base_url: str, *, same_host: bool, limit: int
) -> tuple[list[str], int]:
    """The pages to fetch: http(s), on the sitemap's host when ``same_host``, each once,
    at most ``limit``; and how many were left out."""

    host = (urlsplit(base_url).hostname or "").lower()
    kept: list[str] = []
    seen: set[str] = set()
    for url in urls:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or url in seen:
            continue
        if same_host and (parts.hostname or "").lower() != host:
            continue
        seen.add(url)
        kept.append(url)
    return kept[:limit], len(urls) - len(kept[:limit])


def read_sitemap(
    url: str,
    fetch: Callable[[str], FetchedDocument],
    *,
    same_host: bool,
    limit: int,
) -> SitemapPages:
    """The pages ``url`` lists (one level of a sitemap index followed). A gzipped
    sitemap (``.xml.gz``) is read; a site's ``robots.txt`` — named, or found at the
    root when the root itself isn't a sitemap — gives its ``Sitemap:`` lines."""

    listing = fetch(url)
    if _is_robots(listing.final_url):
        entries: list[tuple[str, float | None]] = []
        children = _robots_sitemaps(_gunzipped(listing.content), listing.final_url)
        if not children:
            raise SitemapError("That robots.txt names no sitemap.")
    else:
        try:
            entries, children = listed_entries(_gunzipped(listing.content), listing.final_url)
        except SitemapError:
            if urlsplit(listing.final_url).path not in ("", "/"):
                raise
            # A site's address: its robots.txt says where the sitemaps are.
            robots = fetch(urljoin(listing.final_url, "/robots.txt"))
            entries = []
            children = _robots_sitemaps(robots.content, robots.final_url)
            if not children:
                raise
    for child in children[:MAX_CHILD_SITEMAPS]:
        if len(entries) >= limit:
            break
        try:
            more, _ = listed_entries(_gunzipped(fetch(child).content), child)
        except (DocumentError, OSError) as exc:
            logger.warning("Sitemap %s could not be read: %s", loggable_url(child), exc)
            continue
        entries.extend(more)
    pages = [page for page, _ in entries]
    selected, left_out = select_pages(pages, listing.final_url, same_host=same_host, limit=limit)
    chosen = set(selected)
    lastmods = {page: when for page, when in entries if page in chosen and when is not None}
    return SitemapPages(selected, left_out, children[:MAX_CHILD_SITEMAPS], lastmods)


# A sitemap decompresses to at most this much (the protocol's own limit is 50 MB).
MAX_SITEMAP_BYTES = 50 * 1024 * 1024


def _gunzipped(content: bytes) -> bytes:
    """A gzipped sitemap's XML (``sitemap.xml.gz``, served as a file, not as
    Content-Encoding), within ``MAX_SITEMAP_BYTES``; anything else as it is."""

    if not content.startswith(b"\x1f\x8b"):
        return content
    decoder = zlib.decompressobj(wbits=31)
    try:
        data = decoder.decompress(content, MAX_SITEMAP_BYTES + 1)
    except zlib.error as exc:
        raise SitemapError("The gzipped sitemap is damaged.") from exc
    if len(data) > MAX_SITEMAP_BYTES or decoder.unconsumed_tail:
        raise SitemapError("The gzipped sitemap expands past 50 MB.")
    return data


def _is_robots(url: str) -> bool:
    return urlsplit(url).path.lower().endswith("/robots.txt")


def _robots_sitemaps(content: bytes, base_url: str) -> list[str]:
    """The ``Sitemap:`` lines of a robots.txt, in order."""

    text = content.decode("utf-8", errors="replace")
    found: list[str] = []
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name.strip().lower() == "sitemap" and value.strip():
            found.append(urljoin(base_url, value.strip()))
    return list(dict.fromkeys(found))


# A stored name's limit (``validate_upload_filename``), in UTF-8 bytes.
_MAX_NAME_BYTES = 255


def page_name(url: str, name: str, known: Mapping[str, str | None]) -> str:
    """The name a sitemap page is stored under: ``name`` (what from-url would pick)
    unless a document from another source already holds it; then one made from the
    URL's host and path ("example.com_docs_intro.html"), with a short hash of the URL
    when even that is taken."""

    if name not in known or known[name] == url:
        return name
    extension = PurePosixPath(name).suffix
    parts = urlsplit(url)
    segments = [segment for segment in unquote(parts.path).split("/") if segment]
    if segments and segments[-1].lower().endswith(extension.lower()) and extension:
        segments[-1] = segments[-1][: -len(extension)]
    stem = "_".join([parts.hostname or "page", *segments])
    if parts.query:
        stem += "_" + parts.query
    stem = re.sub(r"[^\w.\-]+", "_", stem).strip("._") or "page"
    candidate = _within_limit(stem, extension)
    if candidate in known and known[candidate] != url:
        digest = hashlib.sha256(url.encode()).hexdigest()[:8]
        candidate = _within_limit(stem, f"_{digest}{extension}")
    return candidate


def _within_limit(stem: str, suffix: str) -> str:
    room = _MAX_NAME_BYTES - len(suffix.encode())
    return stem.encode()[:room].decode(errors="ignore") + suffix


@dataclass(frozen=True)
class KnownPage:
    """A page already indexed from this URL: when, and the validators its last fetch
    left (for a conditional fetch)."""

    uploaded_at: float | None
    etag: str | None = None
    last_modified: str | None = None


# (url, has_room, etag, last_modified) -> the fetched page (``not_modified`` on a 304).
PageFetch = Callable[[str, Callable[[int], bool], str | None, str | None], FetchedDocument]


@dataclass(frozen=True)
class PageImport:
    """What fetching one sitemap's pages needs: where to queue them, how to fetch."""

    queue: IngestQueue
    raw_store: RawDocumentStore | None
    fetch: PageFetch
    tags: tuple[str, ...] | None
    refresh_minutes: int | None
    # Every name in use, with the URL its document came from (None: an upload, or one
    # in flight): a page whose plain name holds another source's document is renamed.
    known: dict[str, str | None]
    # Pages indexed from these URLs before: skipped when the listing's last change is
    # no newer than that, else fetched conditionally (304: unchanged, not downloaded).
    indexed: Mapping[str, KnownPage] = field(default_factory=dict)
    lastmods: Mapping[str, float] = field(default_factory=dict)
    # (name, url) -> the check run under the filename lock before a page is written:
    # ``known`` was read when the import began, and an upload (or another import) that
    # took the name since must not be replaced by a page queued minutes later.
    still_ours: Callable[[str, str], Callable[[], None]] | None = None


# Problems kept per import (each "url: why"), at most.
_MAX_PROBLEMS = 50
# Imports remembered (newest), finished or not.
MAX_IMPORTS_KEPT = 50


@dataclass
class ImportStatus:
    """One sitemap import as it goes: GET /documents/sitemap-imports/{id}."""

    id: str
    url: str
    workspace: str
    total: int
    started_at: float
    queued: int = 0
    failed: int = 0
    skipped: int = 0
    # Already indexed and not changed since (the listing's date, or a 304).
    unchanged: int = 0
    # Left when the import was cancelled.
    cancelled_pages: int = 0
    finished_at: float | None = None
    problems: list[str] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def done(self) -> int:
        return self.queued + self.failed + self.skipped + self.unchanged + self.cancelled_pages

    @property
    def state(self) -> str:
        if self.finished_at is None:
            return "cancelling" if self.cancel.is_set() else "running"
        return "cancelled" if self.cancelled_pages else "done"

    def note(self, url: str, why: str) -> None:
        if len(self.problems) < _MAX_PROBLEMS:
            self.problems.append(f"{loggable_url(url)}: {why}")


_imports: OrderedDict[str, ImportStatus] = OrderedDict()


def import_status(import_id: str, workspace: str) -> ImportStatus | None:
    """An import of ``workspace``'s (another workspace's reads as unknown)."""

    with _pending_lock:
        status = _imports.get(import_id)
    return status if status is not None and status.workspace == workspace else None


def imports_of(workspace: str) -> list[ImportStatus]:
    """``workspace``'s remembered imports, newest first."""

    with _pending_lock:
        return [status for status in reversed(_imports.values()) if status.workspace == workspace]


def _remember(status: ImportStatus) -> None:
    with _pending_lock:
        _imports[status.id] = status
        while len(_imports) > MAX_IMPORTS_KEPT:
            oldest = next(
                (key for key, kept in _imports.items() if kept.finished_at is not None), None
            )
            if oldest is None:
                break
            del _imports[oldest]


# Imports queued or running, by collection: a workspace can't be deleted under one
# (its later pages would be queued into a workspace already gone).
_pending: dict[str, int] = {}


def importing_into(collection: str) -> bool:
    """A sitemap import for ``collection`` is queued or still fetching pages."""

    with _pending_lock:
        return _pending.get(collection, 0) > 0


def _collection_of(job: PageImport) -> str:
    return str(job.queue.ingest_service.settings.qdrant_collection)


def start_page_import(pages: list[str], job: PageImport, url: str = "") -> ImportStatus:
    """Fetch and queue ``pages`` in the background, one after another; the import's
    status, which it updates as it goes."""

    collection = _collection_of(job)
    status = ImportStatus(
        id=str(uuid.uuid4()),
        url=url,
        workspace=workspace_of(job.queue.ingest_service.settings),
        total=len(pages),
        started_at=time.time(),
    )
    _remember(status)
    with _pending_lock:
        _pending[collection] = _pending.get(collection, 0) + 1

    def run() -> None:
        try:
            _import_pages(pages, job, status=status)
        finally:
            status.finished_at = time.time()
            with _pending_lock:
                left = _pending.get(collection, 1) - 1
                if left > 0:
                    _pending[collection] = left
                else:
                    _pending.pop(collection, None)

    try:
        _executor().submit(run)
    except RuntimeError:  # shutting down
        status.cancelled_pages = len(pages)
        status.finished_at = time.time()
        with _pending_lock:
            _pending[collection] = _pending.get(collection, 1) - 1
            if _pending[collection] <= 0:
                _pending.pop(collection, None)
    return status


def shutdown_imports() -> None:
    """Stop the import worker at shutdown: pages not yet fetched are dropped (the
    interpreter would otherwise wait for every one of them on exit). A later import
    (another app in the same process) starts a new worker."""

    global _pool
    with _pending_lock:
        pool, _pool = _pool, None
        running = list(_imports.values())
    for status in running:
        status.cancel.set()
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def _import_pages(
    pages: list[str],
    job: PageImport,
    sleep: Callable[[float], None] = time.sleep,
    status: ImportStatus | None = None,
) -> dict[str, int]:
    from api.document_names import refuse_near_duplicate_name

    status = status or ImportStatus("", "", "", len(pages), time.time())
    settings = job.queue.ingest_service.settings
    for index, url in enumerate(pages):
        if status.cancel.is_set():
            status.cancelled_pages = len(pages) - index
            break
        before = job.indexed.get(url)
        listed = job.lastmods.get(url)
        if (
            before is not None
            and before.uploaded_at is not None
            and listed is not None
            and listed <= before.uploaded_at
        ):
            status.unchanged += 1
            continue
        fetched: FetchedDocument | None = None
        for _ in range(_BACKLOG_RETRIES):
            try:
                with fetch_slots:
                    fetched = job.fetch(
                        url,
                        job.queue.backlog.has_room,
                        before.etag if before is not None else None,
                        before.last_modified if before is not None else None,
                    )
                break
            except BacklogFullError:
                if status.cancel.is_set():
                    break
                sleep(_BACKLOG_WAIT_SECONDS)
            except Exception as exc:  # noqa: BLE001 - one page's failure is its own
                logger.warning("Sitemap page %s was not fetched: %s", loggable_url(url), exc)
                status.note(url, str(exc)[:200] or type(exc).__name__)
                break
        if fetched is None:
            if status.cancel.is_set():
                status.cancelled_pages = len(pages) - index
                break
            status.failed += 1
            continue
        if fetched.not_modified:
            status.unchanged += 1
            continue
        try:
            # /docs/intro/ and /blog/intro/ are both "intro.html": a page never replaces a
            # document from another source (another page, an upload) — it is named by
            # its URL instead. The same URL updates its own document.
            filename = page_name(url, fetched.filename, job.known)
            if job.raw_store is not None and filename not in job.known:
                refuse_near_duplicate_name(filename, job.known)
            accepted = job.queue.enqueue(
                filename=filename,
                content=fetched.content,
                raw_store=job.raw_store,
                source=DocumentSource(
                    url, fetched.etag, fetched.last_modified, job.refresh_minutes
                ),
                tags=job.tags,
                dedupe=True,
                kind="url",
                precondition=job.still_ours(filename, url) if job.still_ours else None,
            )
        except Exception as exc:  # noqa: BLE001 - a conflict, a full queue, a closing workspace
            logger.warning("Sitemap page %s was not queued: %s", loggable_url(url), exc)
            detail = getattr(exc, "detail", None) or str(exc)
            status.note(url, str(detail)[:200])
            status.skipped += 1
            continue
        job.known[filename] = url
        status.queued += 1
        audit(
            "document.upload",
            workspace=workspace_of(settings),
            filename=accepted.filename,
            job_id=accepted.job_id,
            source="sitemap",
        )
    logger.info(
        "Sitemap import: %d queued, %d unchanged, %d failed, %d skipped.",
        status.queued,
        status.unchanged,
        status.failed,
        status.skipped,
    )
    return {
        "queued": status.queued,
        "failed": status.failed,
        "skipped": status.skipped,
        "unchanged": status.unchanged,
    }
