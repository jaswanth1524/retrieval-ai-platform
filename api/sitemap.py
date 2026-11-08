"""Adding a site's pages from its sitemap or feed (``POST /documents/from-sitemap``).

The sitemap (or sitemap index, RSS or Atom feed) is read when the request arrives; the
pages it lists are then fetched one by one in the background, each through the same
guarded fetch as ``/documents/from-url`` (public addresses only, size and time caps,
the shared fetch slots) and queued as a ``url`` job, so they show in the job list as
they go. Only pages on the sitemap's own host by default, at most SITEMAP_MAX_PAGES.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit
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

# One import at a time, its pages one after another: a sitemap of a hundred pages must
# not take every fetch slot from the users adding single URLs.
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docrag-sitemap")


class SitemapError(DocumentError):
    """Not a sitemap or feed this reads (a 400)."""


@dataclass(frozen=True)
class SitemapPages:
    urls: list[str]
    # Listed but left out: another host, not http(s), a duplicate, or past the cap.
    left_out: int
    # The listing was a sitemap index; these are its sitemaps, read for pages.
    child_sitemaps: list[str]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def listed_urls(xml: bytes, base_url: str) -> tuple[list[str], list[str]]:
    """The page URLs a sitemap, RSS or Atom feed lists, and (for a sitemap index) its
    child sitemaps, in document order."""

    if declares_dtd(xml):
        raise SitemapError("The sitemap declares a DTD or entities; it was not read.")
    try:
        root = ElementTree.fromstring(xml)
    except (ElementTree.ParseError, RecursionError) as exc:
        raise SitemapError("That URL is not a sitemap or feed (not XML).") from exc
    kind = _local(root.tag)
    pages: list[str] = []
    children: list[str] = []
    if kind == "urlset":
        pages = [
            (loc.text or "").strip()
            for entry in root
            if _local(entry.tag) == "url"
            for loc in entry
            if _local(loc.tag) == "loc"
        ]
    elif kind == "sitemapindex":
        children = [
            (loc.text or "").strip()
            for entry in root
            if _local(entry.tag) == "sitemap"
            for loc in entry
            if _local(loc.tag) == "loc"
        ]
    elif kind == "rss":
        pages = [
            (link.text or "").strip()
            for item in root.iter()
            if _local(item.tag) == "item"
            for link in item
            if _local(link.tag) == "link"
        ]
    elif kind == "feed":
        for entry in root:
            if _local(entry.tag) != "entry":
                continue
            links = [link for link in entry if _local(link.tag) == "link"]
            chosen = next((link for link in links if link.get("rel") in (None, "alternate")), None)
            if chosen is not None and chosen.get("href"):
                pages.append(chosen.get("href", "").strip())
    else:
        raise SitemapError("That URL is not a sitemap, sitemap index, RSS or Atom feed.")
    return (
        [urljoin(base_url, url) for url in pages if url],
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
    """The pages ``url`` lists (one level of a sitemap index followed)."""

    listing = fetch(url)
    pages, children = listed_urls(listing.content, listing.final_url)
    for child in children[:MAX_CHILD_SITEMAPS]:
        if len(pages) >= limit:
            break
        try:
            more, _ = listed_urls(fetch(child).content, child)
        except (DocumentError, OSError) as exc:
            logger.warning("Sitemap %s could not be read: %s", loggable_url(child), exc)
            continue
        pages.extend(more)
    selected, left_out = select_pages(pages, listing.final_url, same_host=same_host, limit=limit)
    return SitemapPages(selected, left_out, children[:MAX_CHILD_SITEMAPS])


@dataclass(frozen=True)
class PageImport:
    """What fetching one sitemap's pages needs: where to queue them, how to fetch."""

    queue: IngestQueue
    raw_store: RawDocumentStore | None
    fetch: Callable[[str, Callable[[int], bool]], FetchedDocument]
    tags: tuple[str, ...] | None
    refresh_minutes: int | None
    known: set[str]


def start_page_import(pages: list[str], job: PageImport) -> None:
    """Fetch and queue ``pages`` in the background, one after another."""

    _pool.submit(_import_pages, pages, job)


def _import_pages(
    pages: list[str], job: PageImport, sleep: Callable[[float], None] = time.sleep
) -> dict[str, int]:
    from api.document_names import FilenameConflictError, refuse_near_duplicate_name

    outcome = {"queued": 0, "failed": 0, "skipped": 0}
    settings = job.queue.ingest_service.settings
    for url in pages:
        fetched: FetchedDocument | None = None
        for _ in range(_BACKLOG_RETRIES):
            try:
                with fetch_slots:
                    fetched = job.fetch(url, job.queue.backlog.has_room)
                break
            except BacklogFullError:
                sleep(_BACKLOG_WAIT_SECONDS)
            except Exception as exc:  # noqa: BLE001 - one page's failure is its own
                logger.warning("Sitemap page %s was not fetched: %s", loggable_url(url), exc)
                break
        if fetched is None:
            outcome["failed"] += 1
            continue
        try:
            # A name the server picked must not replace a document already here, and
            # with originals kept a near-identical name is refused (as from-url does).
            if fetched.remote_named and fetched.filename in job.known:
                raise FilenameConflictError(f"'{fetched.filename}' is already indexed.")
            if job.raw_store is not None:
                refuse_near_duplicate_name(fetched.filename, job.known)
            accepted = job.queue.enqueue(
                filename=fetched.filename,
                content=fetched.content,
                raw_store=job.raw_store,
                source=DocumentSource(
                    url, fetched.etag, fetched.last_modified, job.refresh_minutes
                ),
                tags=job.tags,
                dedupe=True,
                kind="url",
            )
        except Exception as exc:  # noqa: BLE001 - a conflict, a full queue, a closing workspace
            logger.warning("Sitemap page %s was not queued: %s", loggable_url(url), exc)
            outcome["skipped"] += 1
            continue
        job.known.add(fetched.filename)
        outcome["queued"] += 1
        audit(
            "document.upload",
            workspace=workspace_of(settings),
            filename=accepted.filename,
            job_id=accepted.job_id,
            source="sitemap",
        )
    logger.info(
        "Sitemap import: %d queued, %d failed, %d skipped.",
        outcome["queued"],
        outcome["failed"],
        outcome["skipped"],
    )
    return outcome
