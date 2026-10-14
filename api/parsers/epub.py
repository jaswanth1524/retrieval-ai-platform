"""EPUB parsing: chapters in reading order, each through the HTML parser."""

from __future__ import annotations

import logging
import posixpath
from dataclasses import replace
from io import BytesIO
from urllib.parse import unquote
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from api.parsers.common import (
    ZIP_MAX_EXPANDED_BYTES,
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    check_zip_expanded_size,
    declares_dtd,
)
from api.parsers.html import parse_html_document

logger = logging.getLogger(__name__)

# Every chapter is read into memory; see ZIP_MAX_EXPANDED_BYTES.
EPUB_MAX_EXPANDED_BYTES = ZIP_MAX_EXPANDED_BYTES

_CONTAINER_PATH = "META-INF/container.xml"
_CHAPTER_MEDIA_TYPES = frozenset({"application/xhtml+xml", "text/html"})


def parse_epub_document(filename: str, content: bytes) -> list[DocumentSection]:
    """Extract an EPUB's chapters in spine (reading) order.

    An EPUB has no fixed pages, so ``page`` is the chapter's position in the reading
    order (1-based) — what a citation can point a reader at. Sections come from each
    chapter's headings, as for HTML. Chapters that fail to parse or hold no text
    (covers, image-only pages) are skipped.
    """

    check_zip_expanded_size(filename, content, "EPUB", EPUB_MAX_EXPANDED_BYTES)
    try:
        with ZipFile(BytesIO(content)) as archive:
            chapters = _chapter_paths(filename, archive)
            sections: list[DocumentSection] = []
            for position, path in enumerate(chapters, start=1):
                try:
                    markup = archive.read(path).decode("utf-8", errors="replace")
                except KeyError:
                    logger.info("EPUB %s lists a missing chapter %s; skipped.", filename, path)
                    continue
                try:
                    chapter = parse_html_document(filename, markup)
                except EmptyDocumentError:
                    continue
                sections.extend(replace(section, page=position) for section in chapter)
    except BadZipFile as exc:
        raise DocumentParseError(f"Could not parse EPUB '{filename}'.") from exc
    if not sections:
        raise EmptyDocumentError(f"EPUB '{filename}' has no extractable text.")
    return sections


def _chapter_paths(filename: str, archive: ZipFile) -> list[str]:
    """Archive paths of the spine's chapters, in reading order."""

    container = _read_xml(filename, archive, _CONTAINER_PATH)
    rootfile = next(
        (
            element.get("full-path")
            for element in container.iter()
            if _local_name(element.tag) == "rootfile" and element.get("full-path")
        ),
        None,
    )
    if not rootfile:
        raise DocumentParseError(f"EPUB '{filename}' has no package document.")
    package = _read_xml(filename, archive, rootfile)
    base = posixpath.dirname(rootfile)

    hrefs: dict[str, str] = {}
    for element in package.iter():
        if _local_name(element.tag) != "item":
            continue
        item_id, href = element.get("id"), element.get("href")
        if item_id and href and element.get("media-type") in _CHAPTER_MEDIA_TYPES:
            hrefs[item_id] = href
    paths: list[str] = []
    for element in package.iter():
        if _local_name(element.tag) != "itemref":
            continue
        href = hrefs.get(element.get("idref") or "")
        if href:
            # Hrefs are relative to the package document and URL-encoded.
            path = posixpath.normpath(posixpath.join(base, unquote(href.split("#")[0])))
            if path not in paths:
                paths.append(path)
    if not paths:
        raise EmptyDocumentError(f"EPUB '{filename}' has no chapters.")
    return paths


def _read_xml(filename: str, archive: ZipFile, path: str) -> ElementTree.Element:
    try:
        raw = archive.read(path)
    except KeyError as exc:
        raise DocumentParseError(f"Could not parse EPUB '{filename}': {path} missing.") from exc
    # Package files never need a DTD; refusing one rules out entity-expansion tricks
    # regardless of what the XML parser would do with it.
    if declares_dtd(raw):
        raise DocumentParseError(f"Could not parse EPUB '{filename}'.")
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise DocumentParseError(f"Could not parse EPUB '{filename}'.") from exc


def _local_name(tag: str) -> str:
    """``{namespace}item`` -> ``item``; EPUBs disagree on namespace prefixes."""

    return tag.rsplit("}", 1)[-1]
