"""What every parser shares: the section type, document errors, and text clean-up."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Literal
from zipfile import BadZipFile, ZipFile

DEFAULT_SECTION = "Document"

# Longest section label any parser emits. The label is part of every chunk's embedded
# prefix ("filename › section"), so an uncapped one — a 2000-character Markdown heading —
# left a token or two of budget per chunk and turned one page into thousands of chunks.
MAX_SECTION_CHARS = 120


# infer_section: candidate first-lines that look like page furniture rather than an
# actual heading, so a PDF page's literal first line (often a page number or footer)
# doesn't become a meaningless citation section label.
_DIGITS_ONLY_RE = re.compile(r"^\d+$")


_PAGE_LABEL_RE = re.compile(r"^page\s+\d+$", re.IGNORECASE)


_ROMAN_NUMERAL_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)


# Zip-based formats (DOCX, EPUB) are inflated in memory by their parsers, while
# MAX_UPLOAD_BYTES caps only the compressed size: a few-MB upload declaring gigabytes of
# XML would OOM the ingest worker. Real documents (images included) sit far below this.
ZIP_MAX_EXPANDED_BYTES = 256 * 1024 * 1024


class DocumentError(ValueError):
    """Base error for document parsing and chunking failures."""


class UnsupportedDocumentError(DocumentError):
    """Raised when an uploaded document type is not supported."""


class EmptyDocumentError(DocumentError):
    """Raised when a document has no extractable text."""


class DocumentParseError(DocumentError):
    """Raised when a supported document cannot be parsed."""


@dataclass(frozen=True)
class DocumentSection:
    """Extracted text with source metadata before chunking.

    ``boundary`` marks whether the edge before the *next* section is a physical
    artifact of the source format (a PDF page break, a plain-text paragraph break —
    safe for chunk windows to cross with overlap) or a semantic one (a Markdown
    heading — a real topic break that chunk windows must not cross).
    """

    filename: str
    page: int
    section: str
    text: str
    boundary: Literal["physical", "semantic"] = "physical"


def normalize_text(text: str) -> str:
    """Normalize line endings and trim surrounding whitespace."""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def cap_section(label: str) -> str:
    """A section label cut to ``MAX_SECTION_CHARS``; ``DEFAULT_SECTION`` when empty."""

    return label[:MAX_SECTION_CHARS] or DEFAULT_SECTION


def infer_section(text: str) -> str:
    """Infer a stable section label when richer structure is unavailable.

    Skips candidate first-lines that look like page furniture — a bare page number,
    a "Page N" label, or a lone roman numeral — since PDF page text often has exactly
    one of these as its literal first line, which would otherwise become a
    meaningless citation section label. Falls back to the first non-blank line if
    every candidate line looks like furniture.
    """

    first_candidate: str | None = None
    for line in text.splitlines():
        candidate = line.strip().strip("#").strip()
        if not candidate:
            continue
        if first_candidate is None:
            first_candidate = candidate
        if _looks_like_page_furniture(candidate):
            continue
        return cap_section(candidate)
    if first_candidate is not None:
        return cap_section(first_candidate)
    return DEFAULT_SECTION


def _looks_like_page_furniture(candidate: str) -> bool:
    """True for page numbers/footers unlikely to be a meaningful heading."""

    if len(candidate) < 3:
        return True
    if _DIGITS_ONLY_RE.match(candidate):
        return True
    if _PAGE_LABEL_RE.match(candidate):
        return True
    if _ROMAN_NUMERAL_RE.match(candidate):
        return True
    return False


def check_zip_expanded_size(filename: str, content: bytes, kind: str, limit: int) -> None:
    """Refuse a zip-based document whose members would inflate past ``limit`` bytes."""

    try:
        with ZipFile(BytesIO(content)) as archive:
            expanded = sum(member.file_size for member in archive.infolist())
    except BadZipFile as exc:
        raise DocumentParseError(f"Could not parse {kind} '{filename}'.") from exc
    if expanded > limit:
        raise DocumentParseError(
            f"{kind} '{filename}' expands to {expanded // (1024 * 1024)} MiB, over the "
            f"{limit // (1024 * 1024)} MiB limit."
        )
