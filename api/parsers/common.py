"""What every parser shares: the section type, document errors, and text clean-up."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DEFAULT_SECTION = "Document"


# infer_section: candidate first-lines that look like page furniture rather than an
# actual heading, so a PDF page's literal first line (often a page number or footer)
# doesn't become a meaningless citation section label.
_DIGITS_ONLY_RE = re.compile(r"^\d+$")


_PAGE_LABEL_RE = re.compile(r"^page\s+\d+$", re.IGNORECASE)


_ROMAN_NUMERAL_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)


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
        return candidate[:120]
    if first_candidate is not None:
        return first_candidate[:120]
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
