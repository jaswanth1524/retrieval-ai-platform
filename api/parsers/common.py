"""What every parser shares: the section type, document errors, and text clean-up."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterator, Sequence
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
    heading — a real topic break that chunk windows must not cross). An ``isolated``
    section (a slide, a range of spreadsheet rows) is a unit of its own: windowed alone
    like a semantic one, and never folded into a neighbour however short it is, so its
    page and label stay exact — a one-line slide is still that slide.
    """

    filename: str
    page: int
    section: str
    text: str
    boundary: Literal["physical", "semantic", "isolated"] = "physical"
    # Table rows (CSV, XLSX): the spreadsheet row number of each line of ``text``. The
    # chunker then keeps each row whole and labels a chunk with the rows it really
    # holds ("<row_label_prefix>Rows 4-17"); ``section`` is used only without them.
    line_rows: tuple[int, ...] | None = None
    row_label_prefix: str = ""


# Ligature characters, written out ("ﬁnance" never matched a search for "finance"), and
# invisible characters that split a word in two for BM25 ("infor\u00admation" was
# "infor" + "mation"): soft hyphen, zero-width space, word joiner, a BOM mid-text. The
# zero-width joiner and non-joiner stay — Persian, Indic scripts and emoji need them —
# and so does everything NFKC would also rewrite (², ½, ™, full-width forms).
_TEXT_CLEANUP = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
        "\u00ad": None,
        "\u200b": None,
        "\u2060": None,
        "\ufeff": None,
    }
)


def clean_text(text: str) -> str:
    """``text`` with ligature characters written out and word-splitting invisible
    characters removed. Applied to every parsed section and to questions alike."""

    return text.translate(_TEXT_CLEANUP)


def normalize_text(text: str) -> str:
    """Normalize line endings and trim surrounding whitespace."""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def cap_section(label: str) -> str:
    """A section label cut to ``MAX_SECTION_CHARS``; ``DEFAULT_SECTION`` when empty."""

    return label[:MAX_SECTION_CHARS] or DEFAULT_SECTION


def infer_section(text: str, skip: Collection[str] = ()) -> str:
    """Infer a stable section label when richer structure is unavailable.

    Skips candidate first-lines that look like page furniture — a bare page number,
    a "Page N" label, or a lone roman numeral — since PDF page text often has exactly
    one of these as its literal first line, which would otherwise become a
    meaningless citation section label. Falls back to the first non-blank line if
    every candidate line looks like furniture. Lines in ``skip`` (casefolded, spaces
    collapsed: a PDF's running line kept on its first page) are passed over too.
    """

    first_candidate: str | None = None
    for line in text.splitlines():
        candidate = line.strip().strip("#").strip()
        if not candidate or " ".join(candidate.split()).casefold() in skip:
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


# XML members of an Office file (XLSX, PPTX) a DTD could hide in.
_OFFICE_XML_SUFFIXES = (".xml", ".rels", ".vml")
# In every encoding an XML parser detects on its own (from a BOM or the first bytes):
# matching only ASCII let a UTF-16 member carry a DTD past the check.
_DTD_MARKERS = tuple(
    marker.encode(encoding)
    for marker in ("<!DOCTYPE", "<!ENTITY")
    for encoding in ("ascii", "utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be")
)
_DTD_MARKER_MAX_BYTES = max(len(marker) for marker in _DTD_MARKERS)
_DTD_SCAN_BLOCK = 64 * 1024


def declares_dtd(xml: bytes) -> bool:
    """True when ``xml`` holds a DTD or entity declaration, in any XML encoding."""

    return any(marker in xml for marker in _DTD_MARKERS)


def refuse_zip_xml_dtd(filename: str, content: bytes, kind: str) -> None:
    """Refuse an Office file whose XML declares a DTD or an entity.

    Office Open XML never needs either, and refusing them up front rules out entity
    expansion and external-entity tricks whatever the XML library would do with them —
    the EPUB parser's stance. Members are scanned in blocks, never read whole.
    """

    try:
        with ZipFile(BytesIO(content)) as archive:
            for info in archive.infolist():
                if not info.filename.lower().endswith(_OFFICE_XML_SUFFIXES):
                    continue
                with archive.open(info) as handle:
                    tail = b""
                    while block := handle.read(_DTD_SCAN_BLOCK):
                        window = tail + block
                        if declares_dtd(window):
                            raise DocumentParseError(
                                f"{kind} '{filename}' declares an XML DTD, which DocRAG "
                                "does not read."
                            )
                        tail = window[-_DTD_MARKER_MAX_BYTES:]
    except DocumentParseError:
        raise
    except Exception as exc:  # BadZipFile, a CRC error, an encrypted member, ...
        raise DocumentParseError(f"Could not parse {kind} '{filename}'.") from exc


_CELL_LINE_BREAK_RE = re.compile(r"[ \t]*[\r\n]+[ \t]*")


def render_row(header: list[str], row: list[str]) -> str:
    """Render one table row as "col1: v1; col2: v2" so any chunk slice self-describes.

    Empty cells are left out; a column without a header is named by its position. A
    line break inside a cell becomes a space: a row is one line, and the chunker's one
    row label per line (``DocumentSection.line_rows``) relies on it.
    """

    parts: list[str] = []
    for index, value in enumerate(row):
        cell = _CELL_LINE_BREAK_RE.sub(" ", value.strip())
        if not cell:
            continue
        column = header[index] if index < len(header) and header[index] else f"col{index + 1}"
        parts.append(f"{column}: {cell}")
    return "; ".join(parts)


def row_groups(
    header: list[str], rows: Sequence[tuple[int, list[str]]], per_group: int
) -> Iterator[tuple[int, int, list[tuple[int, str]]]]:
    """``rows`` (spreadsheet row number, cells) in groups of ``per_group``, each as
    ``(first number, last number, [(number, rendered row)])`` without the rows that
    render empty; a group with nothing left is skipped. CSV and XLSX label their
    sections from the numbers."""

    for start in range(0, len(rows), per_group):
        group = rows[start : start + per_group]
        rendered = [(number, render_row(header, cells)) for number, cells in group]
        rendered = [(number, line) for number, line in rendered if line]
        if rendered:
            yield group[0][0], group[-1][0], rendered
