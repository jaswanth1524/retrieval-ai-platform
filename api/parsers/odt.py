"""OpenDocument text (.odt): LibreOffice's and Google Docs' export format."""

from __future__ import annotations

from io import BytesIO
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from api.parsers.common import (
    ZIP_MAX_EXPANDED_BYTES,
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    cap_section,
    check_zip_expanded_size,
    infer_section,
    normalize_text,
    refuse_zip_xml_dtd,
    render_row,
)

ODT_MAX_EXPANDED_BYTES = ZIP_MAX_EXPANDED_BYTES

_OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
_TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
_TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"

# Containers whose children are read in order as blocks of their own.
_BLOCK_CONTAINERS = frozenset(
    {
        f"{_TEXT}section",
        f"{_TEXT}list",
        f"{_TEXT}list-item",
        f"{_TEXT}list-header",
        f"{_TEXT}index-body",
    }
)
# Never body text: the record of deleted tracked changes, and generated tables of
# contents' templates.
_SKIPPED = frozenset(
    {
        f"{_TEXT}tracked-changes",
        f"{_TEXT}sequence-decls",
        f"{_TEXT}table-of-content-source",
        f"{_TEXT}alphabetical-index-source",
        f"{_OFFICE}forms",
    }
)
# A run of spaces (``text:s``) is at most this long: the count is an attribute, so a
# few bytes could ask for gigabytes of spaces the zip-size check never sees.
_MAX_SPACE_RUN = 100
# Repeated cells with text are written out at most this many times.
_MAX_REPEATED_CELLS = 64
# Elements nested deeper than this are read as plain text, not walked further: a
# crafted file nests far past Python's recursion limit.
_MAX_DEPTH = 64


def _count(element: ElementTree.Element, attribute: str, cap: int) -> int:
    """A repeat count from ``attribute``, 1 when it is missing or not a number, at most
    ``cap``."""

    try:
        return max(1, min(int(element.get(attribute, "1") or 1), cap))
    except ValueError:
        return 1


def parse_odt_document(filename: str, content: bytes) -> list[DocumentSection]:
    """Extract an ODT's paragraphs, lists and tables in reading order, one section per
    heading (text before the first heading is labelled from its own first line).

    ODT has no fixed pages, so every section is page 1, as for DOCX. A table becomes
    one "header: value; ..." line per row.
    """

    check_zip_expanded_size(filename, content, "ODT", ODT_MAX_EXPANDED_BYTES)
    refuse_zip_xml_dtd(filename, content, "ODT")
    try:
        with ZipFile(BytesIO(content)) as archive:
            xml = archive.read("content.xml")
        root = ElementTree.fromstring(xml)
    except (BadZipFile, KeyError, ElementTree.ParseError, OSError) as exc:
        raise DocumentParseError(f"Could not parse ODT '{filename}'.") from exc

    body = root.find(f"{_OFFICE}body/{_OFFICE}text")
    if body is None:
        raise EmptyDocumentError(f"ODT '{filename}' has no text body.")

    sections: list[DocumentSection] = []
    label: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = "\n\n".join(buffer).strip()
        if text:
            sections.append(
                DocumentSection(
                    filename=filename,
                    page=1,
                    section=label if label is not None else infer_section(text),
                    text=text,
                    boundary="semantic",
                )
            )

    def visit(element: ElementTree.Element, depth: int = 0) -> None:
        nonlocal label, buffer
        if depth > _MAX_DEPTH:
            text = normalize_text(" ".join(element.itertext()))
            if text:
                buffer.append(text)
            return
        for child in element:
            if child.tag in _SKIPPED:
                continue
            if child.tag == f"{_TEXT}h":
                heading = normalize_text(_inline_text(child))
                if heading:
                    flush()
                    label = cap_section(heading)
                    buffer = [heading]
            elif child.tag == f"{_TEXT}p":
                paragraph = normalize_text(_inline_text(child))
                if paragraph:
                    buffer.append(paragraph)
            elif child.tag == f"{_TABLE}table":
                buffer.extend(_table_rows(child))
            elif child.tag in _BLOCK_CONTAINERS:
                visit(child, depth + 1)

    visit(body)
    flush()
    if not sections:
        raise EmptyDocumentError(f"ODT '{filename}' has no extractable text.")
    return sections


def _inline_text(element: ElementTree.Element, depth: int = 0) -> str:
    """A paragraph's text with ODF's whitespace elements written out."""

    if depth > _MAX_DEPTH:
        return " ".join(element.itertext())
    parts: list[str] = [element.text or ""]
    for child in element:
        if child.tag == f"{_TEXT}s":
            parts.append(" " * _count(child, f"{_TEXT}c", _MAX_SPACE_RUN))
        elif child.tag == f"{_TEXT}tab":
            parts.append("\t")
        elif child.tag == f"{_TEXT}line-break":
            parts.append("\n")
        elif child.tag in (f"{_TEXT}tracked-changes", f"{_OFFICE}annotation"):
            pass
        else:
            parts.append(_inline_text(child, depth + 1))
        parts.append(child.tail or "")
    return "".join(parts)


def _cell_text(cell: ElementTree.Element) -> str:
    return " ".join(
        text
        for text in (
            normalize_text(_inline_text(paragraph)) for paragraph in cell.iter(f"{_TEXT}p")
        )
        if text
    )


def _table_rows(table: ElementTree.Element) -> list[str]:
    rows: list[list[str]] = []
    for row in table.iter(f"{_TABLE}table-row"):
        cells: list[str] = []
        for cell in row:
            if cell.tag not in (f"{_TABLE}table-cell", f"{_TABLE}covered-table-cell"):
                continue
            text = _cell_text(cell) if cell.tag == f"{_TABLE}table-cell" else ""
            # A repeated empty cell (a sheet-like table's padding) is one cell, not
            # thousands: only cells with text are repeated.
            repeat = _count(cell, f"{_TABLE}number-columns-repeated", _MAX_REPEATED_CELLS)
            cells.extend([text] * (repeat if text else 1))
        if any(cells):
            rows.append(cells)
    if len(rows) < 2:
        return [" | ".join(cell for cell in row if cell) for row in rows]
    header, *body = rows
    return [line for line in (render_row(header, row) for row in body) if line]
