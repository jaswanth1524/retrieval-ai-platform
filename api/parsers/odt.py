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

    def visit(element: ElementTree.Element) -> None:
        nonlocal label, buffer
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
                visit(child)

    visit(body)
    flush()
    if not sections:
        raise EmptyDocumentError(f"ODT '{filename}' has no extractable text.")
    return sections


def _inline_text(element: ElementTree.Element) -> str:
    """A paragraph's text with ODF's whitespace elements written out."""

    parts: list[str] = [element.text or ""]
    for child in element:
        if child.tag == f"{_TEXT}s":
            parts.append(" " * int(child.get(f"{_TEXT}c", "1") or 1))
        elif child.tag == f"{_TEXT}tab":
            parts.append("\t")
        elif child.tag == f"{_TEXT}line-break":
            parts.append("\n")
        elif child.tag in (f"{_TEXT}tracked-changes", f"{_OFFICE}annotation"):
            pass
        else:
            parts.append(_inline_text(child))
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
            repeat = int(cell.get(f"{_TABLE}number-columns-repeated", "1") or 1)
            cells.extend([text] * (min(repeat, 64) if text else 1))
        if any(cells):
            rows.append(cells)
    if len(rows) < 2:
        return [" | ".join(cell for cell in row if cell) for row in rows]
    header, *body = rows
    return [line for line in (render_row(header, row) for row in body) if line]
