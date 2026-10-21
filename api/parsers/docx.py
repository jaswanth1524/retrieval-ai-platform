"""DOCX parsing: headings as sections, tables rendered in place."""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from typing import Any

from api.parsers.common import (
    DEFAULT_SECTION,
    ZIP_MAX_EXPANDED_BYTES,
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    cap_section,
    check_zip_expanded_size,
    normalize_text,
    render_row,
)

# python-docx inflates the whole zip in memory; see ZIP_MAX_EXPANDED_BYTES.
DOCX_MAX_EXPANDED_BYTES = ZIP_MAX_EXPANDED_BYTES


def _check_docx_expanded_size(filename: str, content: bytes) -> None:
    """Refuse a DOCX whose members would inflate past ``DOCX_MAX_EXPANDED_BYTES``."""

    check_zip_expanded_size(filename, content, "DOCX", DOCX_MAX_EXPANDED_BYTES)


def parse_docx_document(filename: str, content: bytes) -> list[DocumentSection]:
    """Extract DOCX text, using heading styles as semantic section boundaries.

    DOCX has no fixed pagination, so ``page`` is always 1 and the section label
    carries the location meaning. Tables are rendered in place, a row per line (see
    ``_table_text``).
    """

    from docx import Document
    from docx.text.paragraph import Paragraph

    _check_docx_expanded_size(filename, content)
    try:
        document = Document(BytesIO(content))
    except Exception as exc:
        raise DocumentParseError(f"Could not parse DOCX '{filename}'.") from exc

    sections: list[DocumentSection] = []
    current_section = DEFAULT_SECTION
    buffer: list[str] = []

    def flush() -> None:
        text = normalize_text("\n".join(buffer))
        if text:
            sections.append(
                DocumentSection(
                    filename=filename,
                    page=1,
                    section=current_section,
                    text=text,
                    boundary="semantic",
                )
            )

    for block in _iter_blocks(document.element.body):
        if block.tag == _W_P:
            text = _paragraph_text(block).strip()
            if not text:
                continue
            paragraph = Paragraph(block, document)
            style_name = paragraph.style.name if paragraph.style is not None else ""
            if style_name.startswith("Heading"):
                flush()
                current_section = cap_section(text)
                buffer = [text]
            else:
                buffer.append(text)
        else:
            table_text = _table_text(block)
            if table_text:
                buffer.append(table_text)

    flush()
    if not sections:
        raise EmptyDocumentError(f"DOCX '{filename}' has no extractable text.")
    return sections


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_W_P, _W_TBL, _W_TR, _W_TC = f"{_W}p", f"{_W}tbl", f"{_W}tr", f"{_W}tc"
# Wrappers whose content is document content: a content control (a form field, a
# template's placeholder block), custom XML markup. python-docx's block and paragraph
# readers skip them, and the text inside was never indexed.
_W_CONTAINERS = frozenset({f"{_W}sdt", f"{_W}sdtContent", f"{_W}customXml", f"{_W}smartTag"})
# Text that is not in the document as it reads: tracked deletions, moved-away text.
_W_REMOVED = frozenset({f"{_W}del", f"{_W}moveFrom"})


def _iter_blocks(element: Any) -> Iterator[Any]:
    """Paragraphs and tables in document order, including those inside containers."""

    for child in element.iterchildren():
        if child.tag in (_W_P, _W_TBL):
            yield child
        elif child.tag in _W_CONTAINERS:
            yield from _iter_blocks(child)


def _paragraph_text(paragraph: Any) -> str:
    """A paragraph's text as it reads: runs inside tracked insertions, hyperlinks,
    simple fields and content controls included, tracked deletions left out (python-
    docx's ``Paragraph.text`` reads only direct runs and hyperlinks)."""

    parts: list[str] = []

    def walk(element: Any) -> None:
        for child in element.iterchildren():
            tag = child.tag
            if tag in _W_REMOVED:
                continue
            if tag == f"{_W}t":
                parts.append(child.text or "")
            elif tag == f"{_W}tab":
                parts.append("\t")
            elif tag in (f"{_W}br", f"{_W}cr"):
                parts.append("\n")
            elif tag != f"{_W}instrText":
                walk(child)

    walk(paragraph)
    return "".join(parts)


def _cell_text(cell: Any) -> str:
    return " ".join(
        text for text in (" ".join(_paragraph_text(p).split()) for p in cell.iter(_W_P)) if text
    )


def _table_text(table: Any) -> str:
    """A table as text, a row per line. Each cell is read once: python-docx repeats a
    merged cell in every column it spans ("Merged | Merged"). A table with a header row
    and at least two rows under it reads like a spreadsheet's ("Days: 20; Grade: B"),
    so a chunk holding part of it still says what each value is."""

    rows = [[_cell_text(cell) for cell in _row_cells(row)] for row in table.iterchildren(_W_TR)]
    rows = [row for row in rows if any(row)]
    if len(rows) >= 3 and len(rows[0]) >= 2 and all(rows[0]):
        header = rows[0]
        lines = [" | ".join(header), *(render_row(header, row) for row in rows[1:])]
    else:
        lines = [" | ".join(cell for cell in row) for row in rows]
    return "\n".join(line for line in lines if line.strip())


def _row_cells(row: Any) -> Iterator[Any]:
    for child in row.iterchildren():
        if child.tag == _W_TC:
            yield child
        elif child.tag in _W_CONTAINERS:
            yield from _row_cells(child)
