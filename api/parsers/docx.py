"""DOCX parsing: headings as sections, tables rendered in place."""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO

from api.parsers.common import (
    DEFAULT_SECTION,
    ZIP_MAX_EXPANDED_BYTES,
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    cap_section,
    check_zip_expanded_size,
    normalize_text,
)

# python-docx inflates the whole zip in memory; see ZIP_MAX_EXPANDED_BYTES.
DOCX_MAX_EXPANDED_BYTES = ZIP_MAX_EXPANDED_BYTES


def _check_docx_expanded_size(filename: str, content: bytes) -> None:
    """Refuse a DOCX whose members would inflate past ``DOCX_MAX_EXPANDED_BYTES``."""

    check_zip_expanded_size(filename, content, "DOCX", DOCX_MAX_EXPANDED_BYTES)


def parse_docx_document(filename: str, content: bytes) -> list[DocumentSection]:
    """Extract DOCX text, using heading styles as semantic section boundaries.

    DOCX has no fixed pagination, so ``page`` is always 1 and the section label
    carries the location meaning. Tables are rendered as one text block of
    ``" | "``-joined cell rows in document order.
    """

    from docx import Document
    from docx.table import Table
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

    for block in _iter_docx_blocks(document, Paragraph, Table):
        if isinstance(block, Paragraph):
            style_name = block.style.name if block.style is not None else ""
            text = block.text.strip()
            if not text:
                continue
            if style_name.startswith("Heading"):
                flush()
                current_section = cap_section(text)
                buffer = [text]
            else:
                buffer.append(text)
        elif isinstance(block, Table):
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in block.rows]
            table_text = "\n".join(row for row in rows if row.strip())
            if table_text:
                buffer.append(table_text)

    flush()
    if not sections:
        raise EmptyDocumentError(f"DOCX '{filename}' has no extractable text.")
    return sections


def _iter_docx_blocks(document: object, paragraph_cls: type, table_cls: type) -> Iterator[object]:
    """Yield a DOCX body's paragraphs and tables in document order.

    Prefers python-docx >= 1.1's ``iter_inner_content`` and falls back to manual XML
    body iteration on older versions.
    """

    iter_inner = getattr(document, "iter_inner_content", None)
    if callable(iter_inner):
        yield from iter_inner()
        return
    body = document.element.body  # type: ignore[attr-defined]
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield paragraph_cls(child, document)
        elif child.tag.endswith("}tbl"):
            yield table_cls(child, document)
