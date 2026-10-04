"""DOCX parsing: headings as sections, tables rendered in place."""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from zipfile import BadZipFile, ZipFile

from api.parsers.common import (
    DEFAULT_SECTION,
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    normalize_text,
)

# A DOCX is a zip that python-docx inflates entirely into memory. MAX_UPLOAD_BYTES caps
# the compressed size only, so a few-MB upload declaring gigabytes of XML would OOM the
# ingest worker. Real documents (images included) sit far below this.
DOCX_MAX_EXPANDED_BYTES = 256 * 1024 * 1024


def _check_docx_expanded_size(filename: str, content: bytes) -> None:
    """Refuse a DOCX whose members would inflate past ``DOCX_MAX_EXPANDED_BYTES``."""

    try:
        with ZipFile(BytesIO(content)) as archive:
            expanded = sum(member.file_size for member in archive.infolist())
    except BadZipFile as exc:
        raise DocumentParseError(f"Could not parse DOCX '{filename}'.") from exc
    if expanded > DOCX_MAX_EXPANDED_BYTES:
        raise DocumentParseError(
            f"DOCX '{filename}' expands to {expanded // (1024 * 1024)} MiB, over the "
            f"{DOCX_MAX_EXPANDED_BYTES // (1024 * 1024)} MiB limit."
        )


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
                current_section = text[:120] or DEFAULT_SECTION
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
