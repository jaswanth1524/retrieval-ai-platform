"""DOCX parsing: headings as sections, tables rendered in place."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# python-docx inflates the whole zip in memory; see ZIP_MAX_EXPANDED_BYTES.
DOCX_MAX_EXPANDED_BYTES = ZIP_MAX_EXPANDED_BYTES


def _check_docx_expanded_size(filename: str, content: bytes) -> None:
    """Refuse a DOCX whose members would inflate past ``DOCX_MAX_EXPANDED_BYTES``."""

    check_zip_expanded_size(filename, content, "DOCX", DOCX_MAX_EXPANDED_BYTES)


def parse_docx_document(
    filename: str, content: bytes, *, hidden: bool = False
) -> list[DocumentSection]:
    """Extract DOCX text, using heading styles as semantic section boundaries.

    DOCX has no fixed pagination, so ``page`` is always 1 and the section label
    carries the location meaning. Tables are rendered in place, a row per line (see
    ``_table_text``). Text formatted hidden (``w:vanish``) is left out unless ``hidden``
    (INDEX_HIDDEN_CONTENT). Footnotes and endnotes follow the paragraph that refers to
    them, and the headers' and footers' text is a last section of its own.
    """

    from docx import Document
    from docx.text.paragraph import Paragraph

    _check_docx_expanded_size(filename, content)
    try:
        document = Document(BytesIO(content))
    except Exception as exc:
        raise DocumentParseError(f"Could not parse DOCX '{filename}'.") from exc

    try:
        notes = _notes(document, hidden)
    except Exception:  # noqa: BLE001 - enrichment, never a parse gate
        logger.info("DOCX %s: footnotes skipped.", filename, exc_info=True)
        notes = {}
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
            referenced: list[str] = []
            text = _paragraph_text(block, hidden, referenced).strip()
            text = "\n".join([text, *_note_lines(referenced, notes)]).strip()
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
            table_text = _table_text(block, hidden)
            if table_text:
                buffer.append(table_text)

    flush()
    try:
        margins = _header_footer_text(document, hidden)
    except Exception:  # noqa: BLE001 - enrichment, never a parse gate
        logger.info("DOCX %s: headers and footers skipped.", filename, exc_info=True)
        margins = ""
    if margins:
        sections.append(
            DocumentSection(
                filename=filename,
                page=1,
                section=HEADER_FOOTER_SECTION,
                text=margins,
                boundary="semantic",
            )
        )
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
# Word writes a text box (and other shapes) twice: the modern drawing in
# ``mc:Choice`` and a VML copy in ``mc:Fallback`` for older readers. Reading both
# indexed the box's text twice; the fallback is skipped.
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
_W_TEXT_BOX = f"{_W}txbxContent"
_W_NOTE_REFS = {f"{_W}footnoteReference": "footnote", f"{_W}endnoteReference": "endnote"}
_NOTES_RELTYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
# The last section of a document with a header or footer: their text, each distinct
# line once (Word repeats them on every page, and a DOCX has no pages to cite).
HEADER_FOOTER_SECTION = "Header and footer"


def _iter_blocks(element: Any) -> Iterator[Any]:
    """Paragraphs and tables in document order, including those inside containers."""

    for child in element.iterchildren():
        if child.tag in (_W_P, _W_TBL):
            yield child
        elif child.tag in _W_CONTAINERS:
            yield from _iter_blocks(child)


def _vanished(run: Any) -> bool:
    """A run formatted hidden: ``<w:rPr><w:vanish/>`` (not ``w:val="0"``/``false``)."""

    properties = run.find(f"{_W}rPr")
    if properties is None:
        return False
    vanish = properties.find(f"{_W}vanish")
    if vanish is None:
        return False
    return vanish.get(f"{_W}val") not in ("0", "false", "off")


def _paragraph_text(paragraph: Any, hidden: bool = False, notes: list[str] | None = None) -> str:
    """A paragraph's text as it reads: runs inside tracked insertions, hyperlinks,
    simple fields and content controls included, tracked deletions left out (python-
    docx's ``Paragraph.text`` reads only direct runs and hyperlinks). A text box's
    paragraphs are read on lines of their own, once. ``notes`` collects the
    ``("footnote"|"endnote") + id`` of each note the paragraph refers to."""

    parts: list[str] = []

    def walk(element: Any) -> None:
        for child in element.iterchildren():
            tag = child.tag
            if tag in _W_REMOVED or tag == _MC_FALLBACK:
                continue
            if tag == f"{_W}r" and not hidden and _vanished(child):
                continue
            if tag == _W_TEXT_BOX:
                for inner in child.iter(_W_P):
                    text = _paragraph_text(inner, hidden, notes).strip()
                    if text:
                        parts.append(f"\n{text}\n")
            elif tag in _W_NOTE_REFS:
                if notes is not None:
                    notes.append(_W_NOTE_REFS[tag] + (child.get(f"{_W}id") or ""))
            elif tag == f"{_W}t":
                parts.append(child.text or "")
            elif tag == f"{_W}tab":
                parts.append("\t")
            elif tag in (f"{_W}br", f"{_W}cr"):
                parts.append("\n")
            elif tag != f"{_W}instrText":
                walk(child)

    walk(paragraph)
    return "".join(parts)


def _notes(document: Any, hidden: bool) -> dict[str, str]:
    """Footnote and endnote text by ``kind + id`` (python-docx doesn't read them)."""

    from docx.oxml import parse_xml

    found: dict[str, str] = {}
    for rel in document.part.rels.values():
        for kind in ("footnote", "endnote"):
            if rel.reltype != f"{_NOTES_RELTYPE}{kind}s" or rel.is_external:
                continue
            try:
                root = parse_xml(rel.target_part.blob)
            except Exception:  # lxml's syntax errors; a broken notes part is skipped
                continue
            for note in root.iterchildren(f"{_W}{kind}"):
                if note.get(f"{_W}type") not in (None, "normal"):
                    continue  # the separator lines, not notes
                text = " ".join(
                    t for t in (_paragraph_text(p, hidden).strip() for p in note.iter(_W_P)) if t
                )
                if text:
                    found[kind + (note.get(f"{_W}id") or "")] = text
    return found


def _note_lines(referenced: list[str], notes: dict[str, str]) -> list[str]:
    """A line per note a paragraph refers to, under it ("Footnote: …"), so the note
    is indexed with the text it explains."""

    lines = []
    for key in referenced:
        text = notes.get(key)
        if text is not None:
            kind = "Footnote" if key.startswith("footnote") else "Endnote"
            lines.append(f"{kind}: {text}")
    return lines


def _header_footer_text(document: Any, hidden: bool) -> str:
    """Every distinct line of the document's headers and footers, in order. A page
    number field's result is left out with the field (``instrText`` isn't read; a
    page number alone is skipped)."""

    seen: dict[str, None] = {}
    for section in document.sections:
        for part in (
            section.header,
            section.first_page_header,
            section.even_page_header,
            section.footer,
            section.first_page_footer,
            section.even_page_footer,
        ):
            if part.is_linked_to_previous:
                continue
            for paragraph in part._element.iter(_W_P):
                line = " ".join(_paragraph_text(paragraph, hidden).split())
                if line and not line.isdigit():
                    seen.setdefault(line, None)
            for table in part._element.iter(_W_TBL):
                for line in _table_text(table, hidden).splitlines():
                    seen.setdefault(line, None)
    return normalize_text("\n".join(seen))


def _cell_text(cell: Any, hidden: bool = False) -> str:
    return " ".join(
        text
        for text in (" ".join(_paragraph_text(p, hidden).split()) for p in cell.iter(_W_P))
        if text
    )


def _table_text(table: Any, hidden: bool = False) -> str:
    """A table as text, a row per line. Each cell is read once: python-docx repeats a
    merged cell in every column it spans ("Merged | Merged"). A table with a header row
    and at least two rows under it reads like a spreadsheet's ("Days: 20; Grade: B"),
    so a chunk holding part of it still says what each value is."""

    rows = _grid_rows(table, hidden)
    rows = [row for row in rows if any(row)]
    if len(rows) >= 3 and len(rows[0]) >= 2 and all(rows[0]):
        header = rows[0]
        lines = [" | ".join(header), *(render_row(header, row) for row in rows[1:])]
    else:
        lines = [" | ".join(cell for cell in row) for row in rows]
    return "\n".join(line for line in lines if line.strip())


def _grid_rows(table: Any, hidden: bool) -> list[list[str]]:
    """Each row's cells' text, a cell under a vertical merge (``w:vMerge`` with no
    ``restart``) given the text of the cell it continues — so "Berlin" keeps the
    "Europe" of the merged Region cell above it. Grid columns are counted with
    ``w:gridSpan`` so a merge is matched to the right column."""

    rows: list[list[str]] = []
    above: dict[int, str] = {}
    for row in table.iterchildren(_W_TR):
        texts: list[str] = []
        column = 0
        for cell in _row_cells(row):
            properties = cell.find(f"{_W}tcPr")
            span, merge = 1, None
            if properties is not None:
                grid_span = properties.find(f"{_W}gridSpan")
                if grid_span is not None:
                    try:
                        span = max(1, int(grid_span.get(f"{_W}val") or 1))
                    except ValueError:
                        span = 1
                v_merge = properties.find(f"{_W}vMerge")
                if v_merge is not None:
                    merge = v_merge.get(f"{_W}val") or "continue"
            text = _cell_text(cell, hidden)
            if merge == "continue" and not text:
                text = above.get(column, "")
            above[column] = text
            texts.append(text)
            column += span
        rows.append(texts)
    return rows


def _row_cells(row: Any) -> Iterator[Any]:
    for child in row.iterchildren():
        if child.tag == _W_TC:
            yield child
        elif child.tag in _W_CONTAINERS:
            yield from _row_cells(child)
