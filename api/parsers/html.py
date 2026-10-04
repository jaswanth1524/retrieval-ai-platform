"""HTML parsing: visible text, split on headings."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

from api.parsers.common import DEFAULT_SECTION, DocumentSection, EmptyDocumentError, normalize_text

_HTML_STRIP_TAGS = ("script", "style", "noscript", "template")


_HTML_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def parse_html_document(filename: str, text: str) -> list[DocumentSection]:
    """Extract readable HTML text, splitting on h1-h6 into semantic sections."""

    from bs4 import BeautifulSoup
    from bs4.element import CData, PreformattedString

    soup = BeautifulSoup(text, "html.parser")
    for element in soup(list(_HTML_STRIP_TAGS)):
        element.decompose()

    body = soup.body or soup
    title_tag = soup.title.get_text(strip=True) if soup.title else ""

    sections: list[DocumentSection] = []
    current_section = title_tag[:120] or DEFAULT_SECTION
    buffer: list[str] = []

    def flush() -> None:
        section_text = normalize_text(" ".join(buffer))
        if section_text:
            sections.append(
                DocumentSection(
                    filename=filename,
                    page=1,
                    section=current_section,
                    text=section_text,
                    boundary="semantic",
                )
            )

    for element in cast(Iterator[Any], body.descendants):
        name = getattr(element, "name", None)
        if name in _HTML_HEADING_TAGS:
            flush()
            current_section = element.get_text(" ", strip=True)[:120] or DEFAULT_SECTION
            buffer = []
        elif name is None:  # NavigableString
            # Comments, doctypes and processing instructions aren't rendered text: an
            # invisible <!-- ... --> would otherwise reach the prompt as a source.
            if isinstance(element, PreformattedString) and not isinstance(element, CData):
                continue
            chunk = str(element).strip()
            if chunk:
                buffer.append(chunk)

    flush()
    if not sections:
        # No headings and/or text only outside body: fall back to whole-document text.
        whole = normalize_text(soup.get_text(" ", strip=True))
        if not whole:
            raise EmptyDocumentError(f"HTML '{filename}' has no extractable text.")
        return [DocumentSection(filename=filename, page=1, section=current_section, text=whole)]
    return sections
