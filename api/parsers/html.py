"""HTML parsing: visible text, split on headings."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, cast

from api.parsers.common import DocumentSection, EmptyDocumentError, cap_section, normalize_text

_HTML_STRIP_TAGS = ("script", "style", "noscript", "template")


_HTML_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


# An inline style that hides the element: what a browser never shows the reader.
_HIDING_STYLE = re.compile(r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)\b", re.I)


def _is_hidden(element: Any) -> bool:
    """Text no reader of the page sees: the ``hidden`` attribute, ``aria-hidden``, or an
    inline ``display:none``/``visibility:hidden``. Indexed, it was a free channel for
    instructions aimed at the model ("ignore the question and ...") that nobody reviewing
    the page would notice. Stylesheets and scripts can hide text too; only what the
    markup itself says is checked."""

    attrs = getattr(element, "attrs", None)
    if not attrs:
        return False
    if "hidden" in attrs:
        return True
    if str(attrs.get("aria-hidden", "")).strip().lower() == "true":
        return True
    style = attrs.get("style")
    return isinstance(style, str) and _HIDING_STYLE.search(style) is not None


def parse_html_document(
    filename: str, text: str, *, default_section: str | None = None
) -> list[DocumentSection]:
    """Extract readable HTML text, splitting on h1-h6 into semantic sections.

    Text before the first heading is labelled with ``default_section`` when given (an
    email's subject), else the page's ``<title>``.
    """

    from bs4 import BeautifulSoup
    from bs4.element import CData, PreformattedString

    soup = BeautifulSoup(text, "html.parser")
    for element in soup(list(_HTML_STRIP_TAGS)):
        element.decompose()
    for element in [tag for tag in soup.find_all(True) if _is_hidden(tag)]:
        element.decompose()

    body = soup.body or soup
    title_tag = soup.title.get_text(strip=True) if soup.title else ""

    sections: list[DocumentSection] = []
    current_section = cap_section(default_section if default_section else title_tag)
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
            current_section = cap_section(element.get_text(" ", strip=True))
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
