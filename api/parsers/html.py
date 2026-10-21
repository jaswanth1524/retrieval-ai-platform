"""HTML parsing: visible text, split on headings."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, cast

from api.parsers.common import (
    DocumentSection,
    EmptyDocumentError,
    cap_section,
    normalize_text,
    render_row,
)

_HTML_STRIP_TAGS = ("script", "style", "noscript", "template")


_HTML_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


# Page chrome, not content: site navigation, the masthead, the footer, sidebars. Left out
# when outside the page's content elements (an article's own <header> holds its title, a
# section's <header> its heading, an <aside> in the main text a note on it).
_CHROME_TAGS = ("nav", "header", "footer", "aside")
_CONTENT_TAGS = ("main", "article", "section")

# A <main> or lone <article> is taken as the page's content only when it holds at least
# this share of the page's text: a <main> holding a welcome line, or one teaser
# <article>, had the page's real text (in a plain <div>) dropped.
_MAIN_CONTENT_MIN_SHARE = 0.5


# Elements that end a block of text: their text is its own paragraph, not run together
# with the next ("Item one Item two" from a list).
_BLOCK_TAGS = (
    "p", "div", "li", "dd", "dt", "blockquote", "pre", "section", "article", "br", "hr",
    "ul", "ol", "dl", "figure", "figcaption", "table", "main",
)  # fmt: skip

_BLOCK_BREAK = "\n\n"
_BREAK_RUN = re.compile(r"[ \t]*\n\n[\s]*")


def _text_length(element: Any) -> int:
    return len(" ".join(element.get_text(" ").split()))


def _content_root(soup: Any) -> Any:
    """Where the page's own content is: the body without its chrome, narrowed to its
    <main> or only <article> when that holds most of the text."""

    for element in soup.find_all(list(_CHROME_TAGS)):
        if not element.decomposed and element.find_parent(list(_CONTENT_TAGS)) is None:
            element.decompose()
    body = soup.body or soup
    articles = soup.find_all("article")
    candidate = soup.find("main") or (articles[0] if len(articles) == 1 else None)
    if candidate is not None and _text_length(candidate) >= _MAIN_CONTENT_MIN_SHARE * (
        _text_length(body)
    ):
        return candidate
    return body


def _rows_of(table: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue  # a nested table's rows belong to it
        cells = row.find_all(["th", "td"], recursive=False)
        rows.append([" ".join(cell.get_text(" ", strip=True).split()) for cell in cells])
    return [row for row in rows if any(row)]


def _render_table(table: Any) -> str:
    """A table as one line per row ("| col: value; ..."), each row self-describing like
    a CSV row; the header is the first row when it is made of <th> cells."""

    rows = _rows_of(table)
    if not rows:
        return ""
    first = table.find("tr")
    header: list[str] = []
    if first is not None and first.find("td", recursive=False) is None:
        header, rows = rows[0], rows[1:]
    lines = [f"| {render_row(header, row)}" for row in rows if render_row(header, row)]
    return "\n".join(lines)


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

    soup = BeautifulSoup(text, "html.parser")
    title_tag = soup.title.get_text(strip=True) if soup.title else ""
    for element in soup(list(_HTML_STRIP_TAGS)):
        element.decompose()
    for element in [tag for tag in soup.find_all(True) if _is_hidden(tag)]:
        element.decompose()
    sections = _sections(filename, _content_root(soup), title_tag, default_section)
    if sections:
        return sections
    # Nothing in the main content (or a page whose text all sat in its chrome): the
    # whole page, as before.
    soup = BeautifulSoup(text, "html.parser")
    for element in soup(list(_HTML_STRIP_TAGS)):
        element.decompose()
    for element in [tag for tag in soup.find_all(True) if _is_hidden(tag)]:
        element.decompose()
    sections = _sections(filename, soup.body or soup, title_tag, default_section)
    if not sections:
        whole = normalize_text(soup.get_text(" ", strip=True))
        if not whole:
            raise EmptyDocumentError(f"HTML '{filename}' has no extractable text.")
        label = cap_section(default_section if default_section else title_tag)
        return [DocumentSection(filename=filename, page=1, section=label, text=whole)]
    return sections


def _sections(
    filename: str, body: Any, title_tag: str, default_section: str | None
) -> list[DocumentSection]:
    """``body``'s text split on h1-h6, with tables as rows and blocks as paragraphs."""

    from bs4.element import CData, NavigableString, PreformattedString

    for table in body.find_all("table"):
        if table.decomposed or table.find_parent("table") is not None:
            continue
        table.replace_with(NavigableString(f"{_BLOCK_BREAK}{_render_table(table)}{_BLOCK_BREAK}"))
    for block in body.find_all(list(_BLOCK_TAGS)):
        block.insert_before(NavigableString(_BLOCK_BREAK))
        block.insert_after(NavigableString(_BLOCK_BREAK))

    sections: list[DocumentSection] = []
    current_section = cap_section(default_section if default_section else title_tag)
    buffer: list[str] = []

    def flush() -> None:
        section_text = normalize_text(_BREAK_RUN.sub(_BLOCK_BREAK, " ".join(buffer)))
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
            raw = str(element)
            chunk = raw.strip()
            if chunk:
                # A rendered table keeps its row lines.
                buffer.append(chunk if "\n| " not in raw and not chunk.startswith("| ") else raw)
            elif _BLOCK_BREAK in raw:
                buffer.append(_BLOCK_BREAK)

    flush()
    return sections
