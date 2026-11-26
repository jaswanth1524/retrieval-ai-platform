"""OpenDocument presentations (.odp): each slide a page, titled by its title frame."""

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
    normalize_text,
    refuse_zip_xml_dtd,
)
from api.parsers.odt import (
    _TABLE,
    _TEXT,
    HiddenStyles,
    _inline_text,
    _table_rows,
    _visible_paragraphs,
    read_odf,
    warn_hidden,
)

_OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
_DRAW = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}"
_PRESENTATION = "{urn:oasis:names:tc:opendocument:xmlns:presentation:1.0}"


def parse_odp_document(
    filename: str, content: bytes, *, hidden: bool = False
) -> list[DocumentSection]:
    """Each slide's text — frames in document order, tables as "header: value; ..."
    rows, then the speaker notes — as PPTX is: a slide is a page, its title the section
    label ("Slide n" without one). Images and media are skipped, and so are hidden
    slides unless ``hidden`` (INDEX_HIDDEN_CONTENT)."""

    check_zip_expanded_size(filename, content, "ODP", ZIP_MAX_EXPANDED_BYTES)
    refuse_zip_xml_dtd(filename, content, "ODP")
    try:
        with ZipFile(BytesIO(content)) as archive:
            root, styles_root = read_odf(archive)
    except (BadZipFile, KeyError, ElementTree.ParseError, OSError) as exc:
        raise DocumentParseError(f"Could not parse ODP '{filename}'.") from exc

    body = root.find(f"{_OFFICE}body/{_OFFICE}presentation")
    if body is None:
        raise EmptyDocumentError(f"ODP '{filename}' has no slides.")
    styles = None if hidden else HiddenStyles(root, styles_root)
    skipped = 0
    sections: list[DocumentSection] = []
    for number, slide in enumerate(body.iter(f"{_DRAW}page"), start=1):
        # A hidden slide isn't presented; numbers stay deck positions.
        if styles is not None and (
            slide.get(f"{_DRAW}style-name") in styles.page
            or slide.get(f"{_PRESENTATION}visibility") == "hidden"
        ):
            skipped += 1
            continue
        title = ""
        lines: list[str] = []
        for frame in _shapes(slide):
            text = _frame_text(frame, styles)
            if not text:
                continue
            if not title and frame.get(f"{_PRESENTATION}class") == "title":
                title = text.splitlines()[0]
            lines.append(text)
        notes = slide.find(f"{_PRESENTATION}notes")
        if notes is not None:
            spoken = "\n".join(
                text
                for text in (_frame_text(frame, styles) for frame in notes.iter(f"{_DRAW}frame"))
                if text
            )
            if spoken:
                lines.append(f"Notes: {spoken}")
        text = "\n".join(lines).strip()
        if text:
            sections.append(
                DocumentSection(
                    filename=filename,
                    page=number,
                    section=cap_section(title or f"Slide {number}"),
                    text=text,
                    # Never merged into the next slide: short slides are the norm.
                    boundary="isolated",
                )
            )
    warn_hidden(filename, skipped, "slide(s)")
    if not sections:
        raise EmptyDocumentError(f"ODP '{filename}' has no extractable text.")
    return sections


# Groups nested deeper than this are not walked (a crafted file nests far past it).
_MAX_GROUP_DEPTH = 32


def _shapes(container: ElementTree.Element, depth: int = 0) -> list[ElementTree.Element]:
    """The shapes holding a slide's text, in order: text frames and custom shapes (a
    drawn box with text in it), including those inside groups."""

    found: list[ElementTree.Element] = []
    for element in container:
        if element.tag in (f"{_DRAW}frame", f"{_DRAW}custom-shape"):
            found.append(element)
        elif element.tag == f"{_DRAW}g" and depth < _MAX_GROUP_DEPTH:
            found.extend(_shapes(element, depth + 1))
    return found


def _frame_text(frame: ElementTree.Element, styles: HiddenStyles | None = None) -> str:
    """A shape's paragraphs and tables, without hidden text when ``styles`` is given."""

    parts: list[str] = []
    for element in frame:
        if styles is not None and styles.hides_text(element):
            continue
        if element.tag in (f"{_TEXT}p", f"{_TEXT}h"):
            # A custom shape holds its paragraphs directly.
            text = normalize_text(_inline_text(element, styles=styles))
            if text:
                parts.append(text)
        elif element.tag in (f"{_TEXT}list", f"{_DRAW}text-box"):
            for paragraph in _visible_paragraphs(element, styles):
                text = normalize_text(_inline_text(paragraph, styles=styles))
                if text:
                    parts.append(text)
        elif element.tag == f"{_TABLE}table":
            parts.extend(_table_rows(element, styles))
    return "\n".join(parts)
