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
from api.parsers.odt import _TABLE, _TEXT, _inline_text, _table_rows

_OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
_DRAW = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}"
_PRESENTATION = "{urn:oasis:names:tc:opendocument:xmlns:presentation:1.0}"


def parse_odp_document(filename: str, content: bytes) -> list[DocumentSection]:
    """Each slide's text — frames in document order, tables as "header: value; ..."
    rows, then the speaker notes — as PPTX is: a slide is a page, its title the section
    label ("Slide n" without one). Images and media are skipped."""

    check_zip_expanded_size(filename, content, "ODP", ZIP_MAX_EXPANDED_BYTES)
    refuse_zip_xml_dtd(filename, content, "ODP")
    try:
        with ZipFile(BytesIO(content)) as archive:
            root = ElementTree.fromstring(archive.read("content.xml"))
    except (BadZipFile, KeyError, ElementTree.ParseError, OSError) as exc:
        raise DocumentParseError(f"Could not parse ODP '{filename}'.") from exc

    body = root.find(f"{_OFFICE}body/{_OFFICE}presentation")
    if body is None:
        raise EmptyDocumentError(f"ODP '{filename}' has no slides.")
    sections: list[DocumentSection] = []
    for number, slide in enumerate(body.iter(f"{_DRAW}page"), start=1):
        title = ""
        lines: list[str] = []
        for frame in slide.findall(f"{_DRAW}frame"):
            text = _frame_text(frame)
            if not text:
                continue
            if not title and frame.get(f"{_PRESENTATION}class") == "title":
                title = text.splitlines()[0]
            lines.append(text)
        notes = slide.find(f"{_PRESENTATION}notes")
        if notes is not None:
            spoken = "\n".join(
                text
                for text in (_frame_text(frame) for frame in notes.iter(f"{_DRAW}frame"))
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
    if not sections:
        raise EmptyDocumentError(f"ODP '{filename}' has no extractable text.")
    return sections


def _frame_text(frame: ElementTree.Element) -> str:
    parts: list[str] = []
    for element in frame:
        if element.tag == f"{_DRAW}text-box":
            for paragraph in element.iter():
                if paragraph.tag in (f"{_TEXT}p", f"{_TEXT}h"):
                    text = normalize_text(_inline_text(paragraph))
                    if text:
                        parts.append(text)
        elif element.tag == f"{_TABLE}table":
            parts.extend(_table_rows(element))
    return "\n".join(parts)
