"""PPTX parsing: each slide a page, its title the section, speaker notes included."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from io import BytesIO
from typing import Any

from api.parsers.common import (
    ZIP_MAX_EXPANDED_BYTES,
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    cap_section,
    check_zip_expanded_size,
    normalize_text,
    refuse_zip_xml_dtd,
    render_row,
)

# python-pptx parses each slide's XML in memory; see ZIP_MAX_EXPANDED_BYTES.
PPTX_MAX_EXPANDED_BYTES = ZIP_MAX_EXPANDED_BYTES

# Group shapes nest; real decks go a few levels deep at most.
_MAX_GROUP_DEPTH = 16


logger = logging.getLogger(__name__)


def parse_pptx_document(
    filename: str, content: bytes, *, hidden: bool = False
) -> list[DocumentSection]:
    """Extract each slide's text: text boxes and placeholders in shape order, tables as
    "header: value; ..." rows, then the speaker notes.

    A slide is a page (its number in the deck) and its title is the section label
    ("Slide n" without one), so a citation reads as the slide a presenter would show.
    A chart is read as its title and a line per category with each series' value.
    Pictures and media are skipped, and so are hidden slides unless ``hidden``
    (INDEX_HIDDEN_CONTENT).
    """

    check_zip_expanded_size(filename, content, "PPTX", PPTX_MAX_EXPANDED_BYTES)
    refuse_zip_xml_dtd(filename, content, "PPTX")

    from pptx import Presentation

    try:
        presentation = Presentation(BytesIO(content))
        sections: list[DocumentSection] = []
        for number, slide in enumerate(presentation.slides, start=1):
            # A hidden slide (show="0") isn't presented; numbers stay deck positions.
            if not hidden and slide._element.get("show") in ("0", "false"):  # noqa: SLF001
                continue
            title_shape = slide.shapes.title
            title = normalize_text(title_shape.text_frame.text) if title_shape is not None else ""
            # By id: python-pptx hands out a new proxy object per lookup. The title is
            # the section label (and in every chunk's embedded prefix), not body text.
            title_id = title_shape.shape_id if title_shape is not None else None
            parts = [
                text
                for shape in _shapes(slide.shapes, 0)
                if shape.shape_id != title_id and (text := _shape_text(shape))
            ]
            if slide.has_notes_slide:
                notes = normalize_text(slide.notes_slide.notes_text_frame.text or "")
                if notes:
                    parts.append(f"Notes: {notes}")
            text = normalize_text("\n".join(parts))
            if not text and not title:
                continue
            sections.append(
                DocumentSection(
                    filename=filename,
                    page=number,
                    section=cap_section(title or f"Slide {number}"),
                    # A title-only slide (a section divider) still says something.
                    text=text or title,
                    # Never merged into the next slide: short slides are the norm.
                    boundary="isolated",
                )
            )
    except DocumentParseError:
        raise
    except Exception as exc:  # python-pptx and its XML layer raise many types.
        raise DocumentParseError(f"Could not parse PPTX '{filename}'.") from exc
    if not sections:
        raise EmptyDocumentError(f"PPTX '{filename}' has no slide text.")
    return sections


def _shapes(shapes: Iterable[Any], depth: int) -> Iterator[Any]:
    """Every shape, group members included, in the order they are drawn."""

    from pptx.enum.shapes import MSO_SHAPE_TYPE

    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            if depth < _MAX_GROUP_DEPTH:
                yield from _shapes(shape.shapes, depth + 1)
            continue
        yield shape


# Categories a chart is read for: a chart of every day of a year says what it
# shows without 365 lines.
_MAX_CHART_POINTS = 200


def _chart_text(chart: Any) -> str:
    """ "Chart: Meal allowance by city" then "Paris: 2025 60; 2026 65" per category."""

    lines: list[str] = []
    if chart.has_title and chart.chart_title.has_text_frame:
        title = normalize_text(chart.chart_title.text_frame.text)
        if title:
            lines.append(f"Chart: {title}")
    for plot in chart.plots:
        categories = [str(c) for c in plot.categories][:_MAX_CHART_POINTS]
        series = [(normalize_text(item.name or ""), list(item.values)) for item in plot.series]
        if not categories:
            for name, _ in series:
                if name:
                    lines.append(name)
            continue
        for index, category in enumerate(categories):
            values = []
            for name, points in series:
                if index < len(points) and points[index] is not None:
                    value = points[index]
                    shown = int(value) if isinstance(value, float) and value.is_integer() else value
                    values.append(f"{name} {shown}".strip())
            if values:
                lines.append(f"{category}: " + "; ".join(values))
    if not lines:
        return ""
    if not lines[0].startswith("Chart:"):
        lines.insert(0, "Chart:")
    return "\n".join(lines)


def _table_rows(table: Any) -> list[list[str]]:
    """Each row's cells' text, a cell under a vertical merge given the text of the
    merged cell above (python-pptx reads it as empty), so each row says its region."""

    rows: list[list[str]] = []
    for row in table.rows:
        texts = []
        for column, cell in enumerate(row.cells):
            text = cell.text
            if not text.strip() and cell.is_spanned and rows:
                above = rows[-1][column] if column < len(rows[-1]) else ""
                # Only a merge down: a merge across already has its text once.
                if above and cell._tc.get("vMerge") in ("1", "true"):  # noqa: SLF001
                    text = above
            texts.append(text)
        rows.append(texts)
    return rows


def _shape_text(shape: Any) -> str:
    if getattr(shape, "has_chart", False) and shape.has_chart:
        try:
            return _chart_text(shape.chart)
        except Exception:  # noqa: BLE001 - enrichment, never a parse gate
            # A chart it can't read is skipped, as every chart used to be; the deck's
            # text still indexes.
            logger.info("PPTX chart skipped: it could not be read.", exc_info=True)
            return ""
    if getattr(shape, "has_table", False) and shape.has_table:
        rows = _table_rows(shape.table)
        if not rows:
            return ""
        header = [cell.strip() for cell in rows[0]]
        rendered = [render_row(header, row) for row in rows[1:]]
        lines = [line for line in rendered if line] or ["; ".join(cell for cell in header if cell)]
        return "\n".join(lines)
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        return normalize_text(shape.text_frame.text)
    return ""
