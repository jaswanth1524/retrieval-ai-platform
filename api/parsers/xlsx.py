"""XLSX parsing: each sheet a page, its rows rendered like a CSV's."""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
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
    refuse_zip_xml_dtd,
    row_groups,
)
from api.parsers.odt import warn_hidden
from api.settings import AppSettings

# openpyxl reads sheet XML as it goes, but shared strings are held in memory; see
# ZIP_MAX_EXPANDED_BYTES.
XLSX_MAX_EXPANDED_BYTES = ZIP_MAX_EXPANDED_BYTES


def parse_xlsx_document(
    filename: str, content: bytes, settings: AppSettings | None = None
) -> list[DocumentSection]:
    """Render each sheet's rows as "header: value; ..." text, in row-range sections.

    A sheet is a page (its position in the workbook), so a citation names the sheet by
    page number and by its section label, ``"<sheet> · Rows 4-53"`` — spreadsheet row
    numbers, blank rows included, as with CSV. The first non-blank row of a sheet is its
    header. Cells hold their last computed values (``data_only``); formulas are not
    evaluated. Charts and images are skipped, and so are hidden sheets unless
    INDEX_HIDDEN_CONTENT is on.
    """

    limits = settings if settings is not None else AppSettings.model_construct()
    rows_per_section = int(limits.csv_rows_per_section)
    max_cells = int(limits.max_spreadsheet_cells)
    hidden = bool(settings is not None and settings.index_hidden_content)
    check_zip_expanded_size(filename, content, "XLSX", XLSX_MAX_EXPANDED_BYTES)
    refuse_zip_xml_dtd(filename, content, "XLSX")

    from openpyxl import load_workbook

    try:
        workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
    except Exception as exc:  # openpyxl raises several reader-specific exceptions.
        raise DocumentParseError(f"Could not parse XLSX '{filename}'.") from exc

    sections: list[DocumentSection] = []
    cells = 0
    skipped_sheets = skipped_rows = 0
    try:
        for page, sheet in enumerate(workbook.worksheets, start=1):
            # Hidden and very hidden sheets (INDEX_HIDDEN_CONTENT): what the workbook
            # keeps from its reader. Pages stay the sheets' positions in the workbook.
            if not hidden and getattr(sheet, "sheet_state", "visible") != "visible":
                skipped_sheets += 1
                continue
            # The declared dimension is only a hint; trusting it padded every row to
            # its width (a sheet claiming A1:XFD1048576 is 17 billion cells).
            sheet.reset_dimensions()
            # Hidden rows and columns (INDEX_HIDDEN_CONTENT) and merged cells, read
            # from the sheet's XML: the read-only reader reports neither. Row numbers
            # stay the sheet's.
            layout = _sheet_layout(content, getattr(sheet, "_worksheet_path", ""))
            hidden_rows, hidden_columns = (
                (frozenset(), frozenset()) if hidden else (layout.rows, layout.columns)
            )
            anchors: dict[tuple[int, int], str] = {}
            records: list[tuple[int, list[str]]] = []
            for number, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                cells += len(row) + 1
                if cells > max_cells:
                    raise DocumentParseError(
                        f"XLSX '{filename}' has more than {max_cells} cells "
                        "(MAX_SPREADSHEET_CELLS). Split it into smaller files."
                    )
                texts = [_cell_text(value) for value in row]
                _fill_merged(texts, number, layout, anchors, header=not records)
                values = [
                    value
                    for column, value in enumerate(texts, start=1)
                    if column not in hidden_columns
                ]
                if not any(value.strip() for value in values):
                    continue
                if number in hidden_rows:
                    skipped_rows += 1
                    continue
                records.append((number, values))
            records = _two_row_header(records, layout, hidden_columns)
            sections.extend(_sheet_sections(filename, page, sheet.title, records, rows_per_section))
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"Could not parse XLSX '{filename}'.") from exc
    finally:
        workbook.close()
    warn_hidden(filename, skipped_sheets, "sheet(s)")
    warn_hidden(filename, skipped_rows, "row(s)")
    if not sections:
        raise EmptyDocumentError(f"XLSX '{filename}' has no cell values.")
    return sections


# Columns a hidden <col> range may name (XFD, the last column Excel has).
_MAX_COLUMN = 16384


# Cells a sheet's merges may cover before the rest are left as they are (a merge of a
# whole column is a million cells, and each would be looked up row by row).
_MAX_MERGED_CELLS = 200_000


@dataclass(frozen=True)
class _SheetLayout:
    rows: frozenset[int] = frozenset()
    columns: frozenset[int] = frozenset()
    # Each merged cell but the top-left one, by row: (its column, the top-left's
    # (row, column)).
    merges: dict[int, list[tuple[int, tuple[int, int]]]] = field(default_factory=dict)
    # The top-left cells of those merges.
    anchors: frozenset[tuple[int, int]] = frozenset()


def _sheet_layout(content: bytes, sheet_path: str) -> _SheetLayout:
    """The row numbers and column numbers (1-based) a sheet hides and its merged
    ranges, streamed from its XML; none when it can't be read (the sheet is then read
    whole and unmerged, as before)."""

    if not sheet_path:
        return _SheetLayout()
    rows: set[int] = set()
    columns: set[int] = set()
    merges: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    merged = 0
    try:
        with ZipFile(BytesIO(content)) as archive, archive.open(sheet_path.lstrip("/")) as xml:
            # A row may leave out its number ("r"): it is then the one after the last.
            # Such rows, and one unreadable attribute, used to make the whole sheet
            # read as hiding nothing.
            last_row = 0
            for _, element in ElementTree.iterparse(xml):
                tag = element.tag.rsplit("}", 1)[-1]
                if tag == "row":
                    number = _int_attribute(element, "r") or last_row + 1
                    last_row = number
                    if element.get("hidden") in ("1", "true"):
                        rows.add(number)
                    element.clear()
                elif tag == "col" and element.get("hidden") in ("1", "true"):
                    low = _int_attribute(element, "min") or 0
                    high = min(_int_attribute(element, "max") or low, _MAX_COLUMN)
                    columns.update(range(max(low, 1), high + 1))
                elif tag == "mergeCell":
                    bounds = _range_bounds(element.get("ref") or "")
                    if bounds is None:
                        continue
                    top, left, bottom, right = bounds
                    size = (bottom - top + 1) * (right - left + 1)
                    if merged + size > _MAX_MERGED_CELLS:
                        continue
                    merged += size
                    for r in range(top, bottom + 1):
                        for c in range(left, right + 1):
                            if (r, c) != (top, left):
                                merges.setdefault(r, []).append((c, (top, left)))
    except (KeyError, OSError, ElementTree.ParseError, BadZipFile):
        return _SheetLayout()
    anchors = frozenset(anchor for row in merges.values() for _, anchor in row)
    return _SheetLayout(frozenset(rows), frozenset(columns), merges, anchors)


_CELL_REF_RE = re.compile(r"^\$?([A-Z]{1,3})\$?(\d{1,7})$")


def _cell_position(ref: str) -> tuple[int, int] | None:
    match = _CELL_REF_RE.match(ref.strip().upper())
    if match is None:
        return None
    column = 0
    for letter in match.group(1):
        column = column * 26 + ord(letter) - 64
    row = int(match.group(2))
    if not 1 <= column <= _MAX_COLUMN or row < 1:
        return None
    return row, column


def _range_bounds(ref: str) -> tuple[int, int, int, int] | None:
    """``"A2:B5"`` as (top, left, bottom, right), 1-based; None when unreadable."""

    first, _, last = ref.partition(":")
    start = _cell_position(first)
    end = _cell_position(last or first)
    if start is None or end is None:
        return None
    return (
        min(start[0], end[0]),
        min(start[1], end[1]),
        max(start[0], end[0]),
        max(start[1], end[1]),
    )


def _fill_merged(
    texts: list[str],
    number: int,
    layout: _SheetLayout,
    anchors: dict[tuple[int, int], str],
    *,
    header: bool,
) -> None:
    """Give a merged cell the value of its range's top-left cell, as the sheet shows
    it: a Region merged down three rows is the region of each of them. Across columns
    only in the header row ("Q1" over two columns heads both); a data row's wide
    merge ("Total" over A:C) keeps its value once. ``anchors`` keeps the top-left
    cells' values seen so far (they always come first)."""

    if not layout.merges:
        return
    for column, text in enumerate(texts, start=1):
        if text and (number, column) in layout.anchors:
            anchors[(number, column)] = text
    for column, anchor in layout.merges.get(number, ()):
        if anchor[0] == number and not header:
            continue
        value = anchors.get(anchor, "")
        if not value:
            continue
        while len(texts) < column:
            texts.append("")
        if not texts[column - 1]:
            texts[column - 1] = value


def _two_row_header(
    records: list[tuple[int, list[str]]],
    layout: _SheetLayout,
    hidden_columns: frozenset[int],
) -> list[tuple[int, list[str]]]:
    """Read a two-row header as one: "Rates" merged over C:D with "Meal" and "Hotel"
    under it heads those columns "Rates Meal" and "Rates Hotel" (each was "Rates",
    and the second row read as data). Only when the first row has a merge across
    columns and the second row's text all sits under such merges (or repeats the cell
    above, merged down)."""

    if len(records) < 3:
        return records
    (first_number, first), (second_number, second) = records[0], records[1]
    if second_number != first_number + 1:
        return records
    merged_here = layout.merges.get(first_number, ())
    widest = max([len(first), len(second), *(column for column, _ in merged_here)])
    visible = [c for c in range(1, widest + len(hidden_columns) + 1) if c not in hidden_columns]
    under: set[int] = set()  # positions (in the visible row) under a header merge
    for column, anchor in merged_here:
        if anchor[0] == first_number and column in visible:
            under.add(visible.index(column))
            under.add(visible.index(anchor[1]) if anchor[1] in visible else -1)
    under.discard(-1)
    if not under:
        return records
    for position, text in enumerate(second):
        cell = text.strip()
        if not cell:
            continue
        above = first[position].strip() if position < len(first) else ""
        if position in under:
            if _looks_numeric(cell):
                return records
        elif cell != above:
            return records
    header = list(first) + [""] * max(0, len(second) - len(first))
    for position in under:
        lower = second[position].strip() if position < len(second) else ""
        if lower and position < len(header):
            header[position] = f"{header[position].strip()} {lower}".strip()
    return [(first_number, header), *records[2:]]


def _looks_numeric(text: str) -> bool:
    try:
        float(text.replace(",", ""))
    except ValueError:
        return False
    return True


def _int_attribute(element: ElementTree.Element, name: str) -> int | None:
    try:
        return int(element.get(name) or "")
    except ValueError:
        return None


def _sheet_sections(
    filename: str,
    page: int,
    title: str,
    records: list[tuple[int, list[str]]],
    rows_per_section: int,
) -> list[DocumentSection]:
    if not records:
        return []
    header = [cell.strip() for cell in records[0][1]]
    data_rows = records[1:]
    if not data_rows:
        return [
            DocumentSection(
                filename=filename,
                page=page,
                section=cap_section(f"{title} · Header"),
                text="; ".join(cell for cell in header if cell),
                boundary="isolated",
            )
        ]
    sections: list[DocumentSection] = []
    for first, last, rendered in row_groups(header, data_rows, rows_per_section):
        label = f"{title} · Row {first}" if first == last else f"{title} · Rows {first}-{last}"
        sections.append(
            DocumentSection(
                filename=filename,
                page=page,
                section=cap_section(label),
                text="\n".join(line for _, line in rendered),
                boundary="isolated",
                # Chunks keep rows whole and are labelled with the rows they hold.
                line_rows=tuple(number for number, _ in rendered),
                row_label_prefix=f"{title[:80]} · ",
            )
        )
    return sections


def _cell_text(value: object) -> str:
    """A cell value as it reads in the sheet: 3 rather than 3.0, dates without a
    midnight time."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, datetime.datetime):
        if value.time() == datetime.time(0):
            return value.date().isoformat()
        return value.isoformat(sep=" ")
    if isinstance(value, datetime.date | datetime.time):
        return value.isoformat()
    return str(value)
