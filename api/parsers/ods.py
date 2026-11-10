"""OpenDocument spreadsheets (.ods): each sheet a page, its rows rendered like a CSV's."""

from __future__ import annotations

from io import BytesIO
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from api.parsers.common import (
    ZIP_MAX_EXPANDED_BYTES,
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    check_zip_expanded_size,
    refuse_zip_xml_dtd,
)
from api.parsers.odt import _TABLE, _count, row_cells
from api.parsers.xlsx import _sheet_sections
from api.settings import AppSettings

_OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
# Repeated rows with values are written out at most this many times; repeated empty
# rows only move the row numbers on (a sheet's padding is often a million of them).
_MAX_REPEATED_ROWS = 64


def parse_ods_document(
    filename: str, content: bytes, settings: AppSettings | None = None
) -> list[DocumentSection]:
    """Render each sheet's rows as "header: value; ..." text in row-range sections,
    labelled ``"<sheet> · Rows 4-53"`` with spreadsheet row numbers — as XLSX is. Cells
    read as they show (their text); formulas are not evaluated."""

    limits = settings if settings is not None else AppSettings.model_construct()
    rows_per_section = int(limits.csv_rows_per_section)
    max_cells = int(limits.max_spreadsheet_cells)
    check_zip_expanded_size(filename, content, "ODS", ZIP_MAX_EXPANDED_BYTES)
    refuse_zip_xml_dtd(filename, content, "ODS")
    try:
        with ZipFile(BytesIO(content)) as archive:
            root = ElementTree.fromstring(archive.read("content.xml"))
    except (BadZipFile, KeyError, ElementTree.ParseError, OSError) as exc:
        raise DocumentParseError(f"Could not parse ODS '{filename}'.") from exc

    body = root.find(f"{_OFFICE}body/{_OFFICE}spreadsheet")
    if body is None:
        raise EmptyDocumentError(f"ODS '{filename}' has no sheets.")
    sections: list[DocumentSection] = []
    cells = 0
    for page, table in enumerate(body.iter(f"{_TABLE}table"), start=1):
        title = table.get(f"{_TABLE}name") or f"Sheet {page}"
        records: list[tuple[int, list[str]]] = []
        number = 0
        for row in table.iter(f"{_TABLE}table-row"):
            values = row_cells(row)
            filled = any(value.strip() for value in values)
            repeat = _count(row, f"{_TABLE}number-rows-repeated", 1 << 30)
            if not filled:
                number += repeat
                continue
            for _ in range(min(repeat, _MAX_REPEATED_ROWS)):
                number += 1
                cells += len(values) + 1
                if cells > max_cells:
                    raise DocumentParseError(
                        f"ODS '{filename}' has more than {max_cells} cells "
                        "(MAX_SPREADSHEET_CELLS). Split it into smaller files."
                    )
                records.append((number, list(values)))
            number += max(repeat - _MAX_REPEATED_ROWS, 0)
        sections.extend(_sheet_sections(filename, page, title, records, rows_per_section))
    if not sections:
        raise EmptyDocumentError(f"ODS '{filename}' has no cell values.")
    return sections
