"""CSV parsing: rows rendered as text, grouped into labelled sections."""

from __future__ import annotations

import io

from api.parsers.common import (
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    render_row,
)
from api.settings import AppSettings

_DEFAULT_CSV_ROWS_PER_SECTION = 50


def parse_csv_document(
    filename: str, text: str, settings: AppSettings | None = None
) -> list[DocumentSection]:
    """Render CSV rows as self-describing text, grouped into row-range sections."""

    import csv

    rows_per_section = (
        int(settings.csv_rows_per_section)
        if settings is not None
        else _DEFAULT_CSV_ROWS_PER_SECTION
    )
    try:
        # Only real separators: left to guess, the sniffer split a one-column file of
        # prose ("title\nalpha beta\ngamma delta") on the letter "l".
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    # A stream with newline="", as the csv module requires: str.splitlines() cut quoted
    # multi-line cells apart and also split on \u2028/\x0c inside a cell.
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        # Numbered before blank rows are dropped, so a section label names the rows a
        # spreadsheet shows ("Rows 4-9"), not positions among the non-blank ones.
        records = [
            (number, row)
            for number, row in enumerate(reader, 1)
            if any(cell.strip() for cell in row)
        ]
    except csv.Error as exc:  # e.g. a field past csv.field_size_limit()
        raise DocumentParseError(f"Could not parse CSV '{filename}': {exc}") from exc
    if not records:
        raise EmptyDocumentError(f"CSV '{filename}' has no rows.")

    header = [cell.strip() for cell in records[0][1]]
    data_rows = records[1:]
    if not data_rows:
        # Header-only file: treat the header row itself as the single section.
        return [
            DocumentSection(
                filename=filename,
                page=1,
                section="Header",
                text="; ".join(header),
            )
        ]

    sections: list[DocumentSection] = []
    for start in range(0, len(data_rows), rows_per_section):
        group = data_rows[start : start + rows_per_section]
        rendered = [(number, render_row(header, row)) for number, row in group]
        rendered = [(number, line) for number, line in rendered if line]
        if not rendered:
            continue
        label = f"Rows {group[0][0]}-{group[-1][0]}"
        sections.append(
            DocumentSection(
                filename=filename,
                page=1,
                section=label,
                text="\n".join(line for _, line in rendered),
                # Chunks keep rows whole and are labelled with the rows they hold.
                line_rows=tuple(number for number, _ in rendered),
            )
        )
    if not sections:
        raise EmptyDocumentError(f"CSV '{filename}' has no data rows.")
    return sections
