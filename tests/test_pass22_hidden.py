"""Pass 22: hidden text the pass-21 checks missed (tables, slide frames, named and
inherited styles, filtered rows, XLSX rows without numbers)."""

from __future__ import annotations

import io
import zipfile

import pytest

from api.documents import parse_document_bytes
from tests.factories import make_test_settings
from tests.test_pass21_hidden import _NS

_HIDDEN_AUTO = (
    '<style:style style:name="T1" style:family="text">'
    '<style:text-properties text:display="none"/></style:style>'
    # Inherits from a named style in styles.xml that hides its text.
    '<style:style style:name="P9" style:family="paragraph" '
    'style:parent-style-name="HiddenNote"/>'
)
_STYLES_XML = (
    f'<?xml version="1.0" encoding="UTF-8"?><office:document-styles {_NS}><office:styles>'
    '<style:style style:name="HiddenNote" style:family="paragraph">'
    '<style:text-properties text:display="none"/></style:style>'
    "</office:styles></office:document-styles>"
)


def _odf(body: str) -> bytes:
    xml = (
        f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {_NS}>'
        f"<office:automatic-styles>{_HIDDEN_AUTO}</office:automatic-styles>"
        f"<office:body>{body}</office:body></office:document-content>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("content.xml", xml)
        archive.writestr("styles.xml", _STYLES_XML)
    return buffer.getvalue()


def _cell(inner: str) -> str:
    return f"<table:table-cell>{inner}</table:table-cell>"


def _odt() -> bytes:
    table = (
        "<table:table>"
        f"<table:table-row>{_cell('<text:p>name</text:p>')}{_cell('<text:p>note</text:p>')}"
        "</table:table-row><table:table-row>"
        f"{_cell('<text:p>alpha</text:p>')}"
        + _cell('<text:p>shown <text:span text:style-name="T1">secretcell</text:span></text:p>')
        + "</table:table-row></table:table>"
    )
    return _odf(
        "<office:text><text:h>Policy</text:h><text:p>Shown paragraph about leave.</text:p>"
        '<text:p text:style-name="HiddenNote">secretnamed</text:p>'
        '<text:p text:style-name="P9">secretinherited</text:p>'
        f"{table}</office:text>"
    )


def _odp() -> bytes:
    return _odf(
        '<office:presentation><draw:page draw:name="a"><draw:frame><draw:text-box>'
        '<text:p>Shown slide text <text:span text:style-name="T1">secretspan</text:span></text:p>'
        '<text:p text:style-name="HiddenNote">secretnamed</text:p>'
        "</draw:text-box></draw:frame></draw:page></office:presentation>"
    )


def _ods() -> bytes:
    def row(*cells: str, visibility: str = "") -> str:
        attr = f' table:visibility="{visibility}"' if visibility else ""
        inner = "".join(_cell(f"<text:p>{c}</text:p>") for c in cells)
        return f"<table:table-row{attr}>{inner}</table:table-row>"

    return _odf(
        f'<office:spreadsheet><table:table table:name="S">{row("name", "value")}'
        f"{row('alpha', '1')}{row('secretfiltered', '2', visibility='filter')}"
        "</table:table></office:spreadsheet>"
    )


def _xlsx_rows_without_numbers() -> bytes:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["name", "value"])
    sheet.append(["alpha", "1"])
    sheet.append(["secretrow", "2"])
    sheet.row_dimensions[3].hidden = True
    buffer = io.BytesIO()
    workbook.save(buffer)
    # Strip every row's "r" (Excel may leave it out: the row is the one after the last).
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as src, zipfile.ZipFile(out, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.startswith("xl/worksheets/"):
                import re

                data = re.sub(rb'<row r="\d+"', b"<row", data)
            dst.writestr(item, data)
    return out.getvalue()


@pytest.mark.parametrize(
    ("name", "build", "shown", "secrets"),
    [
        ("a.odt", _odt, "alpha", ["secretcell", "secretnamed", "secretinherited"]),
        ("a.odp", _odp, "Shown slide", ["secretspan", "secretnamed"]),
        ("a.ods", _ods, "alpha", ["secretfiltered"]),
        ("a.xlsx", _xlsx_rows_without_numbers, "alpha", ["secretrow"]),
    ],
)
def test_more_hidden_text_is_left_out_unless_asked_for(
    name: str, build: object, shown: str, secrets: list[str]
) -> None:
    content = build()  # type: ignore[operator]

    def text(**overrides: object) -> str:
        sections = parse_document_bytes(name, content, make_test_settings(**overrides))
        return " ".join(f"{s.section} {s.text}" for s in sections)

    default = text()
    assert shown in default
    assert not any(secret in default for secret in secrets), default
    everything = text(index_hidden_content=True)
    assert all(secret in everything for secret in secrets), everything
