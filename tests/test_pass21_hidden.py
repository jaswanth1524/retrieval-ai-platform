"""Pass 21: hidden content in OpenDocument files and hidden XLSX rows and columns."""

from __future__ import annotations

import io
import zipfile

import pytest

from api.documents import parse_document_bytes
from api.parsers.common import collect_parse_warnings
from tests.factories import make_test_settings

_NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
    'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"'
)


def _odf(styles: str, body: str) -> bytes:
    xml = (
        f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {_NS}>'
        f"<office:automatic-styles>{styles}</office:automatic-styles>"
        f"<office:body>{body}</office:body></office:document-content>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("content.xml", xml)
    return buffer.getvalue()


def _odt() -> bytes:
    return _odf(
        '<style:style style:name="T1" style:family="text">'
        '<style:text-properties text:display="none"/></style:style>',
        "<office:text><text:h>Policy</text:h>"
        "<text:p>Shown paragraph about leave. "
        '<text:span text:style-name="T1">secretspan</text:span></text:p>'
        '<text:section text:name="S" text:display="none"><text:p>secretsection</text:p>'
        "</text:section></office:text>",
    )


def _ods() -> bytes:
    def row(*cells: str, hidden: bool = False) -> str:
        attr = ' table:visibility="collapse"' if hidden else ""
        inner = "".join(f"<table:table-cell><text:p>{c}</text:p></table:table-cell>" for c in cells)
        return f"<table:table-row{attr}>{inner}</table:table-row>"

    return _odf(
        '<style:style style:name="ta2" style:family="table">'
        '<style:table-properties table:display="false"/></style:style>',
        "<office:spreadsheet>"
        f'<table:table table:name="Shown">{row("name", "value")}{row("alpha", "1")}'
        f"{row('secretrow', '2', hidden=True)}</table:table>"
        f'<table:table table:name="Secret" table:style-name="ta2">{row("h", "v")}'
        f"{row('secretsheet', '3')}</table:table>"
        "</office:spreadsheet>",
    )


def _odp() -> bytes:
    def slide(name: str, text: str, style: str = "") -> str:
        attr = f' draw:style-name="{style}"' if style else ""
        return (
            f'<draw:page draw:name="{name}"{attr}><draw:frame><draw:text-box>'
            f"<text:p>{text}</text:p></draw:text-box></draw:frame></draw:page>"
        )

    return _odf(
        '<style:style style:name="dp2" style:family="drawing-page">'
        '<style:drawing-page-properties presentation:visibility="hidden"/></style:style>',
        f"<office:presentation>{slide('a', 'Shown slide text')}"
        f"{slide('b', 'secretslide', 'dp2')}</office:presentation>",
    )


def _xlsx() -> bytes:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["name", "value", "secretcolumn"])
    sheet.append(["alpha", "1", "x"])
    sheet.append(["secretrow", "2", "y"])
    sheet.row_dimensions[3].hidden = True
    sheet.column_dimensions["C"].hidden = True
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("name", "build", "shown", "secrets"),
    [
        ("a.odt", _odt, "Shown paragraph", ["secretspan", "secretsection"]),
        ("a.ods", _ods, "alpha", ["secretrow", "secretsheet"]),
        ("a.odp", _odp, "Shown slide", ["secretslide"]),
        ("a.xlsx", _xlsx, "alpha", ["secretrow", "secretcolumn"]),
    ],
)
def test_hidden_parts_are_left_out_with_a_warning_unless_asked_for(
    name: str, build: object, shown: str, secrets: list[str]
) -> None:
    content = build()  # type: ignore[operator]

    def text(**overrides: object) -> tuple[str, list[str]]:
        with collect_parse_warnings() as warnings:
            sections = parse_document_bytes(name, content, make_test_settings(**overrides))
        return " ".join(f"{s.section} {s.text}" for s in sections), list(warnings)

    default, warnings = text()
    assert shown in default
    assert not any(secret in default for secret in secrets), default
    assert any("hidden" in warning for warning in warnings), warnings
    everything, none = text(index_hidden_content=True)
    assert all(secret in everything for secret in secrets), everything
    assert not any("hidden" in warning for warning in none)
