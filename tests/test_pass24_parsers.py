"""Pass 24: what real Office files hold that the parsers used to miss or repeat."""

from __future__ import annotations

import io
import zipfile

from docx import Document
from docx.oxml import parse_xml
from openpyxl import Workbook
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

from api.parsers import pdf as pdf_parser
from api.parsers.common import DocumentSection
from api.parsers.docx import HEADER_FOOTER_SECTION, parse_docx_document
from api.parsers.pptx import parse_pptx_document
from api.parsers.xlsx import parse_xlsx_document

_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_BOX_NAMESPACES = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
    'xmlns:v="urn:schemas-microsoft-com:vml" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
)
_BOX_TEXT = (
    "<w:txbxContent><w:p><w:r><w:t>Note: receipts are required above 25 euros."
    "</w:t></w:r></w:p></w:txbxContent>"
)
# A text box as Word saves it: the drawing (mc:Choice) and a VML copy (mc:Fallback).
_TEXT_BOX = (
    f"<w:r {_BOX_NAMESPACES}><mc:AlternateContent>"
    f'<mc:Choice Requires="wps"><w:drawing><wp:anchor><a:graphic><a:graphicData><wps:wsp>'
    f"<wps:txbx>{_BOX_TEXT}</wps:txbx></wps:wsp></a:graphicData></a:graphic></wp:anchor>"
    f"</w:drawing></mc:Choice><mc:Fallback><w:pict><v:shape><v:textbox>{_BOX_TEXT}"
    f"</v:textbox></v:shape></w:pict></mc:Fallback></mc:AlternateContent></w:r>"
)
_FOOTNOTES = (
    f'<w:footnotes {_W}><w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/>'
    f'</w:r></w:p></w:footnote><w:footnote w:id="1"><w:p><w:r><w:t>The allowance is 60 '
    f"euros in Paris and 55 euros in Berlin.</w:t></w:r></w:p></w:footnote></w:footnotes>"
)


def _hard_docx() -> bytes:
    """A header and footer, a footnote, a text box and a table with a cell merged down."""

    document = Document()
    document.sections[0].header.paragraphs[0].text = "ACME Corp — Confidential — Policy v7"
    document.sections[0].footer.paragraphs[0].text = "Approved by the Board, March 2026"
    document.add_heading("Travel", 1)
    paragraph = document.add_paragraph("Employees may claim a daily meal allowance")
    paragraph._p.append(parse_xml(f'<w:r {_W}><w:footnoteReference w:id="1"/></w:r>'))
    paragraph.add_run(" while travelling.")
    document.add_paragraph("See the box.")._p.append(parse_xml(_TEXT_BOX))
    table = document.add_table(rows=4, cols=3)
    for column, heading in enumerate(["Region", "City", "Allowance"]):
        table.cell(0, column).text = heading
    table.cell(1, 0).merge(table.cell(2, 0))
    table.cell(1, 0).text = "Europe"
    for row, values in ((1, ("Paris", "60")), (2, ("Berlin", "55"))):
        table.cell(row, 1).text, table.cell(row, 2).text = values
    table.cell(3, 0).text, table.cell(3, 1).text, table.cell(3, 2).text = "Asia", "Tokyo", "70"
    buffer = io.BytesIO()
    document.save(buffer)

    # python-docx can't write footnotes: add the part and its relationship.
    source = zipfile.ZipFile(io.BytesIO(buffer.getvalue()))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(
                    b"</Types>",
                    b'<Override PartName="/word/footnotes.xml" ContentType="application/'
                    b'vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/></Types>',
                )
            if item.filename == "word/_rels/document.xml.rels":
                data = data.replace(
                    b"</Relationships>",
                    b'<Relationship Id="rIdFn1" Type="http://schemas.openxmlformats.org/'
                    b'officeDocument/2006/relationships/footnotes" Target="footnotes.xml"/>'
                    b"</Relationships>",
                )
            target.writestr(item, data)
        target.writestr("word/footnotes.xml", _FOOTNOTES)
    return out.getvalue()


def _merged_xlsx() -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Allowances"
    sheet.append(["Region", "City", "Rates", None])
    sheet.merge_cells("C1:D1")
    sheet.append([None, None, "Meal", "Hotel"])
    sheet.append(["Europe", "Paris", 60, 180])
    sheet.append([None, "Berlin", 55, 150])
    sheet.merge_cells("A3:A4")
    sheet.append(["Asia", "Tokyo", 70, 200])
    sheet.append(["Total", None, None, 530])
    sheet.merge_cells("A6:C6")
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _chart_pptx() -> bytes:
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Allowances"
    table = slide.shapes.add_table(4, 3, Inches(1), Inches(1.5), Inches(6), Inches(2)).table
    for column, heading in enumerate(["Region", "City", "Meal"]):
        table.cell(0, column).text = heading
    table.cell(1, 0).merge(table.cell(2, 0))
    table.cell(1, 0).text = "Europe"
    table.cell(1, 1).text, table.cell(1, 2).text = "Paris", "60"
    table.cell(2, 1).text, table.cell(2, 2).text = "Berlin", "55"
    table.cell(3, 0).text, table.cell(3, 1).text, table.cell(3, 2).text = "Asia", "Tokyo", "70"
    trend = presentation.slides.add_slide(presentation.slide_layouts[5])
    trend.shapes.title.text = "Trend"
    data = CategoryChartData()
    data.categories = ["Paris", "Berlin"]
    data.add_series("2025", (60, 55))
    data.add_series("2026", (65, 58))
    chart = trend.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1.5), Inches(6), Inches(4), data
    ).chart
    chart.has_title = True
    chart.chart_title.text_frame.text = "Meal allowance by city"
    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def _text(sections: list[DocumentSection]) -> str:
    return "\n".join(section.text for section in sections)


def test_a_docx_text_box_is_read_once_on_its_own_line() -> None:
    text = _text(parse_docx_document("t.docx", _hard_docx()))
    assert text.count("receipts are required above 25 euros") == 1
    assert "See the box.\nNote: receipts" in text


def test_a_docx_footnote_follows_the_paragraph_that_refers_to_it() -> None:
    text = _text(parse_docx_document("t.docx", _hard_docx()))
    assert (
        "meal allowance while travelling.\n"
        "Footnote: The allowance is 60 euros in Paris and 55 euros in Berlin." in text
    )


def test_docx_headers_and_footers_are_a_last_section_once() -> None:
    sections = parse_docx_document("t.docx", _hard_docx())
    assert sections[-1].section == HEADER_FOOTER_SECTION
    assert sections[-1].text == (
        "ACME Corp — Confidential — Policy v7\nApproved by the Board, March 2026"
    )
    plain = Document()
    plain.add_paragraph("Only body text.")
    buffer = io.BytesIO()
    plain.save(buffer)
    assert all(
        s.section != HEADER_FOOTER_SECTION for s in parse_docx_document("p.docx", buffer.getvalue())
    )


def test_a_docx_cell_merged_down_gives_each_row_its_value() -> None:
    text = _text(parse_docx_document("t.docx", _hard_docx()))
    assert "Region: Europe; City: Berlin; Allowance: 55" in text
    assert "Region: Asia; City: Tokyo; Allowance: 70" in text


def test_xlsx_merged_cells_and_a_two_row_header() -> None:
    sections = parse_xlsx_document("t.xlsx", _merged_xlsx())
    text = _text(sections)
    assert "Region: Europe; City: Berlin; Rates Meal: 55; Rates Hotel: 150" in text
    # A data row's merge across columns keeps its value once.
    assert "Region: Total; Rates Hotel: 530" in text
    assert sections[0].section.endswith("Rows 3-6")


def test_an_xlsx_second_row_of_numbers_is_data_not_a_header() -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["City", "Rates", None])
    sheet.merge_cells("B1:C1")
    sheet.append(["Paris", 60, 180])
    sheet.append(["Berlin", 55, 150])
    buffer = io.BytesIO()
    workbook.save(buffer)
    text = _text(parse_xlsx_document("n.xlsx", buffer.getvalue()))
    assert "City: Paris; Rates: 60; Rates: 180" in text


def test_pptx_charts_are_read_and_merged_table_cells_filled() -> None:
    sections = parse_pptx_document("t.pptx", _chart_pptx())
    assert "Region: Europe; City: Berlin; Meal: 55" in sections[0].text
    assert sections[1].text == (
        "Chart: Meal allowance by city\nParis: 2025 60; 2026 65\nBerlin: 2025 55; 2026 58"
    )


TABLE = "| City | Meal |\n| --- | --- |\n| Paris | 60 |\n| Berlin | 55 |"


def test_a_pdf_table_replaces_its_text_layer_copy_in_place() -> None:
    one_cell_a_line = "Rates\nCity\nMeal\nParis\n60\nBerlin\n55\nAfter the table."
    assert pdf_parser._place_tables(one_cell_a_line, [TABLE]) == (
        f"Rates\n\n[Table]\n{TABLE}\n\nAfter the table."
    )
    a_row_a_line = "Rates\nCity Meal\nParis 60\nBerlin 55\nAfter the table."
    assert pdf_parser._place_tables(a_row_a_line, [TABLE]) == (
        f"Rates\n\n[Table]\n{TABLE}\n\nAfter the table."
    )


def test_a_pdf_table_not_found_in_the_text_is_added_after_it() -> None:
    assert pdf_parser._place_tables("Only prose here.", [TABLE]) == (
        f"Only prose here.\n\n[Table]\n{TABLE}"
    )
    # A cell value on its own among prose is not the table.
    assert pdf_parser._place_tables("Paris\nis lovely", [TABLE]) == (
        f"Paris\nis lovely\n\n[Table]\n{TABLE}"
    )


def test_scatter_and_bubble_charts_never_fail_the_deck() -> None:
    from pptx.chart.data import BubbleChartData, XyChartData

    presentation = Presentation()
    xy = XyChartData()
    series = xy.add_series("Trips")
    series.add_data_point(1.0, 60.0)
    series.add_data_point(2.0, 55.0)
    bubble = BubbleChartData()
    bubbles = bubble.add_series("Cities")
    bubbles.add_data_point(1.0, 60.0, 3.0)
    for kind, data in ((XL_CHART_TYPE.XY_SCATTER, xy), (XL_CHART_TYPE.BUBBLE, bubble)):
        slide = presentation.slides.add_slide(presentation.slide_layouts[5])
        slide.shapes.title.text = f"Chart {kind}"
        slide.shapes.add_chart(kind, Inches(1), Inches(1.5), Inches(6), Inches(4), data)
    buffer = io.BytesIO()
    presentation.save(buffer)
    sections = parse_pptx_document("xy.pptx", buffer.getvalue())
    assert len(sections) == 2


def test_a_broken_notes_part_or_header_never_fails_the_document(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from api.parsers import docx as docx_parser

    def broken(*args: object) -> object:
        raise RuntimeError("unreadable part")

    monkeypatch.setattr(docx_parser, "_notes", broken)
    monkeypatch.setattr(docx_parser, "_header_footer_text", broken)
    text = _text(parse_docx_document("t.docx", _hard_docx()))
    assert "Employees may claim a daily meal allowance while travelling." in text
