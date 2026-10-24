"""Text extraction fixes of CHUNKER_VERSION 9: line breaks inside table cells, one bad
page of PDF tables, and stamped scan pages when OCR is off."""

from __future__ import annotations

import re
from typing import Any

import pytest

from api.chunking import HeuristicTokenCounter
from api.documents import chunk_sections, parse_document_bytes
from api.parsers import pdf as pdf_parser
from api.parsers.common import DocumentSection, render_row
from tests.factories import make_test_settings
from tests.samples import pdf_bytes, xlsx_bytes


def test_a_line_break_inside_a_cell_stays_inside_its_row() -> None:
    assert render_row(["Name", "Notes"], ["Ada", "first line\r\n  second line"]) == (
        "Name: Ada; Notes: first line second line"
    )


def _row_spans(labels: list[str]) -> list[tuple[int, int]]:
    spans = []
    for label in labels:
        match = re.search(r"Rows? (\d+)(?:-(\d+))?$", label)
        assert match, label
        spans.append((int(match[1]), int(match[2] or match[1])))
    return spans


@pytest.mark.parametrize("kind", ["csv", "xlsx"])
def test_a_multi_line_cell_keeps_every_chunk_labelled_with_its_own_rows(kind: str) -> None:
    rows = [["Item", "Notes"]] + [
        [f"item {n}", f"line one of {n}\nline two of {n}" if n == 5 else f"note {n}"]
        for n in range(1, 31)
    ]
    if kind == "csv":
        content = "\n".join(
            ",".join(f'"{cell}"' if "\n" in cell else cell for cell in row) for row in rows
        ).encode()
    else:
        content = xlsx_bytes({"Items": rows})
    settings = make_test_settings(chunk_size_tokens=64, chunk_overlap_tokens=0)

    sections = parse_document_bytes(f"items.{kind}", content)
    chunks = chunk_sections(sections, settings, HeuristicTokenCounter())

    assert len(chunks) > 1
    spans = _row_spans([chunk.section for chunk in chunks])
    assert all(last - first < 29 for first, last in spans)  # never "Rows 2-31"
    assert any("line one of 5 line two of 5" in chunk.text for chunk in chunks)


def test_one_page_whose_tables_fail_keeps_the_tables_of_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Page:
        def __init__(self, number: int) -> None:
            self.number = number
            self.rects = [object()]
            self.lines: list[object] = []
            self.curves: list[object] = []
            self.chars = [object()]

        def extract_tables(self) -> list[Any]:
            if self.number == 2:
                raise ValueError("bad page")
            return [[["k", "v"], [f"p{self.number}", "1"]]]

        def close(self) -> None:
            return None

    class Pdf:
        pages = [Page(1), Page(2), Page(3)]

        def __enter__(self) -> Pdf:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    import pdfplumber

    monkeypatch.setattr(pdfplumber, "open", lambda stream: Pdf())
    tables = pdf_parser._extract_pdf_tables(b"%PDF")
    assert sorted(tables) == [1, 3]


def test_stamped_scan_pages_keep_their_stamp_when_ocr_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = pdf_bytes(
        ["Contract terms in full text on this page.", "ACME-000123"],
        image_pages=frozenset({2}),
    )
    asked: list[list[int]] = []

    def fake_ocr(filename: str, data: bytes, pages: list[int]) -> list[DocumentSection]:
        asked.append(pages)
        return []

    monkeypatch.setattr(pdf_parser, "_ocr_pdf_pages", fake_ocr)
    settings = make_test_settings(max_ocr_pages=0)
    sections = parse_document_bytes("contract.pdf", content, settings)
    assert asked == []
    assert [s.text for s in sections] == [
        "Contract terms in full text on this page.",
        "ACME-000123",
    ]
    # A PDF that is only stamped scans still has text to index.
    only_stamp = pdf_bytes(["ACME-000124"], image_pages=frozenset({1}))
    assert [s.text for s in parse_document_bytes("scan.pdf", only_stamp, settings)] == [
        "ACME-000124"
    ]
