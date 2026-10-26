"""JSON, source code, ODT and image (OCR) documents."""

from __future__ import annotations

import io
import json
import sys
import types
from typing import Any

import pytest

from api.chunking import HeuristicTokenCounter
from api.documents import (
    CORE_EXTENSIONS,
    DocumentParseError,
    EmptyDocumentError,
    chunk_sections,
    parse_document_bytes,
)
from api.parsers import image as image_parser
from api.parsers import pdf as pdf_parser
from tests.factories import make_test_settings
from tests.samples import odt_bytes


def test_json_is_one_section_per_top_level_key_with_one_fact_per_line() -> None:
    document = {
        "service": {"name": "billing", "owners": ["ada", "grace"], "limits": {"rps": 50}},
        "contacts": [{"team": "search", "lead": "lin"}],
        "debug": False,
        "empty": {},
    }
    sections = parse_document_bytes("config.json", json.dumps(document).encode())

    assert [section.section for section in sections] == ["service", "contacts", "debug"]
    assert sections[0].text.split("\n") == [
        "service › name: billing",
        "service › owners: ada, grace",
        "service › limits › rps: 50",
    ]
    assert sections[1].text.split("\n") == [
        "contacts [1] › team: search",
        "contacts [1] › lead: lin",
    ]
    assert sections[2].text == "debug: false"


def test_a_json_list_is_one_section_per_item_and_bad_json_is_named() -> None:
    sections = parse_document_bytes("rows.json", b'[{"a": 1}, {"a": 2}]')
    assert [section.section for section in sections] == ["Item 1", "Item 2"]
    with pytest.raises(DocumentParseError, match="not valid JSON"):
        parse_document_bytes("bad.json", b'{"a": ')
    with pytest.raises(EmptyDocumentError):
        parse_document_bytes("nothing.json", b"{}")


def test_deep_json_is_written_out_rather_than_walked_forever() -> None:
    nested: Any = "bottom"
    for _ in range(60):
        nested = {"k": nested}
    (section,) = parse_document_bytes("deep.json", json.dumps({"top": nested}).encode())
    assert "bottom" in section.text


def test_source_code_is_split_at_blank_lines_and_labelled_by_each_block() -> None:
    code = b"def add(a, b):\n    return a + b\n\n\ndef total(items):\n    return sum(items)\n"
    sections = parse_document_bytes("maths.py", code)
    assert [section.section for section in sections] == ["def add(a, b):", "def total(items):"]
    chunks = chunk_sections(sections, make_test_settings(), HeuristicTokenCounter())
    assert any("return sum(items)" in chunk.text for chunk in chunks)


def test_an_odt_reads_headings_paragraphs_lists_and_tables_in_order() -> None:
    body = (
        "<text:p>Preface before any heading.</text:p>"
        '<text:h text:outline-level="1">Leave policy</text:h>'
        '<text:p>Employees accrue<text:s/>twenty<text:s text:c="2"/>days.</text:p>'
        "<text:list><text:list-item><text:p>Carry over five.</text:p></text:list-item></text:list>"
        "<text:tracked-changes><text:changed-region><text:deletion>"
        "<text:p>Deleted text.</text:p></text:deletion></text:changed-region>"
        "</text:tracked-changes>"
        '<text:h text:outline-level="2">Rates</text:h>'
        "<table:table>"
        "<table:table-row><table:table-cell><text:p>Type</text:p></table:table-cell>"
        "<table:table-cell><text:p>Days</text:p></table:table-cell></table:table-row>"
        "<table:table-row><table:table-cell><text:p>Annual</text:p></table:table-cell>"
        "<table:table-cell><text:p>20</text:p></table:table-cell></table:table-row>"
        "</table:table>"
    )
    sections = parse_document_bytes("policy.odt", odt_bytes(body))

    assert [section.section for section in sections] == [
        "Preface before any heading.",
        "Leave policy",
        "Rates",
    ]
    # <text:s/> is a space, <text:s text:c="2"/> two.
    assert "Employees accrue twenty  days." in sections[1].text
    assert "Carry over five." in sections[1].text
    assert "Deleted text" not in " ".join(section.text for section in sections)
    assert "Type: Annual; Days: 20" in sections[2].text


def test_an_odt_declaring_a_dtd_is_refused() -> None:
    sneaky = odt_bytes("<text:p>x</text:p>", extra_members={"styles.xml": b"<!DOCTYPE x []><x/>"})
    with pytest.raises(DocumentParseError, match="DTD"):
        parse_document_bytes("sneaky.odt", sneaky)
    with pytest.raises(DocumentParseError):
        parse_document_bytes("broken.odt", b"not a zip")


def test_images_are_offered_only_with_the_ocr_extra() -> None:
    assert not {".png", ".jpg", ".tiff"} & CORE_EXTENSIONS


@pytest.fixture
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """A stand-in rapidocr_onnxruntime: "reads" each image as its size."""

    seen: list[tuple[int, int]] = []

    class RapidOCR:
        def __call__(self, array: Any) -> tuple[list[Any], float]:
            height, width = array.shape[:2]
            seen.append((width, height))
            return [[None, f"Scanned page {len(seen)} text", 0.9]], 0.1

    module = types.ModuleType("rapidocr_onnxruntime")
    module.RapidOCR = RapidOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", module)
    monkeypatch.setattr(pdf_parser, "_OCR_ENGINE", None)
    return seen


def _image(size: tuple[int, int], frames: int = 1, fmt: str = "PNG") -> bytes:
    from PIL import Image

    images = [Image.new("RGB", size, "white") for _ in range(frames)]
    buffer = io.BytesIO()
    images[0].save(buffer, format=fmt, save_all=frames > 1, append_images=images[1:])
    return buffer.getvalue()


def test_an_image_is_read_with_ocr_one_page_per_frame(fake_ocr: list[tuple[int, int]]) -> None:
    single = image_parser.parse_image_document("scan.png", _image((40, 30)))
    assert [(section.page, section.text) for section in single] == [(1, "Scanned page 1 text")]

    pages = image_parser.parse_image_document("fax.tiff", _image((40, 30), frames=3, fmt="TIFF"))
    assert [section.page for section in pages] == [1, 2, 3]

    limited = image_parser.parse_image_document(
        "fax.tiff", _image((40, 30), frames=3, fmt="TIFF"), make_test_settings(max_ocr_pages=2)
    )
    assert [section.page for section in limited] == [1, 2]


def test_a_huge_image_is_scaled_down_and_a_bad_one_refused(
    fake_ocr: list[tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(image_parser, "OCR_MAX_PAGE_PIXELS", 100)
    image_parser.parse_image_document("big.png", _image((40, 30)))
    width, height = fake_ocr[-1]
    assert width * height <= 100

    with pytest.raises(DocumentParseError, match="Could not read image"):
        image_parser.parse_image_document("bad.png", b"not an image")
    monkeypatch.setattr(image_parser, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(DocumentParseError, match="megapixel"):
        image_parser.parse_image_document("bomb.png", _image((40, 30)))


def test_without_the_ocr_extra_an_image_says_how_to_get_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    with pytest.raises(EmptyDocumentError, match="uv sync --extra ocr"):
        image_parser.parse_image_document("scan.png", _image((4, 3)))


def test_a_json_citation_quotes_the_one_fact() -> None:
    from api.citations import attach_spans, source_citations
    from api.reranking import RerankedChunk

    document = {"service": {"name": "billing", "owners": ["ada", "grace"], "rps_limit": 50}}
    settings = make_test_settings()
    (chunk,) = chunk_sections(
        parse_document_bytes("service.json", json.dumps(document).encode(), settings),
        settings,
        HeuristicTokenCounter(),
    )
    assert chunk.text.split("\n") == [
        "service › name: billing",
        "service › owners: ada, grace",
        "service › rps_limit: 50",
    ]
    ranked = RerankedChunk(
        point_id="p",
        filename="service.json",
        page=1,
        section=chunk.section,
        chunk_id="c",
        text=chunk.text,
        retrieval_score=0.1,
        rerank_score=0.9,
    )
    [cited] = attach_spans(
        "The billing service rps_limit is 50 [1].", source_citations([ranked]), [ranked]
    )
    assert [span.quote for span in cited.spans] == ["service › rps_limit: 50"]
