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


def test_malformed_odt_counts_and_deep_nesting_are_read_safely() -> None:
    spaces = '<text:p>a<text:s text:c="x"/>b<text:s text:c="2000000000"/>c</text:p>'
    table = (
        "<table:table><table:table-row>"
        '<table:table-cell table:number-columns-repeated="many"><text:p>Only</text:p>'
        "</table:table-cell></table:table-row></table:table>"
    )
    deep_inline = "<text:p>" + "<text:span>" * 2000 + "deep inline" + "</text:span>" * 2000
    deep_inline += "</text:p>"
    deep_blocks = "<text:section>" * 2000 + "<text:p>deep block</text:p>"
    deep_blocks += "</text:section>" * 2000
    sections = parse_document_bytes(
        "crafted.odt", odt_bytes(spaces + table + deep_inline + deep_blocks)
    )
    text = " ".join(section.text for section in sections)
    # A bad count is one space; a two-billion one is capped, not gigabytes of spaces.
    assert "a b" in text and len(text) < 500
    assert "Only" in text and "deep inline" in text and "deep block" in text


def test_a_library_error_on_a_malformed_file_is_a_document_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api import documents

    def broken(filename: str, content: bytes, settings: object = None) -> Any:
        raise ValueError("invalid literal for int() with base 10: 'x'")

    monkeypatch.setitem(documents.PARSERS, ".txt", broken)
    # A DocumentError, so the job's error_type is "document" and the folder watcher
    # stops retrying the file instead of retrying it forever.
    with pytest.raises(DocumentParseError, match="Could not parse 'odd.txt'"):
        parse_document_bytes("odd.txt", b"x")


def test_a_json_number_too_long_to_read_is_named() -> None:
    with pytest.raises(DocumentParseError, match="number too long"):
        parse_document_bytes("big.json", b'{"n": 1' + b"1" * 5000 + b"}")


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


def _png_header(width: int, height: int) -> bytes:
    """A PNG that only declares its size: no pixel data, a few bytes on disk."""

    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def test_a_declared_huge_canvas_is_refused_with_its_size(fake_ocr: list[tuple[int, int]]) -> None:
    # Over this app's limit but under Pillow's refusal: the app's check names the size.
    with pytest.raises(DocumentParseError, match="10000 x 10000 pixels"):
        image_parser.parse_image_document("bomb.png", _png_header(10_000, 10_000))
    # Past Pillow's own refusal at open: still the limit, not "could not read".
    with pytest.raises(DocumentParseError, match="megapixel limit"):
        image_parser.parse_image_document("bomb.png", _png_header(20_000, 20_000))
    assert fake_ocr == []


def test_a_tiff_frame_over_the_limit_is_skipped_not_decoded(
    fake_ocr: list[tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    buffer = io.BytesIO()
    small, large = Image.new("RGB", (40, 30), "white"), Image.new("RGB", (100, 100), "white")
    small.save(buffer, format="TIFF", save_all=True, append_images=[large, small])
    monkeypatch.setattr(image_parser, "MAX_IMAGE_PIXELS", 5000)

    sections = image_parser.parse_image_document("fax.tiff", buffer.getvalue())
    assert [section.page for section in sections] == [1, 3]
    assert fake_ocr == [(40, 30), (40, 30)]


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
