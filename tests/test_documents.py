from __future__ import annotations

from typing import Any

import pytest

import api.documents as documents
from api.documents import (
    ChunkConfigError,
    DocumentSection,
    EmptyDocumentError,
    UnsupportedDocumentError,
    chunk_sections,
    infer_section,
    looks_like_markdown,
    merge_tiny_semantic_sections,
    parse_document_bytes,
    parse_markdown_document,
)
from api.settings import AppSettings


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "chunk_size_tokens": 4,
        "chunk_overlap_tokens": 1,
        "min_section_words": 0,
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


def test_parse_text_document_normalizes_filename_and_metadata() -> None:
    sections = parse_document_bytes("../notes.txt", b"Project Overview\nalpha beta gamma")

    assert sections == [
        DocumentSection(
            filename="notes.txt",
            page=1,
            section="Project Overview",
            text="Project Overview\nalpha beta gamma",
        )
    ]


def test_parse_markdown_document_uses_headings_as_sections() -> None:
    sections = parse_document_bytes(
        "guide.md",
        b"# Intro\nAlpha text.\n\n## Setup\nRun docker compose up.",
    )

    assert [section.section for section in sections] == ["Intro", "Setup"]
    assert [section.page for section in sections] == [1, 1]
    assert sections[1].text == "## Setup\nRun docker compose up."


def test_parse_markdown_document_supports_setext_h1_headings() -> None:
    sections = parse_markdown_document("guide.md", "Intro\n===\nAlpha text.")

    assert [section.section for section in sections] == ["Intro"]
    assert sections[0].text == "Intro\n===\nAlpha text."


def test_parse_markdown_document_does_not_treat_thematic_break_as_heading() -> None:
    """A bare '---' is a CommonMark thematic break, not a setext H2 — must not split."""

    sections = parse_markdown_document("guide.md", "Intro\nAlpha.\n\n---\n\nBeta.")

    # No ATX or setext-H1 heading anywhere -> no section split at all (the '---'
    # line is just body text, not a heading trigger).
    assert len(sections) == 1


def test_looks_like_markdown_requires_at_least_two_headings() -> None:
    assert looks_like_markdown("# Just one heading\nSome prose.") is False
    assert looks_like_markdown("# Heading one\nbody\n# Heading two\nbody") is True


def test_looks_like_markdown_recognizes_mixed_atx_and_setext() -> None:
    assert looks_like_markdown("Title\n===\nbody\n## Section\nmore body") is True


def test_txt_upload_with_two_or_more_headings_is_parsed_as_markdown() -> None:
    sections = parse_document_bytes(
        "notes.txt",
        b"# Intro\nHello.\n\n## Details\nMore text here.",
    )

    assert [section.section for section in sections] == ["Intro", "Details"]


def test_txt_upload_without_headings_splits_on_blank_line_paragraphs() -> None:
    sections = parse_document_bytes(
        "notes.txt",
        b"First paragraph line one.\nLine two.\n\nSecond paragraph here.",
    )

    assert len(sections) == 2
    assert sections[0].section == "First paragraph line one."
    assert sections[1].section == "Second paragraph here."


def test_txt_upload_without_blank_lines_stays_a_single_section() -> None:
    """No regression vs. prior behavior when there is nothing to split on."""

    sections = parse_document_bytes("notes.txt", b"Just one block of text, no blank lines.")

    assert len(sections) == 1


def test_chunk_sections_preserves_citation_payload_metadata() -> None:
    settings = make_settings(chunk_size_tokens=4, chunk_overlap_tokens=1)
    sections = [
        DocumentSection(
            filename="guide.md",
            page=2,
            section="Setup",
            text="one two three four five six seven",
        )
    ]

    chunks = chunk_sections(sections, settings)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
    ]
    first_payload = chunks[0].to_payload()
    assert first_payload["filename"] == "guide.md"
    assert first_payload["page"] == 2
    assert first_payload["section"] == "Setup"
    assert first_payload["chunk_id"] == chunks[0].chunk_id
    assert first_payload["text"] == "one two three four"


def test_chunk_ids_are_stable_for_same_source_and_text() -> None:
    settings = make_settings()
    section = DocumentSection(
        filename="same.txt",
        page=1,
        section="Same",
        text="alpha beta gamma delta epsilon",
    )

    first = chunk_sections([section], settings)
    second = chunk_sections([section], settings)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    settings = make_settings(chunk_size_tokens=4, chunk_overlap_tokens=4)
    section = DocumentSection(filename="bad.txt", page=1, section="Bad", text="alpha beta")

    # A ChunkConfigError (DocumentError subclass) maps to 400, not the bare
    # ValueError this used to raise, which surfaced as an unhandled 500.
    with pytest.raises(ChunkConfigError, match="CHUNK_OVERLAP_TOKENS"):
        chunk_sections([section], settings)


def test_parse_text_document_falls_back_to_cp1252_when_not_valid_utf8() -> None:
    """A Windows-exported .txt saved as cp1252 must not be rejected outright — only
    utf-8-sig was tried before, which raised DocumentParseError for this input."""

    content = "café notes".encode("cp1252")

    sections = parse_document_bytes("notes.txt", content)

    assert sections[0].text == "café notes"


def test_parse_text_document_falls_back_to_latin1_as_last_resort() -> None:
    # 0x81 is undefined in cp1252 but valid in latin-1 (maps to U+0081) — proves the
    # fallback chain reaches its final, always-succeeding rung rather than raising.
    content = b"bad\x81byte"

    sections = parse_document_bytes("notes.txt", content)

    assert sections[0].text == "bad\x81byte"


def test_empty_and_unsupported_documents_are_rejected() -> None:
    with pytest.raises(EmptyDocumentError):
        parse_document_bytes("empty.txt", b"  \n")

    with pytest.raises(UnsupportedDocumentError):
        parse_document_bytes("archive.zip", b"content")


def test_pdf_parser_preserves_page_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        def __init__(self, stream: object) -> None:
            self.stream = stream
            self.pages = [
                FakePage("Intro\nalpha beta"),
                FakePage(""),
                FakePage("Appendix\ngamma delta"),
            ]

    monkeypatch.setattr(documents, "PdfReader", FakeReader)

    sections = parse_document_bytes("paper.pdf", b"not real pdf")

    assert [(section.page, section.section) for section in sections] == [
        (1, "Intro"),
        (3, "Appendix"),
    ]


def test_chunk_sections_crosses_physical_page_boundary() -> None:
    """Overlap must carry across a PDF page break, not reset to zero at it."""

    settings = make_settings(chunk_size_tokens=4, chunk_overlap_tokens=1)
    sections = [
        DocumentSection(filename="paper.pdf", page=1, section="Intro", text="one two three"),
        DocumentSection(filename="paper.pdf", page=2, section="Middle", text="four five six seven"),
    ]

    chunks = chunk_sections(sections, settings)

    # A single continuous word stream: one two three four five six seven (7 words),
    # windowed size=4 step=3 -> windows [0:4], [3:7].
    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
    ]
    # Page/section attribution follows the window's first word (window-start rule).
    assert chunks[0].page == 1
    assert chunks[0].section == "Intro"
    assert chunks[1].page == 2
    assert chunks[1].section == "Middle"


def test_chunk_sections_crosses_paragraph_boundary() -> None:
    """Plain-text paragraph breaks are physical too — overlap must cross them."""

    settings = make_settings(chunk_size_tokens=4, chunk_overlap_tokens=1)
    sections = [
        DocumentSection(filename="notes.txt", page=1, section="First para", text="alpha beta"),
        DocumentSection(
            filename="notes.txt", page=1, section="Second para", text="gamma delta epsilon"
        ),
    ]

    chunks = chunk_sections(sections, settings)

    assert [chunk.text for chunk in chunks] == [
        "alpha beta gamma delta",
        "delta epsilon",
    ]


def test_chunk_sections_does_not_cross_markdown_heading_boundary() -> None:
    """Semantic (Markdown heading) boundaries must never be crossed by a window."""

    settings = make_settings(chunk_size_tokens=10, chunk_overlap_tokens=2)
    sections = [
        DocumentSection(
            filename="guide.md",
            page=1,
            section="Intro",
            text="one two three",
            boundary="semantic",
        ),
        DocumentSection(
            filename="guide.md",
            page=1,
            section="Setup",
            text="four five six",
            boundary="semantic",
        ),
    ]

    chunks = chunk_sections(sections, settings)

    # Each section windowed independently -> no chunk mixes words from both.
    assert [chunk.text for chunk in chunks] == ["one two three", "four five six"]
    assert [chunk.section for chunk in chunks] == ["Intro", "Setup"]


def test_chunk_sections_does_not_cross_different_filenames() -> None:
    """Physical sections from different files must never be windowed together."""

    settings = make_settings(chunk_size_tokens=10, chunk_overlap_tokens=2)
    sections = [
        DocumentSection(filename="a.txt", page=1, section="A", text="one two three"),
        DocumentSection(filename="b.txt", page=1, section="B", text="four five six"),
    ]

    chunks = chunk_sections(sections, settings)

    assert [chunk.text for chunk in chunks] == ["one two three", "four five six"]
    assert [chunk.filename for chunk in chunks] == ["a.txt", "b.txt"]


def test_chunk_sections_warns_on_oversized_chunk_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = make_settings(chunk_size_tokens=600, chunk_overlap_tokens=0)
    long_text = " ".join(f"word{i}" for i in range(600))
    sections = [DocumentSection(filename="dense.txt", page=1, section="Dense", text=long_text)]

    with caplog.at_level("WARNING", logger="api.documents"):
        chunks = chunk_sections(sections, settings)

    assert len(chunks) == 1
    assert any("may exceed" in record.message for record in caplog.records)


def test_chunk_sections_does_not_warn_on_ordinary_chunk_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = make_settings(chunk_size_tokens=300, chunk_overlap_tokens=75)
    ordinary_text = " ".join("word" for _ in range(300))
    sections = [
        DocumentSection(filename="normal.txt", page=1, section="Normal", text=ordinary_text)
    ]

    with caplog.at_level("WARNING", logger="api.documents"):
        chunk_sections(sections, settings)

    assert not any("may exceed" in record.message for record in caplog.records)


def test_merge_tiny_semantic_sections_folds_forward_into_next_section() -> None:
    tiny = DocumentSection(
        filename="guide.md", page=1, section="Notes", text="see above", boundary="semantic"
    )
    big = DocumentSection(
        filename="guide.md",
        page=1,
        section="Body",
        text=" ".join(f"word{i}" for i in range(50)),
        boundary="semantic",
    )

    merged = merge_tiny_semantic_sections([tiny, big], min_words=10)

    assert len(merged) == 1
    assert merged[0].section == "Body"
    assert "see above" in merged[0].text
    assert merged[0].text.endswith(big.text)


def test_merge_tiny_semantic_sections_folds_backward_when_trailing() -> None:
    big = DocumentSection(
        filename="guide.md",
        page=1,
        section="Body",
        text=" ".join(f"word{i}" for i in range(50)),
        boundary="semantic",
    )
    tiny = DocumentSection(
        filename="guide.md", page=1, section="Notes", text="see above", boundary="semantic"
    )

    merged = merge_tiny_semantic_sections([big, tiny], min_words=10)

    assert len(merged) == 1
    assert merged[0].section == "Body"
    assert merged[0].text.startswith(big.text)
    assert "see above" in merged[0].text


def test_merge_tiny_semantic_sections_leaves_physical_sections_untouched() -> None:
    physical = DocumentSection(filename="notes.txt", page=1, section="P", text="a")

    merged = merge_tiny_semantic_sections([physical], min_words=10)

    assert merged == [physical]


def test_infer_section_skips_page_number_first_line() -> None:
    assert infer_section("12\nReal Heading\nbody text") == "Real Heading"


def test_infer_section_skips_page_label_first_line() -> None:
    assert infer_section("Page 4\nReal Heading\nbody text") == "Real Heading"


def test_infer_section_skips_roman_numeral_first_line() -> None:
    # "xiv" is long enough to bypass the <3-char filter, isolating the roman-numeral
    # regex specifically (a shorter numeral like "iv" would be caught by length alone).
    assert infer_section("xiv\nReal Heading\nbody text") == "Real Heading"


def test_infer_section_falls_back_to_first_line_when_all_look_like_furniture() -> None:
    assert infer_section("12\n34") == "12"

