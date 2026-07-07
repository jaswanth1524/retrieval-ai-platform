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
    parse_document_bytes,
)
from api.settings import AppSettings


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "chunk_size_tokens": 4,
        "chunk_overlap_tokens": 1,
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


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

