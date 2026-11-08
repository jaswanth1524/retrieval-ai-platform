"""Pass 15 formats: OpenDocument spreadsheets and presentations, mailboxes, notebooks."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from api.documents import DocumentParseError, parse_document_bytes
from api.parsers.common import EmptyDocumentError
from tests.factories import make_test_settings
from tests.samples import ipynb_bytes, mbox_bytes, odf_bytes, odp_slide, ods_row


def _parse(name: str, content: bytes, **settings: object) -> list:  # type: ignore[type-arg]
    return parse_document_bytes(name, content, make_test_settings(**settings))


def test_a_spreadsheet_keeps_row_numbers_and_skips_padding() -> None:
    content = odf_bytes(
        "spreadsheet",
        '<table:table table:name="Team">'
        + ods_row("Name", "Role")
        + ods_row("Ada", "Lead")
        # A sheet's padding: a million empty rows, never written out.
        + ods_row(repeat=1_000_000)
        + ods_row("Lin", "Ops")
        + "</table:table>"
        + '<table:table table:name="Empty"></table:table>',
    )

    (section,) = _parse("team.ods", content)

    assert section.page == 1 and section.section == "Team · Rows 2-1000003"
    assert section.text == "Name: Ada; Role: Lead\nName: Lin; Role: Ops"
    assert section.line_rows == (2, 1000003)


def test_a_spreadsheet_past_the_cell_limit_is_refused() -> None:
    content = odf_bytes(
        "spreadsheet",
        '<table:table table:name="Big">' + ods_row("a", "b", repeat=50) + "</table:table>",
    )
    with pytest.raises(DocumentParseError, match="MAX_SPREADSHEET_CELLS"):
        _parse("big.ods", content, max_spreadsheet_cells=20)


def test_slides_are_pages_titled_by_their_title_frame() -> None:
    content = odf_bytes(
        "presentation",
        odp_slide("Plan", ["Ship it."]) + odp_slide(None, []) + odp_slide(None, ["Q&amp;A"]),
    )

    sections = _parse("deck.odp", content)

    assert [(s.page, s.section, s.text) for s in sections] == [
        (1, "Plan", "Plan\nShip it."),
        (3, "Slide 3", "Q&A"),
    ]
    with pytest.raises(EmptyDocumentError):
        _parse("blank.odp", odf_bytes("presentation", odp_slide(None, [])))


def test_a_doctype_in_an_open_document_is_refused() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("content.xml", '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><x/>')
    for name in ("x.ods", "x.odp"):
        with pytest.raises(DocumentParseError):
            _parse(name, buffer.getvalue())


def test_a_mailbox_is_one_page_per_message() -> None:
    content = mbox_bytes(
        [("Hello", "First line.\nFrom the start, escaped."), ("Second", "More text.")]
    )

    sections = _parse("box.mbox", content)

    assert {s.page for s in sections} == {1, 2}
    bodies = [s.text for s in sections if not s.text.startswith("Subject:")]
    assert bodies[0] == "First line.\nFrom the start, escaped."
    with pytest.raises(DocumentParseError, match="not an mbox"):
        _parse("not.mbox", b"Subject: hi\n\nbody")


def test_a_notebook_reads_as_markdown_with_its_code_and_without_outputs() -> None:
    content = ipynb_bytes([("markdown", "# Intro\nWhy."), ("code", "print(1)"), ("raw", "x")])

    sections = _parse("n.ipynb", content)
    text = "\n".join(section.text for section in sections)

    assert sections[0].section == "Intro"
    assert "```python\nprint(1)\n```" in text and "SECRET OUTPUT" not in text
    with pytest.raises(DocumentParseError):
        _parse("bad.ipynb", b"{not json")
    with pytest.raises(DocumentParseError, match="no cells"):
        _parse("bad.ipynb", json.dumps({"cells": "x"}).encode())
    with pytest.raises(EmptyDocumentError):
        _parse("empty.ipynb", json.dumps({"cells": []}).encode())
