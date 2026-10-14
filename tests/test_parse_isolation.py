"""Parsing in a child process (api/parse_isolation.py), with a real forkserver child."""

from __future__ import annotations

import multiprocessing

import pytest

from api.documents import DocumentParseError, EmptyDocumentError, parse_document_bytes
from api.parse_isolation import parse_with_limit
from tests.factories import make_test_settings
from tests.samples import xlsx_bytes


def test_a_child_parses_the_same_sections_as_the_api_process() -> None:
    settings = make_test_settings(parse_timeout_seconds=60)
    content = xlsx_bytes({"Leave": [["Name", "Days"], ["Ada", 27]]})

    assert parse_with_limit("leave.xlsx", content, settings) == parse_document_bytes(
        "leave.xlsx", content, settings
    )


def test_a_document_error_in_the_child_is_raised_as_itself() -> None:
    with pytest.raises(EmptyDocumentError):
        parse_with_limit("empty.txt", b"   \n  ", make_test_settings(parse_timeout_seconds=60))


def test_a_parse_past_the_limit_is_stopped_and_its_process_killed() -> None:
    before = {child.pid for child in multiprocessing.active_children()}

    with pytest.raises(DocumentParseError, match="PARSE_TIMEOUT_SECONDS"):
        # No child can start, parse and answer within a millisecond.
        parse_with_limit(
            "notes.txt", b"Intro\nalpha", make_test_settings(parse_timeout_seconds=0.001)
        )

    leftover = [
        child
        for child in multiprocessing.active_children()
        if child.pid not in before and child.name == "docrag-parse"
    ]
    assert leftover == []


def test_zero_parses_in_this_process() -> None:
    sections = parse_with_limit("notes.txt", b"Intro\nalpha", make_test_settings())
    assert sections[0].text.endswith("alpha")
