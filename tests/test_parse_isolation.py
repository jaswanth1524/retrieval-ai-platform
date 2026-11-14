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


def exit_without_a_result(connection: object, *args: object) -> None:
    """A child that dies the way an OOM kill or a native crash leaves it: no answer."""

    import os

    os._exit(3)


def test_a_child_that_dies_without_an_answer_is_a_parse_error_with_its_exit_code() -> None:
    with pytest.raises(DocumentParseError, match=r"exited unexpectedly \(exit code 3\)"):
        parse_with_limit(
            "notes.txt",
            b"Intro\nalpha",
            make_test_settings(parse_timeout_seconds=60),
            _target=exit_without_a_result,
        )


def test_a_child_that_cannot_start_closes_its_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    from multiprocessing.connection import Connection

    from api import parse_isolation

    closed: list[Connection] = []
    real_close = Connection.close

    def tracking_close(self: Connection) -> None:
        closed.append(self)
        real_close(self)

    class NoProcess:
        def __init__(self, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            raise OSError("too many open files")

    context = parse_isolation._forkserver_context()
    monkeypatch.setattr(Connection, "close", tracking_close)
    monkeypatch.setattr(type(context), "Process", NoProcess)
    with pytest.raises(OSError, match="too many open files"):
        parse_with_limit("notes.txt", b"x", make_test_settings(parse_timeout_seconds=60))
    assert len(closed) == 2


def report_an_environment_error(connection: object, *args: object) -> None:
    """A child whose parse failed on its surroundings (a full /tmp), not the file."""

    connection.send(("error", "OSError: [Errno 28] No space left on device"))  # type: ignore[attr-defined]


def test_a_child_failing_on_its_environment_is_transient_not_a_bad_document() -> None:
    from api.ingest_jobs import ingest_error_type
    from api.parse_isolation import ParseEnvironmentError

    with pytest.raises(ParseEnvironmentError) as raised:
        parse_with_limit(
            "notes.txt",
            b"Intro\nalpha",
            make_test_settings(parse_timeout_seconds=60),
            _target=report_an_environment_error,
        )
    assert ingest_error_type(raised.value) == "parse_failed"


def test_a_document_fault_in_a_real_child_stays_a_document_error() -> None:
    """Only the child's environment is transient: a file a parser chokes on (here a
    ValueError from a malformed workbook) still gives up as a bad document."""

    from api.ingest_jobs import ingest_error_type

    with pytest.raises(DocumentParseError) as raised:
        parse_with_limit(
            "broken.xlsx",
            b"PK\x03\x04 not really a workbook",
            make_test_settings(parse_timeout_seconds=60),
        )
    assert ingest_error_type(raised.value) == "document"
