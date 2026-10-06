from __future__ import annotations

import logging

import pytest

from api.logging_config import configure_logging


def test_configure_logging_sets_api_logger_level() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger("api").level == logging.DEBUG

    configure_logging("WARNING")
    assert logging.getLogger("api").level == logging.WARNING


def test_configure_logging_configures_eval_namespace() -> None:
    configure_logging("INFO")
    assert logging.getLogger("eval").level == logging.INFO


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("INFO")
    handlers = logging.getLogger("api").handlers
    assert len(handlers) == 1


def test_configure_logging_leaves_api_logger_propagating() -> None:
    # propagate must stay True so pytest's caplog (root-attached) still sees api.* records.
    configure_logging("INFO")
    assert logging.getLogger("api").propagate is True


def test_json_log_lines_carry_the_request_id(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from api.request_id import request_id_var

    configure_logging("INFO", "json")
    token = request_id_var.set("req-42")
    try:
        logging.getLogger("api.test").info("hello %s", "world")
    finally:
        request_id_var.reset(token)
        configure_logging("INFO")

    line = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert line["message"] == "hello world"
    assert line["request_id"] == "req-42"
    assert line["level"] == "INFO"


def test_concurrent_query_variants_keep_the_request_id_in_their_threads() -> None:
    """Pool threads start with an empty context: with query expansion on, repository
    log lines from the variant searches had no request id."""

    import threading

    from api.pipeline import _map_concurrently
    from api.request_id import current_request_id, request_id_var

    gate = threading.Barrier(3)

    def seen(_: int) -> str | None:
        gate.wait(5)  # all three run at once, on pool threads
        return current_request_id()

    token = request_id_var.set("req-123")
    try:
        assert _map_concurrently(seen, [1, 2, 3]) == ["req-123"] * 3
    finally:
        request_id_var.reset(token)
