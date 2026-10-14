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


def _serve_one(app: object, headers: list[tuple[bytes, bytes]]) -> None:
    import asyncio

    from api.request_id import RequestIdMiddleware

    async def endpoint(scope: object, receive: object, send: object) -> None:
        await send({"type": "http.response.start", "status": 204, "headers": []})  # type: ignore[operator]
        await send({"type": "http.response.body", "body": b""})  # type: ignore[operator]

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: object) -> None:
        return None

    scope = {"type": "http", "method": "GET", "path": "/documents", "headers": headers}
    asyncio.run(RequestIdMiddleware(endpoint)(scope, receive, send))  # type: ignore[arg-type]


def test_json_logs_write_one_access_line_per_request_with_its_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """uvicorn's access line can't carry the request id, so under JSON it is replaced."""

    import json

    configure_logging("INFO", "json")
    try:
        assert not logging.getLogger("uvicorn.access").isEnabledFor(logging.INFO)
        _serve_one(object(), [(b"x-request-id", b"req-7")])
    finally:
        configure_logging("INFO")

    line = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert line["logger"] == "api.access"
    assert line["request_id"] == "req-7"
    assert line["http"]["method"] == "GET"
    assert line["http"]["path"] == "/documents"
    assert line["http"]["status"] == 204
    assert line["http"]["duration_ms"] >= 0


def test_text_logs_leave_the_access_line_to_uvicorn(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    _serve_one(object(), [])

    assert "api.access" not in capsys.readouterr().err
    assert not logging.getLogger("api.access").isEnabledFor(logging.INFO)
