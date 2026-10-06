from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.types import Message, Receive, Scope, Send

from api.dependencies import get_app_settings
from api.request_guard import RequestGuardMiddleware, is_guarded_path
from api.settings import AppSettings
from tests.factories import make_test_settings


def _settings(**overrides: Any) -> AppSettings:
    return make_test_settings(**overrides)


class _Downstream:
    """Stands in for the app: records whether it ran and reads the whole body."""

    def __init__(self) -> None:
        self.called = False
        self.body = b""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.called = True
        while True:
            message = await receive()
            self.body += message.get("body", b"")
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _run(
    settings: AppSettings,
    *,
    path: str = "/documents",
    method: str = "POST",
    headers: list[tuple[bytes, bytes]] | None = None,
    chunks: list[bytes] | None = None,
) -> tuple[_Downstream, list[Message], int]:
    downstream = _Downstream()
    host = FastAPI()
    host.dependency_overrides[get_app_settings] = lambda: settings
    guard = RequestGuardMiddleware(downstream)
    body_chunks = list(chunks or [b""])
    reads = 0

    async def receive() -> Message:
        nonlocal reads
        reads += 1
        chunk = body_chunks.pop(0) if body_chunks else b""
        return {"type": "http.request", "body": chunk, "more_body": bool(body_chunks)}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers or [],
        "app": host,
    }
    asyncio.run(guard(scope, receive, send))
    return downstream, sent, reads


def _status(sent: list[Message]) -> int:
    return next(message["status"] for message in sent if message["type"] == "http.response.start")


def test_a_missing_key_is_rejected_before_any_body_is_read() -> None:
    downstream, sent, reads = _run(_settings(api_key="secret"), chunks=[b"x" * 10])

    assert _status(sent) == 401
    assert reads == 0
    assert not downstream.called


def test_a_matching_key_passes_through() -> None:
    downstream, sent, _ = _run(
        _settings(api_key="secret"), headers=[(b"x-api-key", b"secret")], chunks=[b"abc"]
    )

    assert _status(sent) == 200
    assert downstream.body == b"abc"


def test_a_non_ascii_key_is_a_401_not_a_crash() -> None:
    _, sent, _ = _run(_settings(api_key="secret"), headers=[(b"x-api-key", "clé".encode())])

    assert _status(sent) == 401


def test_a_declared_oversized_upload_is_413_before_its_body_is_read() -> None:
    settings = _settings(max_upload_bytes=1000)
    downstream, sent, reads = _run(settings, headers=[(b"content-length", b"999999")])

    assert _status(sent) == 413
    assert reads == 0
    assert not downstream.called


def test_an_undeclared_body_is_cut_off_once_it_passes_the_limit() -> None:
    _, sent, reads = _run(
        _settings(), path="/questions", chunks=[b"x" * 700_000, b"x" * 700_000, b"x" * 700_000]
    )

    assert _status(sent) == 413
    assert reads == 2  # stopped at the chunk that crossed the limit
    assert [message["status"] for message in sent if "status" in message] == [413]


def test_open_routes_are_never_guarded() -> None:
    for path in ("/health", "/health/ready", "/config", "/", "/assets/app.js", "/documentsx"):
        assert not is_guarded_path(path)
    for path in ("/documents", "/documents/a.pdf/original", "/questions/stream", "/metrics"):
        assert is_guarded_path(path)

    downstream, sent, _ = _run(_settings(api_key="secret"), path="/config", method="GET")
    assert downstream.called
    assert _status(sent) == 200


@pytest.mark.parametrize("method", ["OPTIONS"])
def test_cors_preflights_pass_through(method: str) -> None:
    downstream, _, _ = _run(_settings(api_key="secret"), method=method)

    assert downstream.called


def test_the_read_only_key_is_turned_away_from_writes_before_the_body() -> None:
    settings = _settings(api_key="full", api_read_key="read")

    downstream, sent, reads = _run(settings, headers=[(b"x-api-key", b"read")], chunks=[b"x"])
    assert _status(sent) == 403
    assert reads == 0 and not downstream.called

    downstream, sent, _ = _run(
        settings, path="/questions", headers=[(b"x-api-key", b"read")], chunks=[b"{}"]
    )
    assert downstream.called and _status(sent) == 200


def test_full_key_routes() -> None:
    from api.request_guard import needs_full_key

    for method, path in [
        ("POST", "/documents"),
        ("POST", "/documents/reindex"),
        ("POST", "/documents/a.pdf/reindex"),
        ("PATCH", "/documents/a.pdf/tags"),
        ("DELETE", "/documents/a.pdf"),
        ("GET", "/metrics"),
        ("GET", "/feedback"),
        ("GET", "/export"),
    ]:
        assert needs_full_key(method, path), (method, path)
    for method, path in [
        ("GET", "/documents"),
        ("GET", "/documents/a.pdf/content"),
        ("GET", "/documents/jobs/x"),
        ("POST", "/questions/stream"),
        ("POST", "/feedback"),
        ("GET", "/traces"),
    ]:
        assert not needs_full_key(method, path), (method, path)


def test_two_api_key_headers_are_a_400() -> None:
    """The guard judged the last copy and the route's dependency the first, so the two
    checks could see different keys."""

    downstream, sent, _ = _run(
        _settings(api_key="secret"),
        headers=[(b"x-api-key", b"secret"), (b"X-API-Key", b"other")],
    )

    assert _status(sent) == 400
    assert not downstream.called
