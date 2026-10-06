"""ASGI middleware that rejects unauthorized or oversized requests before their body.

FastAPI reads (and for multipart, spools to disk) a request's whole body *before* it
runs route dependencies — ``require_api_key`` included. So with ``API_KEY`` set, an
unauthenticated client could still stream gigabytes at ``POST /documents`` and fill the
temp disk before being told 401, and ``read_upload_within_limit`` bounded memory only
after the spool had already happened. This checks the key and the size from the
headers alone, and counts bytes for a body sent without a ``Content-Length``.

``require_api_key`` stays on every guarded route as defence in depth; both share
``api_key_matches`` so the two can't disagree.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.dependencies import READ_ONLY_KEY_ERROR, get_app_settings, key_access
from api.settings import AppSettings

# Every route under these prefixes (and /metrics) is key-guarded; see CLAUDE.md. /health,
# /health/ready, /config and the static frontend stay open.
GUARDED_PREFIXES = ("/documents", "/questions", "/traces", "/feedback")
GUARDED_EXACT = frozenset({"/metrics", "/export", "/access", "/search", "/import"})
# Room for the multipart boundary and part headers around an upload's own bytes.
UPLOAD_OVERHEAD_BYTES = 64 * 1024
# JSON bodies (a question with its history, feedback) are a few tens of KB at most.
MAX_JSON_BODY_BYTES = 1024 * 1024


def is_guarded_path(path: str) -> bool:
    if path in GUARDED_EXACT:
        return True
    return any(path == prefix or path.startswith(prefix + "/") for prefix in GUARDED_PREFIXES)


_REINDEX_OR_TAGS = re.compile(r"^/documents/[^/]+/(reindex|tags)$")


def needs_full_key(method: str, path: str) -> bool:
    """Routes the read-only key (API_READ_KEY) may not use: changes to the corpus, and
    operator data (feedback, metrics, the full export)."""

    if path in ("/metrics", "/export") or (path == "/feedback" and method == "GET"):
        return True
    if method == "POST" and path == "/import":
        return True
    if method == "DELETE" and path.startswith("/documents/"):
        return True
    if method == "POST" and path in ("/documents", "/documents/reindex"):
        return True
    return method in ("POST", "PATCH") and _REINDEX_OR_TAGS.match(path) is not None


def body_limit(method: str, path: str, settings: AppSettings) -> int:
    if method == "POST" and path == "/documents":
        return int(settings.max_upload_bytes) + UPLOAD_OVERHEAD_BYTES
    if method == "POST" and path == "/import":
        return int(settings.max_import_bytes) + UPLOAD_OVERHEAD_BYTES
    return MAX_JSON_BODY_BYTES


class _BodyTooLarge(Exception):
    pass


class RequestGuardMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def _settings(self, scope: Scope) -> AppSettings:
        # Through the app's dependency overrides, so tests (and anything else that
        # overrides get_app_settings) see the same settings the routes do.
        app = scope.get("app")
        overrides: dict[Callable[..., Any], Callable[..., Any]] = getattr(
            app, "dependency_overrides", {}
        )
        provider = overrides.get(get_app_settings, get_app_settings)
        settings: AppSettings = provider()
        return settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        method = scope.get("method", "")
        if scope["type"] != "http" or method == "OPTIONS" or not is_guarded_path(path):
            await self.app(scope, receive, send)
            return

        settings = self._settings(scope)
        raw_headers = scope.get("headers", [])
        headers = {key.lower(): value for key, value in raw_headers}
        if settings.api_key:
            # The dict above keeps the last copy and FastAPI's dependency reads the
            # first; one header, so both checks judge the same key.
            if sum(1 for key, _ in raw_headers if key.lower() == b"x-api-key") > 1:
                await _reject(send, 400, "Send a single X-API-Key header.")
                return
            access = key_access(settings, headers.get(b"x-api-key"))
            if access == "none":
                await _reject(send, 401, "Missing or invalid API key.")
                return
            if access == "read" and needs_full_key(method, path):
                await _reject(send, 403, READ_ONLY_KEY_ERROR)
                return

        limit = body_limit(method, path, settings)
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                too_large = int(declared) > limit
            except ValueError:
                await _reject(send, 400, "Invalid Content-Length header.")
                return
            if too_large:
                await _reject(send, 413, _too_large_detail(limit))
                return

        received = 0
        exceeded = False
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            # The app may turn our exception into its own error response (FastAPI maps
            # a body it can't read to a 400); answer with the real reason instead.
            if exceeded:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except _BodyTooLarge:
            pass
        if exceeded and not response_started:
            await _reject(send, 413, _too_large_detail(limit))


def _too_large_detail(limit: int) -> str:
    return f"Request body is larger than the {limit}-byte limit."


async def _reject(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
