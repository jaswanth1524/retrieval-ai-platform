"""Request ids: one per HTTP request, on the response and in every log line it causes.

An incoming ``X-Request-ID`` (from a proxy or a client retrying) is kept when it is a
sane token; otherwise one is generated. It lives in a context variable for the
request's duration, which the SSE producer thread and ingest jobs copy, so a question
or an upload can be followed from the access log through retrieval to its trace.
"""

from __future__ import annotations

import contextvars
import logging
import re
import uuid
from collections.abc import Callable
from concurrent.futures import Executor, Future
from contextvars import ContextVar

from starlette.types import ASGIApp, Message, Receive, Scope, Send

HEADER = "X-Request-ID"
_VALID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

request_id_var: ContextVar[str | None] = ContextVar("docrag_request_id", default=None)


def current_request_id() -> str | None:
    return request_id_var.get()


def submit_in_context[R](executor: Executor, function: Callable[[], R]) -> Future[R]:
    """``executor.submit(function)``, run in a copy of the caller's context.

    A pool thread starts with an empty context, so its log lines lost the request id.
    One copy per task: a single Context can't be entered by two threads at once.
    """

    return executor.submit(contextvars.copy_context().run, function)


class RequestIdMiddleware:
    """Outermost middleware: sets the request id before anything else runs."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = dict(scope.get("headers", [])).get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if _VALID.match(supplied) else uuid.uuid4().hex
        token = request_id_var.set(request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


class RequestIdFilter(logging.Filter):
    """Stamps ``record.request_id`` ("-" outside a request) for the log formats."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True
