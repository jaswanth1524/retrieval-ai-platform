"""Security headers: a baseline on every response, a content policy on the UI's.

Plain ASGI rather than ``BaseHTTPMiddleware``, which runs the app in a task and pipes the
body through a stream: that would sit between the answer stream and the client, and
between a question's slot and the background task that frees it.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# On every response. nosniff stops a browser running an upload or an error body as
# script; with no Referer, a link in an answer doesn't tell its site which DocRAG
# instance, document or conversation it was clicked from.
BASELINE_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
)

# The UI's policy (api/static_files.py sets it on the HTML and assets it serves). Not
# applied to API responses: Swagger UI at /docs loads its script from a CDN and runs an
# inline one, and a JSON body never needs a policy to be safe.
#
# The build has no inline script or style element (fonts are bundled, React sets style
# props through the CSSOM), so 'self' covers it. frame-src blob: is the document
# viewer's PDF frame; frame-ancestors 'self' stops another site framing the UI to click
# its delete buttons. Not 'none': a blob: document inherits this policy, and WebKit
# (Safari) then refused to load the viewer's own PDF frame. X-Frame-Options: DENY on
# the UI's responses still keeps them out of any frame. connect-src 'self' means a
# frontend served elsewhere can't use this policy as-is (it is only sent on the UI this
# server serves itself).
# frontend/vite.config.ts sends the same string from `vite preview`, so the e2e suite runs
# under it; tests/routes_http/test_http_security_headers.py keeps the two equal.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
    "font-src 'self'; connect-src 'self'; frame-src blob:; object-src 'none'; "
    "frame-ancestors 'self'; base-uri 'self'; form-action 'self'"
)


class SecurityHeadersMiddleware:
    """Adds ``BASELINE_HEADERS`` to every HTTP response that doesn't set them itself."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {key.lower() for key, _ in headers}
                headers += [(key, value) for key, value in BASELINE_HEADERS if key not in present]
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)
