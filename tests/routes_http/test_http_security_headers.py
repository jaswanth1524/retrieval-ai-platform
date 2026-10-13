"""Security headers on API, stream and UI responses (api/security_headers.py)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.security_headers import CONTENT_SECURITY_POLICY
from tests.routes_http.support import ApiTestContext, hermetic_app, make_settings

VITE_CONFIG = Path(__file__).resolve().parents[2] / "frontend" / "vite.config.ts"


def test_every_api_response_gets_the_baseline_headers(api_context: ApiTestContext) -> None:
    for response in (
        api_context.client.get("/health"),
        api_context.client.get("/documents/missing.txt/content"),
        api_context.client.post("/questions", json={"question": ""}),
    ):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        # The page policy is the UI's alone: Swagger UI at /docs needs a CDN script.
        assert "content-security-policy" not in response.headers
    docs = api_context.client.get("/docs")
    assert "content-security-policy" not in docs.headers


def test_the_answer_stream_keeps_its_headers_and_gains_the_baseline(
    api_context: ApiTestContext,
) -> None:
    with api_context.client.stream(
        "POST", "/questions/stream", json={"question": "alpha"}
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"
        assert response.headers["x-content-type-options"] == "nosniff"
        frames = [line for line in response.iter_lines() if line.startswith("data:")]
    assert frames  # still streamed, not swallowed


def test_the_ui_is_served_with_the_content_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.main as main_module

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>DocRAG</title>")
    script = "console.log('docrag');" * 100
    (dist / "assets" / "app.js").write_text(script)
    (dist / "assets" / "app.js.br").write_bytes(b"compressed")
    monkeypatch.setattr(main_module, "FRONTEND_DIST_DIR", dist)
    clear_dependency_caches()
    app = hermetic_app(make_settings(), QdrantClient(":memory:"))
    with TestClient(app) as client:
        responses = [
            client.get("/"),
            client.get("/assets/app.js", headers={"Accept-Encoding": "br"}),
            client.get("/assets/app.js", headers={"Accept-Encoding": "identity"}),
        ]
    clear_dependency_caches()
    assert responses[1].headers["content-encoding"] == "br"
    for response in responses:
        assert response.status_code == 200
        assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["x-content-type-options"] == "nosniff"


def test_vite_preview_sends_the_same_policy() -> None:
    match = re.search(
        r"const CONTENT_SECURITY_POLICY =\s*((?:\"[^\"]*\"\s*\+?\s*)+);", VITE_CONFIG.read_text()
    )
    assert match is not None
    vite_policy = "".join(re.findall(r"\"([^\"]*)\"", match.group(1)))
    assert vite_policy == CONTENT_SECURITY_POLICY


def test_the_policy_lets_the_page_frame_its_own_pdf_and_nothing_else() -> None:
    # The viewer's PDF is a blob: frame, which inherits this policy: with
    # frame-ancestors 'none' WebKit refused to load it. Other sites still can't frame
    # the UI ('self', plus X-Frame-Options: DENY on every UI response).
    assert "frame-ancestors 'self'" in CONTENT_SECURITY_POLICY
    assert "frame-src blob:" in CONTENT_SECURITY_POLICY
    assert "'unsafe-inline'" not in CONTENT_SECURITY_POLICY
