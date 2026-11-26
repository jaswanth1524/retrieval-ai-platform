"""Pass 22: MCP index_status counts every stale document, not one page's."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from cli.client import DocragClient
from cli.mcp_server import build_server
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait
from tests.test_mcp_server import text_of


@pytest.mark.asyncio
async def test_index_status_counts_every_stale_document(monkeypatch: Any) -> None:
    import api.routes.documents_read as documents_read

    clear_dependency_caches()
    app = hermetic_app(make_settings(), QdrantClient(":memory:"))
    with TestClient(app) as test_client:
        for name in ("a.txt", "b.txt", "c.txt"):
            body = f"Intro\\n{name} text".encode()
            assert upload_and_wait(test_client, name, body)["state"] == "done"
        monkeypatch.setattr(documents_read, "staleness", lambda *a: lambda meta: True)
        page = test_client.get("/documents", params={"limit": 1}).json()
        assert len(page["stale_filenames"]) == 1 and page["stale_total"] == 3
        assert test_client.get("/documents").json()["stale_total"] is None
        server = build_server(lambda: DocragClient("http://testserver", None, http=test_client))
        status = text_of(await server.call_tool("index_status", {}))
        assert "Documents: 3 (3 need re-indexing)" in status
    clear_dependency_caches()
