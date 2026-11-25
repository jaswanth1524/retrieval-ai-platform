"""Pass 21: MCP index_status, tag_document and delete_document."""

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
async def test_index_status_and_the_tag_and_delete_tools() -> None:
    clear_dependency_caches()
    settings = make_settings(api_key="full-key", api_read_key="read-key")
    app = hermetic_app(settings, QdrantClient(":memory:"))
    with TestClient(app) as test_client:
        headers = {"X-API-Key": "full-key"}
        for name in ("a.txt", "b.txt"):
            body = f"Intro\\n{name} text".encode()
            assert upload_and_wait(test_client, name, body, headers=headers)["state"] == "done"
        reader = build_server(
            lambda: DocragClient("http://testserver", "read-key", http=test_client)
        )
        status = text_of(await reader.call_tool("index_status", {}))
        assert status.startswith("Status: ") and "Documents: 2" in status
        assert "Indexing jobs: 0 queued or running" in status
        assert "tag_document" not in {tool.name for tool in await reader.list_tools()}

        writer = build_server(
            lambda: DocragClient("http://testserver", "full-key", http=test_client),
            allow_writes=True,
        )
        tools: dict[str, Any] = {tool.name: tool for tool in await writer.list_tools()}
        assert tools["delete_document"].annotations.destructive_hint is True
        tagged = text_of(
            await writer.call_tool("tag_document", {"filename": "a.txt", "tags": ["hr"]})
        )
        assert tagged == "a.txt: hr"
        refused = text_of(
            await writer.call_tool(
                "delete_document", {"filename": "a.txt", "confirm_filename": "b.txt"}
            )
        )
        assert refused.startswith("Nothing was deleted")
        deleted = text_of(
            await writer.call_tool(
                "delete_document", {"filename": "a.txt", "confirm_filename": "a.txt"}
            )
        )
        assert deleted == "Deleted a.txt."
        assert test_client.get("/documents", headers=headers).json()["filenames"] == ["b.txt"]
    clear_dependency_caches()
