from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.mcpserver import MCPServer
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from cli.client import DocragClient, StreamResult
from cli.mcp_server import build_server, format_answer, format_passages
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait


@pytest.fixture
def server() -> Iterator[MCPServer]:
    clear_dependency_caches()
    settings = make_settings(api_key="full-key", api_read_key="read-key", followups_enabled=True)
    app = hermetic_app(settings, QdrantClient(":memory:"))
    with TestClient(app) as test_client:
        headers = {"X-API-Key": "full-key"}
        for name, body in (("guide.txt", b"Intro\nAlpha is the first letter."),):
            assert upload_and_wait(test_client, name, body, headers=headers)["state"] == "done"
        # The read-only key: every tool works with it.
        yield build_server(lambda: DocragClient("http://testserver", "read-key", http=test_client))
    clear_dependency_caches()


def text_of(result: Any) -> str:
    return "\n".join(part.text for part in result.content if getattr(part, "text", None))


@pytest.mark.asyncio
async def test_every_tool_is_listed_as_read_only(server: MCPServer) -> None:
    tools = {tool.name: tool for tool in await server.list_tools()}

    assert set(tools) == {"list_documents", "search_passages", "ask", "summarize", "read_document"}
    assert all(tool.annotations and tool.annotations.read_only_hint for tool in tools.values())


@pytest.mark.asyncio
async def test_tools_answer_from_the_indexed_documents_with_citations(server: MCPServer) -> None:
    listing = text_of(await server.call_tool("list_documents", {}))
    assert listing == "guide.txt (1 passages)"

    passages = text_of(await server.call_tool("search_passages", {"query": "alpha"}))
    assert passages.startswith("1. guide.txt p.1") and "Alpha is the first letter" in passages

    answer = text_of(await server.call_tool("ask", {"question": "alpha"}))
    assert answer.startswith("Alpha is documented [1].")
    assert "[1] guide.txt p.1" in answer

    summary = text_of(
        await server.call_tool("summarize", {"filenames": ["guide.txt"], "focus": "letters"})
    )
    assert "Sources:" in summary

    document = text_of(await server.call_tool("read_document", {"filename": "guide.txt"}))
    assert "guide.txt: passages of 1" in document and "Alpha is the first letter" in document


@pytest.mark.asyncio
async def test_server_refusals_come_back_as_readable_text(server: MCPServer) -> None:
    missing = text_of(await server.call_tool("summarize", {"filenames": ["missing.txt"]}))
    assert missing == "DocRAG error (404): No indexed document named 'missing.txt'."
    too_many = text_of(await server.call_tool("summarize", {"filenames": ["a", "b", "c"]}))
    assert "one document, or two" in too_many
    past_end = text_of(
        await server.call_tool("read_document", {"filename": "guide.txt", "start": 50})
    )
    assert past_end == "No passages from 50 on (the document has 1)."


def test_formatting_notes_partial_reads_and_status() -> None:
    result = StreamResult(
        {
            "answer": "x",
            "sources": [],
            "answer_status": "uncited",
            "chunks_read": 4,
            "chunks_total": 9,
        },
        ["What else?"],
    )
    text = format_answer(result)
    assert "Answer status: uncited" in text and "Read 4 of 9 passages" in text
    assert text.endswith("Suggested follow-ups: What else?")
    assert format_passages([]) == "No passages matched."
    long = format_passages(
        [
            {
                "rank": 1,
                "filename": "a",
                "page": 2,
                "page_end": 3,
                "text": "w " * 2000,
                "rerank_score": 0.5,
            }
        ]
    )
    assert "a p.2-3" in long and long.endswith("…")


@pytest.mark.asyncio
async def test_the_stdio_server_starts_and_lists_its_tools() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "cli.mcp_server"],
        env={"DOCRAG_URL": "http://127.0.0.1:9", "LITELLM_MODE": "PRODUCTION"},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        assert {tool.name for tool in tools.tools} >= {"ask", "summarize", "search_passages"}
        # Nothing listens on port 9: the tool says so instead of failing the session.
        result = await session.call_tool("list_documents", {})
        assert "Cannot reach DocRAG" in text_of(result)
