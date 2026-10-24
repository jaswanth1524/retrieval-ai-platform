from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from typing import Any

import httpx
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

    assert set(tools) == {
        "list_documents",
        "search_passages",
        "ask",
        "summarize",
        "read_document",
        "list_workspaces",
    }
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
    none = text_of(await server.call_tool("summarize", {"filenames": []}))
    assert "at least one document" in none
    brief = text_of(await server.call_tool("summarize", {"filenames": ["a", "b", "c"]}))
    assert brief.startswith("DocRAG error (404)")
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
    weak_answer = format_answer(
        StreamResult({"answer": "x [1]", "sources": [], "weak_matches": True}, [])
    )
    assert "Weak match: no passage matched the question closely" in weak_answer
    assert text.endswith("Suggested follow-ups: What else?")
    assert format_passages([]) == "No passages matched."
    weak = format_passages([{"rank": 1, "filename": "a.txt", "text": "x"}], weak=True)
    assert weak.startswith("No passage clearly matches") and "1. a.txt" in weak
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


def mock_server(handler: Any) -> MCPServer:
    return build_server(
        lambda: DocragClient("http://x", "", transport=httpx.MockTransport(handler))
    )


@pytest.mark.asyncio
async def test_an_error_after_the_answer_still_returns_the_answer() -> None:
    """The follow-ups failing (or the connection dropping) after `done` lost the answer."""

    done = {"type": "done", "answer": "Kept [1].", "sources": [], "answer_status": "answered"}
    error = {"type": "error", "detail": "follow-ups failed"}
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in (done, error))
    server = mock_server(lambda _: httpx.Response(200, text=body))

    assert text_of(await server.call_tool("ask", {"question": "q"})) == "Kept [1]."
    summary = text_of(await server.call_tool("summarize", {"filenames": ["a.txt"]}))
    assert summary == "Kept [1]."


@pytest.mark.asyncio
async def test_read_document_caps_what_it_returns_and_says_where_to_go_on() -> None:
    """A page of 100 long passages came back whole: hundreds of kilobytes in one result."""

    requested: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        requested.append(params)
        first, last = int(params["start"]), int(params["end"])
        chunks = [
            {
                "chunk_ordinal": n,
                "page": 1,
                "section": "S",
                "text": ("x" * 50_000) if n == 1 else f"passage {n} " + "w" * 2_000,
            }
            for n in range(first, min(last, 300) + 1)
        ]
        return httpx.Response(200, json={"chunks": chunks, "total_chunks": 300})

    server = mock_server(handler)

    first = text_of(await server.call_tool("read_document", {"filename": "a.txt", "count": 100}))
    assert len(first) < 30_000
    assert "[passage cut]" in first and "passage 2 " in first
    shown = int(first.rsplit("more with start=", 1)[1].rstrip(".]")) - 1
    assert 2 <= shown < 100
    assert f"[Showing passages up to {shown} of 300; more with start={shown + 1}.]" in first

    later = text_of(
        await server.call_tool(
            "read_document", {"filename": "a.txt", "start": shown + 1, "count": 100}
        )
    )
    assert f"[{shown + 1}] p.1" in later and "[passage cut]" not in later

    # The last page needs no note.
    tail = text_of(await server.call_tool("read_document", {"filename": "a.txt", "start": 300}))
    assert "more with start" not in tail and "passage 300 " in tail


@pytest.mark.asyncio
async def test_a_cancelled_tool_call_closes_the_answer_stream() -> None:
    """Cancelling the coroutine left the request's thread streaming the answer, holding a
    server question slot, until the answer ended."""

    import asyncio
    import threading
    import time

    started, closed = threading.Event(), threading.Event()

    class Pings(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            started.set()
            while not closed.is_set():
                time.sleep(0.01)  # the server's keepalives, sped up
                yield b": ping\n\n"

        def close(self) -> None:
            closed.set()

    server = mock_server(
        lambda _: httpx.Response(200, stream=Pings(), headers={"content-type": "text/event-stream"})
    )
    call = asyncio.create_task(server.call_tool("ask", {"question": "q"}))
    assert await asyncio.to_thread(started.wait, 5)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert await asyncio.to_thread(closed.wait, 5)


@pytest.mark.asyncio
async def test_parity_tools_read_around_a_citation_and_list_workspaces(server: MCPServer) -> None:
    passages = text_of(await server.call_tool("search_passages", {"query": "alpha"}))
    chunk_id = passages.split("(chunk ")[1].split(",")[0]
    around = text_of(
        await server.call_tool(
            "read_document", {"filename": "guide.txt", "around_chunk_id": chunk_id}
        )
    )
    assert "Alpha is the first letter" in around
    unknown = text_of(
        await server.call_tool(
            "read_document", {"filename": "guide.txt", "around_chunk_id": "nope"}
        )
    )
    assert unknown == "guide.txt has no passage nope."
    # A tag nothing carries: scoped search finds nothing.
    tagged = text_of(await server.call_tool("search_passages", {"query": "alpha", "tags": ["x"]}))
    assert "guide.txt" not in tagged
    workspaces = text_of(await server.call_tool("list_workspaces", {}))
    assert "not enabled" in workspaces


def test_answers_list_the_exact_quotes_their_citations_rest_on() -> None:
    result = StreamResult(
        {
            "answer": "Alpha is first [1].",
            "sources": [
                {
                    "source_number": 1,
                    "filename": "a.txt",
                    "page": 1,
                    "chunk_id": "c1",
                    "spans": [{"start": 0, "end": 10, "quote": "Alpha is\nthe first letter."}],
                }
            ],
        },
        [],
    )
    assert '    "Alpha is the first letter."' in format_answer(result)


@pytest.mark.asyncio
async def test_ask_takes_the_conversation_as_history(
    server: MCPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[Any] = []
    original = DocragClient.ask

    def spy(self: DocragClient, *args: Any, **kwargs: Any) -> StreamResult:
        sent.append(kwargs.get("history"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DocragClient, "ask", spy)
    history = [
        {"role": "user", "content": "What is alpha?"},
        {"role": "assistant", "content": "The first letter."},
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "  "},
    ]
    answer = text_of(await server.call_tool("ask", {"question": "and beta?", "history": history}))
    assert answer.startswith("Alpha is documented [1].")
    assert sent == [history[:2]]


@pytest.mark.asyncio
async def test_ask_trims_a_long_history_to_what_the_server_takes(
    server: MCPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[Any] = []
    original = DocragClient.ask

    def spy(self: DocragClient, *args: Any, **kwargs: Any) -> StreamResult:
        sent.append(kwargs.get("history"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DocragClient, "ask", spy)
    history = [
        {"role": "user", "content": "What is alpha?"},
        {"role": "assistant", "content": "Alpha [1]. " * 1000},  # an answer with its sources
    ]
    answer = text_of(await server.call_tool("ask", {"question": "and beta?", "history": history}))
    assert answer.startswith("Alpha is documented [1].")
    assert len(sent[0][1]["content"]) <= 4000
