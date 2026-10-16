"""DocRAG as an MCP server: search, cited answers and summaries as tools.

Lets an MCP client (Claude Desktop, an IDE agent) use a running DocRAG — the answers keep
DocRAG's citations, so the client can quote filename, page and section. Runs over stdio:

    uv sync --extra mcp
    DOCRAG_URL=http://127.0.0.1:8000 DOCRAG_API_KEY=... uv run python -m cli.mcp_server

Every tool goes through the REST API (``cli.client``), so the server's key, read-only
key, rate limit and question slots apply; the read-only key is enough, since no tool
changes the corpus.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from cli.client import DocragClient, DocragError, StreamResult

# A passage shown to the model is cut here: a search returns up to a dozen of them.
_PASSAGE_CHARS = 1200
# None of the tools change the corpus; "open world" is false — they only read DocRAG.
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

INSTRUCTIONS = (
    "DocRAG answers questions from the user's own indexed documents, with citations. "
    "Use list_documents to see what is indexed, search_passages to find text, ask for a "
    "cited answer, and summarize for a whole document (or two to compare). Quote the "
    "citations (filename, page, section) when you use an answer."
)


def _pages(item: dict[str, Any]) -> str:
    page, end = item.get("page"), item.get("page_end")
    return f"p.{page}-{end}" if end else f"p.{page}"


def format_answer(result: StreamResult) -> str:
    """An answer with its numbered sources, as text a model can quote from."""

    done = result.done
    lines = [str(done.get("answer", "")).strip()]
    sources = done.get("sources") or []
    if sources:
        lines.append("\nSources:")
        for source in sources:
            lines.append(
                f"[{source.get('source_number')}] {source.get('filename')} {_pages(source)}, "
                f"section: {source.get('section') or '-'} (chunk {source.get('chunk_id')})"
            )
    status = done.get("answer_status")
    if status and status != "answered":
        lines.append(f"\nAnswer status: {status} (the documents may not cover this).")
    read, total = done.get("chunks_read"), done.get("chunks_total")
    if isinstance(read, int) and isinstance(total, int) and read < total:
        lines.append(f"\nRead {read} of {total} passages (evenly spaced).")
    if result.followups:
        lines.append("\nSuggested follow-ups: " + " | ".join(result.followups))
    return "\n".join(lines)


def format_passages(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No passages matched."
    blocks = []
    for item in results:
        text = " ".join(str(item.get("text", "")).split())
        if len(text) > _PASSAGE_CHARS:
            text = text[:_PASSAGE_CHARS].rstrip() + " …"
        blocks.append(
            f"{item.get('rank')}. {item.get('filename')} {_pages(item)}, "
            f"section: {item.get('section') or '-'} (chunk {item.get('chunk_id')}, "
            f"score {float(item.get('rerank_score', 0)):.2f})\n{text}"
        )
    return "\n\n".join(blocks)


def build_server(client_factory: Callable[[], DocragClient] = DocragClient) -> MCPServer:
    """The MCP server; ``client_factory`` makes the REST client (tests pass a fake)."""

    server: MCPServer = MCPServer(name="docrag", instructions=INSTRUCTIONS)

    async def call(work: Callable[[DocragClient], str]) -> str:
        def run() -> str:
            with client_factory() as client:
                try:
                    return work(client)
                except DocragError as exc:
                    # Returned as text, not raised: the model can read it and react
                    # (a 404 names the document, a 429 says to retry).
                    code = f" ({exc.status_code})" if exc.status_code else ""
                    return f"DocRAG error{code}: {exc}"

        return await asyncio.to_thread(run)

    @server.tool(annotations=_READ_ONLY)
    async def list_documents() -> str:
        """List the indexed documents with their passage counts and tags."""

        def work(client: DocragClient) -> str:
            listing = client.documents()
            names = listing.get("filenames", [])
            if not names:
                return "No documents are indexed."
            counts, tags = listing.get("chunk_counts", {}), listing.get("tags", {})
            return "\n".join(
                f"{name} ({counts.get(name, '?')} passages)"
                + (f" tags: {', '.join(tags[name])}" if tags.get(name) else "")
                for name in names
            )

        return await call(work)

    @server.tool(annotations=_READ_ONLY)
    async def search_passages(
        query: str, filenames: list[str] | None = None, top_k: int | None = None
    ) -> str:
        """Find the passages that best match a query (hybrid search + rerank, no LLM).

        filenames limits the search to those documents; top_k caps the results."""

        return await call(lambda client: format_passages(client.search(query, filenames, top_k)))

    @server.tool(annotations=_READ_ONLY)
    async def ask(question: str, filenames: list[str] | None = None) -> str:
        """Answer a question from the indexed documents, with numbered citations.

        filenames limits the answer to those documents."""

        return await call(lambda client: format_answer(client.ask(question, filenames)))

    @server.tool(annotations=_READ_ONLY)
    async def summarize(filenames: list[str], focus: str | None = None) -> str:
        """Summarize one whole document, or compare two (pass both names).

        focus narrows it ("deadlines", "risks"). Long documents take minutes."""

        if not 1 <= len(filenames) <= 2:
            return "summarize takes one document, or two to compare."
        return await call(lambda client: format_answer(client.summarize(filenames, focus)))

    @server.tool(annotations=_READ_ONLY)
    async def read_document(filename: str, start: int = 1, count: int = 20) -> str:
        """Read a document's indexed text, passage by passage, from passage `start`."""

        def work(client: DocragClient) -> str:
            first = max(1, start)
            page = client.content(filename, start=first, end=first + max(1, min(count, 100)) - 1)
            chunks = page.get("chunks", [])
            if not chunks:
                return f"No passages from {first} on (the document has {page.get('total_chunks')})."
            body = "\n\n".join(
                f"[{chunk.get('chunk_ordinal')}] {_pages(chunk)} · {chunk.get('section') or '-'}\n"
                f"{chunk.get('text', '')}"
                for chunk in chunks
            )
            return f"{filename}: passages of {page.get('total_chunks')}\n\n{body}"

        return await call(work)

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
