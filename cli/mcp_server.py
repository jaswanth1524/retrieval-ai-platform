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
import threading
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from cli.client import Cancelled, DocragClient, DocragError, StreamResult

# A passage shown to the model is cut here: a search returns up to a dozen of them.
_PASSAGE_CHARS = 1200
# read_document returns whole passages up to this much text in all (a page of 100 long
# passages is ~250k characters, too much for one tool result); a single passage longer
# than _READ_PASSAGE_CHARS (well above a normal chunk) is cut. The note at the end says
# where to go on from.
_READ_TOTAL_CHARS = 24_000
_READ_PASSAGE_CHARS = 4_000
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


def format_passages(results: list[dict[str, Any]], weak: bool = False) -> str:
    """Ranked passages as text. ``weak``: none cleared the server's relevance bar, so
    they are only the closest ones (said first, so the model doesn't quote them as
    answers)."""

    if not results:
        return "No passages matched."
    blocks = (
        [
            "No passage clearly matches; these are the closest (low scores). A fuller "
            "question usually finds better ones."
        ]
        if weak
        else []
    )
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

    async def call(work: Callable[[DocragClient, Callable[[], bool]], str]) -> str:
        """Run ``work(client, should_stop)`` off the event loop.

        A cancelled tool call (the client gave up, or sent a cancel notification)
        cancels this coroutine, but not the thread doing the request: without telling
        it, an answer went on streaming — holding one of the server's question slots —
        until it finished. ``should_stop`` turns true then, and an answer or summary
        stream closes its connection at the next line the server sends.
        """

        cancelled = threading.Event()

        def run() -> str:
            with client_factory() as client:
                try:
                    return work(client, cancelled.is_set)
                except Cancelled:
                    return "Cancelled."
                except DocragError as exc:
                    # Returned as text, not raised: the model can read it and react
                    # (a 404 names the document, a 429 says to retry).
                    code = f" ({exc.status_code})" if exc.status_code else ""
                    return f"DocRAG error{code}: {exc}"

        try:
            return await asyncio.to_thread(run)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    @server.tool(annotations=_READ_ONLY)
    async def list_documents() -> str:
        """List the indexed documents with their passage counts and tags."""

        def work(client: DocragClient, should_stop: Callable[[], bool]) -> str:
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

        def work(client: DocragClient, _: Callable[[], bool]) -> str:
            response = client.search_response(query, filenames, top_k)
            return format_passages(
                response.get("results", []), weak=bool(response.get("weak_matches"))
            )

        return await call(work)

    @server.tool(annotations=_READ_ONLY)
    async def ask(question: str, filenames: list[str] | None = None) -> str:
        """Answer a question from the indexed documents, with numbered citations.

        filenames limits the answer to those documents."""

        return await call(
            lambda client, should_stop: format_answer(
                client.ask(question, filenames, should_stop=should_stop)
            )
        )

    @server.tool(annotations=_READ_ONLY)
    async def summarize(filenames: list[str], focus: str | None = None) -> str:
        """Summarize one whole document, or compare two (pass both names).

        focus narrows it ("deadlines", "risks"). Long documents take minutes."""

        if not 1 <= len(filenames) <= 2:
            return "summarize takes one document, or two to compare."
        return await call(
            lambda client, should_stop: format_answer(
                client.summarize(filenames, focus, should_stop=should_stop)
            )
        )

    @server.tool(annotations=_READ_ONLY)
    async def read_document(filename: str, start: int = 1, count: int = 20) -> str:
        """Read a document's indexed text, passage by passage, from passage `start`.

        A long read stops early and ends with a note giving the `start` to go on from."""

        def work(client: DocragClient, should_stop: Callable[[], bool]) -> str:
            first = max(1, start)
            page = client.content(filename, start=first, end=first + max(1, min(count, 100)) - 1)
            chunks = page.get("chunks", [])
            total = page.get("total_chunks")
            if not chunks:
                return f"No passages from {first} on (the document has {total})."
            blocks: list[str] = []
            used = 0
            last = first - 1
            for chunk in chunks:
                text = str(chunk.get("text", ""))
                if len(text) > _READ_PASSAGE_CHARS:
                    text = text[:_READ_PASSAGE_CHARS].rstrip() + " … [passage cut]"
                block = (
                    f"[{chunk.get('chunk_ordinal')}] {_pages(chunk)} · "
                    f"{chunk.get('section') or '-'}\n{text}"
                )
                # Always at least one passage, so reading on from the note advances.
                if blocks and used + len(block) > _READ_TOTAL_CHARS:
                    break
                blocks.append(block)
                used += len(block)
                ordinal = chunk.get("chunk_ordinal")
                last = ordinal if isinstance(ordinal, int) else last + 1
            body = "\n\n".join(blocks)
            result = f"{filename}: passages of {total}\n\n{body}"
            if isinstance(total, int) and last < total:
                result += (
                    f"\n\n[Showing passages up to {last} of {total}; more with start={last + 1}.]"
                )
            return result

        return await call(work)

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
