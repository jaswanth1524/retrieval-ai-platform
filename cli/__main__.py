"""``docrag`` / ``python -m cli``: DocRAG from a terminal or a script.

Installed with ``uv tool install ./cli`` it is the ``docrag`` command; in the
repository the module runs it directly:

    uv run python -m cli status
    uv run python -m cli ingest ./handbook ./notes.pdf
    uv run python -m cli add-url https://example.com/policy.html
    uv run python -m cli ask "How many days of leave do I get?" --file handbook.pdf
    uv run python -m cli search "leave policy"
    uv run python -m cli summarize handbook.pdf [other.pdf] [--focus "deadlines"]
    uv run python -m cli export backup.zip / import backup.zip

``--url`` / ``DOCRAG_URL`` (default http://127.0.0.1:8000) and ``--api-key`` /
``DOCRAG_API_KEY`` select the server. Exit codes: 0 ok, 1 a request failed (or some
uploads did), 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from cli.client import DocragClient, DocragError, StreamResult


def ingestible(paths: Sequence[Path], extensions: set[str]) -> tuple[list[Path], list[Path]]:
    """Files to upload under ``paths`` (directories walked recursively, sorted) and the
    ones skipped for an unsupported type. Dotfiles and dot-directories are ignored."""

    found: list[Path] = []
    skipped: list[Path] = []
    for path in paths:
        candidates = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for candidate in candidates:
            relative = candidate.relative_to(path) if path.is_dir() else Path(candidate.name)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if candidate.suffix.lower() in extensions:
                found.append(candidate)
            else:
                skipped.append(candidate)
    return found, skipped


def _print_sources(result: StreamResult, out: TextIO) -> None:
    sources = result.done.get("sources") or []
    if sources:
        print("\nSources:", file=out)
        for source in sources:
            page = source.get("page")
            end = source.get("page_end")
            pages = f"p.{page}-{end}" if end else f"p.{page}"
            print(
                f"  [{source.get('source_number')}] {source.get('filename')} {pages}"
                f" · {source.get('section') or '-'} ({source.get('chunk_id')})",
                file=out,
            )
    read, total = result.done.get("chunks_read"), result.done.get("chunks_total")
    if isinstance(read, int) and isinstance(total, int) and read < total:
        print(f"\n(read {read} of {total} passages, evenly spaced)", file=out)
    status = result.done.get("answer_status")
    if status and status != "answered":
        print(f"\n(answer status: {status})", file=out)
    if result.followups:
        print("\nYou could also ask:", file=out)
        for question in result.followups:
            print(f"  - {question}", file=out)


def _stage_printer(err: TextIO) -> Any:
    def show(event: dict[str, Any]) -> None:
        step, steps = event.get("step"), event.get("steps")
        progress = f" {step}/{steps}" if step and steps else ""
        print(f"[{event.get('stage')}{progress}]", file=err)

    return show


def cmd_status(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    health = client.health()
    config = client.config()
    listing = client.documents()
    if args.json:
        print(json.dumps({"health": health, "config": config, "documents": listing}), file=out)
        return 0
    print(f"DocRAG {health.get('version', '?')} — {health.get('status')}", file=out)
    print(
        f"generation: {config.get('llm_provider')} ({config.get('llm_model')}); "
        f"embeddings: {config.get('dense_embedding_model')}",
        file=out,
    )
    print(f"documents: {len(listing.get('filenames', []))}", file=out)
    return 0


def cmd_documents(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    listing = client.documents()
    if args.json:
        print(json.dumps(listing), file=out)
        return 0
    counts = listing.get("chunk_counts", {})
    for name in listing.get("filenames", []):
        print(f"{name}\t{counts.get(name, '?')} chunks", file=out)
    return 0


def cmd_ingest(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    paths = [Path(p) for p in args.paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"error: not found: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2
    extensions = {ext.lower() for ext in client.config().get("supported_extensions", [])}
    files, skipped = ingestible(paths, extensions)
    for path in skipped:
        print(f"skip  {path} (unsupported type)", file=out)
    failed = 0
    for path in files:
        try:
            job = client.upload(path)
            status = client.wait_for_job(job["job_id"])
        except DocragError as exc:
            failed += 1
            print(f"fail  {path}: {exc}", file=out)
            continue
        if status.get("state") == "done":
            chunks = (status.get("result") or {}).get("chunks_ingested", status.get("chunks_done"))
            print(f"ok    {path} → {status.get('filename')} ({chunks} chunks)", file=out)
        else:
            failed += 1
            print(f"fail  {path}: {status.get('error') or 'indexing failed'}", file=out)
    print(f"{len(files) - failed} indexed, {failed} failed, {len(skipped)} skipped", file=out)
    return 1 if failed else 0


def cmd_add_url(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    job = client.upload_url(args.url, args.name)
    status = client.wait_for_job(job["job_id"])
    if status.get("state") != "done":
        print(f"fail  {args.url}: {status.get('error') or 'indexing failed'}", file=out)
        return 1
    print(f"ok    {args.url} → {status.get('filename')}", file=out)
    return 0


def cmd_ask(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    streamed: list[str] = []

    def write(text: str) -> None:
        streamed.append(text)
        out.write(text)
        out.flush()

    result = client.ask(
        args.question,
        filenames=args.file or None,
        provider=args.provider,
        on_delta=None if args.json else write,
        on_stage=None if args.json or args.quiet else _stage_printer(sys.stderr),
    )
    if args.json:
        print(json.dumps({**result.done, "followups": result.followups}), file=out)
        return 0
    print(file=out)
    # A citation retry replaces the streamed text with the answer that cites its sources.
    final = str(result.done.get("answer", ""))
    if final.strip() != "".join(streamed).strip():
        print(f"\n[revised after the citation check]\n{final}", file=out)
    _print_sources(result, out)
    return 0


def cmd_summarize(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if len(args.files) > 2:
        print("error: summarize takes one document, or two to compare", file=sys.stderr)
        return 2
    result = client.summarize(
        args.files,
        instruction=args.focus,
        provider=args.provider,
        on_stage=None if args.json or args.quiet else _stage_printer(sys.stderr),
    )
    if args.json:
        print(json.dumps(result.done), file=out)
        return 0
    print(result.done.get("answer", ""), file=out)
    _print_sources(result, out)
    return 0


def cmd_search(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    results = client.search(args.query, filenames=args.file or None, top_k=args.top_k)
    if args.json:
        print(json.dumps(results), file=out)
        return 0
    for item in results:
        text = " ".join(str(item.get("text", "")).split())
        print(
            f"{item.get('rank')}. {item.get('filename')} p.{item.get('page')} "
            f"({item.get('rerank_score', 0):.2f})\n   {text[:240]}",
            file=out,
        )
    return 0


def cmd_export(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    written = client.export(Path(args.output))
    print(f"wrote {written} bytes to {args.output}", file=out)
    return 0


def cmd_import(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    report = client.restore(Path(args.archive))
    if args.json:
        print(json.dumps(report), file=out)
        return 0
    print(
        f"{len(report.get('jobs', []))} documents queued, {len(report.get('skipped', []))} "
        f"already indexed, {len(report.get('rejected', []))} rejected, "
        f"{report.get('feedback_imported', 0)} feedback rows",
        file=out,
    )
    for job in report.get("jobs", []):
        status = client.wait_for_job(job["job_id"])
        print(f"{status.get('state'):6} {status.get('filename')}", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    # `docrag` once installed (cli/pyproject.toml), `python -m cli` from the repository.
    prog = "docrag" if Path(sys.argv[0]).stem == "docrag" else "python -m cli"
    parser = argparse.ArgumentParser(prog=prog, description="DocRAG over its REST API.")
    parser.add_argument("--url", help="DocRAG base URL (DOCRAG_URL; default 127.0.0.1:8000).")
    parser.add_argument("--api-key", help="X-API-Key (DOCRAG_API_KEY).")
    parser.add_argument("--workspace", help="Workspace to use (DOCRAG_WORKSPACE).")
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Version, provider, document count.")
    sub.add_parser("documents", help="List indexed documents.")

    ingest = sub.add_parser("ingest", help="Upload files or whole directories and wait.")
    ingest.add_argument("paths", nargs="+")

    add_url = sub.add_parser("add-url", help="Have the server fetch and index a URL.")
    add_url.add_argument("url")
    add_url.add_argument("--name", help="Stored filename (with a supported extension).")

    ask = sub.add_parser("ask", help="Ask a question (streamed).")
    ask.add_argument("question")
    ask.add_argument("--file", action="append", help="Limit to this document (repeatable).")
    ask.add_argument("--provider", help="ollama, openai, openai_compatible or anthropic.")
    ask.add_argument("--quiet", action="store_true", help="No progress lines on stderr.")

    summarize = sub.add_parser("summarize", help="Summarize a document, or compare two.")
    summarize.add_argument("files", nargs="+")
    summarize.add_argument("--focus", help="What to concentrate on.")
    summarize.add_argument("--provider")
    summarize.add_argument("--quiet", action="store_true")

    search = sub.add_parser("search", help="Ranked passages, no LLM.")
    search.add_argument("query")
    search.add_argument("--file", action="append")
    search.add_argument("--top-k", type=int)

    export = sub.add_parser("export", help="Download a backup zip (full key).")
    export.add_argument("output")
    restore = sub.add_parser("import", help="Restore a backup zip and wait for it.")
    restore.add_argument("archive")
    return parser


COMMANDS = {
    "status": cmd_status,
    "documents": cmd_documents,
    "ingest": cmd_ingest,
    "add-url": cmd_add_url,
    "ask": cmd_ask,
    "summarize": cmd_summarize,
    "search": cmd_search,
    "export": cmd_export,
    "import": cmd_import,
}


def main(
    argv: Sequence[str] | None = None,
    *,
    client: DocragClient | None = None,
    out: TextIO | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    stream = out or sys.stdout
    owned = client is None
    active = client or DocragClient(args.url, args.api_key, workspace=args.workspace)
    try:
        return COMMANDS[args.command](active, args, stream)
    except DocragError as exc:
        code = f" ({exc.status_code})" if exc.status_code else ""
        print(f"error{code}: {exc}", file=sys.stderr)
        return 1
    finally:
        if owned:
            active.close()


if __name__ == "__main__":
    raise SystemExit(main())
