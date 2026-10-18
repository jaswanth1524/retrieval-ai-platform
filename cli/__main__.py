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
``DOCRAG_API_KEY`` select the server; these flags (and ``--json``) go before or after
the command. Exit codes: 0 ok, 1 a request failed (or some uploads, or some restored
documents, did), 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from cli.client import FINISHED_JOB_STATES, DocragClient, DocragError, StreamResult


def ingestible(paths: Sequence[Path], extensions: set[str]) -> tuple[list[Path], list[Path]]:
    """Files to upload under ``paths`` (directories walked recursively, sorted) and the
    ones skipped for an unsupported type. Dotfiles and dot-directories inside a walked
    directory are ignored; a file named on the command line is always considered."""

    found: list[Path] = []
    skipped: list[Path] = []
    for path in paths:
        if path.is_dir():
            candidates = sorted(
                p
                for p in path.rglob("*")
                if p.is_file()
                and not any(part.startswith(".") for part in p.relative_to(path).parts)
            )
        else:
            candidates = [path]
        for candidate in candidates:
            if candidate.suffix.lower() in extensions:
                found.append(candidate)
            else:
                skipped.append(candidate)
    return found, skipped


def stored_name_key(name: str) -> str:
    """The server keeps only a file's basename, and refuses (or, without stored
    originals, conflates) names that differ only by letter case or Unicode form."""

    return unicodedata.normalize("NFC", name.strip()).casefold()


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


# The /config field naming each provider's model. llm_model is Ollama's: printing it
# for every provider showed "openai (qwen2.5:latest)".
_PROVIDER_MODEL_FIELDS = {
    "ollama": "llm_model",
    "openai": "openai_model",
    "openai_compatible": "openai_compatible_model",
    "anthropic": "anthropic_model",
}


def provider_model(config: dict[str, Any], provider: str) -> str:
    """The model ``provider`` generates with, per the server's /config."""

    field = _PROVIDER_MODEL_FIELDS.get(provider, "llm_model")
    return str(config.get(field) or "?")


def cmd_status(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    health = client.health()
    config = client.config()
    listing = client.documents()
    if args.json:
        print(json.dumps({"health": health, "config": config, "documents": listing}), file=out)
        return 0
    print(f"DocRAG {health.get('version', '?')} — {health.get('status')}", file=out)
    provider = str(config.get("llm_provider", ""))
    print(
        f"generation: {provider} ({provider_model(config, provider)}); "
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


def _wait_limit(args: argparse.Namespace) -> float | None:
    """``--timeout`` in seconds; 0 waits as long as indexing takes."""

    timeout = float(getattr(args, "timeout", 0) or 0)
    return timeout if timeout > 0 else None


def _job_error(status: dict[str, Any]) -> str:
    state = status.get("state")
    if state in ("done", "failed", "unknown"):
        return str(status.get("error") or "indexing failed")
    if state == "not_sent":
        return "not uploaded: the wait ended first"
    return f"still {state} when the wait ended; it goes on on the server"


def cmd_ingest(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Upload and follow several files at a time (``DocragClient.index_files``): the
    server indexes them side by side instead of one file after another."""

    paths = [Path(p) for p in args.paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"error: not found: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2
    config = client.config()
    extensions = {ext.lower() for ext in config.get("supported_extensions", [])}
    max_bytes = config.get("max_upload_bytes")
    files, unsupported = ingestible(paths, extensions)
    text = None if args.json else out
    skipped: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    indexed: list[dict[str, Any]] = []

    def skip(path: Path, reason: str) -> None:
        skipped.append({"path": str(path), "reason": reason})
        if text:
            print(f"skip  {path} ({reason})", file=text)

    def fail(path: Path, error: str) -> None:
        failed.append({"path": str(path), "error": error})
        if text:
            print(f"fail  {path}: {error}", file=text)

    for path in unsupported:
        skip(path, "unsupported type")
    uploads: list[Path] = []
    first_of: dict[str, Path] = {}
    for path in files:
        try:
            size = path.stat().st_size
        except OSError as exc:
            fail(path, f"cannot read it ({exc.strerror or exc})")
            continue
        if isinstance(max_bytes, int) and size > max_bytes:
            skip(path, f"too large: {size} bytes, the limit is {max_bytes}")
            continue
        # Same basename from two folders: the second upload would replace the first.
        first = first_of.setdefault(stored_name_key(path.name), path)
        if first is not path:
            skip(path, f"another file is already indexed as {first.name}: {first}")
            continue
        uploads.append(path)

    def finished(path: Path, status: dict[str, Any]) -> None:
        if status.get("state") != "done":
            fail(path, _job_error(status))
            return
        chunks = (status.get("result") or {}).get("chunks_ingested", status.get("chunks_done"))
        indexed.append({"path": str(path), "filename": status.get("filename"), "chunks": chunks})
        if text:
            print(f"ok    {path} → {status.get('filename')} ({chunks} chunks)", file=text)

    statuses = client.index_files(uploads, timeout=_wait_limit(args), on_finished=finished)
    for path, status in statuses.items():
        if status.get("state") not in FINISHED_JOB_STATES:
            fail(path, _job_error(status))
    if args.json:
        print(json.dumps({"indexed": indexed, "failed": failed, "skipped": skipped}), file=out)
    else:
        print(f"{len(indexed)} indexed, {len(failed)} failed, {len(skipped)} skipped", file=out)
    return 1 if failed else 0


def cmd_add_url(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    job = client.upload_url(args.source_url, args.name)
    status = client.wait_for_jobs([job["job_id"]], timeout=_wait_limit(args))[job["job_id"]]
    succeeded = status.get("state") == "done"
    if args.json:
        print(
            json.dumps(status if succeeded else {**status, "error": _job_error(status)}), file=out
        )
    elif succeeded:
        print(f"ok    {args.source_url} → {status.get('filename')}", file=out)
    else:
        print(f"fail  {args.source_url}: {_job_error(status)}", file=out)
    return 0 if succeeded else 1


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
    response = client.search_response(args.query, filenames=args.file or None, top_k=args.top_k)
    results: list[dict[str, Any]] = response.get("results", [])
    if args.json:
        print(json.dumps(results), file=out)
        return 0
    if response.get("weak_matches") and results:
        print("(no strong matches; the closest passages:)", file=out)
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
    """Restore a backup and wait for every indexing job it started (``--json`` too).

    Exits 1 when a job failed or could not be followed, or the server rejected or
    deferred anything (a deferred document needs the same import run again)."""

    report = client.restore(Path(args.archive))
    jobs = report.get("jobs", [])
    rejected, deferred = report.get("rejected", []), report.get("deferred", [])
    if not args.json:
        print(
            f"{len(jobs)} documents queued, {len(report.get('skipped', []))} already "
            f"indexed, {len(rejected)} rejected, {len(deferred)} deferred (indexing queue "
            f"full; import again later), {len(report.get('missing_originals', []))} without "
            f"an original, {report.get('feedback_imported', 0)} feedback rows",
            file=out,
        )
        for name in rejected:
            print(f"reject {name}", file=out)
        for name in deferred:
            print(f"defer  {name}", file=out)
    names = {job["job_id"]: job.get("filename") for job in jobs}
    failed = 0

    def finished(job_id: str, status: dict[str, Any]) -> None:
        nonlocal failed
        if status.get("state") != "done":
            failed += 1
        if not args.json:
            reason = f": {status.get('error')}" if status.get("error") else ""
            print(
                f"{status.get('state'):6} {status.get('filename') or names[job_id]}{reason}",
                file=out,
            )

    statuses = client.wait_for_jobs(list(names), timeout=_wait_limit(args), on_finished=finished)
    for job_id, status in statuses.items():
        if status.get("state") not in FINISHED_JOB_STATES:
            failed += 1
            if not args.json:
                print(f"{status.get('state'):6} {names[job_id]}: {_job_error(status)}", file=out)
    results = [statuses[job["job_id"]] for job in jobs if job["job_id"] in statuses]
    if args.json:
        print(json.dumps({**report, "results": results}), file=out)
    return 1 if failed or rejected or deferred else 0


def _add_global_flags(parser: argparse.ArgumentParser, default: Any) -> None:
    parser.add_argument(
        "--url", default=default, help="DocRAG base URL (DOCRAG_URL; default 127.0.0.1:8000)."
    )
    parser.add_argument("--api-key", default=default, help="X-API-Key (DOCRAG_API_KEY).")
    parser.add_argument("--workspace", default=default, help="Workspace to use (DOCRAG_WORKSPACE).")
    parser.add_argument(
        "--json", action="store_true", default=default, help="Machine-readable output."
    )


def build_parser() -> argparse.ArgumentParser:
    # `docrag` once installed (cli/pyproject.toml), `python -m cli` from the repository.
    prog = "docrag" if Path(sys.argv[0]).stem == "docrag" else "python -m cli"
    parser = argparse.ArgumentParser(prog=prog, description="DocRAG over its REST API.")
    _add_global_flags(parser, None)
    parser.set_defaults(json=False)
    # The same flags after the subcommand (`docrag ask "q" --json`). SUPPRESS: a
    # subcommand that wasn't given one leaves the value given before it alone.
    shared = argparse.ArgumentParser(add_help=False)
    _add_global_flags(shared, argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help, parents=[shared])

    command("status", "Version, provider, document count.")
    command("documents", "List indexed documents.")

    def wait_flag(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--timeout",
            type=float,
            default=3600.0,
            help="Stop waiting for indexing after this many seconds (default 3600; 0 = "
            "no limit). Jobs still running go on on the server.",
        )

    ingest = command("ingest", "Upload files or whole directories and wait.")
    ingest.add_argument("paths", nargs="+")
    wait_flag(ingest)

    add_url = command("add-url", "Have the server fetch and index a URL.")
    # Not `url`: that is the server's address (--url), which this would overwrite.
    add_url.add_argument("source_url", metavar="url")
    add_url.add_argument("--name", help="Stored filename (with a supported extension).")
    wait_flag(add_url)

    ask = command("ask", "Ask a question (streamed).")
    ask.add_argument("question")
    ask.add_argument("--file", action="append", help="Limit to this document (repeatable).")
    ask.add_argument("--provider", help="ollama, openai, openai_compatible or anthropic.")
    ask.add_argument("--quiet", action="store_true", help="No progress lines on stderr.")

    summarize = command("summarize", "Summarize a document, or compare two.")
    summarize.add_argument("files", nargs="+")
    summarize.add_argument("--focus", help="What to concentrate on.")
    summarize.add_argument("--provider")
    summarize.add_argument("--quiet", action="store_true")

    search = command("search", "Ranked passages, no LLM.")
    search.add_argument("query")
    search.add_argument("--file", action="append")
    search.add_argument("--top-k", type=int)

    export = command("export", "Download a backup zip (full key).")
    export.add_argument("output")
    restore = command("import", "Restore a backup zip and wait for it.")
    restore.add_argument("archive")
    wait_flag(restore)
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


def _tolerate_unencodable_output() -> None:
    """Print ``?`` for what the console can't encode (cp1252 on Windows has no ``→``)
    instead of dying with UnicodeEncodeError halfway through an answer."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # pragma: no cover - a detached or closed stream
            pass


def main(
    argv: Sequence[str] | None = None,
    *,
    client: DocragClient | None = None,
    out: TextIO | None = None,
) -> int:
    _tolerate_unencodable_output()  # first: argparse's own errors echo the arguments
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
    except OSError as exc:  # a local file the client didn't already explain
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if owned:
            active.close()


if __name__ == "__main__":
    raise SystemExit(main())
