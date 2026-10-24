"""``docrag`` / ``python -m cli``: DocRAG from a terminal or a script.

Installed with ``uv tool install ./cli`` it is the ``docrag`` command; in the
repository the module runs it directly:

    uv run python -m cli status
    uv run python -m cli ingest ./handbook ./notes.pdf
    uv run python -m cli add-url https://example.com/policy.html
    uv run python -m cli ask "How many days of leave do I get?" --file handbook.pdf
    uv run python -m cli chat   # follow-up questions, with the conversation as history
    uv run python -m cli search "leave policy"
    uv run python -m cli summarize handbook.pdf [other.pdf] [--focus "deadlines"]
    uv run python -m cli read handbook.pdf [--around CHUNK_ID]
    uv run python -m cli tags add handbook.pdf hr / delete handbook.pdf
    uv run python -m cli reindex --stale / refresh policy.html / jobs [--cancel ID]
    uv run python -m cli workspaces list|create NAME|delete NAME / watch [--scan]
    uv run python -m cli export backup.zip / import backup.zip

``--url`` / ``DOCRAG_URL`` (default http://127.0.0.1:8000) and ``--api-key`` /
``DOCRAG_API_KEY`` select the server; these flags (and ``--json``) go before or after
the command. Exit codes: 0 ok, 1 a request failed (or some uploads, or some restored
documents, did), 2 bad arguments, 3 (``ask --strict``) the documents don't answer it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from cli.client import (
    FINISHED_JOB_STATES,
    REFUSED_STATUSES,
    DocragClient,
    DocragError,
    StreamResult,
    grounding_notes,
    trim_history,
)

# `ask --strict`: the documents don't answer the question (a script can tell).
EXIT_NOT_ANSWERED = 3


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
    for note in grounding_notes(result.done):
        print(f"\n({note})", file=out)
    if result.followups:
        print("\nYou could also ask:", file=out)
        for question in result.followups:
            print(f"  - {question}", file=out)


# The server's stage names, as the web UI words them; an unknown one prints as sent.
STAGE_NAMES = {
    "condensing": "reading the conversation",
    "searching": "searching documents",
    "waiting_for_model": "waiting for the model (another question is using it)",
    "joining_identical_question": "the same question is being answered; waiting for it",
    "reading": "reading the passages",
    "verifying_citations": "checking citations",
    "reading_documents": "reading the document",
    "summarizing": "taking notes",
    "combining": "combining notes",
}
# How often a wait with nothing printed says it is still going.
TICK_SECONDS = 15.0


class StageReporter:
    """Prints each stage to stderr, and every ``TICK_SECONDS`` of silence the stage and
    the time waited — a local model can take minutes before its first word, and a
    terminal showing nothing for that long looks hung."""

    def __init__(
        self, err: TextIO, tick: float = TICK_SECONDS, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._err = err
        self._tick = tick
        self._clock = clock
        self._started = clock()
        self._label = "searching documents"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def stage(self, event: dict[str, Any]) -> None:
        name = str(event.get("stage"))
        step, steps = event.get("step"), event.get("steps")
        progress = f" {step}/{steps}" if step and steps else ""
        self._label = f"{STAGE_NAMES.get(name, name)}{progress}"
        eta = event.get("eta_ms")
        hint = (
            f", about {round(eta / 1000)}s" if isinstance(eta, (int, float)) and eta >= 5000 else ""
        )
        print(f"[{self._label}{hint}]", file=self._err, flush=True)

    def delta(self) -> None:
        """The answer is arriving: no more ticks."""

        self._stop.set()

    def __enter__(self) -> StageReporter:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _run(self) -> None:
        while not self._stop.wait(self._tick):
            waited = round(self._clock() - self._started)
            print(f"[{self._label}… {waited}s]", file=self._err, flush=True)


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


ANSWER_FORMATS = ("default", "bullets", "table", "brief")


def question_options(args: argparse.Namespace) -> dict[str, Any]:
    """The narrowing and answer-style flags ``ask`` and ``chat`` share."""

    return {
        "page_min": args.page_min,
        "page_max": args.page_max,
        "section": args.section,
        "exclude_filenames": args.exclude,
        "answer_format": args.answer_format,
        "answer_language": args.language,
    }


def _question_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--page-min", type=int, help="Only passages from this page on.")
    parser.add_argument("--page-max", type=int, help="Only passages up to this page.")
    parser.add_argument("--section", help="Only passages under this exact section label.")
    parser.add_argument("--exclude", action="append", help="Leave this document out (repeatable).")
    parser.add_argument(
        "--answer-format", choices=ANSWER_FORMATS, help="How the answer is laid out."
    )
    parser.add_argument(
        "--language", help='Answer in this language ("German"), or "match" the question.'
    )


_MARKER_RE = re.compile(r"\[(\d{1,6})\]")


def answer_markdown(result: StreamResult) -> str:
    """The answer as Markdown: its ``[n]`` markers as footnotes, each source a footnote
    with the quotes the answer rests on — ready to paste into a document or a ticket."""

    answer = _MARKER_RE.sub(r"[^\1]", str(result.done.get("answer", "")).strip())
    lines = [answer]
    notes = grounding_notes(result.done)
    if notes:
        lines.append("")
        lines.extend(f"> {note}" for note in notes)
    sources = result.done.get("sources") or []
    if sources:
        lines.append("")
    for source in sources:
        page, end = source.get("page"), source.get("page_end")
        pages = f"p.{page}-{end}" if end else f"p.{page}"
        line = (
            f"[^{source.get('source_number')}]: {source.get('filename')}, {pages}"
            f" · {source.get('section') or '-'}"
        )
        quotes = [
            " ".join(str(span.get("quote", "")).split())
            for span in source.get("spans") or []
            if isinstance(span, dict) and span.get("quote")
        ]
        if quotes:
            line += " — " + " … ".join(f"“{quote}”" for quote in quotes)
        lines.append(line)
    return "\n".join(lines)


def _ask_once(
    client: DocragClient,
    args: argparse.Namespace,
    question: str,
    out: TextIO,
    history: list[dict[str, str]] | None = None,
) -> StreamResult:
    """Ask one question, streaming the answer to ``out`` (unless --json) with progress
    on stderr (unless --quiet), then print its sources and notes."""

    streamed: list[str] = []
    reporter = StageReporter(sys.stderr)
    output_format = getattr(args, "format", None) or ("json" if args.json else "text")
    as_json = output_format == "json"
    # Markdown is written once the answer is final: its [n] markers become footnotes.
    streaming = output_format == "text"
    quiet = as_json or args.quiet

    def write(text: str) -> None:
        reporter.delta()
        streamed.append(text)
        out.write(text)
        out.flush()

    if quiet:
        reporter.delta()  # never ticks
    with reporter:
        result = client.ask(
            question,
            filenames=args.file or None,
            tags=args.tag or None,
            provider=args.provider,
            on_delta=write if streaming else None,
            on_stage=None if quiet else reporter.stage,
            history=history,
            options=question_options(args),
        )
    if output_format == "markdown":
        print(answer_markdown(result), file=out)
        return result
    if as_json:
        print(json.dumps({**result.done, "followups": result.followups}), file=out)
        return result
    print(file=out)
    # A citation retry replaces the streamed text with the answer that cites its sources.
    final = str(result.done.get("answer", ""))
    if final.strip() != "".join(streamed).strip():
        print(f"\n[revised after the citation check]\n{final}", file=out)
    _print_sources(result, out)
    return result


def cmd_ask(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    result = _ask_once(client, args, args.question, out)
    if args.strict and result.done.get("answer_status") in REFUSED_STATUSES:
        return EXIT_NOT_ANSWERED
    return 0


def cmd_chat(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Questions read from stdin, each asked with the conversation so far, so a
    follow-up ("and for contractors?") is read with it. An answer saying the documents
    don't cover it is left out of that history, as in the web UI: sent back, it primed
    the next question to be refused too. ``/new`` starts over; an empty line, ``/quit``
    or end of input stops."""

    history: list[dict[str, str]] = []
    prompt = "" if args.json else "> "
    while True:
        if prompt:
            print(prompt, end="", file=out, flush=True)
        line = sys.stdin.readline()
        question = line.strip()
        if not line or not question or question in ("/quit", "/exit"):
            return 0
        if question == "/new":
            history = []
            print("(new conversation)", file=out)
            continue
        try:
            result = _ask_once(client, args, question, out, list(history) or None)
        except KeyboardInterrupt:
            # Ctrl-C stops this answer (the connection closes, and with it the answer on
            # the server); the conversation goes on. At the prompt it still ends it.
            print("\n(stopped)", file=sys.stderr)
            continue
        except DocragError as exc:
            code = f" ({exc.status_code})" if exc.status_code else ""
            print(f"error{code}: {exc}", file=sys.stderr)
            continue
        if result.done.get("answer_status") in REFUSED_STATUSES:
            continue
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(result.done.get("answer", ""))},
            ]
        )
        # Kept to what the server takes; ask() trims what it sends the same way.
        history = trim_history(history)


def cmd_summarize(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    reporter = StageReporter(sys.stderr)
    if args.json or args.quiet:
        reporter.delta()  # never ticks
    with reporter:
        result = client.summarize(
            args.files,
            instruction=args.focus,
            provider=args.provider,
            on_stage=None if args.json or args.quiet else reporter.stage,
        )
    if args.json:
        print(json.dumps(result.done), file=out)
        return 0
    print(result.done.get("answer", ""), file=out)
    _print_sources(result, out)
    return 0


def cmd_search(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    response = client.search_response(
        args.query, filenames=args.file or None, top_k=args.top_k, tags=args.tag or None
    )
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


def cmd_read(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """A document's indexed passages: around a cited chunk, or from passage --start."""

    if args.around:
        page = client.content(args.file, around=args.around, radius=args.radius)
    else:
        start = max(1, args.start)
        page = client.content(args.file, start=start, end=start + max(1, args.count) - 1)
    if args.json:
        print(json.dumps(page), file=out)
        return 0
    if args.around and not page.get("target_found", True):
        print(f"(no passage {args.around} in {args.file}; showing its start)", file=sys.stderr)
    for chunk in page.get("chunks", []):
        page_no, end = chunk.get("page"), chunk.get("page_end")
        pages = f"p.{page_no}-{end}" if end else f"p.{page_no}"
        print(
            f"[{chunk.get('chunk_ordinal')}] {pages} · {chunk.get('section') or '-'} "
            f"({chunk.get('chunk_id')})\n{chunk.get('text', '')}\n",
            file=out,
        )
    print(f"({page.get('total_chunks')} passages in all)", file=out)
    return 0


def cmd_delete(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Delete each file named; one that fails doesn't stop the rest (exit 1 if any did)."""

    deleted: list[str] = []
    failed: list[dict[str, Any]] = []
    for name in args.files:
        try:
            client.delete(name)
        except DocragError as exc:
            failed.append({"filename": name, "error": str(exc), "status": exc.status_code})
            if not args.json:
                code = f" ({exc.status_code})" if exc.status_code else ""
                print(f"error{code}: {name}: {exc}", file=sys.stderr)
            continue
        deleted.append(name)
        if not args.json:
            print(f"deleted {name}", file=out)
    if args.json:
        print(json.dumps({"deleted": deleted, "failed": failed}), file=out)
    return 1 if failed else 0


def cmd_tags(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """``set`` replaces a document's tags; ``add``/``remove`` change the current ones."""

    if args.action == "set":
        wanted = list(args.tags)
    else:
        current = list(client.documents().get("tags", {}).get(args.file, []))
        if args.action == "add":
            wanted = current + [tag for tag in args.tags if tag not in current]
        else:
            drop = {tag.casefold() for tag in args.tags}
            wanted = [tag for tag in current if tag.casefold() not in drop]
    stored = client.set_tags(args.file, wanted)
    if args.json:
        print(json.dumps({"filename": args.file, "tags": stored}), file=out)
    else:
        print(f"{args.file}: {', '.join(stored) or '(no tags)'}", file=out)
    return 0


def _follow(
    client: DocragClient, jobs: list[dict[str, Any]], args: argparse.Namespace, out: TextIO
) -> int:
    """Wait for ``jobs`` and print each outcome; 1 when any failed or wasn't followed."""

    if not jobs:
        return 0
    names = {job["job_id"]: job.get("filename") for job in jobs}
    statuses = client.wait_for_jobs(list(names), timeout=_wait_limit(args))
    failed = 0
    for job_id, status in statuses.items():
        ok = status.get("state") == "done"
        failed += not ok
        if not args.json:
            note = " (unchanged)" if status.get("unchanged") else ""
            detail = "" if ok else f": {_job_error(status)}"
            label = "ok   " if ok else "fail "
            print(f"{label} {status.get('filename') or names[job_id]}{note}{detail}", file=out)
    if args.json:
        print(json.dumps(list(statuses.values())), file=out)
    return 1 if failed else 0


def cmd_reindex(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if bool(args.stale) == bool(args.file):
        print("error: name a document, or pass --stale (not both)", file=sys.stderr)
        return 2
    if args.stale:
        report = client.reindex_stale()
        jobs, deferred = report.get("jobs", []), report.get("deferred", [])
        busy, failed = report.get("in_progress", []), report.get("failed", [])
        if not args.json:
            print(f"{len(jobs)} queued, {len(deferred)} deferred (queue full)", file=out)
            if busy:
                print(f"already being indexed: {', '.join(busy)}", file=out)
            if failed:
                print(f"could not be queued: {', '.join(failed)}", file=out)
        return _follow(client, jobs, args, out) or (1 if deferred or failed else 0)
    return _follow(client, [client.reindex(args.file)], args, out)


def cmd_refresh(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    result = client.refresh(args.file)
    if result.get("unchanged") and "job_id" not in result:
        if args.json:
            print(json.dumps(result), file=out)
        else:
            print(f"{args.file}: unchanged at its source", file=out)
        return 0
    return _follow(client, [result], args, out)


def cmd_versions(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.restore:
        return _follow(client, [client.restore_version(args.file, args.restore)], args, out)
    versions = client.versions(args.file)
    if args.json:
        print(json.dumps(versions), file=out)
    elif not versions:
        print(f"no earlier versions of {args.file} (RAW_DOCUMENT_VERSIONS)", file=out)
    for version in versions if not args.json else []:
        print(f"{version['version']}  {version['size']} bytes", file=out)
    return 0


def cmd_queries(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    rows = client.query_log(limit=args.limit, uncovered=args.uncovered)
    if args.json:
        print(json.dumps(rows), file=out)
        return 0
    for row in rows:
        score = row.get("top_score")
        shown = f"{score:.2f}" if isinstance(score, int | float) else "-"
        print(f"{row.get('status'):22} {shown:>5}  {row.get('question')}", file=out)
    return 0


def cmd_jobs(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.cancel:
        status = client.cancel_job(args.cancel)
        if args.json:
            print(json.dumps(status), file=out)
        else:
            print(f"cancel requested: {status.get('filename')} ({status.get('state')})", file=out)
        return 0
    jobs = client.active_jobs()
    if args.json:
        print(json.dumps(jobs), file=out)
        return 0
    if not jobs:
        print("no indexing jobs running", file=out)
    for job in jobs:
        progress = (
            f" {job.get('chunks_done')}/{job.get('chunks_total')}"
            if job.get("chunks_total")
            else ""
        )
        print(f"{job.get('job_id')}  {job.get('state')}{progress}  {job.get('filename')}", file=out)
    return 0


def cmd_workspaces(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.action == "list":
        listing = client.workspaces()
        if args.json:
            print(json.dumps(listing), file=out)
        elif not listing.get("enabled"):
            print("workspaces are not enabled on this server (WORKSPACES_ENABLED)", file=out)
        else:
            for name in ["(default)", *listing.get("workspaces", [])]:
                print(name, file=out)
        return 0
    if not args.name:
        print(f"error: workspaces {args.action} needs a name", file=sys.stderr)
        return 2
    if args.action == "create":
        client.create_workspace(args.name)
    else:
        client.delete_workspace(args.name)
    if args.json:
        print(json.dumps({args.action: args.name}), file=out)
    else:
        print(f"{args.action}d workspace {args.name}", file=out)
    return 0


def cmd_watch(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.scan:
        client.watch_scan()
    status = client.watch_status()
    if args.json:
        print(json.dumps(status), file=out)
        return 0
    print(f"watching {status.get('folder')}: {status.get('files', 0)} files", file=out)
    if args.scan:
        print("(a pass was started; run again to see its result)", file=out)
    for retry in status.get("retrying", []):
        print(
            f"retry   {retry.get('filename')} (attempt {retry.get('attempts')}): "
            f"{retry.get('error')}",
            file=out,
        )
    for name in status.get("gave_up", []):
        print(f"gave up {name} (tried again when it changes)", file=out)
    for problem in status.get("problems", []):
        print(f"skip    {problem}", file=out)
    return 0


def cmd_export(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    written = client.export(Path(args.output))
    if args.json:
        print(json.dumps({"output": args.output, "bytes": written}), file=out)
    else:
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
    ask.add_argument("--tag", action="append", help="Limit to documents with this tag.")
    ask.add_argument("--provider", help="ollama, openai, openai_compatible or anthropic.")
    ask.add_argument("--quiet", action="store_true", help="No progress lines on stderr.")
    ask.add_argument(
        "--strict",
        action="store_true",
        help=f"Exit {EXIT_NOT_ANSWERED} when the documents don't answer the question.",
    )
    _question_flags(ask)
    ask.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        help="text (streamed, default), markdown (footnoted sources) or json (= --json).",
    )

    chat = command("chat", "Ask questions from stdin, each with the conversation so far.")
    chat.add_argument("--file", action="append", help="Limit to this document (repeatable).")
    chat.add_argument("--tag", action="append", help="Limit to documents with this tag.")
    chat.add_argument("--provider", help="ollama, openai, openai_compatible or anthropic.")
    chat.add_argument("--quiet", action="store_true", help="No progress lines on stderr.")
    _question_flags(chat)

    summarize = command(
        "summarize", "Summarize a document, compare two, or write one brief across several."
    )
    summarize.add_argument("files", nargs="+")
    summarize.add_argument("--focus", help="What to concentrate on.")
    summarize.add_argument("--provider")
    summarize.add_argument("--quiet", action="store_true")

    search = command("search", "Ranked passages, no LLM.")
    search.add_argument("query")
    search.add_argument("--file", action="append")
    search.add_argument("--tag", action="append")
    search.add_argument("--top-k", type=int)

    read = command("read", "Print a document's indexed passages.")
    read.add_argument("file")
    read.add_argument("--around", metavar="CHUNK_ID", help="Centre on this passage.")
    read.add_argument("--radius", type=int, default=3, help="Passages either side of it.")
    read.add_argument("--start", type=int, default=1, help="First passage (1-based).")
    read.add_argument("--count", type=int, default=20)

    delete = command("delete", "Delete documents from the index (full key).")
    delete.add_argument("files", nargs="+")

    tags = command("tags", "Set, add or remove a document's tags (full key).")
    tags.add_argument("action", choices=("set", "add", "remove"))
    tags.add_argument("file")
    tags.add_argument("tags", nargs="*")

    reindex = command("reindex", "Re-index from stored originals (RAW_DOCUMENT_DIR).")
    reindex.add_argument("file", nargs="?")
    reindex.add_argument("--stale", action="store_true", help="Every out-of-date document.")
    wait_flag(reindex)

    refresh = command("refresh", "Fetch a URL-ingested document again.")
    refresh.add_argument("file")
    wait_flag(refresh)

    versions = command("versions", "A document's earlier originals (RAW_DOCUMENT_VERSIONS).")
    versions.add_argument("file")
    versions.add_argument("--restore", metavar="VERSION", help="Index this version again.")
    wait_flag(versions)

    queries = command("queries", "What was asked (QUERY_LOG_ENABLED, full key).")
    queries.add_argument("--uncovered", action="store_true", help="Only unanswered ones.")
    queries.add_argument("--limit", type=int, default=50)

    jobs = command("jobs", "Indexing jobs running now.")
    jobs.add_argument("--cancel", metavar="JOB_ID", help="Cancel this job.")

    workspaces = command("workspaces", "List, create or delete workspaces.")
    workspaces.add_argument("action", choices=("list", "create", "delete"))
    workspaces.add_argument("name", nargs="?")

    watch = command("watch", "The watched folder's status (WATCH_DIR).")
    watch.add_argument("--scan", action="store_true", help="Start a pass now (full key).")

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
    "chat": cmd_chat,
    "summarize": cmd_summarize,
    "search": cmd_search,
    "read": cmd_read,
    "delete": cmd_delete,
    "tags": cmd_tags,
    "reindex": cmd_reindex,
    "refresh": cmd_refresh,
    "versions": cmd_versions,
    "queries": cmd_queries,
    "jobs": cmd_jobs,
    "workspaces": cmd_workspaces,
    "watch": cmd_watch,
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
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130  # the shell's code for a run ended by Ctrl-C
    finally:
        if owned:
            active.close()


if __name__ == "__main__":
    raise SystemExit(main())
