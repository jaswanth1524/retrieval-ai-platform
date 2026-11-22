"""``docrag ask``, ``chat``, ``summarize``, ``search`` and ``continue``."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from cli.client import (
    REFUSED_STATUSES,
    DocragClient,
    DocragError,
    StreamResult,
    grounding_notes,
    trim_history,
)
from cli.common import EXIT_NOT_ANSWERED


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
    skipped, cut = result.done.get("parts_skipped"), result.done.get("notes_truncated")
    if isinstance(skipped, int) and skipped > 0:
        print(f"\n({skipped} part(s) left out: the model failed on them twice)", file=out)
    if isinstance(cut, int) and cut > 0:
        print(f"\n(notes on {cut} part(s) stopped at the token limit)", file=out)
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
    "continuing": "continuing the answer",
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


def narrowing_options(args: argparse.Namespace) -> dict[str, Any]:
    """The narrowing flags ``ask``, ``chat`` and ``search`` share, as request fields."""

    return {
        "page_min": args.page_min,
        "page_max": args.page_max,
        "section": args.section,
        "exclude_filenames": args.exclude,
        "exclude_chunk_ids": args.exclude_chunk,
        "uploaded_after": args.uploaded_after,
        "uploaded_before": args.uploaded_before,
    }


def question_options(args: argparse.Namespace) -> dict[str, Any]:
    """The narrowing and answer-style flags ``ask`` and ``chat`` share."""

    return {
        **narrowing_options(args),
        "answer_format": args.answer_format,
        "answer_language": args.language,
        # Sent only when asked for (None is left out of the request).
        "passages_only": args.passages_only or None,
        "strong_matches_only": args.strong_only or None,
    }


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
    line_editing = _start_line_editing() if sys.stdin.isatty() and prompt else None
    try:
        return _chat_loop(client, args, out, history, prompt, line_editing)
    finally:
        if line_editing is not None:
            line_editing()


def _start_line_editing() -> Callable[[], None] | None:
    """At a terminal: arrow keys, editing and earlier questions (↑) at the prompt, kept
    in ``$XDG_STATE_HOME/docrag/history``; the returned callable saves them. None where
    Python has no readline (Windows)."""

    try:
        import readline
    except ImportError:  # pragma: no cover - platform dependent
        return None
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    path = Path(base) / "docrag" / "history"
    try:
        readline.read_history_file(path)
    except OSError:
        pass  # none yet
    readline.set_history_length(CHAT_HISTORY_LINES)

    def save() -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(path)
        except OSError:
            pass  # a read-only home: editing still worked

    return save


# Questions remembered for ↑ in `chat` across sessions.
CHAT_HISTORY_LINES = 500


def _chat_loop(
    client: DocragClient,
    args: argparse.Namespace,
    out: TextIO,
    history: list[dict[str, str]],
    prompt: str,
    line_editing: Callable[[], None] | None,
) -> int:
    while True:
        if line_editing is not None:
            try:
                line = input(prompt) + "\n"
            except EOFError:
                print(file=out)
                line = ""
        else:
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
            page_min=args.page_min,
            page_max=args.page_max,
            section=args.section,
        )
    if args.json:
        print(json.dumps(result.done), file=out)
        return 0
    print(result.done.get("answer", ""), file=out)
    _print_sources(result, out)
    return 0


VERIFY_MARKS = {
    "supported": "ok  ",
    "numbers_differ": "num ",
    "unsupported": "??  ",
    "no_claim": "    ",
}


def format_checks(result: dict[str, Any]) -> list[str]:
    """``/questions/verify``'s answer as lines: each sentence, how it fared, and the
    passage words that support it."""

    lines: list[str] = []
    for check in result.get("sentences") or []:
        status = str(check.get("status"))
        lines.append(f"{VERIFY_MARKS.get(status, status + ' ')}{check.get('sentence')}")
        if check.get("filename"):
            where = f"{check['filename']} p.{check.get('page')}, {check.get('section') or '-'}"
            lines.append(f'      {where}: "{" ".join(str(check.get("quote") or "").split())}"')
        if check.get("missing_numbers"):
            lines.append(f"      not in it: {', '.join(check['missing_numbers'])}")
    if result.get("more"):
        lines.append("(only the first sentences were checked; send the rest separately)")
    return lines


def cmd_verify(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Check a text (a file, or - for stdin) against the documents, sentence by
    sentence: exit 1 when a sentence is unsupported or states other numbers."""

    text = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
    result = client.verify(text, args.file, args.tag)
    if args.json:
        print(json.dumps(result), file=out)
    else:
        for line in format_checks(result):
            print(line, file=out)
    bad = {"unsupported", "numbers_differ"}
    return 1 if any(c.get("status") in bad for c in result.get("sentences") or []) else 0


def cmd_search(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if not (args.query or "").strip() and not args.like:
        print("search needs a query, or --like CHUNK_ID", file=sys.stderr)
        return 2
    options = narrowing_options(args)
    if args.like:
        options["like_chunk_id"] = args.like
    if args.explain or args.expect:
        options["explain"] = True
    if args.expect:
        # A chunk id has the form a passage's id has (hex); anything else is a file.
        target = "expect_filename" if "." in args.expect else "expect_chunk_id"
        options[target] = args.expect
    response = client.search_response(
        args.query or "",
        filenames=args.file or None,
        top_k=args.top_k,
        tags=args.tag or None,
        options=options,
    )
    results: list[dict[str, Any]] = response.get("results", [])
    explanation = response.get("explanation")
    if args.json:
        print(json.dumps(response if options.get("explain") else results), file=out)
        return 0
    if response.get("weak_matches") and results:
        print("(no strong matches; the closest passages:)", file=out)
    for item in results:
        text = " ".join(str(item.get("text", "")).split())
        legs = (
            f" dense #{item.get('dense_rank') or '-'} · BM25 #{item.get('sparse_rank') or '-'}"
            if options.get("explain")
            else ""
        )
        print(
            f"{item.get('rank')}. {item.get('filename')} p.{item.get('page')} "
            f"({item.get('rerank_score', 0):.2f}){legs}\n   {text[:240]}",
            file=out,
        )
    if isinstance(explanation, dict):
        print(
            f"\n{explanation.get('target')}: {explanation.get('outcome')} — "
            f"{explanation.get('detail')}",
            file=out,
        )
    return 0


def cmd_continue(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """The rest of an answer cut off at the output limit (its trace id is in ``ask
    --json``'s output and in ``docrag traces``); prints the whole answer."""

    streaming = not args.json

    def write(text: str) -> None:
        out.write(text)
        out.flush()

    result = client.continue_answer(args.trace_id, on_delta=write if streaming else None)
    if args.json:
        print(json.dumps(result.done), file=out)
        return 0
    print(file=out)
    if result.done.get("truncated"):
        print(
            f"(still cut off: docrag continue {result.done.get('trace_id')} goes on)",
            file=sys.stderr,
        )
    return 0
