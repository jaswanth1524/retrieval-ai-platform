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
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from cli.admin_commands import _INIT_SETTINGS as _INIT_SETTINGS
from cli.admin_commands import _PROVIDER_MODEL_FIELDS as _PROVIDER_MODEL_FIELDS
from cli.admin_commands import EXIT_DEGRADED as EXIT_DEGRADED
from cli.admin_commands import _import_in_background as _import_in_background
from cli.admin_commands import _print_check as _print_check
from cli.admin_commands import _section_suffix as _section_suffix
from cli.admin_commands import _setting_value as _setting_value
from cli.admin_commands import cmd_audit as cmd_audit
from cli.admin_commands import cmd_backup as cmd_backup
from cli.admin_commands import cmd_cache as cmd_cache
from cli.admin_commands import cmd_check as cmd_check
from cli.admin_commands import cmd_doctor as cmd_doctor
from cli.admin_commands import cmd_export as cmd_export
from cli.admin_commands import cmd_feedback as cmd_feedback
from cli.admin_commands import cmd_import as cmd_import
from cli.admin_commands import cmd_imports as cmd_imports
from cli.admin_commands import cmd_init as cmd_init
from cli.admin_commands import cmd_migrate as cmd_migrate
from cli.admin_commands import cmd_notify_test as cmd_notify_test
from cli.admin_commands import cmd_queries as cmd_queries
from cli.admin_commands import cmd_rebuild as cmd_rebuild
from cli.admin_commands import cmd_settings as cmd_settings
from cli.admin_commands import cmd_status as cmd_status
from cli.admin_commands import cmd_traces as cmd_traces
from cli.admin_commands import human_size as human_size
from cli.admin_commands import init_env as init_env
from cli.admin_commands import provider_model as provider_model
from cli.client import (
    DocragClient,
    DocragError,
    upload_bound,
)
from cli.common import _NOT_GIVEN as _NOT_GIVEN

# The commands live in question_commands, document_commands and admin_commands
# (shared helpers in common); re-exported here for scripts and tests that import them.
from cli.common import EXIT_NOT_ANSWERED as EXIT_NOT_ANSWERED
from cli.common import _follow_import as _follow_import
from cli.common import _job_error as _job_error
from cli.common import _wait_limit as _wait_limit
from cli.completion import completion_script
from cli.config import Profile, ProfileError, load_profile, resolve
from cli.document_commands import _MAX_TAG_CHARS as _MAX_TAG_CHARS
from cli.document_commands import _SYNC_DIGEST_CHARS as _SYNC_DIGEST_CHARS
from cli.document_commands import _SYNC_STAND_DOWN as _SYNC_STAND_DOWN
from cli.document_commands import _SYNC_TAG_PREFIX as _SYNC_TAG_PREFIX
from cli.document_commands import HashCache as HashCache
from cli.document_commands import _finished_member as _finished_member
from cli.document_commands import _follow as _follow
from cli.document_commands import _follow_archive_import as _follow_archive_import
from cli.document_commands import _matches as _matches
from cli.document_commands import _names_on_disk as _names_on_disk
from cli.document_commands import _sha256 as _sha256
from cli.document_commands import _unchanged_on_server as _unchanged_on_server
from cli.document_commands import cmd_add_url as cmd_add_url
from cli.document_commands import cmd_dedupe as cmd_dedupe
from cli.document_commands import cmd_delete as cmd_delete
from cli.document_commands import cmd_documents as cmd_documents
from cli.document_commands import cmd_ingest as cmd_ingest
from cli.document_commands import cmd_jobs as cmd_jobs
from cli.document_commands import cmd_original as cmd_original
from cli.document_commands import cmd_outline as cmd_outline
from cli.document_commands import cmd_preview as cmd_preview
from cli.document_commands import cmd_read as cmd_read
from cli.document_commands import cmd_refresh as cmd_refresh
from cli.document_commands import cmd_reindex as cmd_reindex
from cli.document_commands import cmd_sync as cmd_sync
from cli.document_commands import cmd_tags as cmd_tags
from cli.document_commands import cmd_versions as cmd_versions
from cli.document_commands import cmd_watch as cmd_watch
from cli.document_commands import cmd_workspaces as cmd_workspaces
from cli.document_commands import document_names as document_names
from cli.document_commands import ingestible as ingestible
from cli.document_commands import legacy_sync_tag as legacy_sync_tag
from cli.document_commands import stored_name_key as stored_name_key
from cli.document_commands import sync_tag as sync_tag
from cli.question_commands import _MARKER_RE as _MARKER_RE
from cli.question_commands import CHAT_HISTORY_LINES as CHAT_HISTORY_LINES
from cli.question_commands import STAGE_NAMES as STAGE_NAMES
from cli.question_commands import TICK_SECONDS as TICK_SECONDS
from cli.question_commands import StageReporter as StageReporter
from cli.question_commands import _ask_once as _ask_once
from cli.question_commands import _chat_loop as _chat_loop
from cli.question_commands import _print_sources as _print_sources
from cli.question_commands import _start_line_editing as _start_line_editing
from cli.question_commands import answer_markdown as answer_markdown
from cli.question_commands import cmd_ask as cmd_ask
from cli.question_commands import cmd_chat as cmd_chat
from cli.question_commands import cmd_continue as cmd_continue
from cli.question_commands import cmd_search as cmd_search
from cli.question_commands import cmd_summarize as cmd_summarize
from cli.question_commands import cmd_verify as cmd_verify
from cli.question_commands import narrowing_options as narrowing_options
from cli.question_commands import question_options as question_options

ANSWER_FORMATS = ("default", "bullets", "table", "brief")


def upload_time(value: str, *, end_of_day: bool = False) -> float:
    """An ``--uploaded-after/--before`` value: an ISO date or date-time (local time when
    it names no zone), as Unix seconds; ``--uploaded-before DATE`` includes that day."""

    try:
        return upload_bound(value, end_of_day=end_of_day)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a date (2024-05-31) or date-time (2024-05-31T14:00)."
        ) from None


def upload_time_until(value: str) -> float:
    return upload_time(value, end_of_day=True)


def _narrowing_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--page-min", type=int, help="Only passages on or after this page.")
    parser.add_argument("--page-max", type=int, help="Only passages up to this page.")
    parser.add_argument("--section", help="Only passages under this exact section label.")
    parser.add_argument("--exclude", action="append", help="Leave this document out (repeatable).")
    parser.add_argument(
        "--exclude-chunk",
        action="append",
        metavar="CHUNK_ID",
        help="Leave this passage out (repeatable).",
    )
    parser.add_argument(
        "--uploaded-after", type=upload_time, metavar="DATE", help="Documents uploaded since."
    )
    parser.add_argument(
        "--uploaded-before",
        type=upload_time_until,
        metavar="DATE",
        help="Documents uploaded up to (and on) this date.",
    )


def _question_flags(parser: argparse.ArgumentParser) -> None:
    _narrowing_flags(parser)
    parser.add_argument(
        "--answer-format", choices=ANSWER_FORMATS, help="How the answer is laid out."
    )
    parser.add_argument(
        "--language", help='Answer in this language ("German"), or "match" the question.'
    )
    parser.add_argument(
        "--passages-only",
        action="store_true",
        help="Answer from the matched passages alone, without their neighbouring text.",
    )
    parser.add_argument(
        "--strong-only",
        action="store_true",
        help="Say not found rather than answer from weak matches.",
    )


def refresh_every(value: str) -> int | None:
    """``refresh --every``: minutes, ``never`` (0) or ``default`` (the server's)."""

    if value == "default":
        return None
    if value == "never":
        return 0
    try:
        minutes = int(value)
    except ValueError:
        minutes = -1
    if minutes < 0:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a number of minutes, 'never' or 'default'."
        )
    return minutes


def _add_global_flags(parser: argparse.ArgumentParser, default: Any) -> None:
    parser.add_argument(
        "--url", default=default, help="DocRAG base URL (DOCRAG_URL; default 127.0.0.1:8000)."
    )
    parser.add_argument("--api-key", default=default, help="X-API-Key (DOCRAG_API_KEY).")
    parser.add_argument("--workspace", default=default, help="Workspace to use (DOCRAG_WORKSPACE).")
    parser.add_argument(
        "--profile",
        default=default,
        help="A [profiles.NAME] of ~/.config/docrag/config.toml (DOCRAG_PROFILE).",
    )
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

    completion = command("completion", "Print a shell completion script.")
    completion.add_argument("shell", choices=("bash", "zsh"))
    init = command(
        "init",
        "Write a starting .env: a generated API key, originals kept, durable jobs, "
        "backups and the audit log.",
    )
    init.add_argument("--output", default=".env", help="Where to write it (default .env).")
    init.add_argument("--force", action="store_true", help="Replace an existing file.")
    init.add_argument(
        "--ollama-in-compose",
        action="store_true",
        help="Point the API at compose's own Ollama (docker compose --profile ollama up).",
    )
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
    ingest.add_argument("--tag", action="append", help="Tag every file (repeatable).")
    ingest.add_argument(
        "--include",
        action="append",
        metavar="GLOB",
        help='Only files matching ("*.pdf", or "drafts/*" under a directory given).',
    )
    ingest.add_argument(
        "--background",
        action="store_true",
        help="Zips: the server queues every document as room frees (none deferred); "
        "this waits until all are queued, not indexed.",
    )
    ingest.add_argument(
        "--exclude",
        action="append",
        metavar="GLOB",
        help='Leave matching files (or directories, by name) out ("*.py", "node_modules").',
    )
    ingest.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Send symbolic links found in a folder too (left out by default).",
    )
    ingest.add_argument(
        "--keep-paths",
        action="store_true",
        help="Name a file found in a folder by its path under it (reports/q1.pdf).",
    )
    ingest.add_argument(
        "--skip-unchanged",
        action="store_true",
        help="Don't send a file the server already holds under that name with the same "
        "bytes (asked before anything is uploaded).",
    )
    wait_flag(ingest)

    sync = command("sync", "Upload a folder's new and changed files; --delete removes ones gone.")
    sync.add_argument("directory")
    sync.add_argument("--tag", action="append", help="Tag every file sent (repeatable).")
    sync.add_argument("--include", action="append", metavar="GLOB")
    sync.add_argument("--exclude", action="append", metavar="GLOB")
    sync.add_argument(
        "--delete",
        action="store_true",
        help="Delete documents this folder's sync added whose file is gone.",
    )
    sync.add_argument("--dry-run", action="store_true", help="Only say what it would do.")
    sync.add_argument(
        "--force",
        action="store_true",
        help="With --delete: delete even when the folder has no files at all.",
    )
    sync.add_argument(
        "--adopt",
        action="store_true",
        help="Replace a same-named document that came from somewhere else.",
    )
    sync.add_argument("--follow-symlinks", action="store_true", help="Send symlinked files too.")
    sync.add_argument(
        "--keep-paths",
        action="store_true",
        help="Name documents by their path under the folder (reports/q1.pdf), so "
        "same-named files in different folders are all indexed.",
    )
    wait_flag(sync)

    add_url = command("add-url", "Have the server fetch and index a URL.")
    # Not `url`: that is the server's address (--url), which this would overwrite.
    add_url.add_argument("source_url", metavar="url")
    add_url.add_argument("--name", help="Stored filename (with a supported extension).")
    add_url.add_argument("--tag", action="append", help="Tag it (repeatable).")
    add_url.add_argument(
        "--folder", help="Keep it (or, with --sitemap, every page) under this folder."
    )
    add_url.add_argument(
        "--refresh-minutes",
        type=int,
        metavar="MINUTES",
        help="Check it again this often (0 = never; default: the server's interval).",
    )
    add_url.add_argument(
        "--sitemap",
        action="store_true",
        help="The URL is a sitemap, sitemap index, RSS or Atom feed: add the pages it lists.",
    )
    add_url.add_argument(
        "--replace",
        action="store_true",
        help="Replace a document of the same name that came from another source.",
    )
    add_url.add_argument("--max-pages", type=int, help="--sitemap: at most this many pages.")
    add_url.add_argument(
        "--dry-run",
        action="store_true",
        help="--sitemap: list the pages and whether each is new or already indexed; "
        "nothing is fetched or added.",
    )
    add_url.add_argument(
        "--wait",
        action="store_true",
        help="--sitemap: wait until every page has been fetched, and report how it went.",
    )
    add_url.add_argument(
        "--any-host", action="store_true", help="--sitemap: pages on other hosts too."
    )
    add_url.add_argument(
        "--archive",
        action="store_true",
        help="The URL is a zip of documents: the server imports them in the background.",
    )
    add_url.add_argument(
        "--keep-paths", action="store_true", help="--archive: name documents by their path."
    )
    wait_flag(add_url)

    ask = command("ask", "Ask a question (streamed), or a file of them (--questions).")
    ask.add_argument("question", nargs="?")
    ask.add_argument(
        "--questions",
        metavar="PATH",
        help="A file of questions, one per line (# starts a comment; - reads stdin), "
        "answered one after another with the same scope.",
    )
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
    summarize.add_argument("--page-min", type=int, help="One document: from this page.")
    summarize.add_argument("--page-max", type=int, help="One document: up to this page.")
    summarize.add_argument("--section", help="One document: only this section (exact label).")
    summarize.add_argument("--provider")
    summarize.add_argument("--quiet", action="store_true")

    verify = command("verify", "Check a text against the documents, sentence by sentence (no LLM).")
    verify.add_argument("path", help="A text file, or - to read stdin.")
    verify.add_argument("--file", action="append")
    verify.add_argument("--tag", action="append")

    search = command("search", "Ranked passages, no LLM.")
    search.add_argument("query", nargs="?", default="")
    search.add_argument(
        "--like", metavar="CHUNK_ID", help="Passages most like this one (instead of a query)."
    )
    search.add_argument("--file", action="append")
    search.add_argument("--tag", action="append")
    search.add_argument("--top-k", type=int)
    search.add_argument(
        "--explain",
        action="store_true",
        help="Show each result's rank on the dense and the BM25 search.",
    )
    search.add_argument(
        "--expect",
        metavar="CHUNK_ID_OR_FILE",
        help="Say where this passage (or the best passage of this file) was lost "
        "(implies --explain).",
    )
    _narrowing_flags(search)

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
    refresh.add_argument(
        "--every",
        type=refresh_every,
        default=_NOT_GIVEN,
        metavar="MINUTES|never|default",
        help="Set how often it is checked instead of fetching it now.",
    )
    wait_flag(refresh)

    versions = command("versions", "A document's earlier originals (RAW_DOCUMENT_VERSIONS).")
    versions.add_argument("file")
    versions.add_argument("--restore", metavar="VERSION", help="Index this version again.")
    wait_flag(versions)

    queries = command("queries", "What was asked (QUERY_LOG_ENABLED, full key).")
    queries.add_argument("--uncovered", action="store_true", help="Only unanswered ones.")
    queries.add_argument("--limit", type=int, default=50)
    queries.add_argument(
        "--recheck",
        action="store_true",
        help="Search the latest unanswered ones again now (no LLM call; at most 100).",
    )

    preview = command("preview", "How a file would be indexed, without indexing it.")
    preview.add_argument("file")
    preview.add_argument("--count", type=int, default=5, help="Passages to show (max 20).")

    jobs = command("jobs", "Indexing jobs running now (or finished ones).")
    jobs.add_argument("--cancel", metavar="JOB_ID", help="Cancel this job.")
    jobs.add_argument(
        "--retry", metavar="JOB_ID", help="Run this failed job again (a URL, a re-index)."
    )
    wait_flag(jobs)
    jobs.add_argument("--recent", action="store_true", help="Finished jobs, newest first.")
    jobs.add_argument("--failed", action="store_true", help="Only failed jobs, newest first.")
    jobs.add_argument("--limit", type=int, default=50)

    traces = command(
        "traces",
        "Recent question traces: list, show, compare two, score report, export, or run one "
        "again under other ranking settings.",
    )
    traces.add_argument(
        "action",
        nargs="?",
        choices=("list", "show", "diff", "report", "export", "rerun", "what-if"),
        default="list",
    )
    traces.add_argument("target", nargs="?", help="The trace id (show, diff) or file (export).")
    traces.add_argument("other", nargs="?", help="diff: the second trace id.")
    traces.add_argument(
        "--status",
        action="append",
        choices=("answered", "uncited", "insufficient_context", "no_context"),
        help="Only traces answered this way (repeatable).",
    )
    traces.add_argument(
        "--apply",
        action="store_true",
        help="report: make the suggested RERANK_MIN_SCORE this workspace's own (full key).",
    )
    traces.add_argument(
        "--labels",
        choices=("model", "feedback"),
        default="model",
        help="report --apply: what decides each side (the answers, or people's ratings).",
    )
    traces.add_argument("--min-score", type=float, help="rerun: RERANK_MIN_SCORE to try.")
    traces.add_argument("--top-k", type=int, help="rerun: RERANK_TOP_K to try.")
    traces.add_argument("--relative-floor", type=float, help="rerun: RERANK_RELATIVE_FLOOR to try.")
    traces.add_argument("--recency", type=float, help="rerun: RECENCY_BOOST_WEIGHT to try.")
    traces.add_argument("--fallback", type=int, help="what-if: RERANK_FALLBACK_COUNT to try.")
    traces.add_argument(
        "--last", type=int, default=20, help="what-if: how many traced questions (default 20)."
    )
    traces.add_argument(
        "--all",
        action="store_true",
        help="what-if: answered questions too, not only those that went unanswered.",
    )

    feedback = command("feedback", "Recorded answer ratings (FEEDBACK_ENABLED, full key).")
    feedback.add_argument("--limit", type=int, default=50)

    original = command("original", "Download a document's stored original.")
    original.add_argument("file")
    original.add_argument("-o", "--output", help="Where to write it (default: its name).")

    workspaces = command("workspaces", "List, create or delete workspaces.")
    workspaces.add_argument("action", choices=("list", "create", "delete"))
    workspaces.add_argument("name", nargs="?")

    watch = command("watch", "The watched folder's status (WATCH_DIR).")
    watch.add_argument("--scan", action="store_true", help="Start a pass now (full key).")

    backup = command("backup", "Scheduled backups in BACKUP_DIR (full key).")
    backup.add_argument(
        "action", choices=["list", "run", "verify", "drill", "restore", "download", "delete"]
    )
    backup.add_argument(
        "target", nargs="?", help="A backup's name (from list), or a local zip to verify."
    )
    backup.add_argument(
        "-o", "--output", help="download: where to write it (default: its name, here)."
    )
    backup.add_argument(
        "--only",
        action="append",
        metavar="NAME",
        help="restore: just this document (repeatable).",
    )
    backup.add_argument(
        "--sample",
        type=int,
        default=20,
        help="drill: how many of its documents to read and parse (0 = all; default 20).",
    )
    dedupe = command("dedupe", "Documents holding exactly the same bytes under different names.")
    dedupe.add_argument(
        "--delete",
        action="store_true",
        help="Delete every copy but the first name of each group (full key).",
    )
    command("doctor", "Risky server settings: backups, disk, memory, keys (full key).")

    audit = command("audit", "Stored audit events, newest first (AUDIT_LOG_PATH; full key).")
    audit.add_argument("--event", help='An event, or a prefix ending in "." (document.).')
    audit.add_argument("--since", metavar="DATE", help="Only events since this date (ISO).")
    audit.add_argument("--limit", type=int, default=50)
    audit.add_argument("--all-workspaces", action="store_true")

    check = command("check", "How the index and stored originals agree (full key; report only).")
    check.add_argument(
        "--repair",
        action="append",
        choices=("originals", "versions", "incomplete"),
        help="Put right one kind of finding first (repeatable): delete orphaned originals "
        "or version folders, or re-index incomplete documents.",
    )

    migrate = command(
        "migrate",
        "Move the whole server: pack every workspace into one file, or unpack one (full key).",
    )
    migrate.add_argument("action", choices=("pack", "unpack"))
    migrate.add_argument("file")
    migrate.add_argument(
        "--background",
        action="store_true",
        help="pack: have the server write the bundle in the background, then download it "
        "(for a large server, past a proxy's request timeout).",
    )
    wait_flag(migrate)
    command("notify-test", "Send a test notification through the server's NOTIFY_URL (full key).")
    rebuild = command(
        "rebuild",
        "Index every stored original that has no document: after a lost index (full key).",
    )
    wait_flag(rebuild)

    export = command("export", "Download a backup zip (full key).")
    export.add_argument("output")
    export.add_argument(
        "--include-versions",
        action="store_true",
        help="With every document's kept earlier versions (RAW_DOCUMENT_VERSIONS).",
    )
    imports = command("imports", "Background imports in this workspace (zips, backups, rebuilds).")
    imports.add_argument("--cancel", metavar="IMPORT_ID", help="Stop this import.")
    settings_parser = command(
        "settings", "A workspace's own settings: get, or set NAME=VALUE (full key to set)."
    )
    settings_parser.add_argument("action", choices=("get", "set"))
    settings_parser.add_argument("values", nargs="*", metavar="NAME=VALUE")
    settings_parser.add_argument(
        "--name", help="The workspace (default: --workspace, else the default one)."
    )
    cache = command("cache", "This workspace's cached answers (full key).")
    cache.add_argument("--clear", action="store_true", help="Empty them.")
    outline = command("outline", "A document's sections and the page each starts on.")
    outline.add_argument("filename")
    cont = command("continue", "The rest of an answer cut off at the output limit.")
    cont.add_argument("trace_id")
    restore = command("import", "Restore a backup zip and wait for it.")
    restore.add_argument("archive")
    restore.add_argument(
        "--background",
        action="store_true",
        help="Let the server queue every document as room frees (a large backup), and "
        "follow that instead of sending the deferred ones again.",
    )
    wait_flag(restore)
    return parser


COMMANDS = {
    "status": cmd_status,
    "backup": cmd_backup,
    "check": cmd_check,
    "rebuild": cmd_rebuild,
    "notify-test": cmd_notify_test,
    "imports": cmd_imports,
    "settings": cmd_settings,
    "cache": cmd_cache,
    "outline": cmd_outline,
    "continue": cmd_continue,
    "migrate": cmd_migrate,
    "doctor": cmd_doctor,
    "sync": cmd_sync,
    "audit": cmd_audit,
    "documents": cmd_documents,
    "ingest": cmd_ingest,
    "add-url": cmd_add_url,
    "ask": cmd_ask,
    "chat": cmd_chat,
    "summarize": cmd_summarize,
    "search": cmd_search,
    "verify": cmd_verify,
    "preview": cmd_preview,
    "read": cmd_read,
    "delete": cmd_delete,
    "tags": cmd_tags,
    "reindex": cmd_reindex,
    "refresh": cmd_refresh,
    "versions": cmd_versions,
    "queries": cmd_queries,
    "jobs": cmd_jobs,
    "traces": cmd_traces,
    "dedupe": cmd_dedupe,
    "feedback": cmd_feedback,
    "original": cmd_original,
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
    if args.command == "completion":
        # Needs no server: the script is made from this parser.
        print(completion_script(build_parser(), args.shell), file=stream)
        return 0
    if args.command == "init":
        return cmd_init(args, stream)  # needs no server either
    owned = client is None
    if client is None:
        try:
            profile = load_profile(args.profile)
        except ProfileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        profile = profile or Profile()
        client = DocragClient(
            resolve(args.url, "DOCRAG_URL", profile.url),
            resolve(args.api_key, "DOCRAG_API_KEY", profile.api_key()),
            workspace=resolve(args.workspace, "DOCRAG_WORKSPACE", profile.workspace),
        )
    active = client
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
