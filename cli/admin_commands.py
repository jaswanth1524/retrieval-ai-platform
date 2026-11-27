"""``docrag`` commands that look after the server: status, settings, traces, backups,
imports, migration and setup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote

from cli.client import (
    FINISHED_JOB_STATES,
    DocragClient,
    DocragError,
    upload_bound,
)
from cli.common import _follow_import, _job_error, _wait_limit

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


# `docrag status` exits with this when the server answers but isn't ready (a model that
# didn't load, Qdrant or the LLM provider unreachable, an incompatible index).
EXIT_DEGRADED = 1


def cmd_status(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    health = client.health()
    ready = client.readiness()
    config = client.config()
    listing = client.documents()
    degraded = ready.get("status") != "ok"
    if args.json:
        print(
            json.dumps({"health": health, "ready": ready, "config": config, "documents": listing}),
            file=out,
        )
        return EXIT_DEGRADED if degraded else 0
    print(f"DocRAG {health.get('version', '?')} — {ready.get('status')}", file=out)
    provider = str(config.get("llm_provider", ""))
    print(
        f"generation: {provider} ({provider_model(config, provider)}); "
        f"embeddings: {config.get('dense_embedding_model')}",
        file=out,
    )
    checks = [
        ("qdrant", ready.get("qdrant")),
        ("llm provider", ready.get("generation_provider")),
        ("index", ready.get("index_compatible", True)),
    ]
    print("  ".join(f"{name}: {'ok' if ok else 'DOWN'}" for name, ok in checks), file=out)
    models = ready.get("models") or {}
    if models:
        print(
            "models: " + ", ".join(f"{name} {state}" for name, state in sorted(models.items())),
            file=out,
        )
    if ready.get("index_detail"):
        print(f"index: {ready['index_detail']}", file=out)
    print(f"documents: {len(listing.get('filenames', []))}", file=out)
    return EXIT_DEGRADED if degraded else 0


def _section_suffix(change: dict[str, Any]) -> str:
    """ " · <section>" for a changed passage, so two on one page read apart (older
    servers send no section)."""

    section = change.get("section")
    return f" · {section}" if section else ""


def cmd_audit(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    try:
        since = upload_bound(args.since) if args.since else None
    except ValueError:
        print(f"error: --since {args.since!r} isn't a date (2026-11-01)", file=sys.stderr)
        return 2
    entries = client.audit(
        event=args.event, since=since, limit=args.limit, all_workspaces=args.all_workspaces
    )
    if args.json:
        print(json.dumps(entries), file=out)
        return 0
    for entry in entries:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(entry["created_at"])))
        fields = " ".join(f"{name}={value}" for name, value in (entry.get("fields") or {}).items())
        where = f"[{entry.get('workspace')}, {entry.get('key')}]"
        print(f"{when}  {entry.get('event')}  {where}  {fields}", file=out)
    return 0


# What `docrag init` turns on: the durable pieces a fresh install leaves off.
_INIT_SETTINGS = (
    ("RAW_DOCUMENT_DIR", "./data/originals", "Keep uploaded files: re-index, backups, rebuild."),
    ("JOB_STORE_BACKEND", "sqlite", "Indexing jobs survive a restart."),
    ("JOB_STORE_PATH", "./data/jobs.db", None),
    ("REQUEUE_INTERRUPTED_JOBS", "true", "A re-index or URL fetch a restart cut off runs again."),
    ("BACKUP_DIR", "./data/backups", "A backup a day (BACKUP_INTERVAL_HOURS), the last 7 kept."),
    ("AUDIT_LOG_PATH", "./data/audit.db", "Who changed what (GET /admin/audit, `docrag audit`)."),
)


def init_env(*, ollama_in_compose: bool = False, key: str | None = None) -> str:
    """The text of a starting ``.env``: a generated API key and the durable stores on."""

    import secrets

    lines = [
        "# Written by `docrag init`. Every other setting keeps its default (.env.example).",
        "# Required on every request as X-API-Key (the UI asks for it once).",
        f"API_KEY={key or secrets.token_urlsafe(32)}",
    ]
    for name, value, why in _INIT_SETTINGS:
        if why:
            lines.append(f"# {why}")
        lines.append(f"{name}={value}")
    if ollama_in_compose:
        lines.append("# Ollama from compose's own profile: docker compose --profile ollama up")
        lines.append("OLLAMA_BASE_URL=http://ollama:11434")
    return "\n".join(lines) + "\n"


def cmd_init(args: argparse.Namespace, out: TextIO) -> int:
    """Write a starting ``.env`` (never over an existing one without ``--force``)."""

    target = Path(args.output)
    if target.exists() and not args.force:
        print(f"error: {target} already exists (--force replaces it)", file=sys.stderr)
        return 2
    text = init_env(ollama_in_compose=args.ollama_in_compose)
    try:
        # Readable by its owner only: it holds the API key.
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(target, 0o600)
    except OSError as exc:
        print(f"error: cannot write {target}: {exc.strerror or exc}", file=sys.stderr)
        return 1
    print(f"wrote {target} (API key inside; keep it private)", file=out)
    print(
        "next: docker compose"
        + (" --profile ollama" if args.ollama_in_compose else "")
        + " up. `docrag doctor` then checks the setup: it notes the backups share a "
        "disk with the originals (set BACKUP_DIR to another disk to copy them off it)",
        file=out,
    )
    return 0


def cmd_doctor(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Exit 1 when there is a warning, so a script can check a deployment."""

    findings = client.doctor()
    if args.json:
        print(json.dumps(findings), file=out)
    elif not findings:
        print("nothing to report", file=out)
    for finding in [] if args.json else findings:
        print(
            f"{finding.get('level')}: {finding.get('subject')}: {finding.get('message')}", file=out
        )
    return 1 if any(f.get("level") == "warning" for f in findings) else 0


def cmd_queries(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.recheck:
        report = client.recheck_unanswered(limit=min(args.limit, 100))
        if args.json:
            print(json.dumps(report), file=out)
            return 0
        print(
            f"{report['covered_now']} of {report['checked']} unanswered questions now find "
            "a passage above the bar:",
            file=out,
        )
        for item in report.get("items", []):
            mark = "covered" if item.get("covered_now") else "still not"
            where = f"  ({item['top_filename']})" if item.get("top_filename") else ""
            print(f"  {mark:9}  {item.get('question')}{where}", file=out)
        return 0
    rows = client.query_log(limit=args.limit, uncovered=args.uncovered)
    if args.json:
        print(json.dumps(rows), file=out)
        return 0
    for row in rows:
        score = row.get("top_score")
        shown = f"{score:.2f}" if isinstance(score, int | float) else "-"
        print(f"{row.get('status'):22} {shown:>5}  {row.get('question')}", file=out)
    return 0


def cmd_traces(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.action == "report":
        apply = bool(getattr(args, "apply", False))
        report = (
            client.calibrate(apply=True, labels=getattr(args, "labels", "model"))
            if apply
            else client.score_report()
        )
        if args.json:
            print(json.dumps(report), file=out)
            return 0
        print(
            f"answered {report['answered_count']}, not answered {report['unanswered_count']}; "
            f"RERANK_MIN_SCORE now {report['current_min_score']:g}",
            file=out,
        )
        edges = report["bin_edges"]
        for index, (yes, no) in enumerate(
            zip(report["answered"], report["unanswered"], strict=True)
        ):
            if yes or no:
                print(
                    f"  {edges[index]:.2f}-{edges[index + 1]:.2f}  {'+' * yes}{'-' * no}", file=out
                )
        suggested = report.get("suggested_min_score")
        if suggested is not None:
            print(
                f"suggested RERANK_MIN_SCORE={suggested:g} "
                f"({report['would_change']} traced question(s) decided differently)",
                file=out,
            )
            if report.get("applied"):
                print(f"applied: this workspace now uses RERANK_MIN_SCORE={suggested:g}", file=out)
        print(report["note"], file=out)
        return 0
    if args.action == "diff":
        if not args.target or not args.other:
            print("traces diff needs two trace ids", file=sys.stderr)
            return 2
        diff = client.trace_diff(args.target, args.other)
        if args.json:
            print(json.dumps(diff), file=out)
            return 0
        print(f"same answer: {'yes' if diff['same_answer'] else 'no'}", file=out)
        for change in diff.get("settings", []):
            print(f"  {change['name']}: {change['a']!r} -> {change['b']!r}", file=out)
        for change in diff.get("candidates", []):
            print(
                f"  {change['filename']} p.{change['page']}{_section_suffix(change)}: "
                f"rank {change.get('rank_a')} -> "
                f"{change.get('rank_b')}"
                f"{'  ' + str(change['drop_reason_b']) if change.get('drop_reason_b') else ''}",
                file=out,
            )
        return 0
    if args.action == "rerun":
        if not args.target:
            print("traces rerun needs a trace id", file=sys.stderr)
            return 2
        overrides = {
            name: value
            for name, value in (
                ("rerank_min_score", args.min_score),
                ("rerank_top_k", args.top_k),
                ("rerank_relative_floor", args.relative_floor),
                ("recency_boost_weight", args.recency),
            )
            if value is not None
        }
        result = client.rerun_trace(args.target, overrides)
        if args.json:
            print(json.dumps(result), file=out)
            return 0
        verdict = (
            "weak matches only"
            if result.get("weak_matches")
            else ("would answer" if result.get("would_answer") else "nothing found")
        )
        asked_with = overrides or "the settings it was asked with"
        print(f"{result.get('query')}\n  {verdict} with {asked_with}", file=out)
        for change in result.get("diff", {}).get("candidates", []):
            print(
                f"  {change['filename']} p.{change['page']}{_section_suffix(change)}: "
                f"rank {change.get('rank_a')} -> "
                f"{change.get('rank_b')}"
                f"{'  ' + str(change['drop_reason_b']) if change.get('drop_reason_b') else ''}",
                file=out,
            )
        return 0
    if args.action == "what-if":
        body: dict[str, Any] = {
            name: value
            for name, value in (
                ("rerank_min_score", args.min_score),
                ("rerank_relative_floor", args.relative_floor),
                ("rerank_fallback_count", args.fallback),
            )
            if value is not None
        }
        body |= {"limit": args.last, "only_unanswered": not args.all}
        report = client.trace_what_if(body)
        if args.json:
            print(json.dumps(report), file=out)
            return 0
        for item in report.get("items", []):
            mark = (
                "error"
                if item.get("error")
                else ("would answer" if item.get("would_answer") else "not answered")
            )
            score = item.get("top_score")
            shown = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
            print(f"  {mark:<13} {shown}  {item.get('question')}", file=out)
        print(
            f"{report.get('would_answer')} of {report.get('checked')} would be answered "
            f"({report.get('newly_answered')} newly, {report.get('newly_unanswered')} no "
            f"longer){' — stopped early' if report.get('stopped_early') else ''}",
            file=out,
        )
        return 0
    if args.action == "export":
        if not args.target:
            print("traces export needs a file to write", file=sys.stderr)
            return 2
        written = client.export_traces(Path(args.target))
        print(f"wrote {written} bytes to {args.target}", file=sys.stderr)
        return 0
    if args.action == "show":
        if not args.target:
            print("traces show needs a trace id", file=sys.stderr)
            return 2
        trace = client.trace(args.target)
        if args.json:
            print(json.dumps(trace), file=out)
            return 0
        print(f"{trace.get('question')}\n  {trace.get('status')} · {trace.get('mode')}", file=out)
        for candidate in trace.get("candidates") or []:
            score = candidate.get("rerank_score")
            shown = f"{score:.2f}" if isinstance(score, int | float) else "  - "
            mark = "*" if candidate.get("selected_for_context") else " "
            reason = candidate.get("drop_reason") or ""
            print(
                f"  {mark} {shown}  {candidate.get('filename')} p.{candidate.get('page')}"
                f"  {reason}",
                file=out,
            )
        if trace.get("answer"):
            print(f"\n{trace['answer']}", file=out)
        return 0
    traces = client.traces(args.status or ())
    if args.json:
        print(json.dumps(traces), file=out)
        return 0
    for trace in traces:
        total = trace.get("total_ms")
        took = f"{total / 1000:.1f}s" if isinstance(total, int | float) else "-"
        status = trace.get("answer_status") or trace.get("status") or ""
        print(
            f"{trace.get('trace_id')}  {status:20} {took:>6}  {trace.get('question')}",
            file=out,
        )
    return 0


def cmd_feedback(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    rows = client.feedback(limit=args.limit)
    if args.json:
        print(json.dumps(rows), file=out)
        return 0
    for row in rows:
        thumb = "+" if row.get("rating") == "up" else "-"
        print(f"{thumb} {row.get('kind', 'question'):9} {row.get('question')}", file=out)
    return 0


def human_size(size: int) -> str:
    """A byte count as KB/MB/GB, one decimal."""

    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"  # pragma: no cover - the loop always returns


def _print_check(check: dict[str, Any], out: TextIO) -> None:
    print(
        f"{'ok' if check.get('ok') else 'DAMAGED'}: {check.get('documents', 0)} documents, "
        f"{check.get('originals', 0)} originals ({check.get('originals_missing', 0)} missing), "
        f"{check.get('versions', 0)} earlier versions, "
        f"feedback {'included' if check.get('feedback') else 'none'}",
        file=out,
    )
    for problem in check.get("problems", []):
        print(f"  {problem}", file=out)
    refused = check.get("rejected") or []
    if refused:
        print(f"  a restore would refuse {len(refused)}:", file=out)
        for item in refused:
            print(f"    {item}", file=out)


def cmd_backup(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """BACKUP_DIR's backups: list, run one now, verify one (stored, or a local file the
    server reads through), restore one. Exits 1 when a verified backup is damaged."""

    if args.action == "list":
        listing = client.backups()
        if args.json:
            print(json.dumps(listing), file=out)
            return 0
        print(
            f"every {listing.get('interval_hours')} h, newest {listing.get('keep')} kept",
            file=out,
        )
        for backup in listing.get("backups", []):
            verified = {True: "verified", False: "DAMAGED", None: "not checked"}[
                backup.get("verified")
            ]
            print(
                f"{backup.get('name')}  {human_size(int(backup.get('size', 0)))}  {verified}",
                file=out,
            )
        return 0
    if args.action == "run":
        result = client.run_backup()
        if args.json:
            print(json.dumps(result), file=out)
        else:
            print(
                f"backup of {result.get('workspace')} started; `docrag backup list` shows it",
                file=out,
            )
        return 0
    if not args.target:
        raise DocragError(f"`docrag backup {args.action}` needs a backup name or file.")
    if args.action == "verify":
        local = Path(args.target)
        check = (
            client.verify_backup(local)
            if local.is_file()
            else client.verify_stored_backup(args.target)
        )
        if args.json:
            print(json.dumps(check), file=out)
        else:
            _print_check(check, out)
        return 0 if check.get("ok") else 1
    if args.action == "download":
        destination = Path(args.output or args.target)
        written = client.download(f"/backups/{quote(args.target, safe='')}", destination)
        print(f"wrote {written} bytes to {destination}", file=sys.stderr)
        return 0
    if args.action == "delete":
        deleted = client.delete_stored_backup(args.target)
        if args.json:
            print(json.dumps(deleted), file=out)
        else:
            print(f"deleted {deleted.get('name')}", file=out)
        return 0
    if args.action == "drill":
        drill = client.drill_stored_backup(args.target, sample=args.sample)
        if args.json:
            print(json.dumps(drill), file=out)
            return 0 if drill.get("ok") else 1
        for problem in drill.get("problems", []):
            print(f"problem: {problem}", file=out)
        for document in drill.get("documents", []):
            if not document.get("ok"):
                print(f"FAILED  {document.get('filename')}: {document.get('error')}", file=out)
        print(
            f"{drill.get('checked', 0)} of {drill.get('listed', 0)} document(s) read and "
            f"parsed, {drill.get('failed', 0)} failed"
            + (" — the backup restores" if drill.get("ok") else ""),
            file=out,
        )
        return 0 if drill.get("ok") else 1
    only = getattr(args, "only", None)
    if getattr(args, "dry_run", False):
        client.require_feature("restore_dry_run", "a restore dry run")
    elif getattr(args, "background", False):
        client.require_feature("restore_background", "a background restore")
    if getattr(args, "background", False) and not getattr(args, "dry_run", False):
        started = client.restore_stored_backup(args.target, only=only, background=True)
        return _follow_import(client, str(started["import_id"]), args, out)
    dry_run = bool(getattr(args, "dry_run", False))
    report = client.restore_stored_backup(args.target, only=only, dry_run=dry_run)
    rejected = list(report.get("rejected", []))
    if dry_run and report.get("dry_run") is not True:
        print(
            f"error: the server ignored --dry-run and restored for real: "
            f"{len(report.get('jobs', []))} documents queued",
            file=sys.stderr,
        )
        return 1
    if dry_run and not args.json:
        for name in report.get("would_restore", []):
            print(f"would restore: {name}", file=out)
        for name in report.get("skipped", []):
            print(f"already indexed: {name}", file=out)
        for name in report.get("missing_originals", []):
            print(f"no original kept: {name}", file=out)
        for name in rejected:
            print(f"not restored: {name}", file=out)
        print(
            f"{len(report.get('would_restore', []))} would be restored, "
            f"{len(report.get('skipped', []))} already indexed, {len(rejected)} not restored "
            "(nothing was changed)",
            file=out,
        )
        return 1 if rejected else 0
    if args.json:
        print(json.dumps(report), file=out)
    else:
        for name in rejected:
            print(f"not restored: {name}", file=out)
        print(
            f"{len(report.get('jobs', []))} documents queued, "
            f"{len(report.get('skipped', []))} already indexed, "
            f"{len(report.get('deferred', []))} deferred"
            + (f", {len(rejected)} not restored" if rejected else ""),
            file=out,
        )
    # A name given to --only that the backup doesn't hold (a typo) is a failure: it
    # used to print "0 documents queued" and exit 0.
    return 1 if rejected else 0


def cmd_check(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """The consistency report; exits 1 when it found problems (nothing is changed),
    or with ``--repair``, those remedies applied first."""

    if args.repair:
        remedies = {
            "originals": "delete_orphan_originals",
            "versions": "delete_orphan_versions",
            "incomplete": "reindex_incomplete",
        }
        fixed = client.repair(**{remedies[name]: True for name in args.repair})
        for name in fixed.get("originals_deleted", []):
            print(f"deleted original: {name}", file=out)
        for name in fixed.get("versions_deleted", []):
            print(f"deleted earlier versions: {name}", file=out)
        for job in fixed.get("reindex_jobs", []):
            print(f"re-indexing: {job.get('filename')}", file=out)
        for name in fixed.get("left", []):
            print(f"left as it was: {name}", file=out)
    report = client.consistency()
    if args.json:
        print(json.dumps(report), file=out)
        return 1 if report.get("problems") else 0
    print(
        f"{report.get('documents', 0)} documents, "
        f"{report.get('problems', 0)} problem(s)"
        + ("" if report.get("originals_kept") else " (originals aren't kept)"),
        file=out,
    )
    sections = [
        ("originals_without_document", "original, no document"),
        ("documents_without_original", "document, no original"),
        ("versions_without_document", "earlier versions, no document"),
        ("incomplete", "incomplete (re-index repairs it)"),
        ("stale", "stale (re-index recommended)"),
        ("in_flight", "being indexed now"),
    ]
    for key, label in sections:
        for name in report.get(key, []):
            print(f"{label}: {name}", file=out)
    for group in report.get("duplicate_groups", []):
        print(f"same bytes: {', '.join(group)}", file=out)
    return 1 if report.get("problems") else 0


def cmd_export(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    written = client.export(Path(args.output), include_versions=args.include_versions)
    if args.json:
        print(json.dumps({"output": args.output, "bytes": written}), file=out)
    else:
        print(f"wrote {written} bytes to {args.output}", file=out)
    return 0


def _import_in_background(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """``import --background``: every document is queued as room frees (none deferred);
    this follows the import until it has queued them all."""

    started = client.restore(Path(args.archive), background=True)
    import_id = started.get("import_id")
    if not import_id:
        print("error: the server didn't start a background import (too old?)", file=sys.stderr)
        return 1
    return _follow_import(client, import_id, args, out)


# The bundle format this CLI reads (api/migrate.py's MIGRATE_VERSION).
MIGRATE_VERSION = 1
_READ_BLOCK = 1 << 20


def _read_through(handle: Any) -> None:
    """Every byte of a zip member read: its CRC is checked as the end is reached."""

    while handle.read(_READ_BLOCK):
        pass


def verify_bundle(path: Path, args: argparse.Namespace, out: TextIO) -> int:
    """``migrate verify FILE``: a bundle read through before it is carried anywhere —
    its manifest, each workspace's backup (its own manifest and every member's
    checksum) and snapshot. Nothing is sent. Exit 1 when something is wrong."""

    import zipfile

    problems: list[str] = []
    workspaces: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as bundle:
            try:
                manifest = json.loads(bundle.read("manifest.json"))
            except KeyError:
                raise ValueError("it has no manifest.json; is it a migration bundle?") from None
            if not isinstance(manifest, dict) or not isinstance(manifest.get("workspaces"), list):
                raise ValueError("its manifest.json is not a migration manifest")
            if manifest.get("migrate_version") != MIGRATE_VERSION:
                problems.append(
                    f"bundle version {manifest.get('migrate_version')!r}; this CLI knows "
                    f"version {MIGRATE_VERSION}"
                )
            for entry in manifest["workspaces"]:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "?")
                found: dict[str, Any] = {"name": name, "documents": entry.get("documents")}
                member = entry.get("backup")
                try:
                    with bundle.open(str(member)) as raw, zipfile.ZipFile(raw) as backup:
                        listed = json.loads(backup.read("manifest.json")).get("documents", [])
                        # Those it holds, not those it lists: and each it says it
                        # holds must be there.
                        held = set(backup.namelist())
                        named = [
                            str(document["original"])
                            for document in listed
                            if isinstance(document, dict) and document.get("original")
                        ]
                        found["originals"] = sum(member in held for member in named)
                        absent = [member for member in named if member not in held]
                        if absent:
                            problems.append(
                                f"{name}: its backup lacks {len(absent)} original(s) it "
                                f"lists, {absent[0]} first"
                            )
                        damaged = backup.testzip()
                        if damaged is not None:
                            problems.append(f"{name}: {damaged} in its backup is damaged")
                except (
                    KeyError,
                    zipfile.BadZipFile,
                    ValueError,
                    OSError,
                    AttributeError,  # a manifest that isn't an object
                    TypeError,
                ) as exc:
                    problems.append(f"{name}: its backup can't be read ({exc})")
                snapshot = entry.get("snapshot")
                found["snapshot"] = bool(snapshot)
                if snapshot:
                    try:
                        with bundle.open(str(snapshot)) as handle:
                            _read_through(handle)
                    except (KeyError, zipfile.BadZipFile, OSError) as exc:
                        problems.append(f"{name}: its index snapshot can't be read ({exc})")
                workspaces.append(found)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        problems.append(f"{path}: {exc}")
    if args.json:
        print(
            json.dumps({"ok": not problems, "problems": problems, "workspaces": workspaces}),
            file=out,
        )
    else:
        for found in workspaces:
            print(
                f"{found['name']}: {found.get('originals', '?')} originals"
                + (", index snapshot" if found.get("snapshot") else ", no snapshot (re-indexed)"),
                file=out,
            )
        for problem in problems:
            print(f"problem: {problem}", file=out)
        print("the bundle reads through" if not problems else "the bundle has problems", file=out)
    return 0 if not problems else 1


def cmd_migrate(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """``pack FILE``: the whole server (every workspace's documents, settings and index
    snapshot) into one file. ``unpack FILE``: put one back and follow it."""

    target = Path(args.file)
    if args.action == "verify":
        return verify_bundle(target, args, out)
    if args.action == "pack":
        # The bundle (and its background export) belongs to the default workspace,
        # whichever one this command names: polled under another, it read as unknown.
        client.use_workspace("")
        written = (
            client.migrate_pack_in_background(target, timeout=_wait_limit(args))
            if args.background
            else client.migrate_pack(target)
        )
        if args.json:
            print(json.dumps({"output": str(target), "bytes": written}), file=out)
        else:
            print(f"wrote {written} bytes to {target}", file=out)
        return 0
    started = client.migrate_unpack(target)
    import_id = started.get("import_id")
    if not import_id:
        print("error: the server didn't start unpacking (too old?)", file=sys.stderr)
        return 1
    if not args.json:
        print(f"unpacking: {', '.join(started.get('workspaces', [])) or 'no workspaces'}", file=out)
    # The import belongs to the default workspace, whichever one this command names.
    client.use_workspace("")
    return _follow_import(client, import_id, args, out)


def cmd_imports(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Background imports (zips, backups, rebuilds, bundles); --cancel stops one."""

    if args.cancel:
        stopped = client.cancel_import(args.cancel)
        print(json.dumps(stopped) if args.json else f"stopping {args.cancel}", file=out)
        return 0
    imports = client.background_imports()
    if args.json:
        print(json.dumps(imports), file=out)
        return 0
    for item in imports:
        print(
            f"{item.get('id')}  {item.get('kind'):8} {item.get('state'):10} "
            f"queued {item.get('queued', 0)}, skipped {item.get('skipped', 0)}, "
            f"failed {item.get('failed', 0)}  {item.get('name')}"
            + ("  (resumed)" if item.get("resumed") else ""),
            file=out,
        )
    if not imports:
        print("no background imports", file=out)
    return 0


def _setting_value(raw: str) -> Any:
    """A NAME=VALUE's value as JSON when it reads as JSON (0.3, 5, null), else text."""

    try:
        return json.loads(raw)
    except ValueError:
        return raw


def cmd_settings(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """A workspace's own settings: ``get`` shows them and the ones in effect; ``set
    NAME=VALUE...`` changes those named (``NAME=null`` gives one back to the server)."""

    name = args.name or client.workspace or "default"
    if args.action == "set":
        current = dict(client.workspace_settings(name).get("settings", {}))
        for pair in args.values:
            key, sep, raw = pair.partition("=")
            if not sep or not key:
                print(f"error: expected NAME=VALUE, got {pair!r}", file=sys.stderr)
                return 2
            current[key.strip().lower()] = _setting_value(raw.strip())
        result = client.set_workspace_settings(name, current)
    else:
        result = client.workspace_settings(name)
    if args.json:
        print(json.dumps(result), file=out)
        return 0
    own = result.get("settings", {})
    for key, value in sorted(result.get("effective", {}).items()):
        mark = "" if own.get(key) is None else "  (this workspace's)"
        print(f"{key} = {json.dumps(value)}{mark}", file=out)
    return 0


def cmd_cache(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """The workspace's cached answers; --clear empties them."""

    if args.clear:
        removed = client.clear_answer_cache()
        print(
            json.dumps(removed) if args.json else f"removed {removed.get('removed', 0)}", file=out
        )
        return 0
    cache = client.answer_cache()
    if args.json:
        print(json.dumps(cache), file=out)
        return 0
    if not cache.get("enabled"):
        print("the answer cache is off (ANSWER_CACHE_SIZE=0)", file=out)
        return 0
    print(
        f"{cache.get('total_entries', 0)} of {cache.get('max_entries', 0)} cached "
        f"(kept {cache.get('ttl_seconds', 0):.0f} s), {cache.get('in_flight', 0)} being answered",
        file=out,
    )
    for entry in cache.get("entries", []):
        print(f"  {entry.get('age_seconds', 0):6.0f} s  {entry.get('question')}", file=out)
    return 0


def cmd_notify_test(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Send one notification through the server's NOTIFY_URL; exits 1 when it failed."""

    result = client.notify_test()
    if args.json:
        print(json.dumps(result), file=out)
    elif result.get("sent"):
        print("sent: check where your notifications arrive", file=out)
    else:
        print(f"not sent: {result.get('error') or 'the server gave no reason'}", file=out)
    return 0 if result.get("sent") else 1


def cmd_rebuild(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Index every stored original that has no document (after a lost index)."""

    started = client.rebuild()
    import_id = started.get("import_id")
    if not import_id:
        if args.json:
            print(json.dumps(started), file=out)
        else:
            print("nothing to rebuild: every stored original has its document", file=out)
        return 0
    if not args.json:
        print(f"rebuilding {started.get('originals', 0)} documents from their originals", file=out)
    return _follow_import(client, import_id, args, out)


def cmd_import(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Restore a backup and wait for every indexing job it started (``--json`` too).

    Exits 1 when a job failed or could not be followed, or the server rejected or
    deferred anything (a deferred document needs the same import run again)."""

    if args.background:
        return _import_in_background(client, args, out)
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
