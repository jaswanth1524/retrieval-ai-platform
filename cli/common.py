"""Helpers the ``docrag`` command groups share (exit codes, following jobs)."""

from __future__ import annotations

import argparse
import json
from typing import Any, TextIO

from cli.client import (
    DocragClient,
)

# `ask --strict`: the documents don't answer the question (a script can tell).
EXIT_NOT_ANSWERED = 3


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


# An option left out (``refresh --every``, whose None means "the server's interval").
_NOT_GIVEN: Any = object()


def _follow_import(
    client: DocragClient, import_id: str, args: argparse.Namespace, out: TextIO
) -> int:
    """Follow a background import (a backup, a zip, a rebuild) until it has queued
    everything; exits 1 unless it finished with nothing failed."""

    seen = -1

    def progress(status: dict[str, Any]) -> None:
        nonlocal seen
        if not args.json and status.get("queued", 0) != seen:
            seen = status.get("queued", 0)
            print(f"queued {seen} so far…", file=out)

    status = client.wait_for_import(import_id, timeout=_wait_limit(args), on_progress=progress)
    if args.json:
        print(json.dumps(status), file=out)
    else:
        for problem in status.get("problems", []):
            print(f"  {problem}", file=out)
        print(
            f"{status.get('queued', 0)} queued, {status.get('skipped', 0)} skipped, "
            f"{status.get('failed', 0)} failed ({status.get('state')}); `docrag jobs` "
            "follows their indexing",
            file=out,
        )
    return 0 if status.get("state") == "done" and not status.get("failed") else 1
