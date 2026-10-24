"""Two eval runs side by side, question by question, and one run as an HTML report.

``_check_baseline`` compares aggregates, so a run could keep its mean hit@1 while
several questions flipped; a reviewer had nothing that named them. ``diff_runs`` pairs
the questions of two outputs of the same command (``harness retrieval`` results, or
``harness answer`` rows) and lists every one that got better or worse;
``render_report`` writes one run (optionally with RAGAS scores and a diff) as a single
self-contained HTML page.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The retrieval metrics a flip is judged on (more would only repeat the same change).
RETRIEVAL_METRICS = ("hit@1", "mrr")
_RETRIEVAL_LEVELS = ("filename", "chunk")
# An answer status that counts as answered from the documents.
_ANSWERED = "answered"


class RunFileError(ValueError):
    """A file that isn't the output of ``harness retrieval`` or ``harness answer``."""


@dataclass(frozen=True)
class Run:
    kind: str  # "retrieval" or "answer"
    data: Any

    def questions(self) -> dict[str, dict[str, Any]]:
        rows = self.data["per_question"] if self.kind == "retrieval" else self.data
        keyed: dict[str, dict[str, Any]] = {}
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("question"), str):
                keyed.setdefault(row["question"], row)
        return keyed


def load_run(path: Path) -> Run:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunFileError(f"{path}: {exc}") from exc
    if isinstance(data, dict) and isinstance(data.get("per_question"), list):
        return Run("retrieval", data)
    if isinstance(data, list) and all(isinstance(row, dict) and "answer" in row for row in data):
        return Run("answer", data)
    raise RunFileError(f"{path}: not a harness retrieval result or answer run")


@dataclass
class Change:
    question: str
    # "regressed" or "improved".
    direction: str
    details: list[str]


@dataclass
class RunDiff:
    kind: str
    regressed: list[Change] = field(default_factory=list)
    improved: list[Change] = field(default_factory=list)
    only_before: list[str] = field(default_factory=list)
    only_after: list[str] = field(default_factory=list)
    # Aggregate metrics, before and after (retrieval only).
    aggregates: list[tuple[str, float, float]] = field(default_factory=list)


def _retrieval_change(before: dict[str, Any], after: dict[str, Any]) -> tuple[int, list[str]]:
    """(-1, 0 or 1) for worse / same / better, and what moved."""

    direction = 0
    details: list[str] = []
    for level in _RETRIEVAL_LEVELS:
        old, new = before.get(level), after.get(level)
        if not isinstance(old, dict) or not isinstance(new, dict):
            continue
        for metric in RETRIEVAL_METRICS:
            a, b = old.get(metric), new.get(metric)
            if not isinstance(a, int | float) or not isinstance(b, int | float) or a == b:
                continue
            details.append(f"{level} {metric} {a:.2f}→{b:.2f}")
            # The first metric that moved decides (hit@1 before mrr).
            if direction == 0:
                direction = 1 if b > a else -1
    old_nc, new_nc = before.get("no_context"), after.get("no_context")
    if isinstance(old_nc, bool) and isinstance(new_nc, bool) and old_nc != new_nc:
        details.append(f"refused {old_nc}→{new_nc}")
        if direction == 0:
            direction = 1 if new_nc else -1
    return direction, details


def _answer_change(before: dict[str, Any], after: dict[str, Any]) -> tuple[int, list[str]]:
    direction = 0
    details: list[str] = []

    def score(row: dict[str, Any]) -> tuple[int, int]:
        answered = 0 if row.get("error") else 1 if row.get("answer_status") == _ANSWERED else 0
        fact = 1 if row.get("fact_hit") is True else 0
        return fact, answered

    old_status = "error" if before.get("error") else before.get("answer_status")
    new_status = "error" if after.get("error") else after.get("answer_status")
    if old_status != new_status:
        details.append(f"status {old_status}→{new_status}")
    if before.get("fact_hit") != after.get("fact_hit") and after.get("fact_hit") is not None:
        details.append(f"fact {before.get('fact_hit')}→{after.get('fact_hit')}")
    old, new = score(before), score(after)
    if new != old:
        direction = 1 if new > old else -1
    return direction, details


def diff_runs(before: Run, after: Run) -> RunDiff:
    if before.kind != after.kind:
        raise RunFileError(f"can't compare a {before.kind} run with a {after.kind} run")
    result = RunDiff(before.kind)
    old, new = before.questions(), after.questions()
    result.only_before = sorted(set(old) - set(new))
    result.only_after = sorted(set(new) - set(old))
    judge = _retrieval_change if before.kind == "retrieval" else _answer_change
    for question in old:
        if question not in new:
            continue
        direction, details = judge(old[question], new[question])
        if direction < 0:
            result.regressed.append(Change(question, "regressed", details))
        elif direction > 0:
            result.improved.append(Change(question, "improved", details))
    if before.kind == "retrieval":
        for level, metrics in sorted(before.data.get("aggregate", {}).items()):
            later = after.data.get("aggregate", {}).get(level, {})
            for metric, value in sorted(metrics.items()):
                other = later.get(metric)
                if isinstance(value, int | float) and isinstance(other, int | float):
                    if value != other:
                        result.aggregates.append((f"{level} {metric}", float(value), float(other)))
    return result


def format_diff(diff: RunDiff) -> str:
    lines = [f"{len(diff.regressed)} regressed, {len(diff.improved)} improved ({diff.kind} run)"]
    for change in diff.regressed + diff.improved:
        marker = "-" if change.direction == "regressed" else "+"
        lines.append(f"{marker} {change.question}  [{'; '.join(change.details)}]")
    for question in diff.only_before:
        lines.append(f"? only in the first run: {question}")
    for question in diff.only_after:
        lines.append(f"? only in the second run: {question}")
    for name, before, after in diff.aggregates:
        lines.append(f"  {name}: {before:.3f} → {after:.3f} ({after - before:+.3f})")
    return "\n".join(lines)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def ragas_scores(data: Any) -> dict[str, float]:
    """Mean score per metric from ``ragas_runner`` output: a ``{metric: score}`` object,
    or ``{"rows": [...]}`` (one record per question; non-numeric columns ignored)."""

    if not isinstance(data, dict):
        return {}
    rows = data.get("rows")
    if isinstance(rows, list):
        sums: dict[str, list[float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    sums.setdefault(key, []).append(float(value))
        return {key: sum(values) / len(values) for key, values in sums.items() if values}
    return {
        key: float(value)
        for key, value in data.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _ragas_rows(path: Path | None) -> str:
    if path is None:
        return ""
    scores = ragas_scores(json.loads(path.read_text(encoding="utf-8")))
    rows = [[name, f"{value:.3f}"] for name, value in sorted(scores.items())]
    return "<h2>RAGAS</h2>" + _table(["metric", "mean score"], rows) if rows else ""


_STYLE = """
body { font: 14px/1.5 system-ui, sans-serif; margin: 24px auto; max-width: 1100px;
  padding: 0 16px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 24px; }
th, td { border-bottom: 1px solid #ddd; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f5f5f5; }
.bad { color: #b3261e; } .good { color: #1b7f3b; }
"""


def render_report(
    run: Run, *, title: str, ragas: Path | None = None, base: Run | None = None
) -> str:
    """One self-contained HTML page: aggregates, every question, and (with ``base``)
    what changed since it. Every value from the files is escaped."""

    parts = [f"<h1>{html.escape(title)}</h1>"]
    if run.kind == "retrieval":
        aggregate_rows = [
            [level, metric, f"{value:.3f}"]
            for level, metrics in sorted(run.data.get("aggregate", {}).items())
            for metric, value in sorted(metrics.items())
            if isinstance(value, int | float)
        ]
        parts.append("<h2>Aggregates</h2>" + _table(["level", "metric", "value"], aggregate_rows))
        question_rows = []
        for row in run.data["per_question"]:
            cells = [str(row.get("question", ""))]
            for level in _RETRIEVAL_LEVELS:
                scores = row.get(level) or {}
                cells.append(
                    " ".join(f"{m}={scores[m]:.2f}" for m in RETRIEVAL_METRICS if m in scores)
                )
            cells.append("" if "no_context" not in row else str(row["no_context"]))
            question_rows.append(cells)
        parts.append(
            "<h2>Questions</h2>"
            + _table(["question", "filename", "chunk", "refused"], question_rows)
        )
    else:
        rows = run.data
        answered = sum(
            1 for row in rows if not row.get("error") and row.get("answer_status") == _ANSWERED
        )
        facts = [row for row in rows if row.get("fact_hit") is not None]
        summary = [
            ["questions", str(len(rows))],
            ["answered", f"{answered} ({answered / len(rows):.0%})" if rows else "0"],
            ["errors", str(sum(1 for row in rows if row.get("error")))],
        ]
        if facts:
            hits = sum(1 for row in facts if row.get("fact_hit") is True)
            summary.append(["facts stated", f"{hits} of {len(facts)}"])
        parts.append("<h2>Summary</h2>" + _table(["", ""], summary))
        parts.append(
            "<h2>Questions</h2>"
            + _table(
                ["question", "status", "fact", "cited", "answer"],
                [
                    [
                        str(row.get("question", "")),
                        "error: " + str(row["error"])
                        if row.get("error")
                        else str(row.get("answer_status", "")),
                        "" if row.get("fact_hit") is None else str(row["fact_hit"]),
                        ", ".join(str(name) for name in row.get("cited_filenames", [])),
                        str(row.get("answer", ""))[:600],
                    ]
                    for row in rows
                ],
            )
        )
    parts.append(_ragas_rows(ragas))
    if base is not None:
        diff = diff_runs(base, run)
        changes = [
            [change.direction, change.question, "; ".join(change.details)]
            for change in diff.regressed + diff.improved
        ]
        parts.append(
            f"<h2>Since the base run: <span class='bad'>{len(diff.regressed)} regressed</span>, "
            f"<span class='good'>{len(diff.improved)} improved</span></h2>"
            + _table(["change", "question", "what moved"], changes)
        )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<meta http-equiv='Content-Security-Policy' "
        "content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>"
        + "".join(parts)
        + "</body></html>\n"
    )
