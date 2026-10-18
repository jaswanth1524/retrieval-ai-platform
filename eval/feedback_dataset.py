"""Turn captured answer feedback into a draft retrieval-eval dataset.

Ratings from ``POST /feedback`` (FEEDBACK_ENABLED) are only useful once they reach an
evaluation. A thumbs-up on an answer says its cited documents were the right ones for
that question, which is exactly a ``relevant_filenames`` label for the retrieval
harness. A thumbs-down says something was wrong but not what the right documents were,
so those questions go to a review file for a person to label instead of into the
dataset with a guess.

Reads the feedback sqlite file (``FEEDBACK_STORE_PATH``) or a ``GET /export`` zip's
``feedback.jsonl``. Only answer-level ratings of questions count: a rating of one
citation carries the answer's cited filenames, not which of them that citation was, and
a rated summary or follow-up suggestion is not a question retrieval answered (those go to
the review file, marked as such).
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A rating recorded before kinds were (no ``kind``) whose question reads like the task
# text of a summary or comparison (api/summarize.task_description) was one.
_SUMMARY_TASK_PREFIXES = ("Summarize ", "Compare ")


class FeedbackDataError(ValueError):
    """The feedback source can't be read."""


@dataclass(frozen=True)
class FeedbackRow:
    question: str
    answer_excerpt: str
    cited_filenames: tuple[str, ...]
    rating: str
    citation_source_number: int | None
    created_at: float
    # "question", "follow_up" or "summary"; None for a rating recorded before kinds.
    kind: str | None = None


@dataclass
class FeedbackDataset:
    """The draft dataset, the questions to review, and what was left out and why."""

    examples: list[dict[str, Any]] = field(default_factory=list)
    review: list[dict[str, Any]] = field(default_factory=list)
    ratings_read: int = 0
    citation_ratings: int = 0
    uncited: int = 0
    # Latest ratings of summaries and follow-up suggestions (sent to review, not labeled).
    not_questions: int = 0


def _row(data: dict[str, Any]) -> FeedbackRow | None:
    question = data.get("question")
    rating = data.get("rating")
    cited = data.get("cited_filenames")
    if not isinstance(question, str) or not question.strip() or rating not in ("up", "down"):
        return None
    if not isinstance(cited, list):
        cited = []
    number = data.get("citation_source_number")
    created = data.get("created_at")
    excerpt = data.get("answer_excerpt")
    kind = data.get("kind")
    return FeedbackRow(
        question=" ".join(question.split()),
        answer_excerpt=excerpt if isinstance(excerpt, str) else "",
        cited_filenames=tuple(name for name in cited if isinstance(name, str) and name),
        rating=rating,
        citation_source_number=number if isinstance(number, int) else None,
        created_at=float(created) if isinstance(created, (int, float)) else 0.0,
        kind=kind if isinstance(kind, str) else None,
    )


def rows_from_store(path: Path, workspace: str | None = None) -> Iterator[FeedbackRow]:
    """Every rating in a feedback sqlite file, oldest first; only ``workspace``'s
    ("" = the default one) when it is given. A labeled dataset is run against one
    corpus, and each workspace is a corpus of its own.

    Opened read-only, with plain SELECTs: ``FeedbackStore`` would create the file's
    table, migrate an older layout in place and switch it to WAL — writes to a live
    server's store from an offline tool. A file from before workspaces or kinds is read
    as it is (no workspace column = all the default workspace's).
    """

    if not path.is_file():
        raise FeedbackDataError(f"No feedback store at {path}.")
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
            if not columns:
                raise FeedbackDataError(f"{path} has no feedback table.")
            kind = "kind" if "kind" in columns else "NULL"
            where = ""
            params: tuple[str, ...] = ()
            if workspace is not None:
                if "workspace" in columns:
                    where, params = "WHERE workspace = ? ", (workspace,)
                elif workspace:
                    return  # every row is the default workspace's
            rows = conn.execute(
                "SELECT question, answer_excerpt, cited_filenames_json, rating, "
                f"citation_source_number, created_at, {kind} FROM feedback {where}"
                "ORDER BY created_at ASC, rowid ASC",
                params,
            )
            for question, excerpt, cited_json, rating, number, created, row_kind in rows:
                try:
                    cited = json.loads(cited_json)
                except (TypeError, ValueError):
                    cited = []
                row = _row(
                    {
                        "question": question,
                        "answer_excerpt": excerpt,
                        "cited_filenames": cited,
                        "rating": rating,
                        "citation_source_number": number,
                        "created_at": created,
                        "kind": row_kind,
                    }
                )
                if row is not None:
                    yield row
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise FeedbackDataError(f"{path} is not a readable feedback store ({exc}).") from exc


def rows_from_export(path: Path) -> Iterator[FeedbackRow]:
    """Every rating in a ``GET /export`` zip's ``feedback.jsonl``, in file order."""

    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise FeedbackDataError(f"{path} is not a readable export zip.") from exc
    with archive:
        if "feedback.jsonl" not in archive.namelist():
            raise FeedbackDataError(f"{path} has no feedback.jsonl (feedback was off).")
        with archive.open("feedback.jsonl") as member:
            for line in member:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and (row := _row(data)) is not None:
                    yield row


def _kind_of(row: FeedbackRow) -> str:
    if row.kind is not None:
        return row.kind
    return "summary" if row.question.startswith(_SUMMARY_TASK_PREFIXES) else "question"


def build_dataset(rows: Iterable[FeedbackRow]) -> FeedbackDataset:
    """The latest answer-level rating of each question decides where it goes: a
    thumbs-up with cited documents becomes a labeled example, a thumbs-down a question
    to review. Questions are matched ignoring case and spacing.

    Only ratings of answers to questions are labels. A rated summary or follow-up
    suggestion goes to review, marked as not a question: its cited documents are every
    passage a summary read, or say nothing about what retrieval should have found.
    """

    result = FeedbackDataset()
    latest: dict[tuple[str, str], FeedbackRow] = {}
    for row in rows:
        result.ratings_read += 1
        if row.citation_source_number is not None:
            result.citation_ratings += 1
            continue
        key = (_kind_of(row), " ".join(row.question.split()).casefold())
        if key not in latest or row.created_at >= latest[key].created_at:
            latest[key] = row
    for (kind, _), row in sorted(latest.items(), key=lambda item: item[1].created_at):
        if kind != "question":
            result.not_questions += 1
            result.review.append(
                {
                    "question": " ".join(row.question.split()),
                    "relevant_filenames": [],
                    "answer_excerpt": row.answer_excerpt,
                    "cited_filenames": sorted(set(row.cited_filenames)),
                    "source": f"feedback (a rated {kind.replace('_', '-')}, not a question: "
                    "left out of the dataset)",
                }
            )
        elif row.rating == "up":
            if not row.cited_filenames:
                result.uncited += 1
                continue
            result.examples.append(
                {
                    "question": " ".join(row.question.split()),
                    "relevant_filenames": sorted(set(row.cited_filenames)),
                    "relevant_chunk_ids": [],
                    "source": "feedback",
                }
            )
        else:
            result.review.append(
                {
                    "question": " ".join(row.question.split()),
                    "relevant_filenames": [],
                    "answer_excerpt": row.answer_excerpt,
                    "cited_filenames": sorted(set(row.cited_filenames)),
                    "source": "feedback (rated down: label the right documents)",
                }
            )
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return len(lines)
