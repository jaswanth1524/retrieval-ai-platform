"""Turn captured answer feedback into a draft retrieval-eval dataset.

Ratings from ``POST /feedback`` (FEEDBACK_ENABLED) are only useful once they reach an
evaluation. A thumbs-up on an answer says its cited documents were the right ones for
that question, which is exactly a ``relevant_filenames`` label for the retrieval
harness. A thumbs-down says something was wrong but not what the right documents were,
so those questions go to a review file for a person to label instead of into the
dataset with a guess.

Reads the feedback sqlite file (``FEEDBACK_STORE_PATH``) or a ``GET /export`` zip's
``feedback.jsonl``. Only answer-level ratings count: a rating of one citation carries
the answer's cited filenames, not which of them that citation was.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from api.feedback import FeedbackStore

_STORE_PAGE = 500


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


@dataclass
class FeedbackDataset:
    """The draft dataset, the questions to review, and what was left out and why."""

    examples: list[dict[str, Any]] = field(default_factory=list)
    review: list[dict[str, Any]] = field(default_factory=list)
    ratings_read: int = 0
    citation_ratings: int = 0
    uncited: int = 0


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
    return FeedbackRow(
        question=" ".join(question.split()),
        answer_excerpt=excerpt if isinstance(excerpt, str) else "",
        cited_filenames=tuple(name for name in cited if isinstance(name, str) and name),
        rating=rating,
        citation_source_number=number if isinstance(number, int) else None,
        created_at=float(created) if isinstance(created, (int, float)) else 0.0,
    )


def rows_from_store(path: Path, workspace: str | None = None) -> Iterator[FeedbackRow]:
    """Every rating in a feedback sqlite file, oldest first; only ``workspace``'s
    ("" = the default one) when it is given. A labeled dataset is run against one
    corpus, and each workspace is a corpus of its own."""

    if not path.is_file():
        raise FeedbackDataError(f"No feedback store at {path}.")
    store = FeedbackStore(str(path))
    cursor = None
    while True:
        page, cursor = store.rows_oldest_first(cursor, _STORE_PAGE, workspace)
        for item in page:
            row = _row(
                {
                    "question": item.question,
                    "answer_excerpt": item.answer_excerpt,
                    "cited_filenames": item.cited_filenames,
                    "rating": item.rating,
                    "citation_source_number": item.citation_source_number,
                    "created_at": item.created_at,
                }
            )
            if row is not None:
                yield row
        if cursor is None or len(page) < _STORE_PAGE:
            return


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


def build_dataset(rows: Iterable[FeedbackRow]) -> FeedbackDataset:
    """The latest answer-level rating of each question decides where it goes: a
    thumbs-up with cited documents becomes a labeled example, a thumbs-down a question
    to review. Questions are matched ignoring case and spacing."""

    result = FeedbackDataset()
    latest: dict[str, FeedbackRow] = {}
    for row in rows:
        result.ratings_read += 1
        if row.citation_source_number is not None:
            result.citation_ratings += 1
            continue
        key = " ".join(row.question.split()).casefold()
        if key not in latest or row.created_at >= latest[key].created_at:
            latest[key] = row
    for row in sorted(latest.values(), key=lambda item: item.created_at):
        if row.rating == "up":
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
