"""Eval sets without labelling by hand (``harness generate`` / ``harness from-queries``).

``generate``: passages are sampled across the indexed corpus and the configured LLM
writes, for each, one question it answers and the short phrase of the passage that
answers it. A row is kept only when that phrase really is in the passage, so every
label resolves (``relevant_passages``) and the answer can be checked
(``answer_contains``). Generated questions lean on the passage's wording more than real
ones do: they are a start, worth reading through, not a substitute for asked questions.

``from-queries``: the query log (QUERY_LOG_ENABLED) as a dataset draft — answered
questions labelled with the documents their answers cited, the uncovered ones in a
review file (each may be a gap in the documents or a question they don't cover).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from api.generation import ChatGenerator, ChatMessage
from api.settings import AppSettings

QUESTION_SYSTEM_PROMPT = (
    "You write test questions for a document search system. Given one passage, write "
    "one question a person might ask that this passage answers — phrased the way a user "
    "would ask it, not copied from the passage — and the short phrase from the passage "
    "that answers it, copied exactly, at most eight words. Reply with JSON only: "
    '{"question": "...", "answer_phrase": "..."}. If the passage answers no useful '
    'question (a heading, a table of contents), reply {"question": ""}.'
)
# Passages shorter than this say too little to ask about.
MIN_PASSAGE_WORDS = 12
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Passage:
    filename: str
    chunk_id: str
    text: str


def sample_passages(
    repository: Any, settings: AppSettings, count: int, seed: int = 0
) -> list[Passage]:
    """Up to ``count`` passages, spread over the documents (round robin over each one's
    shuffled passages), skipping ones under ``MIN_PASSAGE_WORDS``."""

    rng = random.Random(seed)
    by_file: list[list[Passage]] = []
    for filename in sorted(repository.filename_metadata(settings)):
        passages = [
            Passage(filename, str(p.get("chunk_id", "")), str(p.get("text", "")))
            for p in repository.chunks_for_filename(settings, filename)
            if len(str(p.get("text", "")).split()) >= MIN_PASSAGE_WORDS
        ]
        rng.shuffle(passages)
        if passages:
            by_file.append(passages)
    picked: list[Passage] = []
    while len(picked) < count and any(by_file):
        for passages in by_file:
            if passages and len(picked) < count:
                picked.append(passages.pop())
    return picked


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def parse_question(reply: str, passage: str) -> tuple[str, str] | None:
    """(question, answer phrase) from the model's reply, when it gave a question and a
    phrase that is in ``passage``; else None."""

    match = _JSON_RE.search(reply)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    question = " ".join(str(data.get("question", "")).split())
    phrase = " ".join(str(data.get("answer_phrase", "")).split()).strip(" .")
    if not question or not phrase or len(phrase.split()) > 12:
        return None
    if _normalized(phrase) not in _normalized(passage):
        return None
    if not question.endswith("?"):
        question += "?"
    return question, phrase


def generate_rows(
    passages: Iterable[Passage], generator: ChatGenerator, settings: AppSettings
) -> tuple[list[dict[str, Any]], int]:
    """Dataset rows for ``passages``, and how many passages gave no usable question."""

    rows: list[dict[str, Any]] = []
    skipped = 0
    for passage in passages:
        messages: list[ChatMessage] = [
            {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Passage:\n{passage.text}"},
        ]
        try:
            reply = generator.complete(messages, settings)
        except Exception:  # noqa: BLE001 - one passage's failure skips that passage
            skipped += 1
            continue
        parsed = parse_question(reply, passage.text)
        if parsed is None:
            skipped += 1
            continue
        question, phrase = parsed
        rows.append(
            {
                "question": question,
                "relevant_filenames": [passage.filename],
                "relevant_passages": [phrase],
                "answer_contains": [phrase],
            }
        )
    return rows, skipped


def rows_from_queries(
    queries: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(labelled rows, review rows) from query-log rows (``GET /query-log`` shape), the
    latest of each question (case and spacing ignored); summaries left out."""

    latest: dict[str, dict[str, Any]] = {}
    for row in sorted(queries, key=lambda r: float(r.get("created_at") or 0.0)):
        if row.get("kind", "question") != "question":
            continue
        question = " ".join(str(row.get("question", "")).split())
        if question:
            latest[question.casefold()] = {**row, "question": question}
    labelled: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for row in latest.values():
        if row.get("status") == "error":
            continue  # a failure, not a sign the documents lack the answer
        filenames = sorted({str(name) for name in row.get("filenames") or []})
        if row.get("status") == "ok" and filenames:
            labelled.append({"question": row["question"], "relevant_filenames": filenames})
        else:
            review.append(
                {
                    "question": row["question"],
                    "status": row.get("status"),
                    "top_score": row.get("top_score"),
                    # A draft label: confirm the documents really don't cover it.
                    "expect_no_context": True,
                }
            )
    return labelled, review


def write_jsonl(path: Any, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
