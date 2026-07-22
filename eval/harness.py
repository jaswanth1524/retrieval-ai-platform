"""Live evaluation harness: drive the real pipeline over a labeled question set.

Two modes:

* ``retrieval`` — embed + hybrid-search + rerank each question and score the ranked
  filenames / chunk_ids against labeled relevant sets (hit@k, recall@k, MRR). No LLM.
  Optionally fails against a committed baseline for regression gating.
* ``answer`` — run the full pipeline (retrieval + neighbor expansion + generation) and
  write ``{question, answer, contexts, reference}`` rows directly consumable by
  ``eval.ragas_runner`` for offline RAGAS scoring.

Both modes need a running Qdrant with an ingested corpus (and ``answer`` also an LLM),
so this is a **local** tool, not a CI step. CI unit-tests ``eval.metrics`` and this
module's dataset loading / arg parsing only. The pipeline-driving functions take their
collaborators as parameters so tests can inject in-memory fakes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from api.generation import ChatGenerator, generate_grounded_answer
from api.pipeline import expand_with_neighbors
from api.reranking import Reranker, rerank_candidates_detailed
from api.retrieval import QueryEmbeddingProvider, embed_query, points_to_chunks
from api.settings import AppSettings, get_settings
from eval.metrics import aggregate, compare_to_baseline, score_ranking

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 8)


class HarnessDataError(ValueError):
    """Raised when a harness dataset is malformed."""


@dataclass(frozen=True)
class RetrievalExample:
    """One labeled retrieval-evaluation example."""

    question: str
    reference: str | None
    relevant_filenames: frozenset[str]
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class RankedResult:
    """The ranked identifiers a single retrieval run returned."""

    filenames: list[str]
    chunk_ids: list[str]


@dataclass(frozen=True)
class AnswerRow:
    """A generated answer row, in ``eval.ragas_runner`` input shape."""

    question: str
    answer: str
    contexts: list[str]
    reference: str | None

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
        }
        if self.reference is not None:
            row["reference"] = self.reference
        return row


class Collaborators:
    """The four pipeline seams the harness drives, built from settings for the CLI."""

    def __init__(self, settings: AppSettings) -> None:
        # Imported lazily so the module (and its CI unit tests) never pull in heavy
        # model/qdrant construction just to parse args or load a dataset.
        from api.embeddings import LocalEmbeddingProvider
        from api.generation import LiteLLMGenerator
        from api.qdrant_schema import make_qdrant_client
        from api.repository import VectorRepository
        from api.reranking import LocalCrossEncoderReranker

        self.settings = settings
        self.repository = VectorRepository(make_qdrant_client(settings))
        self.embedding_provider: QueryEmbeddingProvider = LocalEmbeddingProvider(settings)
        self.reranker: Reranker = LocalCrossEncoderReranker(settings)
        self.generator: ChatGenerator = LiteLLMGenerator(settings)


def rank_for_question(
    question: str,
    *,
    repository: Any,
    embedding_provider: QueryEmbeddingProvider,
    reranker: Reranker,
    settings: AppSettings,
    filenames: Sequence[str] | None = None,
) -> RankedResult:
    """Embed, hybrid-search, and rerank one question — return the kept ranking."""

    embedding = embed_query(question, embedding_provider)
    points = repository.hybrid_search(settings, embedding, filenames)
    candidates = points_to_chunks(points)
    outcome = rerank_candidates_detailed(
        query=question, candidates=candidates, reranker=reranker, settings=settings
    )
    return RankedResult(
        filenames=[chunk.filename for chunk in outcome.kept],
        chunk_ids=[chunk.chunk_id for chunk in outcome.kept],
    )


def answer_for_question(
    example: RetrievalExample,
    *,
    repository: Any,
    embedding_provider: QueryEmbeddingProvider,
    reranker: Reranker,
    generator: ChatGenerator,
    settings: AppSettings,
) -> AnswerRow:
    """Run the full pipeline for one question and return a RAGAS-shaped row."""

    embedding = embed_query(example.question, embedding_provider)
    points = repository.hybrid_search(settings, embedding, None)
    candidates = points_to_chunks(points)
    outcome = rerank_candidates_detailed(
        query=example.question, candidates=candidates, reranker=reranker, settings=settings
    )
    expanded = expand_with_neighbors(outcome.kept, repository, settings)
    grounded = generate_grounded_answer(
        query=example.question,
        context_chunks=expanded,
        generator=generator,
        settings=settings,
    )
    limit = int(settings.max_context_chunks)
    contexts = [chunk.expanded_text or chunk.text for chunk in expanded[:limit]]
    return AnswerRow(
        question=example.question,
        answer=grounded.answer,
        contexts=contexts or [""],  # RAGAS requires a non-empty contexts list
        reference=example.reference,
    )


def load_retrieval_dataset(path: Path) -> list[RetrievalExample]:
    """Load labeled retrieval examples from JSON (list / {examples}) or JSONL."""

    return [_example_from_mapping(item, index) for index, item in enumerate(_load_rows(path), 1)]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    rows: list[Any]
    if suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise HarnessDataError(f"Invalid JSON on line {line_number}.") from exc
    elif suffix == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HarnessDataError("Dataset is not valid JSON.") from exc
        if isinstance(parsed, dict):
            parsed = parsed.get("examples")
        if not isinstance(parsed, list):
            raise HarnessDataError("JSON dataset must be a list or contain an 'examples' list.")
        rows = parsed
    else:
        raise HarnessDataError("Dataset must be a .json or .jsonl file.")

    result = [row for row in rows if isinstance(row, dict)]
    if len(result) != len(rows):
        raise HarnessDataError("Every dataset row must be a JSON object.")
    if not result:
        raise HarnessDataError("Dataset must contain at least one example.")
    return result


def _example_from_mapping(data: dict[str, Any], index: int) -> RetrievalExample:
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HarnessDataError(f"Example {index}: 'question' must be a non-empty string.")
    reference = data.get("reference")
    if reference is not None and not isinstance(reference, str):
        raise HarnessDataError(f"Example {index}: 'reference' must be a string when present.")
    return RetrievalExample(
        question=question.strip(),
        reference=reference.strip() if isinstance(reference, str) and reference.strip() else None,
        relevant_filenames=_string_set(data.get("relevant_filenames"), index, "relevant_filenames"),
        relevant_chunk_ids=_string_set(data.get("relevant_chunk_ids"), index, "relevant_chunk_ids"),
    )


def _string_set(value: Any, index: int, field: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise HarnessDataError(f"Example {index}: '{field}' must be a list of strings.")
    return frozenset(item.strip() for item in value if item.strip())


def run_retrieval_eval(
    examples: Sequence[RetrievalExample],
    collaborators: Collaborators,
    ks: Sequence[int],
) -> dict[str, Any]:
    """Score every example's filename ranking (and chunk ranking when labeled)."""

    per_question: list[dict[str, Any]] = []
    filename_rows: list[dict[str, float]] = []
    chunk_rows: list[dict[str, float]] = []
    for example in examples:
        ranked = rank_for_question(
            example.question,
            repository=collaborators.repository,
            embedding_provider=collaborators.embedding_provider,
            reranker=collaborators.reranker,
            settings=collaborators.settings,
        )
        entry: dict[str, Any] = {"question": example.question}
        if example.relevant_filenames:
            file_scores = score_ranking(ranked.filenames, example.relevant_filenames, ks)
            entry["filename"] = file_scores
            filename_rows.append(file_scores)
        if example.relevant_chunk_ids:
            chunk_scores = score_ranking(ranked.chunk_ids, example.relevant_chunk_ids, ks)
            entry["chunk"] = chunk_scores
            chunk_rows.append(chunk_scores)
        per_question.append(entry)

    return {
        "k_values": list(ks),
        "question_count": len(examples),
        "aggregate": {
            "filename": aggregate(filename_rows),
            "chunk": aggregate(chunk_rows),
        },
        "per_question": per_question,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the harness CLI parser."""

    parser = argparse.ArgumentParser(
        description="Live retrieval/answer evaluation harness for DocRAG (local, needs Qdrant)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    retrieval = sub.add_parser("retrieval", help="Score retrieval ranking against labels.")
    retrieval.add_argument("dataset", type=Path, help="Path to .json or .jsonl labeled dataset.")
    retrieval.add_argument(
        "--k",
        type=int,
        action="append",
        dest="ks",
        help="Cutoff for hit@k/recall@k (repeatable). Defaults to 1,3,5,8.",
    )
    retrieval.add_argument("--output", type=Path, help="Optional JSON results path.")
    retrieval.add_argument(
        "--baseline", type=Path, help="Baseline results JSON to compare against."
    )
    retrieval.add_argument(
        "--max-regression",
        type=float,
        default=0.05,
        help="Max allowed drop per metric vs baseline before failing (default 0.05).",
    )

    answer = sub.add_parser("answer", help="Generate answers for RAGAS scoring.")
    answer.add_argument("dataset", type=Path, help="Path to .json or .jsonl labeled dataset.")
    answer.add_argument("--output", type=Path, required=True, help="Answers JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns non-zero on data errors or baseline regressions."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        examples = load_retrieval_dataset(args.dataset)
    except HarnessDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    settings = get_settings()

    if args.command == "retrieval":
        ks = tuple(args.ks) if args.ks else DEFAULT_KS
        results = run_retrieval_eval(examples, Collaborators(settings), ks)
        _emit(results, args.output)
        if args.baseline is not None:
            return _check_baseline(results, args.baseline, args.max_regression)
        return 0

    # answer mode
    collaborators = Collaborators(settings)
    rows = [
        answer_for_question(
            example,
            repository=collaborators.repository,
            embedding_provider=collaborators.embedding_provider,
            reranker=collaborators.reranker,
            generator=collaborators.generator,
            settings=settings,
        ).to_dict()
        for example in examples
    ]
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} answer rows to {args.output}")
    return 0


def _emit(results: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(results, indent=2, sort_keys=True)
    if output is not None:
        output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


def _check_baseline(results: dict[str, Any], baseline_path: Path, max_regression: float) -> int:
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not read baseline: {exc}", file=sys.stderr)
        return 2
    baseline_agg = baseline.get("aggregate", {}) if isinstance(baseline, dict) else {}
    regressions: list[str] = []
    for level in ("filename", "chunk"):
        regressions.extend(
            f"{level}: {message}"
            for message in compare_to_baseline(
                results["aggregate"].get(level, {}),
                baseline_agg.get(level, {}),
                max_regression,
            )
        )
    if regressions:
        print("retrieval regressions vs baseline:", file=sys.stderr)
        for message in regressions:
            print(f"  {message}", file=sys.stderr)
        return 1
    print("no regressions vs baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
