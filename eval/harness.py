"""Live evaluation harness: drive the real pipeline over a labeled question set.

Two modes:

* ``retrieval`` — run retrieval exactly as the API does (``RagPipeline.retrieve``: query
  expansion when enabled, hybrid search, rerank, diversity filter) and score the ranked
  filenames / chunk_ids against labeled relevant sets (hit@k, recall@k, MRR). No LLM
  unless query expansion is on. ``--arm`` scores an earlier stage instead (``fused``,
  ``dense``, ``sparse``), so a change can be traced to the stage it helped or hurt.
  Optionally fails against a committed baseline for regression gating.
* ``answer`` — run the API's own answer path (``RagPipeline.answer``: condensing a
  follow-up with its ``history``, retrieval, prompt fitting, generation, the citation
  retry and ``answer_status``) and write ``{question, answer, contexts, reference}`` rows
  directly consumable by ``eval.ragas_runner`` for offline RAGAS scoring. Rows labeled
  with ``answer_contains`` are also scored on whether the answer states the fact — no
  judge LLM needed. Rows are saved as they finish, so ``--resume`` picks a crashed or
  interrupted run up where it stopped.
* ``validate`` — check a dataset's shape only (CI runs this; no Qdrant or models).

``eval/corpus/`` holds the documents the sample dataset is labeled against; ingest it
into a fresh collection before running against ``eval/baselines/``.

Both modes need a running Qdrant with an ingested corpus (and ``answer`` also an LLM).
CI's ``qdrant`` job runs ``retrieval`` on the handbook set against its committed
baseline; ``answer`` stays a local tool. The pipeline-driving functions take their
collaborators as parameters so tests can inject in-memory fakes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from api.documents import CHUNKER_VERSION, SUPPORTED_EXTENSIONS
from api.generation import ChatGenerator, ChatMessage
from api.pipeline import RagPipeline
from api.reranking import Reranker
from api.retrieval import QueryEmbeddingProvider, embed_query, points_to_chunks
from api.settings import AppSettings, get_settings
from eval.answer_eval import AnswerRow as AnswerRow
from eval.answer_eval import _LastTrace as _LastTrace
from eval.answer_eval import _normalized
from eval.answer_eval import _prompt_contexts as _prompt_contexts
from eval.answer_eval import answer_for_question as answer_for_question
from eval.answer_eval import answer_metrics as answer_metrics
from eval.answer_eval import partial_path as partial_path
from eval.answer_eval import run_answer_eval as run_answer_eval
from eval.answer_eval import states_fact as states_fact
from eval.feedback_dataset import (
    FeedbackDataError,
    build_dataset,
    rows_from_export,
    rows_from_store,
    write_jsonl,
)
from eval.metrics import aggregate, baseline_shape_problems, compare_to_baseline, score_ranking

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 8)
# reranked is what the API serves; the others score one earlier stage on its own.
ARMS: tuple[str, ...] = ("reranked", "fused", "dense", "sparse")


class _NoGenerator:
    """Stands in when no LLM is configured: retrieval needs one only for query expansion."""

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        raise RuntimeError("Query expansion needs an LLM; pass a generator.")

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Any:
        raise RuntimeError("Query expansion needs an LLM; pass a generator.")


class HarnessDataError(ValueError):
    """Raised when a harness dataset is malformed."""


@dataclass(frozen=True)
class RetrievalExample:
    """One labeled retrieval-evaluation example."""

    question: str
    reference: str | None
    relevant_filenames: frozenset[str]
    relevant_chunk_ids: frozenset[str]
    # Text the right chunk contains (whitespace/case-insensitive). Resolved to chunk ids
    # against the indexed corpus at run time, so labels survive chunking changes that
    # would change every chunk id.
    relevant_passages: frozenset[str] = frozenset()
    # A question the corpus does not answer: scored on whether retrieval keeps nothing
    # (nothing clears RERANK_MIN_SCORE, so the answer is "not in your documents")
    # rather than on a ranking. Carries no relevance labels.
    expect_no_context: bool = False
    # Earlier turns of a conversation: the question is a follow-up ("and how long does
    # that take?"), rewritten into a standalone query before retrieval as the API does,
    # so the labels measure the rewriting too.
    history: tuple[dict[str, str], ...] = ()
    # Answer mode: the answer must state one of these (case/space-insensitive), e.g.
    # ["sixty euros", "60 euros"]. Scored as ``fact_hit_rate`` without a judge LLM.
    answer_contains: tuple[str, ...] = ()
    # A question about several things (a comparison): the answer must state one of
    # each group, e.g. [["sixty euros", "60 euros"], ["forty-five euros", "45 euros"]].
    answer_contains_all: tuple[tuple[str, ...], ...] = ()

    @property
    def fact_groups(self) -> tuple[tuple[str, ...], ...]:
        """What the answer must state: one of each group (``answer_contains`` is one)."""

        if self.answer_contains_all:
            return self.answer_contains_all
        return (self.answer_contains,) if self.answer_contains else ()

    @property
    def labeled(self) -> bool:
        return bool(
            self.relevant_filenames
            or self.relevant_chunk_ids
            or self.relevant_passages
            or self.expect_no_context
        )


@dataclass(frozen=True)
class RankedResult:
    """The ranked identifiers a single retrieval run returned."""

    filenames: list[str]
    chunk_ids: list[str]
    # Nothing cleared RERANK_MIN_SCORE: the reranked arm returns the closest passages
    # (RERANK_FALLBACK_COUNT), which an answer is told match only weakly.
    below_min_score: bool = False
    # Wall time of the ranking (condense included for a follow-up), and the pipeline's
    # own stage timings when the arm ran it.
    total_ms: float = 0.0
    stages_ms: dict[str, float] = field(default_factory=dict)


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


def _pipeline(
    repository: Any,
    embedding_provider: QueryEmbeddingProvider,
    reranker: Reranker,
    settings: AppSettings,
    generator: ChatGenerator | None,
) -> RagPipeline:
    return RagPipeline(
        repository=repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
        generator=generator or _NoGenerator(),
        settings=settings,
    )


def rank_for_question(
    question: str,
    *,
    repository: Any,
    embedding_provider: QueryEmbeddingProvider,
    reranker: Reranker,
    settings: AppSettings,
    filenames: Sequence[str] | None = None,
    arm: str = "reranked",
    generator: ChatGenerator | None = None,
    history: Sequence[dict[str, str]] = (),
) -> RankedResult:
    """The ranking one question gets from ``arm`` (default: what the API serves).

    With ``history`` the question is a follow-up: rewritten into a standalone query by
    the pipeline's own condense step first (needs ``generator``), and the raw follow-up
    searched next to the rewrite, as the API does."""

    started = time.monotonic()
    raw_question = question
    if history:
        pipeline = _pipeline(repository, embedding_provider, reranker, settings, generator)
        turns: list[ChatMessage] = [
            {"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
            for m in history
        ]
        question = pipeline.retrieval_query(question, turns)

    if arm in ("dense", "sparse"):
        embedding = embed_query(question, embedding_provider)
        chunks = points_to_chunks(repository.search_leg(settings, embedding, arm))
        return RankedResult(
            filenames=[chunk.filename for chunk in chunks],
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            total_ms=(time.monotonic() - started) * 1000,
        )
    if arm not in ("reranked", "fused"):
        raise ValueError(f"Unknown arm '{arm}'; expected one of {', '.join(ARMS)}.")
    phase = _pipeline(repository, embedding_provider, reranker, settings, generator).retrieve(
        question,
        filenames=filenames,
        also_search=raw_question if question != raw_question else None,
    )
    ranked = phase.fused_candidates if arm == "fused" else phase.context_chunks
    return RankedResult(
        filenames=[chunk.filename for chunk in ranked],
        chunk_ids=[chunk.chunk_id for chunk in ranked],
        below_min_score=arm == "reranked" and phase.below_min_score,
        total_ms=(time.monotonic() - started) * 1000,
        stages_ms={
            "embed_ms": phase.embed_ms,
            "search_ms": phase.search_ms,
            "rerank_ms": phase.rerank_ms,
        },
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
    expect_no_context = data.get("expect_no_context", False)
    if not isinstance(expect_no_context, bool):
        raise HarnessDataError(f"Example {index}: 'expect_no_context' must be true or false.")
    example = RetrievalExample(
        question=question.strip(),
        reference=reference.strip() if isinstance(reference, str) and reference.strip() else None,
        relevant_filenames=_string_set(data.get("relevant_filenames"), index, "relevant_filenames"),
        relevant_chunk_ids=_string_set(data.get("relevant_chunk_ids"), index, "relevant_chunk_ids"),
        relevant_passages=_string_set(data.get("relevant_passages"), index, "relevant_passages"),
        expect_no_context=expect_no_context,
        history=_history(data.get("history"), index),
        answer_contains=tuple(
            sorted(_string_set(data.get("answer_contains"), index, "answer_contains"))
        ),
        answer_contains_all=_fact_groups(data.get("answer_contains_all"), index),
    )
    if expect_no_context and example.fact_groups:
        raise HarnessDataError(
            f"Example {index}: an 'expect_no_context' question has no fact to state "
            "('answer_contains')."
        )
    if expect_no_context and (
        example.relevant_filenames or example.relevant_chunk_ids or example.relevant_passages
    ):
        raise HarnessDataError(
            f"Example {index}: an 'expect_no_context' question can't also name relevant "
            "files, chunks or passages."
        )
    return example


def _fact_groups(value: Any, index: int) -> tuple[tuple[str, ...], ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise HarnessDataError(
            f"Example {index}: 'answer_contains_all' must be a non-empty list of lists."
        )
    return tuple(tuple(sorted(_string_set(group, index, "answer_contains_all"))) for group in value)


def _history(value: Any, index: int) -> tuple[dict[str, str], ...]:
    """A follow-up's earlier turns: [{"role": "user"|"assistant", "content": "..."}]."""

    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
        and m["content"].strip()
        for m in value
    ):
        raise HarnessDataError(
            f"Example {index}: 'history' must be a list of {{role: user|assistant, content}}."
        )
    return tuple({"role": m["role"], "content": m["content"]} for m in value)


def _string_set(value: Any, index: int, field: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise HarnessDataError(f"Example {index}: '{field}' must be a list of strings.")
    return frozenset(item.strip() for item in value if item.strip())


def settings_fingerprint(settings: AppSettings, arm: str) -> dict[str, Any]:
    """What a baseline was measured with, so a comparison can say what changed."""

    return {
        "arm": arm,
        "embedding_model_tag": settings.embedding_model_tag,
        "sparse_embedding_model": settings.sparse_embedding_model,
        "reranker_model": settings.reranker_model,
        "chunker_version": CHUNKER_VERSION,
        "rerank_top_k": int(settings.rerank_top_k),
        "query_expansion_enabled": bool(settings.query_expansion_enabled),
        # Chunk ids (and so chunk-level scores) depend on these.
        "chunk_size_tokens": int(settings.chunk_size_tokens),
        "chunk_overlap_tokens": int(settings.chunk_overlap_tokens),
        "min_section_words": int(settings.min_section_words),
        # What ranks the results. int8 vs fp32 reranking reorders borderline questions
        # (the CI comment on the baseline's CPU says as much), and the rest decide which
        # candidates are scored, kept and shown.
        "reranker_onnx_file": settings.reranker_onnx_file,
        "rerank_candidates": int(settings.rerank_candidates),
        "rerank_min_score": float(settings.rerank_min_score),
        "rerank_fallback_count": int(settings.rerank_fallback_count),
        # Trims the kept passages relative to the best one (0 = off).
        "rerank_relative_floor": float(settings.rerank_relative_floor),
        "recency_boost_weight": float(settings.recency_boost_weight),
        "fused_top_n": int(settings.fused_top_n),
        "context_neighbor_radius": int(settings.context_neighbor_radius),
        # With a cap, adjacent passages are no longer dropped as duplicates of a kept one.
        "context_neighbor_max_chars": int(settings.context_neighbor_max_chars),
        "context_diversity_enabled": bool(settings.context_diversity_enabled),
        "context_diversity_max_similarity": float(settings.context_diversity_max_similarity),
        "dense_query_instruction": settings.dense_query_instruction,
        # Which candidates reach the reranker, and how they are fused.
        "rrf_k": int(settings.rrf_k),
        # int8 search re-scored against the originals: close to, not the same as, float32.
        "qdrant_quantization": settings.qdrant_quantization,
        "dense_retrieval_limit": int(settings.dense_retrieval_limit),
        "sparse_retrieval_limit": int(settings.sparse_retrieval_limit),
        "query_expansion_count": int(settings.query_expansion_count),
        # The int8 reranker quantizes activations per batch, so the batch size can move
        # borderline scores.
        "reranker_batch_size": int(settings.reranker_batch_size),
        # The multilingual profile's knobs: BM25's stemmer and the dense passage prefix
        # change every stored vector, and so every score.
        "sparse_language": settings.sparse_language,
        "dense_passage_prefix": settings.dense_passage_prefix,
        # Recorded, not enforced (metrics.baseline_shape_problems): amd64 and arm64 score
        # a borderline question differently, which explains a small drift.
        "platform_machine": platform.machine(),
    }


def run_retrieval_eval(
    examples: Sequence[RetrievalExample],
    collaborators: Collaborators,
    ks: Sequence[int],
    arm: str = "reranked",
) -> dict[str, Any]:
    """Score every example's filename ranking (and chunk ranking when labeled)."""

    per_question: list[dict[str, Any]] = []
    filename_rows: list[dict[str, float]] = []
    chunk_rows: list[dict[str, float]] = []
    no_context_rows: list[dict[str, float]] = []
    latencies: list[float] = []
    # Filename and chunk scores by the format of the files a question is labelled with.
    by_format: dict[str, dict[str, list[dict[str, float]]]] = {}
    passage_ids = resolve_passages(examples, collaborators.repository, collaborators.settings)
    for index, example in enumerate(examples):
        ranked = rank_for_question(
            example.question,
            history=example.history,
            repository=collaborators.repository,
            embedding_provider=collaborators.embedding_provider,
            reranker=collaborators.reranker,
            settings=collaborators.settings,
            arm=arm,
            generator=getattr(collaborators, "generator", None),
        )
        entry: dict[str, Any] = {
            "question": example.question,
            "latency_ms": {"total_ms": round(ranked.total_ms, 1)}
            | {name: round(value, 1) for name, value in ranked.stages_ms.items()},
        }
        latencies.append(ranked.total_ms)
        if example.expect_no_context:
            # Only the reranked arm applies RERANK_MIN_SCORE; the others always rank.
            # Passages below it (the answer fallback) don't count as found.
            kept_nothing = not ranked.chunk_ids or ranked.below_min_score
            entry["no_context"] = kept_nothing
            no_context_rows.append({"no_context_rate": 1.0 if kept_nothing else 0.0})
        formats = sorted(
            {PurePosixPath(name).suffix.lower() or "(none)" for name in example.relevant_filenames}
        )
        if formats:
            entry["formats"] = formats
        if example.relevant_filenames:
            file_scores = score_ranking(ranked.filenames, example.relevant_filenames, ks)
            entry["filename"] = file_scores
            filename_rows.append(file_scores)
            for suffix in formats:
                by_format.setdefault(suffix, {}).setdefault("filename", []).append(file_scores)
        relevant_chunks = example.relevant_chunk_ids | passage_ids.get(index, frozenset())
        if relevant_chunks:
            chunk_scores = score_ranking(ranked.chunk_ids, relevant_chunks, ks)
            entry["chunk"] = chunk_scores
            chunk_rows.append(chunk_scores)
            for suffix in formats:
                by_format.setdefault(suffix, {}).setdefault("chunk", []).append(chunk_scores)
        per_question.append(entry)

    levels = {"filename": aggregate(filename_rows), "chunk": aggregate(chunk_rows)}
    if no_context_rows:
        # Only with such questions, so a baseline from a dataset without them keeps its
        # shape.
        levels["unanswerable"] = aggregate(no_context_rows)
    result: dict[str, Any] = {
        "fingerprint": settings_fingerprint(collaborators.settings, arm),
        "k_values": list(ks),
        "question_count": len(examples),
        "aggregate": levels,
        # Informational, never compared with a baseline: it depends on the machine.
        "latency_ms": latency_summary(latencies),
        "per_question": per_question,
    }
    if len(by_format) > 1:
        # A corpus of several formats (corpus_mixed): how each parser's documents rank.
        # Beside "aggregate", not in it, so baselines keep their shape and aren't judged
        # on it.
        result["by_format"] = {
            suffix: {
                level: aggregate(rows) | {"questions": float(len(rows))}
                for level, rows in sorted(levels_by.items())
            }
            for suffix, levels_by in sorted(by_format.items())
        }
    return result


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    """p50, p95 and the mean of per-question wall times (ms), nearest-rank."""

    if not values:
        return {}
    ordered = sorted(values)

    def percentile(share: float) -> float:
        rank = max(1, math.ceil(share * len(ordered)))
        return round(ordered[rank - 1], 1)

    return {
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "mean": round(sum(ordered) / len(ordered), 1),
    }


def resolve_passages(
    examples: Sequence[RetrievalExample], repository: Any, settings: AppSettings
) -> dict[int, frozenset[str]]:
    """Each example's ``relevant_passages``, as the ids of the chunks containing them.

    Keyed by the example's position (two rows may share a question). Looks only in the
    example's relevant files when it names any, and in every indexed file when it
    doesn't. A passage found in no chunk is a broken label (the text changed, or a typo)
    and fails the run rather than silently scoring that question as unlabeled.
    """

    wanted = [
        (index, example) for index, example in enumerate(examples) if example.relevant_passages
    ]
    if not wanted:
        return {}
    every_file = (
        frozenset(repository.filename_metadata(settings))
        if any(not example.relevant_filenames for _, example in wanted)
        else frozenset()
    )
    filenames = sorted(
        every_file | {name for _, example in wanted for name in example.relevant_filenames}
    )
    chunks: dict[str, list[tuple[str, str]]] = {
        name: [
            (str(payload.get("chunk_id", "")), _normalized(str(payload.get("text", ""))))
            for payload in repository.chunks_for_filename(settings, name)
        ]
        for name in filenames
    }
    resolved: dict[int, frozenset[str]] = {}
    for index, example in wanted:
        search = example.relevant_filenames or every_file
        ids: set[str] = set()
        for passage in example.relevant_passages:
            needle = _normalized(passage)
            found = {
                chunk_id
                for name in search
                for chunk_id, text in chunks.get(name, [])
                if needle in text
            }
            if not found:
                raise HarnessDataError(
                    f"Passage {passage!r} (question {example.question!r}) is in no indexed "
                    "chunk; ingest the corpus first, or fix the label."
                )
            ids |= found
        resolved[index] = frozenset(ids)
    return resolved


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
        "--arm",
        choices=ARMS,
        default="reranked",
        help="Stage to score: reranked (what the API serves, default), fused, dense, sparse.",
    )
    retrieval.add_argument(
        "--baseline", type=Path, help="Baseline results JSON to compare against."
    )
    retrieval.add_argument(
        "--max-regression",
        type=float,
        default=0.05,
        help="Max allowed drop per metric vs baseline before failing (default 0.05).",
    )

    ingest = sub.add_parser("ingest", help="Index every document in a directory.")
    ingest.add_argument("directory", type=Path, help="e.g. eval/corpus")

    validate = sub.add_parser("validate", help="Check a dataset's shape only (no Qdrant).")
    validate.add_argument("dataset", type=Path, help="Path to .json or .jsonl labeled dataset.")

    answer = sub.add_parser("answer", help="Generate answers for RAGAS scoring.")
    answer.add_argument("dataset", type=Path, help="Path to .json or .jsonl labeled dataset.")
    answer.add_argument("--output", type=Path, required=True, help="Answers JSON output path.")
    answer.add_argument(
        "--limit", type=int, default=0, help="Answer only the first N questions (0 = all)."
    )
    answer.add_argument(
        "--resume",
        action="store_true",
        help="Keep the rows a previous run of this dataset and output already saved "
        "(OUTPUT.partial.jsonl) and answer only the rest.",
    )

    diff = sub.add_parser(
        "diff", help="List the questions that got better or worse between two runs (no Qdrant)."
    )
    diff.add_argument("before", type=Path, help="A retrieval result or answer run (JSON).")
    diff.add_argument("after", type=Path, help="The same command's output to compare.")
    diff.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 when any question got worse.",
    )

    report = sub.add_parser("report", help="Write one run as a self-contained HTML page.")
    report.add_argument("run", type=Path, help="A retrieval result or answer run (JSON).")
    report.add_argument("--output", type=Path, required=True, help="The HTML file to write.")
    report.add_argument("--ragas", type=Path, help="ragas_runner output for the same answers.")
    report.add_argument("--diff", type=Path, dest="base", help="A base run to compare with.")
    report.add_argument("--title", help="Page title (default: the run's file name).")

    feedback = sub.add_parser(
        "from-feedback",
        help="Draft a labeled dataset from answer ratings (no Qdrant).",
    )
    source = feedback.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--store", type=Path, help="The feedback sqlite file (FEEDBACK_STORE_PATH)."
    )
    source.add_argument("--export", type=Path, help="A GET /export zip (its feedback.jsonl).")
    feedback.add_argument(
        "--output", type=Path, required=True, help="Labeled .jsonl from thumbs-up ratings."
    )
    feedback.add_argument(
        "--review", type=Path, help="Optional .jsonl of thumbs-down questions to label by hand."
    )
    scope = feedback.add_mutually_exclusive_group()
    scope.add_argument(
        "--workspace",
        default=None,
        help="With --store: only this workspace's ratings (default: the default workspace, "
        "''). An export holds one workspace already.",
    )
    scope.add_argument(
        "--all-workspaces",
        action="store_true",
        help="With --store: every workspace's ratings in one dataset.",
    )
    return parser


def dataset_from_feedback(args: argparse.Namespace) -> int:
    """``from-feedback``: write the draft dataset, then validate it like ``validate``."""

    if args.export and (args.workspace is not None or args.all_workspaces):
        # Silently ignored, it would read as "this workspace's ratings" when the
        # dataset holds whichever workspace the backup was taken from.
        print(
            "error: --workspace/--all-workspaces apply to --store only; an export holds the "
            "one workspace it was taken from",
            file=sys.stderr,
        )
        return 2
    # One corpus per dataset: the default workspace unless another (or all) is asked for.
    # Reading every workspace by default mixed labels from corpora the dataset is never
    # run against together.
    workspace = None if args.all_workspaces else (args.workspace or "")
    try:
        rows = (
            rows_from_store(args.store, workspace) if args.store else rows_from_export(args.export)
        )
        dataset = build_dataset(rows)
    except FeedbackDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"read {dataset.ratings_read} ratings: {len(dataset.examples)} labeled examples, "
        f"{len(dataset.review) - dataset.not_questions} rated-down questions to review "
        f"(skipped {dataset.citation_ratings} citation ratings, "
        f"{dataset.uncited} thumbs-up answers that cited nothing; "
        f"{dataset.not_questions} summary or follow-up ratings went to the review only)"
    )
    if args.review is not None:
        write_jsonl(args.review, dataset.review)
        print(f"wrote {len(dataset.review)} questions to review to {args.review}")
    if not dataset.examples:
        print("error: no thumbs-up answer with citations to label from", file=sys.stderr)
        return 2
    write_jsonl(args.output, dataset.examples)
    # The same check `validate` runs, so the file is known to load in `retrieval`.
    try:
        examples = load_retrieval_dataset(args.output)
    except HarnessDataError as exc:  # pragma: no cover - build_dataset writes valid rows
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(examples)} examples to {args.output}: OK")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns non-zero on data errors or baseline regressions."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return ingest_directory(args.directory, get_settings())
    if args.command == "from-feedback":
        return dataset_from_feedback(args)
    if args.command in ("diff", "report"):
        return compare_runs(args)

    try:
        examples = load_retrieval_dataset(args.dataset)
    except HarnessDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "validate":
        labeled = sum(1 for e in examples if e.labeled)
        print(f"{len(examples)} examples, {labeled} labeled: OK")
        return 0 if labeled else 2

    settings = get_settings()

    if args.command == "retrieval":
        ks = tuple(args.ks) if args.ks else DEFAULT_KS
        try:
            results = run_retrieval_eval(examples, Collaborators(settings), ks, args.arm)
        except HarnessDataError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _emit(results, args.output)
        if args.baseline is not None:
            return _check_baseline(results, args.baseline, args.max_regression)
        return 0

    # answer mode
    if args.limit > 0:
        examples = examples[: args.limit]
    return run_answer_eval(examples, Collaborators(settings), args.output, resume=args.resume)


def compare_runs(args: argparse.Namespace) -> int:
    """``diff`` and ``report``: read run files only (no Qdrant, no models)."""

    from eval.compare import RunFileError, diff_runs, format_diff, load_run, render_report

    try:
        if args.command == "diff":
            result = diff_runs(load_run(args.before), load_run(args.after))
            print(format_diff(result))
            return 1 if args.fail_on_regression and result.regressed else 0
        run = load_run(args.run)
        base = load_run(args.base) if args.base is not None else None
        page = render_report(run, title=args.title or args.run.name, ragas=args.ragas, base=base)
        args.output.write_text(page, encoding="utf-8")
    except (RunFileError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output}")
    return 0


def ingestible_files(directory: Path) -> tuple[list[Path], list[Path]]:
    """The files in ``directory`` the API could index (sorted), and the ones it skips.

    Without the filter a stray ``.DS_Store`` or README raised UnsupportedDocumentError
    and stopped the run halfway through the corpus.
    """

    files: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        ingestible = not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_EXTENSIONS
        (files if ingestible else skipped).append(path)
    return files, skipped


def ingest_directory(directory: Path, settings: AppSettings) -> int:
    """Index each file in ``directory`` (sorted) into the configured collection."""

    from api.chunking import make_token_counter
    from api.embeddings import LocalEmbeddingProvider
    from api.pipeline import IngestService
    from api.qdrant_schema import make_qdrant_client
    from api.repository import VectorRepository

    files, skipped = ingestible_files(directory)
    for path in skipped:
        print(f"skipped {path.name}: not a supported document type")
    if not files:
        print(f"error: no supported documents in {directory}", file=sys.stderr)
        return 2
    service = IngestService(
        VectorRepository(make_qdrant_client(settings)),
        LocalEmbeddingProvider(settings),
        settings,
        make_token_counter(settings.dense_embedding_model),
    )
    stamps = upload_times(directory)
    for path in files:
        outcome = service.ingest(path.name, path.read_bytes(), uploaded_at=stamps.get(path.name))
        print(f"{outcome.filename}: {outcome.chunks_ingested} chunks")
    return 0


# A corpus may say when each of its documents was uploaded (a dotfile, so it is never
# ingested itself): {"policy.md": "2021-03-01", ...}. Documents it doesn't name are
# stamped with the time they are ingested, as an upload is.
UPLOAD_TIMES_FILE = ".uploaded_at.json"


def upload_times(directory: Path) -> dict[str, float]:
    """``directory``'s ``UPLOAD_TIMES_FILE`` as filename -> Unix seconds (UTC dates)."""

    path = directory / UPLOAD_TIMES_FILE
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object of filename -> ISO date")
    stamps: dict[str, float] = {}
    for name, value in raw.items():
        when = datetime.datetime.fromisoformat(str(value))
        if when.tzinfo is None:
            when = when.replace(tzinfo=datetime.UTC)
        stamps[str(name)] = when.timestamp()
    return stamps


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
    if not isinstance(baseline, dict):
        print("error: baseline is not a results object", file=sys.stderr)
        return 2
    # A baseline from a different dataset, cutoffs or stage compares nothing: every
    # mismatched metric used to be skipped and the gate passed on no evidence at all.
    problems = baseline_shape_problems(results, baseline)
    if problems:
        print("baseline is not comparable to this run:", file=sys.stderr)
        for message in problems:
            print(f"  {message}", file=sys.stderr)
        return 2
    changed = {
        key: (baseline.get("fingerprint", {}).get(key), value)
        for key, value in results.get("fingerprint", {}).items()
        if baseline.get("fingerprint", {}).get(key) != value
    }
    for key, (before, after) in changed.items():
        print(f"note: {key} changed since the baseline: {before!r} -> {after!r}")
    baseline_agg = baseline.get("aggregate", {})
    regressions: list[str] = []
    # Every level the baseline measured: "unanswerable" (no_context_rate) too, which a
    # fixed filename/chunk list let regress unnoticed.
    for level, metrics in baseline_agg.items():
        if not isinstance(metrics, dict):
            continue
        regressions.extend(
            f"{level}: {message}"
            for message in compare_to_baseline(
                results["aggregate"].get(level, {}),
                metrics,
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
