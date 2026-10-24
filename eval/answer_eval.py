"""Answer mode of the eval harness (``python -m eval.harness answer``).

Runs the API's own answer path (``RagPipeline.answer``) over a labeled question set and
writes ``{question, answer, contexts, reference}`` rows ``eval.ragas_runner`` scores,
saving each row as it finishes (``--resume``). ``eval.harness`` re-exports everything
here; split out so the harness module holds retrieval scoring and the CLI.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from api.citations import strip_citation_markers
from api.generation import ChatGenerator, ChatMessage
from api.pipeline import RagPipeline
from api.reranking import Reranker
from api.retrieval import QueryEmbeddingProvider
from api.settings import AppSettings
from api.tracing import QueryTrace

if TYPE_CHECKING:
    from eval.harness import Collaborators, RetrievalExample


@dataclass(frozen=True)
class AnswerRow:
    """A generated answer row, in ``eval.ragas_runner`` input shape, plus what the
    answer cited and how it was grounded (for ``answer_metrics``)."""

    question: str
    answer: str
    contexts: list[str]
    reference: str | None
    cited_filenames: list[str] = field(default_factory=list)
    answer_status: str = "answered"
    # Whether the answer states one of the example's ``answer_contains`` facts; None
    # when the example names none.
    fact_hit: bool | None = None
    seconds: float = 0.0
    # Answered from below-the-bar passages; citations added by matching, not the model.
    weak_matches: bool = False
    citation_backfilled: bool = False
    # The pipeline failed (a provider timeout, say): the row is kept, so one bad
    # question doesn't end a long run, and counts as not answered.
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "cited_filenames": self.cited_filenames,
            "answer_status": self.answer_status,
            "seconds": round(self.seconds, 1),
            "weak_matches": self.weak_matches,
            "citation_backfilled": self.citation_backfilled,
        }
        if self.reference is not None:
            row["reference"] = self.reference
        if self.fact_hit is not None:
            row["fact_hit"] = self.fact_hit
        if self.error is not None:
            row["error"] = self.error
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> AnswerRow:
        """The row ``to_dict`` wrote (``--resume`` reads saved rows back)."""

        return cls(
            question=str(row["question"]),
            answer=str(row.get("answer", "")),
            contexts=[str(c) for c in row.get("contexts", [""])],
            reference=row.get("reference"),
            cited_filenames=[str(f) for f in row.get("cited_filenames", [])],
            answer_status=str(row.get("answer_status", "answered")),
            fact_hit=row.get("fact_hit"),
            seconds=float(row.get("seconds", 0.0)),
            weak_matches=bool(row.get("weak_matches", False)),
            citation_backfilled=bool(row.get("citation_backfilled", False)),
            error=row.get("error"),
        )


class _LastTrace:
    """Trace sink keeping only the latest trace: what the answer was given to read."""

    def __init__(self) -> None:
        self.trace: QueryTrace | None = None

    def add(self, trace: QueryTrace) -> None:
        self.trace = trace


def states_fact(answer: str, facts: Sequence[str]) -> bool | None:
    """Whether ``answer`` states one of ``facts`` (case/space-insensitive, citation
    markers ignored); None when there are no facts to check."""

    if not facts:
        return None
    text = _normalized(strip_citation_markers(answer))
    return any(_normalized(fact) in text for fact in facts)


def states_facts(answer: str, groups: Sequence[Sequence[str]]) -> bool | None:
    """Whether ``answer`` states one fact of every group (a comparison names both
    sides); None when there are none to check."""

    if not groups:
        return None
    return all(states_fact(answer, group) for group in groups)


def answer_for_question(
    example: RetrievalExample,
    *,
    repository: Any,
    embedding_provider: QueryEmbeddingProvider,
    reranker: Reranker,
    generator: ChatGenerator,
    settings: AppSettings,
    texts: dict[str, dict[str, str]] | None = None,
) -> AnswerRow:
    """Answer one question the way the API does and return a RAGAS-shaped row.

    Goes through ``RagPipeline.answer`` (condense with the example's ``history``,
    retrieval, fitting, generation, citation retry, status), so the eval can't drift
    from what users get. The contexts are the passages that went into the prompt, read
    back from the trace.
    """

    sink = _LastTrace()
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
        generator=generator,
        settings=settings,
        trace_store=sink,
    )
    history: list[ChatMessage] = [
        {"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
        for m in example.history
    ]
    started = time.monotonic()
    grounded = pipeline.answer(example.question, history=history or None, use_cache=False)
    contexts = _prompt_contexts(sink.trace, repository, settings, texts)
    return AnswerRow(
        question=example.question,
        answer=grounded.answer,
        contexts=contexts or [""],  # RAGAS requires a non-empty contexts list
        reference=example.reference,
        cited_filenames=sorted({source.filename for source in grounded.sources}),
        answer_status=grounded.answer_status,
        fact_hit=states_facts(grounded.answer, example.fact_groups),
        seconds=time.monotonic() - started,
        weak_matches=grounded.weak_matches,
        citation_backfilled=grounded.citation_backfilled,
    )


def _prompt_contexts(
    trace: QueryTrace | None,
    repository: Any,
    settings: AppSettings,
    texts: dict[str, dict[str, str]] | None = None,
) -> list[str]:
    """The text of each passage the answer's prompt held, in rank order.

    ``texts`` caches each file's passages by chunk id across a run: every question
    re-read whole documents otherwise."""

    if trace is None:
        return []
    # Candidates are listed in fused order; the prompt holds them by rerank score.
    selected = sorted(
        (c for c in trace.candidates if c.selected_for_context),
        key=lambda c: -(c.rerank_score if c.rerank_score is not None else float("-inf")),
    )
    cache = texts if texts is not None else {}
    by_id: dict[str, str] = {}
    for filename in sorted({c.filename for c in selected}):
        if filename not in cache:
            cache[filename] = {
                str(payload.get("chunk_id", "")): str(payload.get("text", ""))
                for payload in repository.chunks_for_filename(settings, filename)
            }
        by_id.update(cache[filename])
    return [by_id[c.chunk_id] for c in selected if c.chunk_id in by_id]


def answer_metrics(
    examples: Sequence[RetrievalExample], rows: Sequence[AnswerRow]
) -> dict[str, float]:
    """How well the answers were grounded, over the examples that can say:

    - ``cited_relevant_rate``: of the questions labeled with relevant files, the share
      whose answer cites at least one of them;
    - ``answered_rate``: of those, the share answered (not "not in your documents");
    - ``refused_unanswerable_rate``: of the ``expect_no_context`` questions, the share
      answered with "the documents don't cover this" rather than an answer;
    - ``uncited_rate``: the share of all answers that cite nothing while answering;
    - ``fact_hit_rate``: of the questions labeled with ``answer_contains``, the share
      whose answer states one of the facts;
    - ``error_rate``: the share of questions the pipeline failed on (a provider
      timeout, say).

    A metric with no examples to measure it is left out.
    """

    labeled = [(e, r) for e, r in zip(examples, rows, strict=True) if e.relevant_filenames]
    unanswerable = [(e, r) for e, r in zip(examples, rows, strict=True) if e.expect_no_context]
    metrics: dict[str, float] = {}
    if labeled:
        metrics["cited_relevant_rate"] = sum(
            bool(set(r.cited_filenames) & e.relevant_filenames) for e, r in labeled
        ) / len(labeled)
        metrics["answered_rate"] = sum(r.answer_status == "answered" for _, r in labeled) / len(
            labeled
        )
    if unanswerable:
        metrics["refused_unanswerable_rate"] = sum(
            r.answer_status in ("no_context", "insufficient_context") for _, r in unanswerable
        ) / len(unanswerable)
    facts = [r for e, r in zip(examples, rows, strict=True) if e.fact_groups]
    if facts:
        metrics["fact_hit_rate"] = sum(bool(r.fact_hit) for r in facts) / len(facts)
    if rows:
        metrics["uncited_rate"] = sum(r.answer_status == "uncited" for r in rows) / len(rows)
        metrics["error_rate"] = sum(r.error is not None for r in rows) / len(rows)
    return metrics


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def partial_path(output: Path) -> Path:
    """Where answer mode saves each row as it finishes."""

    return output.with_name(output.name + ".partial.jsonl")


def run_answer_eval(
    examples: Sequence[RetrievalExample],
    collaborators: Collaborators,
    output: Path,
    *,
    resume: bool = False,
) -> int:
    """Answer every example, saving each row as it finishes, then write ``output``.

    At ~90 s per answer on a local model a run takes half an hour; saving as it goes
    means a crash or Ctrl-C loses one row, not the run (``--resume``).
    """

    partial = partial_path(output)
    saved: dict[int, AnswerRow] = {}
    if resume and partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                index = int(record["index"])
                row = AnswerRow.from_dict(record["row"])
            except (ValueError, KeyError, TypeError):
                # The last line of a run killed mid-write: that question is asked again.
                continue
            if 0 <= index < len(examples) and examples[index].question == row.question:
                # Scored against the dataset as it is now (its facts may have changed).
                saved[index] = replace(
                    row, fact_hit=states_facts(row.answer, examples[index].fact_groups)
                )
        print(f"resuming: {len(saved)} of {len(examples)} rows already answered")
    else:
        partial.unlink(missing_ok=True)
    rows: list[AnswerRow] = []
    texts: dict[str, dict[str, str]] = {}  # passage text by file, read once per run
    with partial.open("a", encoding="utf-8") as sink:
        for index, example in enumerate(examples):
            if index in saved:
                rows.append(saved[index])
                continue
            started = time.monotonic()
            try:
                row = answer_for_question(
                    example,
                    repository=collaborators.repository,
                    embedding_provider=collaborators.embedding_provider,
                    reranker=collaborators.reranker,
                    generator=collaborators.generator,
                    settings=collaborators.settings,
                    texts=texts,
                )
            except Exception as exc:  # noqa: BLE001 - one failed question must not end the run
                row = AnswerRow(
                    question=example.question,
                    answer="",
                    contexts=[""],
                    reference=example.reference,
                    answer_status="error",
                    fact_hit=False if example.fact_groups else None,
                    seconds=time.monotonic() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            sink.write(json.dumps({"index": index, "row": row.to_dict()}) + "\n")
            sink.flush()
            rows.append(row)
            print(
                f"[{index + 1}/{len(examples)}] {row.answer_status} {row.seconds:.1f}s",
                flush=True,
            )
    output.write_text(json.dumps([r.to_dict() for r in rows], indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"wrote {len(rows)} answer rows to {output}")
    for name, value in answer_metrics(examples, rows).items():
        print(f"{name}: {value:.3f}")
    if rows and all(row.error for row in rows):
        print("error: every question failed (see the error field of each row)", file=sys.stderr)
        return 1
    return 0
