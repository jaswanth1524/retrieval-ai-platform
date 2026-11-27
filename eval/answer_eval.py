"""Answer mode of the eval harness (``python -m eval.harness answer``).

Runs the API's own answer path (``RagPipeline.answer``) over a labeled question set and
writes ``{question, answer, contexts, reference}`` rows ``eval.ragas_runner`` scores,
saving each row as it finishes (``--resume``). ``eval.harness`` re-exports everything
here; split out so the harness module holds retrieval scoring and the CLI.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
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
    # The passages (chunk ids) the answer cited, for chunk-level citation precision.
    cited_chunk_ids: list[str] = field(default_factory=list)
    answer_status: str = "answered"
    # Whether the answer states one of the example's ``answer_contains`` facts; None
    # when the example names none.
    fact_hit: bool | None = None
    seconds: float = 0.0
    # Answered from below-the-bar passages; citations added by matching, not the model.
    weak_matches: bool = False
    citation_backfilled: bool = False
    # Citing sentences no passage of their cited sources supports (no span matched).
    unsupported_claims: int = 0
    # Citing sentences stating a number none of their cited passages contains.
    unsupported_numbers: int = 0
    # Whether the answer states one of the example's ``answer_excludes`` (a stale or
    # wrong value); None when the example names none.
    excluded_hit: bool | None = None
    # The answer stopped at the output limit; the model server read only part of the
    # prompt. Either explains a missed fact better than "retrieval failed".
    truncated: bool = False
    prompt_cut: bool = False
    top_rerank_score: float | None = None
    # (prompt, completion) tokens and their price, when the provider reported them.
    usage: tuple[int, int] | None = None
    cost_usd: float | None = None
    # The pipeline failed (a provider timeout, say): the row is kept, so one bad
    # question doesn't end a long run, and counts as not answered.
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "cited_filenames": self.cited_filenames,
            "cited_chunk_ids": self.cited_chunk_ids,
            "answer_status": self.answer_status,
            "seconds": round(self.seconds, 1),
            "weak_matches": self.weak_matches,
            "citation_backfilled": self.citation_backfilled,
            "unsupported_claims": self.unsupported_claims,
            "unsupported_numbers": self.unsupported_numbers,
            "truncated": self.truncated,
            "prompt_cut": self.prompt_cut,
        }
        if self.top_rerank_score is not None:
            row["top_rerank_score"] = round(self.top_rerank_score, 4)
        if self.usage is not None:
            row["usage"] = list(self.usage)
        if self.cost_usd is not None:
            row["cost_usd"] = self.cost_usd
        if self.reference is not None:
            row["reference"] = self.reference
        if self.fact_hit is not None:
            row["fact_hit"] = self.fact_hit
        if self.excluded_hit is not None:
            row["excluded_hit"] = self.excluded_hit
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
            cited_chunk_ids=[str(c) for c in row.get("cited_chunk_ids", [])],
            answer_status=str(row.get("answer_status", "answered")),
            fact_hit=row.get("fact_hit"),
            seconds=float(row.get("seconds", 0.0)),
            weak_matches=bool(row.get("weak_matches", False)),
            citation_backfilled=bool(row.get("citation_backfilled", False)),
            unsupported_claims=_count(row.get("unsupported_claims")),
            unsupported_numbers=_count(row.get("unsupported_numbers")),
            excluded_hit=row.get("excluded_hit"),
            truncated=bool(row.get("truncated", False)),
            prompt_cut=bool(row.get("prompt_cut", False)),
            top_rerank_score=_number(row.get("top_rerank_score")),
            usage=_usage(row.get("usage")),
            cost_usd=_number(row.get("cost_usd")),
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


def states_excluded(answer: str, excludes: Sequence[str]) -> bool | None:
    """Whether ``answer`` states one of ``excludes``: a phrase as a case/space-
    insensitive substring, a number (or a phrase starting with one) only as whole words
    — "30 days" must not match "130 days". None when there are none."""

    if not excludes:
        return None
    text = _normalized(strip_citation_markers(answer))
    for value in excludes:
        wanted = _normalized(value)
        if wanted[:1].isdigit() or wanted[-1:].isdigit():
            if re.search(rf"(?<![\w.,]){re.escape(wanted)}(?![\w]|[.,]\d)", text):
                return True
        elif wanted in text:
            return True
    return False


def scored_facts(answer: str, example: RetrievalExample) -> tuple[bool | None, bool | None]:
    """(fact_hit, excluded_hit): stating an excluded value fails the facts too."""

    excluded = states_excluded(answer, example.answer_excludes)
    facts = states_facts(answer, example.fact_groups)
    if facts and excluded:
        facts = False
    return facts, excluded


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
    if example.task:
        return _summary_row(example, repository=repository, generator=generator, settings=settings)
    history: list[ChatMessage] = [
        {"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
        for m in example.history
    ]
    started = time.monotonic()
    grounded = pipeline.answer(
        example.question,
        filenames=sorted(example.filenames) or None,
        history=history or None,
        tags=sorted(example.tags) or None,
        use_cache=False,
    )
    contexts = _prompt_contexts(sink.trace, repository, settings, texts)
    fact_hit, excluded_hit = scored_facts(grounded.answer, example)
    return AnswerRow(
        question=example.question,
        answer=grounded.answer,
        contexts=contexts or [""],  # RAGAS requires a non-empty contexts list
        reference=example.reference,
        cited_filenames=sorted({source.filename for source in grounded.sources}),
        cited_chunk_ids=sorted({source.chunk_id for source in grounded.sources}),
        answer_status=grounded.answer_status,
        fact_hit=fact_hit,
        excluded_hit=excluded_hit,
        seconds=time.monotonic() - started,
        weak_matches=grounded.weak_matches,
        citation_backfilled=grounded.citation_backfilled,
        unsupported_claims=grounded.unsupported_claims,
        unsupported_numbers=grounded.unsupported_numbers,
        truncated=grounded.truncated,
        prompt_cut=grounded.prompt_cut,
        top_rerank_score=grounded.top_rerank_score,
        usage=grounded.usage,
        cost_usd=grounded.cost_usd,
    )


def _summary_row(
    example: RetrievalExample, *, repository: Any, generator: ChatGenerator, settings: AppSettings
) -> AnswerRow:
    """A summary task, through the API's own summarizer (``POST /questions/summary``)."""

    from api.summarize import DocumentSummarizer, SummaryRequest

    instruction = example.question
    request = SummaryRequest(
        mode=example.task,  # type: ignore[arg-type]
        filenames=tuple(sorted(example.filenames)),
        instruction=instruction if instruction.casefold() not in ("summary", "summarize") else None,
    )
    started = time.monotonic()
    result = DocumentSummarizer(repository, generator, settings).summarize(request, use_cache=False)
    grounded = result.grounded
    fact_hit, excluded_hit = scored_facts(grounded.answer, example)
    return AnswerRow(
        question=example.question,
        answer=grounded.answer,
        contexts=[""],
        reference=example.reference,
        cited_filenames=sorted({source.filename for source in grounded.sources}),
        cited_chunk_ids=sorted({source.chunk_id for source in grounded.sources}),
        answer_status=grounded.answer_status,
        fact_hit=fact_hit,
        excluded_hit=excluded_hit,
        seconds=time.monotonic() - started,
        citation_backfilled=grounded.citation_backfilled,
        unsupported_claims=grounded.unsupported_claims,
        truncated=grounded.truncated,
        prompt_cut=grounded.prompt_cut,
        usage=grounded.usage,
        cost_usd=grounded.cost_usd,
    )


_SOURCE_HEADER = re.compile(r"^\[(\d+)\] filename=[^\n]*\n", re.MULTILINE)


def _contexts_from_prompt(messages: Sequence[ChatMessage] | None) -> list[str]:
    """Each numbered source's text as the answer prompt gave it, in source order; [] when
    the prompt has no ``Context:`` block to read (an older trace, a summary)."""

    if not messages:
        return []
    content = str(messages[-1].get("content", ""))
    start = content.find("Context:\n")
    end = content.rfind("\n\nInstructions:\n")
    if start < 0 or end <= start:
        return []
    block = content[start + len("Context:\n") : end]
    headers = list(_SOURCE_HEADER.finditer(block))
    texts: list[str] = []
    for index, header in enumerate(headers):
        stop = headers[index + 1].start() if index + 1 < len(headers) else len(block)
        texts.append(block[header.end() : stop].strip())
    return texts


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
    from_prompt = _contexts_from_prompt(trace.prompt_messages)
    if from_prompt:
        # What the model read, neighbour text included (``neighbor_expanded``): the
        # passage alone under-credited answers that used it.
        return from_prompt
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


# Answers that say the documents don't cover the question: a scope naming no document
# (``empty_scope``) refuses as plainly as one whose documents lack it.
REFUSED_STATUSES = ("no_context", "insufficient_context", "empty_scope")


def question_category(example: RetrievalExample) -> str:
    """The kind of question an example is, for the per-category metrics."""

    if example.expect_no_context:
        return "unanswerable"
    if example.history or example.chain:
        return "follow-up"
    if example.answer_contains_all:
        return "compound"
    if example.filenames or example.tags:
        return "scoped"
    return "single"


def metrics_by_category(
    examples: Sequence[RetrievalExample],
    rows: Sequence[AnswerRow],
    passage_ids: Mapping[int, frozenset[str]] | None = None,
) -> dict[str, dict[str, float]]:
    """``answer_metrics`` per ``question_category`` (only when there are two or more)."""

    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(question_category(example), []).append(index)
    if len(groups) < 2:
        return {}
    return {
        category: answer_metrics(
            [examples[i] for i in indexes],
            [rows[i] for i in indexes],
            # Keyed by position, so by the position within the category.
            {n: passage_ids[i] for n, i in enumerate(indexes) if i in (passage_ids or {})}
            if passage_ids
            else None,
        )
        for category, indexes in sorted(groups.items())
    }


def answer_metrics(
    examples: Sequence[RetrievalExample],
    rows: Sequence[AnswerRow],
    passage_ids: Mapping[int, frozenset[str]] | None = None,
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
      timeout, say);
    - ``unsupported_claims_mean``: citing sentences per answer that no passage of
      their cited sources supports; ``unsupported_numbers_mean``: citing sentences per
      answer stating a number none of their cited passages contains;
    - ``excluded_rate``: of the questions labeled with ``answer_excludes``, the share
      whose answer states one (a stale or wrong value);
    - ``truncated_rate`` / ``prompt_cut_rate``: the share of answers that stopped at the
      output limit, or whose prompt the model server cut (a missed fact in such a row
      is the budget's doing, not retrieval's);
    - ``cited_chunk_precision``: of the answers to questions labelled with relevant
      passages that cite any, the mean share of cited passages that are relevant ones
      (filename-level ``cited_relevant_rate`` saturates: right file, wrong passage).
      ``passage_ids`` are the examples' ``relevant_passages`` resolved to chunk ids.

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
            r.answer_status in REFUSED_STATUSES for _, r in unanswerable
        ) / len(unanswerable)
    facts = [r for e, r in zip(examples, rows, strict=True) if e.fact_groups]
    if facts:
        metrics["fact_hit_rate"] = sum(bool(r.fact_hit) for r in facts) / len(facts)
    excluded = [r for e, r in zip(examples, rows, strict=True) if e.answer_excludes]
    if excluded:
        metrics["excluded_rate"] = sum(bool(r.excluded_hit) for r in excluded) / len(excluded)
    if rows:
        metrics["uncited_rate"] = sum(r.answer_status == "uncited" for r in rows) / len(rows)
        metrics["error_rate"] = sum(r.error is not None for r in rows) / len(rows)
    # Of the answers given: a question that errored has no answer to measure, and
    # counted as a clean 0 it made a provider outage look like fewer unsupported claims.
    given = [r for r in rows if r.error is None]
    if given:
        metrics["unsupported_claims_mean"] = sum(r.unsupported_claims for r in given) / len(given)
        metrics["unsupported_numbers_mean"] = sum(r.unsupported_numbers for r in given) / len(given)
        metrics["truncated_rate"] = sum(r.truncated for r in given) / len(given)
        metrics["prompt_cut_rate"] = sum(r.prompt_cut for r in given) / len(given)
    precisions = []
    for index, (example, row) in enumerate(zip(examples, rows, strict=True)):
        relevant = example.relevant_chunk_ids | (passage_ids or {}).get(index, frozenset())
        if relevant and row.cited_chunk_ids:
            precisions.append(
                len(set(row.cited_chunk_ids) & relevant) / len(set(row.cited_chunk_ids))
            )
    if precisions:
        metrics["cited_chunk_precision"] = sum(precisions) / len(precisions)
    return metrics


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _usage(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return _count(value[0]), _count(value[1])
    return None


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


# Hyphen, non-breaking hyphen, figure dash: read as "-" (gpt-5-mini writes "forty‑five"
# with U+2011, which missed "twenty-eight" facts and let excluded values through). En and
# em dashes stay: between words they are punctuation, not a hyphen.
_HYPHENS = str.maketrans({"‐": "-", "‑": "-", "‒": "-"})


def _normalized(text: str) -> str:
    return " ".join(text.translate(_HYPHENS).split()).casefold()


def partial_path(output: Path) -> Path:
    """Where answer mode saves each row as it finishes."""

    return output.with_name(output.name + ".partial.jsonl")


def run_answer_eval(
    examples: Sequence[RetrievalExample],
    collaborators: Collaborators,
    output: Path,
    *,
    resume: bool = False,
    replay: Any = None,
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
            # A row that failed (a provider timeout) is asked again, not kept: the
            # re-run is how a transient failure gets fixed.
            if (
                0 <= index < len(examples)
                and examples[index].question == row.question
                and not row.error
            ):
                # Scored against the dataset as it is now (its facts may have changed).
                fact_hit, excluded_hit = scored_facts(row.answer, examples[index])
                saved[index] = replace(row, fact_hit=fact_hit, excluded_hit=excluded_hit)
        print(f"resuming: {len(saved)} of {len(examples)} rows already answered")
    else:
        partial.unlink(missing_ok=True)
    rows: list[AnswerRow] = []
    texts: dict[str, dict[str, str]] = {}  # passage text by file, read once per run
    # Each chain's turns so far: the questions and the answers this run gave them.
    chains: dict[str, list[dict[str, str]]] = {}

    def remember_turn(example: RetrievalExample, row: AnswerRow) -> None:
        # A turn that errored got no answer: the next one in its chain is asked as if it
        # hadn't happened, not after an empty reply.
        if example.chain and row.error is None:
            chains.setdefault(example.chain, []).extend(
                [
                    {"role": "user", "content": example.question},
                    {"role": "assistant", "content": row.answer},
                ]
            )

    with partial.open("a", encoding="utf-8") as sink:
        for index, example in enumerate(examples):
            if index in saved:
                rows.append(saved[index])
                remember_turn(example, saved[index])
                continue
            if example.chain:
                example = replace(example, history=tuple(chains.get(example.chain, [])))
            started = time.monotonic()
            missed = len(replay.misses) if replay is not None else 0
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
            if replay is not None and len(replay.misses) > missed and row.error is None:
                # A call the recording lacks (a condense the pipeline fell back from,
                # say): the answer isn't the recorded one, so it isn't scored as one.
                row = replace(row, answer_status="error", error="replay miss")
            sink.write(json.dumps({"index": index, "row": row.to_dict()}) + "\n")
            sink.flush()
            rows.append(row)
            remember_turn(example, row)
            print(
                f"[{index + 1}/{len(examples)}] {row.answer_status} {row.seconds:.1f}s",
                flush=True,
            )
    from eval.harness import resolve_passages, settings_fingerprint

    try:
        passage_ids = resolve_passages(examples, collaborators.repository, collaborators.settings)
    except Exception:  # noqa: BLE001 - a broken label costs this metric, not the run
        passage_ids = {}
    metrics = answer_metrics(examples, rows, passage_ids)
    settings = collaborators.settings
    # With the rows: what they were measured with and how they scored, so a later run
    # can be compared without re-deriving either. ``examples`` is the key the RAGAS
    # runner reads.
    result = {
        "fingerprint": {
            **settings_fingerprint(settings, "reranked", follow_ups=True),
            "llm_temperature": float(settings.llm_temperature),
            "llm_max_tokens": int(settings.llm_max_tokens),
            "citation_retry_mode": settings.citation_retry_mode,
            "prompt_token_budget": int(settings.prompt_token_budget),
            # What the model saw and how it was asked to write: a baseline measured
            # with other values judges a different run.
            "max_context_chunks": int(settings.max_context_chunks),
            "answer_format": settings.answer_format,
            "answer_language": settings.answer_language,
            # How much of a conversation a follow-up sees, and whether a cut-off
            # answer is continued: both change the answers measured.
            "conversation_max_history_messages": int(settings.conversation_max_history_messages),
            "auto_continue_max": int(settings.auto_continue_max),
            "question_count": len(rows),
        },
        "metrics": {name: round(value, 4) for name, value in metrics.items()},
        # The same metrics per kind of question (follow-ups, compound, unanswerable…):
        # a set whose overall rate held can hide one kind getting worse. Not gated.
        "metrics_by_category": {
            category: {name: round(value, 4) for name, value in scores.items()}
            for category, scores in metrics_by_category(examples, rows, passage_ids).items()
        },
        # ``labelled``: whether the question names relevant files, which answered_rate
        # is measured over (eval/stats.py reads it for the same questions).
        "examples": [
            {**row.to_dict(), "labelled": bool(example.relevant_filenames)}
            for example, row in zip(examples, rows, strict=True)
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"wrote {len(rows)} answer rows to {output}")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    if replay is not None and replay.loose_hits:
        print(
            f"replay: {replay.loose_hits} call(s) answered from a reply recorded for the "
            "same question with other passages"
        )
    if rows and all(row.error for row in rows):
        print("error: every question failed (see the error field of each row)", file=sys.stderr)
        return 1
    return 0
