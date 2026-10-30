"""Per-query debug traces: retrieval candidates, rerank decisions, the full generation
prompt, and stage timings for one answered question.

Traces are process-local and lost on restart — the same tradeoff as ``api.jobs``'
``IngestJobStore`` — unless TRACE_STORE_PATH also keeps them on disk
(``api.trace_archive``).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, Literal, Protocol, get_args

from api.generation import ChatMessage
from api.metrics import traces_evicted_total
from api.reranking import RerankedChunk
from api.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

TraceStatus = Literal["ok", "insufficient_context", "error"]
TraceMode = Literal["sync", "stream"]
# context_budget: kept by the reranker but left out of the prompt so it would fit the
# model's context window (api.generation.fit_prompt_to_context_window).
DropReason = Literal[
    "below_min_score",
    "near_duplicate",
    "top_k_cut",
    "not_scored",
    "context_budget",
    "context_limit",
    # Over the floor, but far below the best passage (RERANK_RELATIVE_FLOOR).
    "relative_cut",
]
TRACE_DROP_REASONS: tuple[str, ...] = get_args(DropReason)


class TraceNotFoundError(RuntimeError):
    """Raised when a trace id has no known trace (never existed, evicted, or restart)."""


@dataclass(frozen=True)
class TraceCandidate:
    """One fused retrieval candidate's journey through rerank and context selection."""

    point_id: str
    filename: str
    page: int
    section: str
    chunk_id: str
    retrieval_score: float
    # None means the candidate was never sent to the cross-encoder (beyond
    # rerank_candidates' cutoff of the fused list).
    rerank_score: float | None
    kept: bool
    drop_reason: DropReason | None
    selected_for_context: bool
    neighbor_expanded: bool


@dataclass(frozen=True)
class TraceConfig:
    """Effective per-request settings a trace was generated under, after override folding."""

    llm_provider: str
    rerank_top_k: int
    max_context_chunks: int
    llm_temperature: float
    rerank_min_score: float
    fused_top_n: int
    filenames: list[str] | None
    tags: list[str] | None = None
    # The cross-encoder that scored the candidates: rerank_min_score was tuned for one.
    reranker_model: str | None = None
    answer_format: str = "default"
    answer_language: str = ""
    # The model the provider was asked for (LiteLLM's name, e.g. "ollama_chat/qwen2.5").
    llm_model: str | None = None
    # The question's passage narrowing (page range, section, picked passages,
    # documents left out), only the parts given; None when it had none.
    narrowing: dict[str, Any] | None = None
    # PASSAGES_ONLY sets the radius to 0, STRONG_MATCHES_ONLY the fallback count.
    context_neighbor_radius: int | None = None
    rerank_fallback_count: int | None = None


@dataclass
class QueryTrace:
    """A single answered (or failed) question, captured end to end."""

    trace_id: str
    created_at: float
    question: str
    mode: TraceMode
    status: TraceStatus
    config: TraceConfig | None
    candidates: list[TraceCandidate]
    prompt_messages: list[ChatMessage] | None
    answer: str | None
    cited_source_numbers: list[int]
    timings: dict[str, float] | None
    error: str | None
    # None when no history was sent, condense was disabled, or the condense call
    # failed/returned empty and the pipeline fell back to the raw question — in the
    # error path specifically, None also covers "the failure preceded condense."
    condensed_question: str | None = None
    # Why the condense step did not produce condensed_question as the model returned it:
    # skipped ("standalone", "disabled"), failed, or the output was cleaned or rejected.
    condense_note: str | None = None
    # Nothing cleared rerank_min_score: the kept candidates are the closest ones
    # (RERANK_FALLBACK_COUNT), sent to the model as weak matches.
    weak_matches: bool = False
    history_message_count: int = 0
    # Alternative query phrasings used for multi-query retrieval (empty when expansion
    # is disabled — the default).
    query_variants: list[str] = field(default_factory=list)
    # True when the answer initially lacked citations and a stricter retry supplied them.
    citation_retry_used: bool = False
    # The answer's citations were added by matching it to the sources, not by the model.
    citation_backfilled: bool = False
    # (prompt, completion) tokens reported for the answer's LLM calls, when reported.
    usage: tuple[int, int] | None = None
    cost_usd: float | None = None
    # The HTTP request it came from (X-Request-ID), to find its log lines.
    request_id: str | None = None
    # The workspace it was asked in ("" = the default): GET /traces shows a workspace
    # only its own, as the prompt and candidates quote its documents.
    workspace: str = ""
    # The finished answer's ``citations.answer_status`` ("answered", "uncited",
    # "insufficient_context", "no_context", ...): finer than ``status``, which only
    # separates answered from not. None on errors and on traces from before it existed.
    answer_status: str | None = None
    # The answer stopped at the token limit (finish_reason "length").
    truncated: bool = False
    # History messages the prompt fitter left out to fit the context window.
    dropped_history_messages: int = 0


@dataclass(frozen=True)
class TraceSummary:
    """One trace as the list shows it: enough to pick and filter traces without
    reading their prompts and candidates."""

    trace_id: str
    created_at: float
    question: str
    mode: str
    status: str
    llm_provider: str | None
    total_ms: float | None
    candidate_count: int
    kept_count: int
    answer_status: str | None
    top_rerank_score: float | None


def summarize_trace(trace: QueryTrace) -> TraceSummary:
    scores = [c.rerank_score for c in trace.candidates if c.rerank_score is not None]
    return TraceSummary(
        trace_id=trace.trace_id,
        created_at=trace.created_at,
        question=trace.question,
        mode=trace.mode,
        status=trace.status,
        llm_provider=trace.config.llm_provider if trace.config is not None else None,
        total_ms=trace.timings.get("total_ms") if trace.timings is not None else None,
        candidate_count=len(trace.candidates),
        kept_count=sum(1 for candidate in trace.candidates if candidate.kept),
        answer_status=trace_answer_status(trace),
        top_rerank_score=max(scores) if scores else None,
    )


def trace_answer_status(trace: QueryTrace) -> str | None:
    """The trace's answer status; for a trace from before it was recorded, what its
    status says (an "insufficient_context" trace was one), else None."""

    if trace.answer_status is not None:
        return trace.answer_status
    return "insufficient_context" if trace.status == "insufficient_context" else None


class TraceSink(Protocol):
    """Minimal surface ``RagPipeline`` needs to record a trace."""

    def add(self, trace: QueryTrace) -> None: ...


class TraceArchiveStore(Protocol):
    """Where traces are also kept on disk (``api.trace_archive.TraceArchive``)."""

    def add(self, trace: QueryTrace) -> None: ...

    def get(self, trace_id: str) -> QueryTrace | None: ...

    def recent(self, workspace: str | None, limit: int) -> list[QueryTrace]: ...

    def summaries(
        self, workspace: str | None, limit: int, answer_statuses: AbstractSet[str] | None
    ) -> list[TraceSummary]: ...

    def export_lines(self, workspace: str) -> Iterator[str]: ...


def trace_to_json(trace: QueryTrace) -> str:
    return json.dumps(dataclasses.asdict(trace), ensure_ascii=False)


def _known_fields[T](cls: type[T], data: dict[str, Any]) -> dict[str, Any]:
    # A row written by another version may carry a field this one doesn't know (or
    # lack one it added, which then takes its default).
    names = {field.name for field in dataclasses.fields(cls)}  # type: ignore[arg-type]
    return {key: value for key, value in data.items() if key in names}


def trace_from_json(body: str) -> QueryTrace:
    data = json.loads(body)
    config = data.get("config")
    data["config"] = TraceConfig(**_known_fields(TraceConfig, config)) if config else None
    data["candidates"] = [
        TraceCandidate(**_known_fields(TraceCandidate, candidate))
        for candidate in data.get("candidates") or []
    ]
    usage = data.get("usage")
    data["usage"] = tuple(usage) if isinstance(usage, list) and len(usage) == 2 else None
    return QueryTrace(**_known_fields(QueryTrace, data))


class TraceStore:
    """Thread-safe in-memory ring buffer of recent query traces, optionally backed by
    an on-disk archive (TRACE_STORE_PATH) that outlives the ring and restarts."""

    def __init__(self, max_retained: int, archive: TraceArchiveStore | None = None) -> None:
        self._max_retained = max_retained
        self._archive = archive
        self._lock = Lock()
        self._traces: OrderedDict[str, QueryTrace] = OrderedDict()

    def add(self, trace: QueryTrace) -> None:
        with self._lock:
            self._traces[trace.trace_id] = trace
            self._traces.move_to_end(trace.trace_id)
            while len(self._traces) > self._max_retained:
                self._traces.popitem(last=False)
                traces_evicted_total.inc()
        if self._archive is not None:
            try:
                self._archive.add(trace)
            except Exception:  # noqa: BLE001 - a full disk must not fail the answer
                logger.warning("Could not archive trace %s.", trace.trace_id, exc_info=True)

    def get(self, trace_id: str) -> QueryTrace | None:
        """Return a snapshot of a trace, or None if unknown/evicted.

        Returns a shallow copy (not the live object) so a caller can never observe a
        trace mutated after it was stored — traces are written once via ``add`` and
        never updated in place, but this keeps that guarantee explicit.
        """

        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is not None:
                return replace(trace)
        if self._archive is None:
            return None
        try:
            return self._archive.get(trace_id)
        except Exception:  # noqa: BLE001 - an unreadable archive is a miss
            logger.warning("Could not read trace %s from the archive.", trace_id, exc_info=True)
            return None

    def list_traces(self, workspace: str | None = None) -> list[QueryTrace]:
        """Return the retained traces, newest first; only ``workspace``'s when given.
        With an archive, its newest ``max_retained`` (they outlive a restart)."""

        if self._archive is not None:
            try:
                return self._archive.recent(workspace, self._max_retained)
            except Exception:  # noqa: BLE001 - fall back to what memory holds
                logger.warning("Could not list traces from the archive.", exc_info=True)
        with self._lock:
            return [
                replace(trace)
                for trace in reversed(self._traces.values())
                if workspace is None or trace.workspace == workspace
            ]

    def list_summaries(
        self,
        workspace: str | None = None,
        answer_statuses: AbstractSet[str] | None = None,
    ) -> list[TraceSummary]:
        """The trace list's rows, newest first: ``workspace``'s, and with
        ``answer_statuses`` only traces with one of them. With an archive, its newest
        ``max_retained`` matching ones, read from its summary columns (no prompts)."""

        if self._archive is not None:
            try:
                return self._archive.summaries(workspace, self._max_retained, answer_statuses)
            except Exception:  # noqa: BLE001 - fall back to what memory holds
                logger.warning("Could not list traces from the archive.", exc_info=True)
        with self._lock:
            traces = [
                trace
                for trace in reversed(self._traces.values())
                if workspace is None or trace.workspace == workspace
            ]
        summaries = [summarize_trace(trace) for trace in traces]
        if answer_statuses is not None:
            summaries = [s for s in summaries if s.answer_status in answer_statuses]
        return summaries

    def export_lines(self, workspace: str) -> Iterator[str]:
        """``workspace``'s traces as JSON Lines, oldest first: every archived one, or
        the ring's without an archive."""

        if self._archive is not None:
            yield from self._archive.export_lines(workspace)
            return
        for trace in reversed(self.list_traces(workspace)):
            yield trace_to_json(trace) + "\n"


def build_trace_candidates(
    fused_candidates: Sequence[RetrievedChunk],
    scored_chunks: Sequence[RerankedChunk],
    kept_chunks: Sequence[RerankedChunk],
    selected_chunks: Sequence[RerankedChunk],
    *,
    min_score: float,
    diversity_dropped_ids: AbstractSet[str] = frozenset(),
    context_budget_dropped_ids: AbstractSet[str] = frozenset(),
    relative_cut_ids: AbstractSet[str] = frozenset(),
) -> list[TraceCandidate]:
    """Join the fused, scored, kept, and context-selected candidate lists by point id.

    ``fused_candidates`` is the full RRF-fused list (before any candidate is dropped
    for exceeding ``rerank_candidates``); ``scored_chunks`` is everything the
    cross-encoder actually scored (``RerankOutcome.scored``); ``kept_chunks`` is what
    survived the ``min_score``/``rerank_top_k`` cut; ``selected_chunks`` is the
    ``max_context_chunks`` prefix of ``kept_chunks`` that actually reached generation.
    """

    scored_by_id = {chunk.point_id: chunk for chunk in scored_chunks}
    kept_ids = {chunk.point_id for chunk in kept_chunks}
    selected_ids = {chunk.point_id for chunk in selected_chunks}
    expanded_ids = {chunk.point_id for chunk in selected_chunks if chunk.expanded_text is not None}

    result: list[TraceCandidate] = []
    for candidate in fused_candidates:
        scored = scored_by_id.get(candidate.point_id)
        kept = candidate.point_id in kept_ids
        rerank_score = scored.rerank_score if scored is not None else None
        drop_reason: DropReason | None = None
        if not kept:
            if scored is None:
                drop_reason = "not_scored"
            elif scored.rerank_score < min_score:
                drop_reason = "below_min_score"
            elif candidate.point_id in diversity_dropped_ids:
                drop_reason = "near_duplicate"
            elif candidate.point_id in relative_cut_ids:
                drop_reason = "relative_cut"
            else:
                drop_reason = "top_k_cut"
        elif candidate.point_id in context_budget_dropped_ids:
            drop_reason = "context_budget"
        elif candidate.point_id not in selected_ids:
            # Kept by rerank, but past the max_context_chunks prefix the prompt takes.
            drop_reason = "context_limit"
        result.append(
            TraceCandidate(
                point_id=candidate.point_id,
                filename=candidate.filename,
                page=candidate.page,
                section=candidate.section,
                chunk_id=candidate.chunk_id,
                retrieval_score=candidate.score,
                rerank_score=rerank_score,
                kept=kept,
                drop_reason=drop_reason,
                selected_for_context=candidate.point_id in selected_ids,
                neighbor_expanded=candidate.point_id in expanded_ids,
            )
        )
    return result
