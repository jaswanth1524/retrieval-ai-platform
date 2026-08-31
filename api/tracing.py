"""Per-query debug traces: retrieval candidates, rerank decisions, the full generation
prompt, and stage timings for one answered question.

Traces are process-local and lost on restart — the same tradeoff as ``api.jobs``'
``IngestJobStore``: there is no requirement for durable trace history across process
restarts, only visibility into a running instance for debugging retrieval quality.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Literal, Protocol

from api.generation import ChatMessage
from api.metrics import traces_evicted_total
from api.reranking import RerankedChunk
from api.retrieval import RetrievedChunk

TraceStatus = Literal["ok", "insufficient_context", "error"]
TraceMode = Literal["sync", "stream"]
DropReason = Literal["below_min_score", "near_duplicate", "top_k_cut", "not_scored"]


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
    history_message_count: int = 0
    # Alternative query phrasings used for multi-query retrieval (empty when expansion
    # is disabled — the default).
    query_variants: list[str] = field(default_factory=list)
    # True when the answer initially lacked citations and a stricter retry supplied them.
    citation_retry_used: bool = False


class TraceSink(Protocol):
    """Minimal surface ``RagPipeline`` needs to record a trace."""

    def add(self, trace: QueryTrace) -> None: ...


class TraceStore:
    """Thread-safe in-memory ring buffer of recent query traces."""

    def __init__(self, max_retained: int) -> None:
        self._max_retained = max_retained
        self._lock = Lock()
        self._traces: OrderedDict[str, QueryTrace] = OrderedDict()

    def add(self, trace: QueryTrace) -> None:
        with self._lock:
            self._traces[trace.trace_id] = trace
            self._traces.move_to_end(trace.trace_id)
            while len(self._traces) > self._max_retained:
                self._traces.popitem(last=False)
                traces_evicted_total.inc()

    def get(self, trace_id: str) -> QueryTrace | None:
        """Return a snapshot of a trace, or None if unknown/evicted.

        Returns a shallow copy (not the live object) so a caller can never observe a
        trace mutated after it was stored — traces are written once via ``add`` and
        never updated in place, but this keeps that guarantee explicit.
        """

        with self._lock:
            trace = self._traces.get(trace_id)
            return replace(trace) if trace is not None else None

    def list_traces(self) -> list[QueryTrace]:
        """Return all retained traces, newest first."""

        with self._lock:
            return [replace(trace) for trace in reversed(self._traces.values())]


def build_trace_candidates(
    fused_candidates: Sequence[RetrievedChunk],
    scored_chunks: Sequence[RerankedChunk],
    kept_chunks: Sequence[RerankedChunk],
    selected_chunks: Sequence[RerankedChunk],
    *,
    min_score: float,
    diversity_dropped_ids: AbstractSet[str] = frozenset(),
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
    expanded_ids = {
        chunk.point_id for chunk in selected_chunks if chunk.expanded_text is not None
    }

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
            else:
                drop_reason = "top_k_cut"
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
