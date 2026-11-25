"""Per-question state and the small helpers every answer path shares: request
overrides, the retrieval phase, trace and usage records, cache replays. Moved out
of ``api/pipeline.py``, which re-exports these names.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from api.answer_cache import (
    CachedAnswer,
)
from api.answer_events import (
    DoneEvent,
    StreamEvent,
)
from api.generation import (
    ChatGenerator,
    ChatMessage,
    GenerationConfigError,
    SourceCitation,
    StageTimings,
    completion_model_and_kwargs,
)
from api.llm import (
    GenerationSlot,
)
from api.prompt_cache import CachingGenerator
from api.prompting import (
    FittedPrompt,
)
from api.query_log import LoggedQuery
from api.repository import PassageFilter
from api.reranking import (
    RerankedChunk,
)
from api.retrieval import (
    RetrievalConfigError,
    RetrievedChunk,
)
from api.settings import AppSettings
from api.tracing import (
    TraceConfig,
    TraceMode,
    TraceStatus,
)


@dataclass(frozen=True)
class AnswerOverrides:
    """Optional per-request overrides for one ``RagPipeline.answer`` call.

    Defined here rather than in ``api.schemas`` so the pipeline layer never depends on
    the FastAPI request layer. ``rrf_k``, ``fused_top_n``, and the dense/sparse
    retrieval limits are intentionally not overridable here — hybrid retrieval and
    RRF k=60 are non-negotiable per CLAUDE.md.
    """

    llm_provider: str | None = None
    rerank_top_k: int | None = None
    max_context_chunks: int | None = None
    llm_temperature: float | None = None
    # Folded into the settings copy like the fields above (ANSWER_FORMAT,
    # ANSWER_LANGUAGE), so the prompt and the answer cache key read them from there.
    answer_format: str | None = None
    answer_language: str | None = None
    # Narrows retrieval to a page range, a section, picked passages, or away from some
    # documents — on every search leg (rewrite, raw follow-up, expansion variants).
    narrow: PassageFilter | None = None
    # Stricter answers; each only takes something away. passages_only: no neighbouring
    # text in the prompt (context_neighbor_radius 0). strong_matches_only: no answer
    # from below-the-bar passages (rerank_fallback_count 0).
    passages_only: bool = False
    strong_matches_only: bool = False


def narrowing_record(narrow: PassageFilter | None) -> dict[str, Any] | None:
    """A question's passage narrowing as the trace shows it (only the parts set)."""

    if narrow is None:
        return None
    record = {
        "page_min": narrow.page_min,
        "page_max": narrow.page_max,
        "section": narrow.section,
        "exclude_filenames": list(narrow.exclude_filenames),
        "chunk_ids": list(narrow.chunk_ids),
        "exclude_chunk_ids": list(narrow.exclude_chunk_ids),
        "uploaded_after": narrow.uploaded_after,
        "uploaded_before": narrow.uploaded_before,
    }
    return {key: value for key, value in record.items() if value not in (None, [])} or None


# The settings a traced question was answered under that a re-run repeats (ranking
# and the prompt's budget; never the reranker or embedding model, which a re-run can't
# change). A field an older trace didn't record is left at the current setting.
_RERUN_FROM_TRACE = (
    "llm_provider",
    "rerank_top_k",
    "max_context_chunks",
    "rerank_min_score",
    "answer_format",
    "answer_language",
    "context_neighbor_radius",
    "rerank_fallback_count",
    "rerank_candidates",
    "rerank_relative_floor",
    "recency_boost_weight",
    "context_neighbor_max_chars",
    "context_diversity_enabled",
    "context_diversity_max_similarity",
)


def _recorded_settings(config: TraceConfig | None) -> dict[str, Any]:
    if config is None:
        return {}
    recorded = {name: getattr(config, name) for name in _RERUN_FROM_TRACE}
    return {name: value for name, value in recorded.items() if value is not None}


def filter_from_record(record: Mapping[str, Any]) -> PassageFilter:
    """The ``PassageFilter`` a trace's narrowing record (``narrowing_record``) came
    from."""

    fields: dict[str, Any] = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in record.items()
        if key in PassageFilter.__dataclass_fields__
    }
    return PassageFilter(**fields)


@dataclass(frozen=True)
class RetrievalPhase:
    """Result of retrieve -> rerank -> neighbor-expand, kept as a single value so
    trace capture (see api.tracing) can see the full fused candidate list and every
    scored chunk, not just the ones that survived to generation.
    """

    context_chunks: list[RerankedChunk]
    fused_candidates: list[RetrievedChunk]
    scored_chunks: list[RerankedChunk]
    embed_ms: float
    search_ms: float
    rerank_ms: float
    # Query-variant generation (before embedding) and neighbour expansion (after
    # rerank) are separate stages at opposite ends of retrieval; they are timed apart
    # so a latency breakdown can't attribute one to the other.
    query_expansion_ms: float = 0.0
    context_expansion_ms: float = 0.0
    # Part of rerank_ms spent waiting for the reranker (another question was scoring).
    rerank_wait_ms: float = 0.0
    diversity_dropped_ids: frozenset[str] = frozenset()
    # Cleared the floor, cut for scoring far below the best (RERANK_RELATIVE_FLOOR).
    relative_cut_ids: frozenset[str] = frozenset()
    query_variants: list[str] = field(default_factory=list)
    # Search only: nothing cleared rerank_min_score, so context_chunks are the closest
    # passages instead (see _retrieve_and_rerank).
    below_min_score: bool = False
    # Point id -> [dense rank, sparse rank] (repository.LegRanks), for the trace.
    leg_ranks: dict[str, list[int | None]] = field(default_factory=dict)


def _cited_chunks(
    cited: Sequence[SourceCitation], selected: Sequence[RerankedChunk]
) -> list[RerankedChunk]:
    """The prompt chunks an answer cites (source ``n`` is ``selected[n - 1]``)."""

    return [
        selected[source.source_number - 1]
        for source in cited
        if 0 < source.source_number <= len(selected)
    ]


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _reply_known(
    generator: ChatGenerator, messages: Sequence[ChatMessage], settings: AppSettings
) -> bool:
    """The prompt cache holds the reply to ``messages``: no model call will be made."""

    return (
        isinstance(generator, CachingGenerator)
        and generator.cached_reply(messages, settings) is not None
    )


def trace_status(status: str) -> TraceStatus:
    """A finished answer's trace status: a model that said the sources don't answer it
    is "insufficient_context" too, not "ok" (the query log counts those as uncovered)."""

    return (
        "insufficient_context"
        if status in ("no_context", "insufficient_context", "empty_scope")
        else "ok"
    )


def _top_rerank_score(phase: RetrievalPhase) -> float | None:
    """The best cross-encoder score among the scored candidates, kept or not."""

    return phase.scored_chunks[0].rerank_score if phase.scored_chunks else None


def _cached_timings(started: float) -> StageTimings:
    """Timings for an answer served from the cache: no stage ran."""

    return StageTimings(
        embed_ms=0.0,
        search_ms=0.0,
        rerank_ms=0.0,
        generate_ms=0.0,
        total_ms=(time.monotonic() - started) * 1000,
    )


def _citation_dict(source: SourceCitation) -> dict[str, Any]:
    # Every field, spans included (as lists of dicts), as CitationResponse has them.
    return dataclasses.asdict(source)


def replay_cached(hit: CachedAnswer, started: float, **done_fields: Any) -> Iterator[StreamEvent]:
    """A cached answer (or summary) streamed as it first went out: its sources, the
    whole text as one delta, ``done`` marked ``cached`` (``done_fields`` adds what only
    one kind has), then its follow-ups, if any."""

    sources = [_citation_dict(source) for source in hit.sources]
    yield {"type": "sources", "sources": sources, "trace_id": hit.trace_id}
    yield {"type": "delta", "text": hit.answer}
    done: dict[str, Any] = {
        "type": "done",
        "answer": hit.answer,
        "sources": sources,
        "timings": timings_dict(_cached_timings(started)),
        "trace_id": hit.trace_id,
        "cached": True,
        "answer_status": hit.answer_status,
        "truncated": hit.truncated,
        "citation_backfilled": hit.citation_backfilled,
        "unsupported_claims": hit.unsupported_claims,
        "unsupported_numbers": hit.unsupported_numbers,
        "passages_left_out": hit.passages_left_out,
        **done_fields,
    }
    yield cast(DoneEvent, done)
    if hit.followups:
        yield {"type": "followups", "questions": list(hit.followups)}


def served_query(
    hit: CachedAnswer,
    *,
    question: str,
    mode: TraceMode,
    workspace: str,
    kind: str = "question",
    filenames: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    narrowing: dict[str, Any] | None = None,
) -> LoggedQuery:
    """The query-log row of a question (or summary) served from the answer cache."""

    status = trace_status(hit.answer_status)
    cited = (
        tuple(filenames or ())
        if kind == "summary"
        else tuple(dict.fromkeys(source.filename for source in hit.sources))
    )
    scope = {
        key: value
        for key, value in (
            ("filenames", sorted(filenames or ())),
            ("tags", sorted(tags or ())),
            ("narrowing", narrowing),
        )
        if value
    }
    return LoggedQuery(
        created_at=time.time(),
        workspace=workspace,
        question=question,
        status=status,
        top_score=hit.top_rerank_score,
        mode=mode,
        trace_id=hit.trace_id,
        filenames=cited if status == "ok" else (),
        kind=kind,
        top_filename=None,
        scope=scope or None,
        cached=True,
    )


def usage_dict(
    usage: tuple[int, int] | None, cost_usd: float | None = None
) -> dict[str, Any] | None:
    """Token usage (and its estimated cost, when known) as ``done`` frames carry it."""

    if usage is None:
        return None
    record: dict[str, Any] = {"prompt_tokens": usage[0], "completion_tokens": usage[1]}
    if cost_usd is not None:
        record["cost_usd"] = cost_usd
    return record


def trace_model_name(settings: AppSettings) -> str | None:
    """The model a trace records, or None when the provider isn't configured."""

    try:
        return completion_model_and_kwargs(settings)[0]
    except GenerationConfigError:
        return None


def usage_cost(usage: object) -> float | None:
    """The estimated cost in a ``done`` frame's usage, if any."""

    cost = usage.get("cost_usd") if isinstance(usage, dict) else None
    return float(cost) if isinstance(cost, int | float) else None


def usage_tuple(usage: object) -> tuple[int, int] | None:
    """``usage_dict``'s inverse (a ``done`` frame's usage back to the pair)."""

    if not isinstance(usage, dict):
        return None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt, completion
    return None


def timings_dict(timings: StageTimings) -> dict[str, float]:
    # Every StageTimings field, so a new stage reaches traces and the done frame without
    # another hand-kept list (TimingsResponse is tested to have the same fields).
    return dataclasses.asdict(timings)


def _picked_left_out(narrow: PassageFilter | None, selected: Sequence[RerankedChunk]) -> int:
    """How many picked passages (``chunk_ids``) the prompt went without: the reranker
    scores at most RERANK_CANDIDATES, only MAX_CONTEXT_CHUNKS reach the prompt, and the
    window may cut more. Said in the answer, not left to the trace alone."""

    if narrow is None or not narrow.chunk_ids:
        return 0
    return len(set(narrow.chunk_ids) - {chunk.chunk_id for chunk in selected})


def check_rerank_top_k(rerank_top_k: int | None, settings: AppSettings) -> None:
    """A per-request ``rerank_top_k`` past the server's ``fused_top_n`` is refused: the
    reranker can't keep more than fusion hands it. The question routes ask before taking
    a slot, so the stream refuses with a 400 too, not an error frame after its 200."""

    if rerank_top_k is not None and rerank_top_k > int(settings.fused_top_n):
        raise RetrievalConfigError(
            f"rerank_top_k ({rerank_top_k}) cannot exceed the server's "
            f"fused_top_n ({settings.fused_top_n})."
        )


@dataclass
class AnswerRun:
    """One question being answered: what it was asked with, and what its steps found,
    so the error and closed-early traces can be recorded from wherever it stops."""

    question: str
    overrides: AnswerOverrides
    filenames: Sequence[str] | None
    tags: Sequence[str] | None
    mode: TraceMode
    trace_id: str | None
    started: float
    # False for Regenerate (use_cache=False): the condense step asks the model again,
    # and the prompt cache is neither read nor written.
    use_memo: bool = True
    # The answer's generator: the pipeline's, wrapped by the prompt cache when it is on.
    generator: ChatGenerator | None = None
    # Set as the steps run; an error before then leaves them as they are.
    settings: AppSettings | None = None
    history: list[ChatMessage] = field(default_factory=list)
    condensed_question: str | None = None
    condense_note: str | None = None
    # The answer streamed so far (kept on the trace of a stream that broke off).
    parts: list[str] = field(default_factory=list)
    # The answer's prompt once generation starts: kept on the trace of a stream that
    # broke off, so a stopped answer can be continued from where it stopped.
    prompt_messages: list[ChatMessage] | None = None
    # The model finished the answer (not cut off at the output limit): an error after
    # that (the citation retry, the trace) leaves nothing to continue.
    answer_complete: bool = False
    # The stream's model slot, held from generation through follow-ups.
    slot: GenerationSlot | None = None
    # What retrieval found and what of it fit the prompt (kept on an error or
    # closed-early trace, so it shows what the question got as far as).
    phase: RetrievalPhase | None = None
    fitted: FittedPrompt | None = None
    # The thread the run began on: its usage counters (``llm.usage_totals``) are that
    # thread's, so a trace stored from another thread leaves usage out.
    thread: int = field(default_factory=threading.get_ident)
    # A trace has been stored (finished, no context, or an error): the closed-early
    # trace is for a stream that ends with none of those.
    finalized: bool = False

    @property
    def narrow(self) -> PassageFilter | None:
        return self.overrides.narrow


@dataclass(frozen=True)
class _Streamed:
    """What the stream's generation step hands to its finish step."""

    messages: list[ChatMessage]
    sources: list[SourceCitation]
    answer: str
    generate_ms: float
    first_token_ms: float
