"""Orchestration seams: query (retrieve -> rerank -> generate) and ingest (parse ->
chunk -> embed -> index), each behind a single call so handlers stay thin.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Any, Literal, NotRequired, TypedDict, cast

from api.answer_cache import (
    AnswerCache,
    CachedAnswer,
    CacheKey,
    answer_cache_key,
)
from api.citations import (
    answer_status,
    cited_numbers,
    declines_to_answer,
    opens_with_a_refusal,
    states_insufficiency,
    strip_citation_markers,
)

# Neighbour expansion and ingestion live in their own modules now; their names are
# re-exported here (``x as x``) so every existing ``from api.pipeline import`` works.
from api.context_expansion import _TAG_LOOKUP_EXECUTOR as _TAG_LOOKUP_EXECUTOR
from api.context_expansion import _VARIANT_EXECUTOR as _VARIANT_EXECUTOR
from api.context_expansion import _clip_head as _clip_head
from api.context_expansion import _clip_tail as _clip_tail
from api.context_expansion import _map_concurrently as _map_concurrently
from api.context_expansion import _strip_overlap as _strip_overlap
from api.context_expansion import expand_with_neighbors as expand_with_neighbors
from api.diversity import select_diverse
from api.documents import contextual_text
from api.embeddings import EmbeddedText
from api.errors import public_error_message
from api.generation import (
    INSUFFICIENT_CONTEXT_ANSWER,
    ChatGenerator,
    ChatMessage,
    GenerationConfigError,
    GenerationError,
    GroundedAnswer,
    SourceCitation,
    StageTimings,
    answer_messages,
    completion_model_and_kwargs,
    finish_answer,
    fit_prompt_to_context_window,
    generate_followups,
    generate_grounded_answer,
    generate_query_variants,
    needs_condense,
    source_citations,
    will_call_llm_for_citations,
)
from api.ingest_service import DocumentIngestOutcome as DocumentIngestOutcome
from api.ingest_service import IngestService as IngestService
from api.llm import (
    GenerationSlot,
    begin_usage,
    expected_prefill_ms,
    reset_finish_reason,
    stopped_at_token_limit,
    usage_cost_usd,
    usage_totals,
    was_degraded,
    was_prompt_cut,
)
from api.metrics import answer_inflight_joins_total
from api.prompt_cache import CachingGenerator, shared_prompt_cache
from api.prompting import (
    FittedPrompt,
    clip_text,
    condense_question_detailed,
    estimate_prompt_tokens,
    one_line,
)
from api.qdrant_schema import check_dense_dimension
from api.query_log import LoggedQuery, log_served
from api.repository import LegRanks, PassageFilter, VectorRepository, reciprocal_rank_fusion
from api.request_id import current_request_id, submit_in_context
from api.reranking import (
    RerankedChunk,
    Reranker,
    RerankOutcome,
    relative_cut,
    rerank_candidates_detailed,
)
from api.retrieval import (
    QueryEmbeddingProvider,
    RetrievalConfigError,
    RetrievalError,
    RetrievedChunk,
    embed_query,
    points_to_chunks,
)
from api.settings import AppSettings
from api.tracing import (
    QueryTrace,
    TraceConfig,
    TraceMode,
    TraceSink,
    TraceStatus,
    build_trace_candidates,
)
from api.workspaces import workspace_of

logger = logging.getLogger(__name__)


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


class SourcesEvent(TypedDict):
    """SSE event: citation metadata for the selected context, sent before generation.

    Carries ``trace_id`` (None when tracing is disabled) so the UI can link this turn
    to its debug trace even if generation fails before the ``done`` event is reached.
    """

    type: Literal["sources"]
    sources: list[dict[str, Any]]
    trace_id: str | None


class StageEvent(TypedDict):
    """SSE event: which pipeline stage just started, for the UI's progress indicator.

    Emitted *before* the stage runs, not after — the whole point is telling the reader
    what the server is doing during the seconds before any token exists. Retrieval can
    take tens of seconds on a whole-corpus question, and the UI previously showed one
    static label for all of it.

    ``stage`` is a free-form lowercase label rather than an enum: a client that doesn't
    recognise one must fall back to its own default, so adding a stage later can never
    break an older frontend.
    """

    type: Literal["stage"]
    stage: str
    # Progress through a multi-step stage (a summary's parts): step ``step`` of
    # ``steps``, both 1-based. Absent on single-step stages.
    step: NotRequired[int]
    steps: NotRequired[int]
    # "reading": the prompt the model is about to read, in estimated tokens, and how
    # long the local model should take to read it at its measured rate (absent before
    # one is measured, and for hosted providers).
    prompt_tokens: NotRequired[int]
    eta_ms: NotRequired[float]


class DeltaEvent(TypedDict):
    """SSE event: one incremental fragment of the generated answer."""

    type: Literal["delta"]
    text: str


class FollowupsEvent(TypedDict):
    """SSE event after ``done`` (FOLLOWUPS_ENABLED): questions to suggest next.

    After ``done``, not inside it, so the answer and its citations are final while the
    extra LLM call runs; a client that doesn't know the frame ignores it.
    """

    type: Literal["followups"]
    questions: list[str]


class DoneEvent(TypedDict):
    """SSE event: the final answer, cited-only sources, and stage timings."""

    type: Literal["done"]
    answer: str
    sources: list[dict[str, Any]]
    timings: dict[str, float]
    trace_id: str | None
    # Present (True) only when the answer came from the answer cache.
    cached: NotRequired[bool]
    # citations.answer_status, and the best cross-encoder score among the candidates.
    answer_status: NotRequired[str]
    top_rerank_score: NotRequired[float | None]
    # The relevance bar (RERANK_MIN_SCORE, or the workspace's) the answer was given
    # under: what the client compares top_rerank_score against.
    rerank_min_score: NotRequired[float]
    # Answered from below-the-bar passages (RetrievalPhase.below_min_score).
    weak_matches: NotRequired[bool]
    # The provider cut the answer off at the token limit.
    truncated: NotRequired[bool]
    # The answer's citations were added by matching it to the sources (backfill).
    citation_backfilled: NotRequired[bool]
    unsupported_claims: NotRequired[int]
    prompt_cached: NotRequired[bool]
    prompt_cut: NotRequired[bool]
    # Picked passages (chunk_ids) the prompt had no room for (GroundedAnswer).
    passages_left_out: NotRequired[int]
    usage: NotRequired[dict[str, int] | None]
    # Summaries only: passages read, of the passages the documents hold. Fewer than
    # the total when a long document was read at evenly spaced passages.
    chunks_read: NotRequired[int]
    chunks_total: NotRequired[int]
    # Summaries only: parts left out after their call failed twice, and parts whose
    # notes stopped at the token limit (the summary may miss what they held).
    parts_skipped: NotRequired[int]
    notes_truncated: NotRequired[int]


StreamEvent = StageEvent | SourcesEvent | DeltaEvent | DoneEvent | FollowupsEvent


def _cited_chunks(
    cited: Sequence[SourceCitation], selected: Sequence[RerankedChunk]
) -> list[RerankedChunk]:
    """The prompt chunks an answer cites (source ``n`` is ``selected[n - 1]``)."""

    return [
        selected[source.source_number - 1]
        for source in cited
        if 0 < source.source_number <= len(selected)
    ]


def _without_unanswered_exchanges(history: list[ChatMessage]) -> list[ChatMessage]:
    """``history`` minus every exchange whose answer was "not enough information".

    Sent back, a refusal primed the next question twice over: the condense step folded
    it into the search query ("What if they miss it?" became "What are the consequences
    if they miss the key points?") and the answering model read it as the
    conversation's own conclusion and refused again. A refusal that cites the sources
    it looked at counts too (``declines_to_answer``); an answer that says something is
    missing but gives something else from a source answered part of it, and stays.
    """

    kept: list[ChatMessage] = []
    for message in history:
        content = message["content"]
        if message["role"] == "assistant" and (
            (
                states_insufficiency(content)
                and not cited_numbers(content, _HISTORY_CITATION_CHECK_MAX)
            )
            or declines_to_answer(content)
        ):
            if kept and kept[-1]["role"] == "user":
                kept.pop()
            continue
        kept.append(message)
    return kept


def _from_first_question(history: list[ChatMessage]) -> list[ChatMessage]:
    """``history`` from its first user message on."""

    start = next(
        (index for index, message in enumerate(history) if message["role"] == "user"),
        len(history),
    )
    return history[start:]


# Only whether an earlier answer cites anything matters, so ranges are clamped small.
_HISTORY_CITATION_CHECK_MAX = 1000


def _clip_earlier_answers(history: list[ChatMessage], settings: AppSettings) -> list[ChatMessage]:
    """``history`` with each earlier answer cut to GENERATION_HISTORY_ASSISTANT_CHARS
    (0: unchanged). The answering model reads all of it before the first token."""

    limit = int(settings.generation_history_assistant_chars)
    if limit <= 0:
        return history
    return [
        {"role": "assistant", "content": clip_text(message["content"], limit)}
        if message["role"] == "assistant"
        else message
        for message in history
    ]


def _with_interpretation(question: str, condensed: str | None) -> str:
    """The question as the answering prompt shows it: a follow-up that was rewritten for
    search carries the rewrite, so a small model reading "Explain in more detail" next
    to sources about vacation days knows what "more" is about."""

    if not condensed or condensed.casefold() == question.casefold():
        return question
    return f"{question}\n(Read with the conversation as: {one_line(condensed)})"


class CondenseMemo:
    """Rewrites of follow-ups already made: Retry, an edited follow-up asked again or
    the same follow-up asked in another tab paid the condense call (2.5-7 s at a local
    model) every time. Keyed on everything the call is sent — the question, the history
    as truncated, the provider and model, the condense settings — so only an identical
    call is skipped. Only usable rewrites are kept (a failure or a rejected rewrite is
    tried again). CONDENSE_CACHE_SIZE entries, least recently used dropped; 0 = off."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def key(question: str, history: Sequence[ChatMessage], settings: AppSettings) -> str:
        parts = [
            question,
            str(settings.llm_provider),
            trace_model_name(settings) or "",
            str(settings.condense_max_tokens),
            *(f"{message['role']}:{message['content']}" for message in history),
        ]
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            found = self._entries.get(key)
            if found is not None:
                self._entries.move_to_end(key)
            return found

    def put(self, key: str, rewrite: str, size: int) -> None:
        if size <= 0:
            return
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = rewrite
            while len(self._entries) > size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


condense_memo = CondenseMemo()


CONTINUE_INSTRUCTION = (
    "Your answer above was cut off before it finished. Continue it from exactly where it "
    "stopped: do not repeat anything already written, and keep citing the sources by "
    "their [n] numbers."
)


class ContinueRefused(Exception):
    """A cut-off answer that can't be continued, with the HTTP status that says why."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ContinuePlan:
    trace: QueryTrace
    chunks: list[RerankedChunk]


# Providers whose chat API continues a trailing assistant message (prefill) — measured
# on Ollama (qwen2.5); Anthropic documents it. Asked in words instead ("continue where
# you stopped"), qwen2.5 started its answer over every time. Hosted OpenAI ignores a
# trailing assistant message, so it is asked in words.
_PREFILL_PROVIDERS = frozenset({"ollama", "anthropic", "openai_compatible"})


def continuation_messages(
    base: Sequence[ChatMessage], answer_so_far: str, settings: AppSettings
) -> list[ChatMessage]:
    """The prompt that asks for the rest of ``answer_so_far``: the answer as a trailing
    assistant message to continue, or, where the provider can't, followed by a request
    to go on."""

    if settings.llm_provider.lower().strip() in _PREFILL_PROVIDERS:
        # Anthropic refuses a prefill ending in whitespace; the join puts it back.
        return [*base, {"role": "assistant", "content": answer_so_far.rstrip()}]
    return [
        *base,
        {"role": "assistant", "content": answer_so_far},
        {"role": "user", "content": CONTINUE_INSTRUCTION},
    ]


def base_prompt(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """A trace's prompt without what a Continue added (the answer so far, the request to
    go on), so continuing again starts from the original prompt, not a nested one."""

    trimmed = list(messages)
    if trimmed and trimmed[-1]["role"] == "user" and trimmed[-1]["content"] == CONTINUE_INSTRUCTION:
        trimmed = trimmed[:-1]
    if trimmed and trimmed[-1]["role"] == "assistant":
        trimmed = trimmed[:-1]
    return trimmed


def drop_restart(before: str, after: str) -> str:
    """``after`` without a restart: a model asked to continue that writes the answer
    again from the top has the part it repeats removed (the rest is the new text)."""

    head = after.lstrip()
    probe = before.lstrip()[:40]
    if len(probe) < 20 or not head.startswith(probe):
        return after
    original = before.lstrip()
    shared = 0
    while shared < min(len(original), len(head)) and original[shared] == head[shared]:
        shared += 1
    return head[shared:]


def join_continuation(before: str, after: str, *, prefilled: bool = False) -> str:
    """A cut-off answer and its continuation as one text. Continuing a prefill, the
    model writes its own leading space where a word ends (" can be", or "2]" inside a
    marker): the two are joined as they are. Asked in words, a space goes between them
    unless either side already has one or the continuation starts with punctuation."""

    if not after:
        return before
    if prefilled:
        return before.rstrip() + after
    if before[-1:].isspace() or after[:1].isspace() or after[:1] in ".,;:!?)]}'\"":
        return before + after
    return f"{before} {after}"


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


def _condense_skip_reason(question: str, settings: AppSettings) -> str | None:
    """Why a question with history is not condensed, or None when it is. The one copy
    of the gate: ``_will_condense`` announces the stage from it, ``_condense_query``
    runs (or skips) the call from it."""

    if not settings.conversation_condense_enabled:
        return "disabled"
    if not needs_condense(question):
        return "standalone"
    return None


def trace_status(status: str) -> TraceStatus:
    """A finished answer's trace status: a model that said the sources don't answer it
    is "insufficient_context" too, not "ok" (the query log counts those as uncovered)."""

    return "insufficient_context" if status in ("no_context", "insufficient_context") else "ok"


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


@dataclass(frozen=True)
class _PassageSelection:
    outcome: RerankOutcome
    diversity_dropped: set[str]
    below_min_score: bool
    relative_cut_ids: set[str]


def _select_passages(
    outcome: RerankOutcome,
    settings: AppSettings,
    *,
    for_generation: bool,
    picked_only: bool = False,
) -> _PassageSelection:
    """The reranked passages kept for the prompt (or a search): near-duplicates dropped,
    the closest ones kept when none clears the floor, and the relative cut applied.

    With ``picked_only`` (a question about passages the user picked, ``chunk_ids``)
    every picked passage is kept, best first: the user chose them, and the context isn't
    neighbour-expanded, so a picked passage next to another was lost for good when the
    diversity filter dropped it as a neighbour, as was one under the score floor.
    """

    if picked_only:
        return _PassageSelection(
            RerankOutcome(scored=outcome.scored, kept=list(outcome.scored)), set(), False, set()
        )
    # A chunk next to a kept one is dropped on the promise that neighbour expansion
    # puts its text back — true only when expansion brings the whole neighbour.
    # With CONTEXT_NEIGHBOR_MAX_CHARS capping it, a well-scored passage was lost.
    diverse_outcome, diversity_dropped = select_diverse(
        outcome,
        settings,
        drop_neighbours=for_generation and int(settings.context_neighbor_max_chars) == 0,
    )
    # Nothing cleared rerank_min_score: the closest passages are used anyway,
    # flagged. A search (no answer) shows up to rerank_top_k of them; the cross-
    # encoder scores a bare keyword low against a passage that has it ("laptop":
    # 0.11 for the IT equipment policy, which it ranks first). An answer gets
    # rerank_fallback_count of them, and the prompt tells the model they match only
    # weakly, so it answers from them only when they hold the answer: the floor is
    # on one model's score scale, and a fixed one refused questions the documents
    # answer ("what are foundational models": 0.006 with MiniLM, 0.154 with jina).
    below_min_score = False
    fallback_count = min(
        int(settings.rerank_top_k),
        int(settings.rerank_fallback_count) if for_generation else int(settings.rerank_top_k),
    )
    if fallback_count > 0 and not diverse_outcome.kept and outcome.scored:
        # rerank_top_k too: select_diverse fills up to it from the scored list, so
        # an answer got 8 below-the-bar passages instead of rerank_fallback_count.
        unfloored = settings.model_copy(
            update={"rerank_min_score": 0.0, "rerank_top_k": fallback_count}
        )
        diverse_outcome, diversity_dropped = select_diverse(
            RerankOutcome(scored=outcome.scored, kept=outcome.scored[:fallback_count]),
            unfloored,
            drop_neighbours=False,
        )
        below_min_score = bool(diverse_outcome.kept)
    relative_cut_ids: set[str] = set()
    if for_generation and not below_min_score:
        trimmed, relative_cut_ids = relative_cut(
            diverse_outcome.kept,
            outcome.scored,
            float(settings.rerank_relative_floor),
        )
        diverse_outcome = RerankOutcome(
            scored=diverse_outcome.scored,
            kept=recency_ordered(trimmed, float(settings.recency_boost_weight)),
        )
    return _PassageSelection(diverse_outcome, diversity_dropped, below_min_score, relative_cut_ids)


def select_passages(
    scored: Sequence[RerankedChunk], settings: AppSettings
) -> tuple[list[RerankedChunk], bool]:
    """The passages an answer would be given from ``scored`` (one rerank pass's scored
    candidates, best first) under ``settings``: the floor and top-K cut as the reranker
    applies them, then ``_select_passages``. With whether they are the weak-match
    fallback. The seam the eval harness's sweep replays settings through without
    retrieving or reranking again."""

    ordered = list(scored)
    floor = float(settings.rerank_min_score)
    kept = [chunk for chunk in ordered if chunk.rerank_score >= floor][: int(settings.rerank_top_k)]
    selection = _select_passages(
        RerankOutcome(scored=ordered, kept=kept), settings, for_generation=True
    )
    return list(selection.outcome.kept), selection.below_min_score


# How far back the recency boost looks: a document this much older than the newest kept
# one gets none of it. A fixed span, not the kept passages' own: documents uploaded
# seconds apart (a folder indexed in one go) must not be told apart by it.
RECENCY_HORIZON_SECONDS = 365 * 24 * 3600.0


def recency_ordered(chunks: Sequence[RerankedChunk], weight: float) -> list[RerankedChunk]:
    """``chunks`` ordered by rerank score plus ``weight`` x how recent each one's document
    is — 1 for the newest kept, falling to 0 a year (``RECENCY_HORIZON_SECONDS``) before
    it — RECENCY_BOOST_WEIGHT. The same passages, only reordered; unchanged when the
    weight is 0 or none is dated."""

    stamps = [chunk.uploaded_at for chunk in chunks if chunk.uploaded_at is not None]
    if weight <= 0 or len(stamps) < 2:
        return list(chunks)
    newest = max(stamps)

    def boosted(chunk: RerankedChunk) -> float:
        if chunk.uploaded_at is None:
            return chunk.rerank_score
        age = newest - chunk.uploaded_at
        return chunk.rerank_score + weight * max(0.0, 1.0 - age / RECENCY_HORIZON_SECONDS)

    return sorted(chunks, key=boosted, reverse=True)


class QuestionAbandoned(Exception):
    """The client left mid-question; stop before the next expensive stage."""


def wait_for_model(
    settings: AppSettings, should_stop: Callable[[], bool] | None
) -> Generator[StreamEvent, None, GenerationSlot]:
    """Take this thread's place at the local model (OLLAMA_MAX_PARALLEL), held.

    Yields a ``waiting_for_model`` frame first when another request has it, so the
    reader sees why nothing moves, and raises ``QuestionAbandoned`` when the client
    leaves while waiting. The caller releases the returned slot; generator calls made
    while it is held take no second place (the slot is re-entrant per thread). With no
    limit (hosted providers, 0) it returns at once and holds nothing.
    """

    slot = GenerationSlot(settings)
    if not slot.try_acquire():
        yield {"type": "stage", "stage": "waiting_for_model"}
        if not slot.acquire(should_stop):
            raise QuestionAbandoned
    return slot


# How often a follower waiting for an identical run checks whether its reader left.
_JOIN_POLL_SECONDS = 0.5


def lead_or_join(
    cache: AnswerCache, key: CacheKey, should_stop: Callable[[], bool] | None
) -> Generator[StreamEvent, None, threading.Event | CachedAnswer | None]:
    """Lead the run of ``key`` (returns the event the caller must ``finish``), or wait
    for the identical one already running and return its cached answer.

    A follower announces itself (``joining_identical_question``) and waits for as long
    as the leader runs — a leader may legitimately wait minutes for a local model, and a
    follower that gave up would only queue behind it. None: the reader left while
    waiting. A leader that leaves no answer (failed, abandoned) makes the follower lead
    the question itself.
    """

    announced = False
    while True:
        event, leading = cache.inflight.claim(key)
        if leading:
            # The previous leader may have finished between our lookup and the claim.
            hit = cache.get(key)
            if hit is None:
                return event
            cache.inflight.finish(key, event)
            return hit
        if not announced:
            announced = True
            yield {"type": "stage", "stage": "joining_identical_question"}
        while not event.wait(_JOIN_POLL_SECONDS):
            if should_stop is not None and should_stop():
                return None
        hit = cache.get(key)
        if hit is not None:
            answer_inflight_joins_total.labels(outcome="replayed").inc()
            return hit
        answer_inflight_joins_total.labels(outcome="ran").inc()


def cached_run(
    cache: AnswerCache | None,
    key: CacheKey | None,
    *,
    use_cache: bool,
    should_stop: Callable[[], bool] | None,
    replay: Callable[[CachedAnswer], Iterator[StreamEvent]],
    run: Callable[[], Generator[StreamEvent]],
    keep: Callable[[DoneEvent], bool] = lambda done: True,
    on_hit: Callable[[CachedAnswer], None] | None = None,
) -> Iterator[StreamEvent]:
    """A run through the answer cache, for answers and summaries alike: a hit replayed
    (``replay``), or the identical run already going joined (``lead_or_join``), or
    ``run`` led — its ``done`` stored (unless the answer is degraded or ``keep`` says
    no) before the frame goes out, and its follow-ups added once they come.

    Stored at ``done``, not after the loop: follow-ups are generated after it, and a
    client that leaves then (or as ``done`` arrives) closes this generator at the yield,
    so a store after the loop never ran. Followers are released right after the store;
    they never wait for the follow-ups. ``use_cache=False`` (Regenerate) skips the
    lookup but still stores the result. ``on_hit`` is told of a hit (or a join) before
    it is replayed: it has no trace of its own, so the query log is told here.
    """

    lead: threading.Event | None = None
    if key is not None and use_cache and cache is not None:
        hit = cache.get(key)
        if hit is None:
            joined = yield from lead_or_join(cache, key, should_stop)
            if joined is None:
                return
            if isinstance(joined, CachedAnswer):
                hit = joined
            else:
                lead = joined
        if hit is not None:
            if on_hit is not None:
                on_hit(hit)
            yield from replay(hit)
            return
    done: DoneEvent | None = None
    events = run()

    def finish_lead() -> None:
        if lead is not None and key is not None and cache is not None:
            cache.inflight.finish(key, lead)

    stored: CachedAnswer | None = None

    def store(followups: Sequence[str]) -> None:
        nonlocal stored
        if key is None or done is None or cache is None or was_degraded() or not keep(done):
            return
        stored = CachedAnswer.from_done(done, followups)
        cache.put(key, stored)

    try:
        for event in events:
            if event["type"] == "done":
                done = event
                store(())
                finish_lead()
            elif event["type"] == "followups":
                store(event["questions"])
            yield event
        # Stored at ``done``, before the follow-ups: when they failed (no frame comes
        # then), the answer is degraded after all and isn't kept, as on the sync path.
        if stored is not None and key is not None and cache is not None and was_degraded():
            cache.discard(key, stored)
    finally:
        # Closing this generator (the client left) must close the inner one now: its
        # `finally` records the "closed early" trace. Left to garbage collection, it ran
        # whenever the inner generator was freed — never, while anything (a tracer, a
        # traceback) still held it.
        events.close()
        finish_lead()


def drain[T](events: Generator[StreamEvent, None, T]) -> T:
    """Run a stage-yielding generator to its return value (the sync route has no
    stream to send its frames to)."""

    while True:
        try:
            next(events)
        except StopIteration as stop:
            return cast(T, stop.value)


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


class RagPipeline:
    """Composes retrieval, reranking, and generation behind a single call."""

    def __init__(
        self,
        repository: VectorRepository,
        embedding_provider: QueryEmbeddingProvider,
        reranker: Reranker,
        generator: ChatGenerator,
        settings: AppSettings,
        trace_store: TraceSink | None = None,
        answer_cache: AnswerCache | None = None,
    ) -> None:
        self._repository = repository
        # None disables answer caching (ANSWER_CACHE_SIZE=0).
        self._answer_cache = answer_cache
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._generator = generator
        self._settings = settings
        # None disables trace capture entirely (the default) — every trace-related
        # call below is skipped rather than merely producing an empty trace, so
        # tracing costs nothing when the operator has turned it off.
        self._trace_store = trace_store

    def effective_settings(self, overrides: AnswerOverrides) -> AppSettings:
        """The settings a question with ``overrides`` runs under (a search's explanation
        names the bar and cuts it was given)."""

        return self._effective_settings(overrides)

    def _effective_settings(self, overrides: AnswerOverrides) -> AppSettings:
        """Fold per-request overrides into a settings copy without mutating the base."""

        settings = self._settings
        check_rerank_top_k(overrides.rerank_top_k, settings)

        update: dict[str, Any] = {
            key: value
            for key, value in {
                "llm_provider": overrides.llm_provider,
                "rerank_top_k": overrides.rerank_top_k,
                "max_context_chunks": overrides.max_context_chunks,
                "llm_temperature": overrides.llm_temperature,
                "answer_format": overrides.answer_format,
                "answer_language": overrides.answer_language,
            }.items()
            if value is not None
        }
        if overrides.passages_only:
            update["context_neighbor_radius"] = 0
        if overrides.strong_matches_only:
            update["rerank_fallback_count"] = 0
        return settings.model_copy(update=update) if update else settings

    def _trace_config(
        self,
        effective_settings: AppSettings,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None = None,
        narrow: PassageFilter | None = None,
    ) -> TraceConfig:
        return TraceConfig(
            llm_provider=effective_settings.llm_provider,
            rerank_top_k=int(effective_settings.rerank_top_k),
            max_context_chunks=int(effective_settings.max_context_chunks),
            llm_temperature=float(effective_settings.llm_temperature),
            rerank_min_score=float(effective_settings.rerank_min_score),
            fused_top_n=int(effective_settings.fused_top_n),
            # Sorted: the same scope given in another order is the same setting.
            filenames=sorted(filenames) if filenames else None,
            tags=sorted(tags) if tags else None,
            reranker_model=effective_settings.reranker_model,
            answer_format=effective_settings.answer_format,
            answer_language=effective_settings.answer_language,
            narrowing=narrowing_record(narrow),
            llm_model=trace_model_name(effective_settings),
            context_neighbor_radius=int(effective_settings.context_neighbor_radius),
            rerank_fallback_count=int(effective_settings.rerank_fallback_count),
            rerank_candidates=int(effective_settings.rerank_candidates),
            rerank_relative_floor=float(effective_settings.rerank_relative_floor),
            recency_boost_weight=float(effective_settings.recency_boost_weight),
            query_expansion_enabled=bool(effective_settings.query_expansion_enabled),
            reranker_onnx_file=effective_settings.reranker_onnx_file,
            context_neighbor_max_chars=int(effective_settings.context_neighbor_max_chars),
            context_diversity_enabled=bool(effective_settings.context_diversity_enabled),
            context_diversity_max_similarity=float(
                effective_settings.context_diversity_max_similarity
            ),
            citation_retry_mode=effective_settings.citation_retry_mode,
            llm_max_tokens=int(effective_settings.llm_max_tokens),
            query_expansion_count=int(effective_settings.query_expansion_count),
        )

    def _phase_timings(
        self,
        phase: RetrievalPhase,
        *,
        total_start: float,
        condense_ms: float,
        generate_ms: float = 0.0,
        citation_retry_ms: float = 0.0,
        first_token_ms: float = 0.0,
    ) -> StageTimings:
        """Stage timings for a finished question; ``total_ms`` measured now."""

        return StageTimings(
            embed_ms=phase.embed_ms,
            search_ms=phase.search_ms,
            rerank_ms=phase.rerank_ms,
            generate_ms=generate_ms,
            total_ms=(time.monotonic() - total_start) * 1000,
            condense_ms=condense_ms,
            query_expansion_ms=phase.query_expansion_ms,
            context_expansion_ms=phase.context_expansion_ms,
            citation_retry_ms=citation_retry_ms,
            rerank_wait_ms=phase.rerank_wait_ms,
            first_token_ms=first_token_ms,
        )

    def _error_trace(
        self,
        *,
        trace_id: str,
        question: str,
        mode: TraceMode,
        effective_settings: AppSettings | None,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        error: str,
        answer: str | None,
        condensed_question: str | None,
        history_message_count: int,
        condense_note: str | None = None,
        narrow: PassageFilter | None = None,
        phase: RetrievalPhase | None = None,
        fitted: FittedPrompt | None = None,
        usage: tuple[int, int] | None = None,
    ) -> QueryTrace:
        """The trace of a question that failed (or was abandoned) before finishing:
        with what retrieval found (``phase``, ``fitted``) when it got that far, and the
        tokens already spent."""

        candidates = (
            build_trace_candidates(
                phase.fused_candidates,
                phase.scored_chunks,
                phase.context_chunks,
                fitted.context_chunks if fitted is not None else [],
                min_score=float(effective_settings.rerank_min_score),
                diversity_dropped_ids=phase.diversity_dropped_ids,
                relative_cut_ids=phase.relative_cut_ids,
                context_budget_dropped_ids=(
                    fitted.dropped_point_ids if fitted is not None else frozenset()
                ),
                leg_ranks=phase.leg_ranks,
            )
            if phase is not None and effective_settings is not None
            else []
        )
        return QueryTrace(
            trace_id=trace_id,
            created_at=time.time(),
            question=question,
            mode=mode,
            status="error",
            config=self._trace_config(effective_settings, filenames, tags, narrow)
            if effective_settings is not None
            else None,
            candidates=candidates,
            prompt_messages=None,
            answer=answer,
            cited_source_numbers=[],
            timings=None,
            error=error,
            condensed_question=condensed_question,
            condense_note=condense_note,
            history_message_count=history_message_count,
            request_id=current_request_id(),
            workspace=workspace_of(self._settings),
            weak_matches=phase.below_min_score if phase is not None else False,
            usage=usage,
            cost_usd=(
                usage_cost_usd(effective_settings, usage)
                if usage is not None and effective_settings is not None
                else None
            ),
        )

    def _finished_trace(
        self,
        *,
        trace_id: str,
        question: str,
        mode: TraceMode,
        status: TraceStatus,
        effective_settings: AppSettings,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        phase: RetrievalPhase,
        selected: Sequence[RerankedChunk],
        context_budget_dropped_ids: AbstractSet[str] = frozenset(),
        prompt_messages: list[ChatMessage] | None,
        answer: str,
        cited_source_numbers: list[int],
        timings: StageTimings,
        condensed_question: str | None,
        history_message_count: int,
        citation_retry_used: bool = False,
        usage: tuple[int, int] | None = None,
        condense_note: str | None = None,
        citation_backfilled: bool = False,
        unsupported_claims: int = 0,
        prompt_cached: bool = False,
        prompt_cut: bool = False,
        narrow: PassageFilter | None = None,
        answer_status: str | None = None,
        truncated: bool = False,
        dropped_history_messages: int = 0,
    ) -> QueryTrace:
        """The trace of a question that ran to an answer (or to "insufficient context")."""

        return QueryTrace(
            trace_id=trace_id,
            created_at=time.time(),
            question=question,
            mode=mode,
            status=status,
            config=self._trace_config(effective_settings, filenames, tags, narrow),
            candidates=build_trace_candidates(
                phase.fused_candidates,
                phase.scored_chunks,
                phase.context_chunks,
                selected,
                min_score=float(effective_settings.rerank_min_score),
                diversity_dropped_ids=phase.diversity_dropped_ids,
                relative_cut_ids=phase.relative_cut_ids,
                context_budget_dropped_ids=context_budget_dropped_ids,
                cited_source_numbers=frozenset(cited_source_numbers),
                leg_ranks=phase.leg_ranks,
            ),
            prompt_messages=prompt_messages,
            answer=answer,
            cited_source_numbers=cited_source_numbers,
            timings=timings_dict(timings),
            error=None,
            condensed_question=condensed_question,
            condense_note=condense_note,
            weak_matches=phase.below_min_score,
            history_message_count=history_message_count,
            query_variants=phase.query_variants,
            citation_retry_used=citation_retry_used,
            citation_backfilled=citation_backfilled,
            unsupported_claims=unsupported_claims,
            prompt_cached=prompt_cached,
            prompt_cut=prompt_cut,
            request_id=current_request_id(),
            workspace=workspace_of(self._settings),
            usage=usage,
            cost_usd=usage_cost_usd(effective_settings, usage),
            answer_status=answer_status,
            truncated=truncated,
            dropped_history_messages=dropped_history_messages,
        )

    def _truncate_history(
        self, history: Sequence[ChatMessage] | None, effective_settings: AppSettings
    ) -> list[ChatMessage]:
        """Keep only the most recent history messages, per server-side config.

        Earlier answers lose their ``[n]`` markers: those numbered the previous turn's
        sources, and this turn's prompt numbers its own from 1 again, so a model
        continuing the conversation could copy a stale ``[2]`` onto an unrelated source.
        An exchange whose answer was "not enough information" is left out (see
        ``_without_unanswered_exchanges``).
        """

        if not history:
            return []
        cap = int(effective_settings.conversation_max_history_messages)
        kept = _without_unanswered_exchanges(list(history)[-cap:] if cap > 0 else [])
        # An odd cap (or a client's odd-length history) could leave an answer first,
        # with its question cut away: the model read an answer to nothing, and
        # Anthropic's API refuses a conversation that doesn't open with the user.
        kept = _from_first_question(kept)
        return [
            {
                "role": "assistant",
                # A bare "[1]" answer still holds its place, so the turns keep alternating.
                "content": strip_citation_markers(message["content"]) or "(answer omitted)",
            }
            if message["role"] == "assistant"
            else message
            for message in kept
        ]

    def _will_condense(
        self,
        question: str,
        truncated_history: Sequence[ChatMessage],
        effective_settings: AppSettings,
        use_memo: bool = True,
    ) -> bool:
        """Whether ``_condense_query`` will actually make its LLM call.

        Split out purely so ``answer_stream`` can announce the ``condensing`` stage
        *before* the call blocks it, without restating the gate — two copies of this
        condition would drift the first time either side changed.
        """

        return (
            bool(truncated_history)
            and _condense_skip_reason(question, effective_settings) is None
            and self._remembered_rewrite(question, truncated_history, effective_settings, use_memo)
            is None
        )

    @staticmethod
    def _remembered_rewrite(
        question: str,
        truncated_history: Sequence[ChatMessage],
        settings: AppSettings,
        use_memo: bool,
    ) -> str | None:
        if not use_memo or int(settings.condense_cache_size) <= 0:
            return None
        return condense_memo.get(CondenseMemo.key(question, truncated_history, settings))

    def _condense_query(
        self,
        question: str,
        truncated_history: Sequence[ChatMessage],
        effective_settings: AppSettings,
        use_memo: bool = True,
    ) -> tuple[str, float, str | None, str | None]:
        """Rewrite a follow-up into a standalone retrieval query, when history exists.

        ``truncated_history`` is expected to already be capped (see
        ``_truncate_history``) — the same truncated list is what generation sees.
        Returns ``(retrieval_query, condense_ms, condensed_question, note)`` — the third
        element is the trace-visible rewrite, or None when condense was skipped
        (no history, or disabled) or failed/returned nothing usable (falls back to the
        raw question); ``note`` says why for the trace. A condense failure must never
        fail the whole request — retrieval just proceeds on the raw question, exactly
        as it did before this feature.
        """

        if not truncated_history:
            return question, 0.0, None, None
        skip_reason = _condense_skip_reason(question, effective_settings)
        if skip_reason is not None:
            return question, 0.0, None, skip_reason
        # The same call made before: its rewrite, with no LLM call (Regenerate, which
        # passes use_memo=False, asks again — the earlier rewrite may be the problem).
        remembered = self._remembered_rewrite(
            question, truncated_history, effective_settings, use_memo
        )
        if remembered is not None:
            return remembered, 0.0, remembered, "remembered"

        condense_start = time.monotonic()
        try:
            condensed = condense_question_detailed(
                question, truncated_history, self._generator, effective_settings
            )
        except GenerationError as exc:
            logger.warning("Condense step failed, falling back to raw question: %s", exc)
            return question, (time.monotonic() - condense_start) * 1000, None, "failed"
        condense_ms = (time.monotonic() - condense_start) * 1000
        if (
            condensed.text
            and not condensed.text.rstrip().endswith("?")
            and opens_with_a_refusal(condensed.text)
        ):
            # The model answered from the history ("The documents do not contain ...")
            # instead of rewriting; searched for, it matches nothing. A question that
            # merely uses the words ("What if a receipt does not contain a date?") is
            # a rewrite.
            condensed = replace(condensed, text="", note="rejected: not a question")
        if not condensed.text:
            logger.warning(
                "Condense step returned nothing usable (%s), falling back to raw question.",
                condensed.note,
            )
            return question, condense_ms, None, condensed.note
        condense_memo.put(
            CondenseMemo.key(question, truncated_history, effective_settings),
            condensed.text,
            int(effective_settings.condense_cache_size),
        )
        return condensed.text, condense_ms, condensed.text, condensed.note

    def retrieval_query(self, question: str, history: Sequence[ChatMessage] | None = None) -> str:
        """The query a question is retrieved with: a follow-up rewritten into a
        standalone one, as ``answer`` does (the eval harness scores this)."""

        settings = self._effective_settings(AnswerOverrides())
        truncated = self._truncate_history(history, settings)
        query, _, _, _ = self._condense_query(question.strip(), truncated, settings)
        return query

    def _stored_tag_variants(self, tags: Sequence[str], settings: AppSettings) -> list[str]:
        """The stored spellings of ``tags``, matched the way tags are de-duplicated.

        Tags are stored as first typed ("Finance") and de-duplicated case-insensitively,
        but the payload filter matches exactly, so asking for "finance" found nothing.
        Tags no document carries are kept as given: they still match nothing, where an
        empty list would have dropped the filter and searched every document.
        """

        wanted = {" ".join(tag.split()).casefold() for tag in tags}
        stored = self._repository.stored_tags(settings)
        return sorted(tag for tag in stored if tag.casefold() in wanted) or list(tags)

    def _expand_context(
        self,
        kept: Sequence[RerankedChunk],
        settings: AppSettings,
        for_generation: bool,
        narrow: PassageFilter | None = None,
    ) -> tuple[list[RerankedChunk], float]:
        """The kept passages with neighbour text for the prompt, and the time taken.

        Timed apart from rerank: neighbour expansion is a Qdrant round trip, not model
        scoring, and folding it into rerank_ms hid it from every latency breakdown.
        Only the max_context_chunks prefix reaches the prompt, so only it is expanded;
        the tail stays in the list unexpanded because traces report it as kept.
        """

        if not for_generation:
            return list(kept), 0.0
        start = time.monotonic()
        limit = int(settings.max_context_chunks)
        expanded = [
            # Neighbours stay inside the question's narrowing: a passage left out, or one
            # outside its pages or section, doesn't come back as neighbour text.
            *expand_with_neighbors(kept[:limit], self._repository, settings, narrow=narrow),
            *kept[limit:],
        ]
        return expanded, (time.monotonic() - start) * 1000

    def _expand_query(self, query: str, settings: AppSettings) -> tuple[list[str], float]:
        """QUERY_EXPANSION_ENABLED's variant phrasings of ``query`` and the time taken."""

        start = time.monotonic()
        variants = generate_query_variants(
            query.strip(), int(settings.query_expansion_count), self._generator, settings
        )
        return variants, (time.monotonic() - start) * 1000

    def _search(
        self,
        queries: Sequence[str],
        settings: AppSettings,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        tag_variants: Future[list[str]] | None,
        narrow: PassageFilter | None,
        seed: EmbeddedText | None = None,
        leg_ranks: LegRanks | None = None,
    ) -> tuple[list[RetrievedChunk], float, float]:
        """Every query embedded and hybrid-searched side by side, the result lists
        RRF-fused when there are several; the candidates, embed_ms and search_ms.
        With ``seed`` (a stored passage's own vectors) that is the one query, as is.
        ``leg_ranks`` is filled for the trace (``VectorRepository.hybrid_search``)."""

        embed_start = time.monotonic()
        query_embeddings = (
            [seed]
            if seed is not None
            else _map_concurrently(
                lambda query: embed_query(query, self._embedding_provider), queries
            )
        )
        embed_ms = (time.monotonic() - embed_start) * 1000
        check_dense_dimension(len(query_embeddings[0].dense), settings)
        if tag_variants is not None:
            tags = tag_variants.result()

        search_start = time.monotonic()
        # Passed only when set, so a repository double without the parameters still works.
        extra: dict[str, Any] = {"narrow": narrow} if narrow is not None else {}
        if leg_ranks is not None:
            extra["leg_ranks"] = leg_ranks
        result_lists = _map_concurrently(
            lambda embedding: self._repository.hybrid_search(
                settings, embedding, filenames, tags, **extra
            ),
            query_embeddings,
        )
        points = (
            result_lists[0]
            if len(result_lists) == 1
            else reciprocal_rank_fusion(
                result_lists,
                k=int(settings.rrf_k),
                limit=int(settings.fused_top_n),
            )
        )
        candidates = points_to_chunks(points)
        search_ms = (time.monotonic() - search_start) * 1000
        return candidates, embed_ms, search_ms

    def _retrieve_and_rerank(
        self,
        question: str,
        effective_settings: AppSettings,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None = None,
        should_stop: Callable[[], bool] | None = None,
        *,
        for_generation: bool = True,
        narrow: PassageFilter | None = None,
        also_search: str | None = None,
        expansion: tuple[list[str], float] | None = None,
        seed: EmbeddedText | None = None,
        traced: bool | None = None,
        collect_leg_ranks: bool = False,
    ) -> RetrievalPhase:
        """Run retrieve -> rerank -> neighbor-expand.

        ``traced`` says whether a trace will show this retrieval (default: when it
        feeds an answer): only then are each candidate's leg ranks collected, which
        can cost Qdrant one more query (TRACE_LEG_RANKS).

        ``expansion`` is ``_expand_query``'s result when the caller already ran it (the
        stream does, so it can wait for the local model with a frame first).

        ``also_search`` is a second query searched alongside ``question`` and fused
        with it (the raw follow-up next to its condensed rewrite); the rerank scores
        against ``question`` alone.

        ``for_generation=False`` (search without an answer) skips the two stages that
        only serve the prompt: LLM query expansion and neighbour expansion.

        Embedding and search are separate steps so each can be timed for
        observability.
        ``should_stop`` is checked before the rerank, the costliest stage, and raises
        ``QuestionAbandoned`` when the client has gone.
        """

        normalized_query = question.strip()
        if not normalized_query:
            raise RetrievalError("Query text is required.")
        self._repository.ensure_ready(effective_settings)
        # Looked up alongside embedding (and, for a sync answer, query expansion), which
        # don't depend on it; only the search below needs the stored spellings.
        tag_variants = (
            submit_in_context(
                _TAG_LOOKUP_EXECUTOR,
                functools.partial(self._stored_tag_variants, tags, effective_settings),
            )
            if tags
            else None
        )

        # Multi-query expansion (opt-in): rewrite into variant phrasings, retrieve each,
        # and RRF-fuse the result lists with the same pinned rrf_k before one rerank.
        # Left un-timed (query_expansion_ms stays 0.0) when disabled, like condense_ms.
        variants: list[str] = []
        query_expansion_ms = 0.0
        if expansion is not None:
            variants, query_expansion_ms = expansion
        elif effective_settings.query_expansion_enabled and for_generation:
            variants, query_expansion_ms = self._expand_query(normalized_query, effective_settings)
        queries = [normalized_query, *variants]
        # The raw follow-up searched next to its rewrite: a rewrite that lost or changed
        # what the user asked no longer takes the raw question's matches with it.
        extra_query = (also_search or "").strip()
        if extra_query and extra_query.casefold() not in {q.casefold() for q in queries}:
            queries.insert(1, extra_query)

        # Which leg found each candidate, for its trace (none is kept without one).
        traced = for_generation if traced is None else traced
        leg_ranks: LegRanks | None = (
            {} if (traced and self._trace_store is not None) or collect_leg_ranks else None
        )
        candidates, embed_ms, search_ms = self._search(
            queries, effective_settings, filenames, tags, tag_variants, narrow, seed, leg_ranks
        )
        if should_stop is not None and should_stop():
            raise QuestionAbandoned

        rerank_start = time.monotonic()
        outcome = rerank_candidates_detailed(
            query=question,
            candidates=candidates,
            reranker=self._reranker,
            settings=effective_settings,
        )
        # Picked passages (chunk_ids) are the context exactly: their neighbours were not
        # picked, and the question was asked about these alone.
        picked_only = for_generation and narrow is not None and bool(narrow.chunk_ids)
        selection = _select_passages(
            outcome, effective_settings, for_generation=for_generation, picked_only=picked_only
        )
        diverse_outcome = selection.outcome
        rerank_ms = (time.monotonic() - rerank_start) * 1000
        # Read on this thread right after the call: the reranker keeps it per thread.
        rerank_wait_ms = float(getattr(self._reranker, "last_wait_ms", 0.0))
        expanded, context_expansion_ms = self._expand_context(
            diverse_outcome.kept,
            effective_settings,
            for_generation and not picked_only,
            narrow=narrow,
        )

        return RetrievalPhase(
            context_chunks=expanded,
            fused_candidates=candidates,
            scored_chunks=outcome.scored,
            embed_ms=embed_ms,
            search_ms=search_ms,
            rerank_ms=rerank_ms,
            rerank_wait_ms=rerank_wait_ms,
            query_expansion_ms=query_expansion_ms,
            context_expansion_ms=context_expansion_ms,
            diversity_dropped_ids=frozenset(selection.diversity_dropped),
            relative_cut_ids=frozenset(selection.relative_cut_ids),
            query_variants=queries[1:],
            below_min_score=selection.below_min_score,
            leg_ranks=leg_ranks or {},
        )

    def retrieve(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        *,
        for_generation: bool = True,
        narrow: PassageFilter | None = None,
        also_search: str | None = None,
        explain: bool = False,
    ) -> RetrievalPhase:
        """Retrieval exactly as ``answer`` runs it — query expansion, hybrid search,
        rerank, diversity filter, neighbour expansion — without generating. The eval
        harness scores this, so it measures the pipeline users actually get.
        ``for_generation=False`` is ``POST /search``: the same hybrid search, RRF and
        rerank, minus the LLM query expansion and the neighbour expansion that only
        feed a prompt. ``explain`` (``POST /search``'s): each candidate's rank on the
        dense and the BM25 search is recorded too (``phase.leg_ranks``)."""

        effective_settings = self._effective_settings(overrides or AnswerOverrides())
        if explain:
            # With Qdrant's own fusion the leg ranks cost one more query; asked for here.
            effective_settings = effective_settings.model_copy(update={"trace_leg_ranks": True})
        return self._retrieve_and_rerank(
            question,
            effective_settings,
            filenames,
            tags,
            for_generation=for_generation,
            narrow=narrow,
            also_search=also_search,
            # Search, re-check and the eval harness keep no trace of it.
            traced=False,
            collect_leg_ranks=explain,
        )

    def rerun_retrieval(
        self, trace: QueryTrace, settings_update: Mapping[str, Any]
    ) -> tuple[RetrievalPhase, QueryTrace]:
        """``trace``'s question searched again — what it searched (a follow-up's
        rewrite, next to the question as asked), in its scope and narrowing — under the
        settings it was asked with (``_RERUN_FROM_TRACE``; the server's where the trace
        has none) with ``settings_update`` (ranking settings only), and the
        trace that retrieval would have left: ``POST /traces/{id}/rerun``. No LLM call:
        query expansion is left out (its variants came from the model)."""

        config = trace.config
        settings = self._effective_settings(AnswerOverrides()).model_copy(
            update={
                **_recorded_settings(config),
                **settings_update,
                "query_expansion_enabled": False,
            }
        )
        record = config.narrowing if config is not None else None
        narrow = filter_from_record(record) if record else None
        query = trace.condensed_question or trace.question
        phase = self._retrieve_and_rerank(
            query,
            settings,
            config.filenames if config is not None else None,
            config.tags if config is not None else None,
            for_generation=True,
            narrow=narrow,
            also_search=trace.question if query != trace.question else None,
        )
        # What reached the prompt, fitted as the answer was (without its history, which
        # a trace doesn't keep): passages the window left out are left out again.
        fitted = fit_prompt_to_context_window(
            query,
            phase.context_chunks[: int(settings.max_context_chunks)],
            None,
            settings,
            weak_matches=phase.below_min_score,
        )
        selected = fitted.context_chunks
        rerun = dataclasses.replace(
            trace,
            config=self._trace_config(
                settings,
                config.filenames if config is not None else None,
                config.tags if config is not None else None,
                narrow,
            ),
            candidates=build_trace_candidates(
                phase.fused_candidates,
                phase.scored_chunks,
                phase.context_chunks,
                selected,
                min_score=float(settings.rerank_min_score),
                diversity_dropped_ids=phase.diversity_dropped_ids,
                relative_cut_ids=phase.relative_cut_ids,
                context_budget_dropped_ids=fitted.dropped_point_ids,
                leg_ranks=phase.leg_ranks,
            ),
            answer=None,
            weak_matches=phase.below_min_score,
        )
        return phase, rerun

    def retrieve_similar(
        self,
        chunk_id: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        *,
        narrow: PassageFilter | None = None,
    ) -> RetrievalPhase | None:
        """``POST /search`` from a passage ("more like this"), or None when no passage
        has ``chunk_id``.

        The same hybrid search, RRF and rerank as a search: both legs are queried with
        the passage's stored dense and sparse vectors, and the cross-encoder scores
        each candidate against the passage as it was embedded. The passage itself is
        left out; the request's narrowing still applies.
        """

        effective_settings = self._effective_settings(overrides or AnswerOverrides())
        seed = self._repository.seed_passage(effective_settings, chunk_id)
        if seed is None:
            return None
        without_seed = dataclasses.replace(
            narrow or PassageFilter(),
            exclude_chunk_ids=tuple(
                sorted({*(narrow.exclude_chunk_ids if narrow else ()), chunk_id})
            ),
        )
        return self._retrieve_and_rerank(
            contextual_text(seed.filename, seed.section, seed.text),
            effective_settings,
            filenames,
            tags,
            for_generation=False,
            narrow=without_seed,
            seed=seed.embedding,
        )

    def _cache_key(
        self,
        question: str,
        overrides: AnswerOverrides | None,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        history: Sequence[ChatMessage] | None,
    ) -> CacheKey | None:
        """The answer-cache key for this request, or None when it must not be cached."""

        if self._answer_cache is None or history or not question.strip():
            return None
        try:
            overrides = overrides or AnswerOverrides()
            settings = self._effective_settings(overrides)
            return answer_cache_key(question, settings, filenames, tags, overrides.narrow)
        except (RetrievalConfigError, GenerationConfigError):
            # The uncached path raises it with its proper handling, error trace
            # included (a provider without its key failed here, before the trace).
            return None

    def answer(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
        *,
        tags: Sequence[str] | None = None,
        use_cache: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> GroundedAnswer:
        """``_answer_uncached``, served from the answer cache when it has this question.

        ``use_cache=False`` (Regenerate) skips the lookup but still stores the new
        answer, so the next plain ask gets the regenerated one. ``should_stop`` (the
        client has gone) is checked between stages, as on the stream: ``QuestionAbandoned``
        is raised instead of finishing work nobody will read.
        """

        started = time.monotonic()
        key = self._cache_key(question, overrides, filenames, tags, history)
        lead: threading.Event | None = None
        if key is not None and use_cache and self._answer_cache is not None:
            hit = self._answer_cache.get(key)
            if hit is None:
                joined = drain(lead_or_join(self._answer_cache, key, should_stop))
                if joined is None:
                    raise QuestionAbandoned
                if isinstance(joined, CachedAnswer):
                    hit = joined
                elif joined is not None:
                    lead = joined
            if hit is not None:
                self._log_hit(hit, question, "sync", filenames, tags, overrides)
                return hit.as_grounded(_cached_timings(started))
        try:
            grounded = self._answer_uncached(
                question, overrides, filenames, history, tags, should_stop, use_memo=use_cache
            )
            # A cut-off answer isn't kept: asked again, it may come out whole.
            if (
                key is not None
                and self._answer_cache is not None
                and not was_degraded()
                and not grounded.truncated
                and not grounded.prompt_cut
            ):
                self._answer_cache.put(key, CachedAnswer.from_grounded(grounded))
        finally:
            # After the store, so followers find the answer; on a failure they find
            # none and each asks itself.
            if lead is not None and key is not None and self._answer_cache is not None:
                self._answer_cache.inflight.finish(key, lead)
        return grounded

    def answer_stream(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
        *,
        tags: Sequence[str] | None = None,
        use_cache: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[StreamEvent]:
        """``_answer_stream_uncached``, replayed from the answer cache on a hit.

        ``should_stop`` (the client has gone) is polled between stages, so an
        abandoned question stops before the rerank or the LLM call instead of
        finishing work nobody will read.
        """

        started = time.monotonic()
        yield from cached_run(
            self._answer_cache,
            self._cache_key(question, overrides, filenames, tags, history),
            use_cache=use_cache,
            should_stop=should_stop,
            replay=lambda hit: replay_cached(
                hit,
                started,
                top_rerank_score=hit.top_rerank_score,
                # In the cache key: the bar it was answered under is today's.
                rerank_min_score=float(self._settings.rerank_min_score),
                weak_matches=hit.weak_matches,
            ),
            run=lambda: self._answer_stream_uncached(
                question, overrides, filenames, history, tags, should_stop, use_memo=use_cache
            ),
            # A cut-off answer isn't kept: asked again, it may come out whole.
            # Nor one whose prompt the provider cut short (its grounding rules gone).
            keep=lambda done: not done.get("truncated") and not done.get("prompt_cut"),
            on_hit=lambda hit: self._log_hit(hit, question, "stream", filenames, tags, overrides),
        )

    def continuable(
        self, trace_id: str, trace: QueryTrace | None, settings: AppSettings
    ) -> ContinuePlan:
        """What continuing the cut-off answer of trace ``trace_id`` (``trace``, as the
        trace store has it) needs: its prompt, its answer so far and its passages in
        source order. ``ContinueRefused`` (with the HTTP status to give) when it can't
        be continued — checked before the question slot and the 200."""

        if self._trace_store is None:
            raise ContinueRefused(
                404, "Traces are off (TRACE_ENABLED), so answers can't be continued."
            )
        if trace is None or trace.workspace != workspace_of(settings):
            raise ContinueRefused(404, f"No trace found with id '{trace_id}'.")
        if not trace.truncated or not trace.answer or not trace.prompt_messages:
            raise ContinueRefused(409, "That answer wasn't cut off; there is nothing to continue.")
        prompt_sources = sorted(
            (c for c in trace.candidates if c.source_number is not None),
            key=lambda c: c.source_number or 0,
        )
        wanted: dict[str, list[str]] = {}
        for candidate in prompt_sources:
            wanted.setdefault(candidate.filename, []).append(candidate.chunk_id)
        payloads: dict[str, dict[str, object]] = {}
        for filename, chunk_ids in wanted.items():
            for payload in self._repository.chunks_for_filename(
                settings, filename, PassageFilter(chunk_ids=tuple(chunk_ids))
            ):
                payloads[str(payload.get("chunk_id"))] = payload
        chunks: list[RerankedChunk] = []
        for candidate in prompt_sources:
            found = payloads.get(candidate.chunk_id)
            if found is None:
                # Numbered sources the answer cites by number: one gone shifts them all.
                raise ContinueRefused(
                    409,
                    "A passage that answer was written from has changed or been deleted "
                    "since; ask the question again instead.",
                )
            chunks.append(
                RerankedChunk(
                    point_id=candidate.point_id,
                    filename=candidate.filename,
                    page=candidate.page,
                    section=candidate.section,
                    chunk_id=candidate.chunk_id,
                    text=str(found.get("text", "")),
                    retrieval_score=candidate.retrieval_score,
                    rerank_score=candidate.rerank_score or 0.0,
                    page_end=_int_or_none(found.get("page_end")),
                )
            )
        return ContinuePlan(trace, chunks)

    def continue_stream(
        self,
        plan: ContinuePlan,
        settings: AppSettings,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[StreamEvent]:
        """The rest of a cut-off answer: the model is shown its prompt and its answer so
        far, asked to go on, and the two are joined. Citations and spans are worked out
        again over the whole answer (the sources are the same, in the same order). A new
        trace is stored; nothing is cached."""

        trace = plan.trace
        assert trace.answer is not None and trace.prompt_messages is not None
        started = time.monotonic()
        config = trace.config
        if config is not None:
            settings = settings.model_copy(
                update={
                    "llm_provider": config.llm_provider,
                    "llm_temperature": config.llm_temperature,
                }
            )
        begin_usage()
        base = base_prompt(trace.prompt_messages)
        messages = continuation_messages(base, trace.answer, settings)
        all_sources = source_citations(plan.chunks)
        new_trace_id = str(uuid.uuid4())
        yield {
            "type": "sources",
            "sources": [_citation_dict(source) for source in all_sources],
            "trace_id": new_trace_id,
        }
        slot = yield from wait_for_model(settings, should_stop)
        try:
            yield {
                "type": "stage",
                "stage": "reading",
                "prompt_tokens": estimate_prompt_tokens(messages),
            }
            generate_start = time.monotonic()
            reset_finish_reason()
            parts: list[str] = []
            for delta in self._generator.stream(messages, settings):
                parts.append(delta)
                yield {"type": "delta", "text": delta}
            generate_ms = (time.monotonic() - generate_start) * 1000
            truncated = stopped_at_token_limit()
            answer = join_continuation(
                trace.answer,
                drop_restart(trace.answer, "".join(parts)),
                prefilled=messages[-1]["role"] == "assistant",
            )
            outcome, cited = finish_answer(
                base,
                answer,
                plan.chunks,
                self._generator,
                settings,
                weak_matches=trace.weak_matches,
            )
        finally:
            slot.release()
        status = answer_status(outcome.answer, len(cited), len(plan.chunks))
        usage = usage_totals()
        cited_numbers = {source.source_number for source in cited}
        timings = StageTimings(
            embed_ms=0.0,
            search_ms=0.0,
            rerank_ms=0.0,
            generate_ms=generate_ms,
            total_ms=(time.monotonic() - started) * 1000,
        )
        if self._trace_store is not None:
            self._trace_store.add(
                replace(
                    trace,
                    trace_id=new_trace_id,
                    created_at=time.time(),
                    candidates=[
                        replace(c, cited=c.source_number in cited_numbers) for c in trace.candidates
                    ],
                    prompt_messages=messages,
                    answer=outcome.answer,
                    cited_source_numbers=sorted(cited_numbers),
                    timings=timings_dict(timings),
                    status=trace_status(status),
                    answer_status=status,
                    truncated=truncated,
                    citation_retry_used=outcome.retry_used,
                    citation_backfilled=outcome.backfilled,
                    unsupported_claims=outcome.unsupported_claims,
                    prompt_cached=False,
                    prompt_cut=was_prompt_cut(),
                    usage=usage,
                    cost_usd=usage_cost_usd(settings, usage),
                    error=None,
                )
            )
        yield {
            "type": "done",
            "answer": outcome.answer,
            "sources": [_citation_dict(source) for source in cited],
            "timings": timings_dict(timings),
            "trace_id": new_trace_id,
            "answer_status": status,
            "top_rerank_score": max(
                (c.rerank_score for c in trace.candidates if c.rerank_score is not None),
                default=None,
            ),
            "rerank_min_score": float(settings.rerank_min_score),
            "weak_matches": trace.weak_matches,
            "truncated": truncated,
            "citation_backfilled": outcome.backfilled,
            "unsupported_claims": outcome.unsupported_claims,
            "prompt_cut": was_prompt_cut(),
            "usage": usage_dict(usage, usage_cost_usd(settings, usage)),
        }

    def _store_error_trace(self, run: AnswerRun, error: str) -> None:
        """Record ``run`` as failed (or abandoned), with the answer streamed so far."""

        if self._trace_store is None or run.trace_id is None:
            return
        self._trace_store.add(
            self._error_trace(
                trace_id=run.trace_id,
                narrow=run.narrow,
                question=run.question,
                mode=run.mode,
                effective_settings=run.settings,
                filenames=run.filenames,
                tags=run.tags,
                error=error,
                answer="".join(run.parts).strip() or None,
                condensed_question=run.condensed_question,
                condense_note=run.condense_note,
                history_message_count=len(run.history),
                phase=run.phase,
                fitted=run.fitted,
                # Begun with the run's settings (``begin_usage`` comes first).
                usage=(
                    usage_totals()
                    if run.settings is not None and run.thread == threading.get_ident()
                    else None
                ),
            )
        )

    def _store_finished_trace(
        self, run: AnswerRun, settings: AppSettings, phase: RetrievalPhase, **fields: Any
    ) -> None:
        """Record ``run`` as finished; ``fields`` are what only this outcome has."""

        if self._trace_store is None or run.trace_id is None:
            return
        self._trace_store.add(
            self._finished_trace(
                trace_id=run.trace_id,
                narrow=run.narrow,
                question=run.question,
                mode=run.mode,
                effective_settings=settings,
                filenames=run.filenames,
                tags=run.tags,
                phase=phase,
                condensed_question=run.condensed_question,
                condense_note=run.condense_note,
                history_message_count=len(run.history),
                **fields,
            )
        )

    def _answer_uncached(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
        tags: Sequence[str] | None = None,
        should_stop: Callable[[], bool] | None = None,
        *,
        use_memo: bool = True,
    ) -> GroundedAnswer:
        """Retrieve, rerank, and generate a grounded, cited answer.

        ``overrides`` are folded into a single ``model_copy`` of the shared settings
        (``_effective_settings``) — the singleton is never mutated, so concurrent
        requests with different overrides stay isolated. The dense/sparse retrieval
        limits, ``fused_top_n``, and ``rrf_k`` always come from the base settings
        (embeddings and hybrid fusion stay fixed regardless). ``filenames``, when given,
        restricts retrieval to those documents. ``history``, when given, drives one
        extra condense call that rewrites ``question`` into a standalone retrieval
        query — the server itself stores no conversation state.
        """

        run = self._start_run(question, overrides, filenames, tags, "sync")
        run.use_memo = use_memo
        # Held from the answer through its citation retry and follow-ups, as the stream
        # does, so another question's call can't slip in between them at a local model.
        slot: GenerationSlot | None = None
        # Every LLM call this question makes counts: the condense, query expansion,
        # the answer, its citation retry and the follow-ups.
        begin_usage()

        try:
            effective_settings = run.settings = self._effective_settings(run.overrides)
            run.history = self._truncate_history(history, effective_settings)
            # The condense and query-expansion calls wait their turn at a local model
            # like the answer does, and give the place back for retrieval (as the
            # stream does): sent unguarded, they queued inside Ollama with their
            # timeout running behind another question's generation.
            condense_slot: GenerationSlot | None = None
            try:
                if self._will_condense(question, run.history, effective_settings, run.use_memo):
                    condense_slot = drain(wait_for_model(effective_settings, should_stop))
                retrieval_query, condense_ms, run.condensed_question, run.condense_note = (
                    self._condense_query(question, run.history, effective_settings, run.use_memo)
                )
            finally:
                if condense_slot is not None:
                    condense_slot.release()
            expansion: tuple[list[str], float] | None = None
            if effective_settings.query_expansion_enabled and retrieval_query.strip():
                expansion_slot = drain(wait_for_model(effective_settings, should_stop))
                try:
                    expansion = self._expand_query(retrieval_query, effective_settings)
                finally:
                    expansion_slot.release()
            phase = self._retrieve_and_rerank(
                retrieval_query,
                effective_settings,
                filenames,
                tags,
                should_stop,
                narrow=run.narrow,
                also_search=question if run.condensed_question else None,
                expansion=expansion,
            )
            prompt_question, fitted = self._fit_prompt(run, effective_settings, phase)
            selected = fitted.context_chunks
            generator = self._answer_generator(run, effective_settings)

            # Nothing to answer from: the canned answer makes no LLM call, so it must not
            # queue behind another question's generation at a local model — nor must an
            # answer the prompt cache already holds.
            if selected and not _reply_known(
                generator,
                answer_messages(
                    prompt_question.strip(),
                    list(selected[: int(effective_settings.max_context_chunks)]),
                    fitted.history or None,
                    effective_settings,
                    weak_matches=phase.below_min_score,
                ),
                effective_settings,
            ):
                slot = drain(wait_for_model(effective_settings, should_stop))
                if should_stop is not None and should_stop():
                    raise QuestionAbandoned
            generate_start = time.monotonic()
            reset_finish_reason()
            grounded = generate_grounded_answer(
                query=prompt_question,
                context_chunks=selected,
                generator=generator,
                settings=effective_settings,
                history=fitted.history or None,
                weak_matches=phase.below_min_score,
            )
            # generate_grounded_answer runs the zero-citation retry internally, so the
            # block just timed covers two LLM calls when it fired. Split it back out so
            # generate_ms means the same thing here as it does on the streaming path.
            generate_ms = (time.monotonic() - generate_start) * 1000 - grounded.citation_retry_ms
            # The last call made the answer: the citation retry's when it fired.
            truncated = stopped_at_token_limit()
            timings = self._phase_timings(
                phase,
                total_start=run.started,
                condense_ms=condense_ms,
                generate_ms=generate_ms,
                citation_retry_ms=grounded.citation_retry_ms,
            )
        except Exception as exc:
            if slot is not None:
                slot.release()
            # Traces are readable with the read-only key: the same rule as a response
            # body, so no internal detail.
            self._store_error_trace(run, public_error_message(exc))
            raise

        # The answer's usage: everything up to it (condense, expansion, the answer and
        # its citation retry), as the stream's `done` reports it. Follow-up tokens are
        # still counted in the token and cost metrics.
        usage = usage_totals()
        cost_usd = usage_cost_usd(effective_settings, usage)
        try:
            status = answer_status(grounded.answer, len(grounded.sources), len(selected))
            wants_followups = effective_settings.followups_enabled and status == "answered"
            followups = (
                generate_followups(
                    question.strip(),
                    grounded.answer,
                    _cited_chunks(grounded.sources, selected),
                    generator,
                    effective_settings,
                )
                if wants_followups
                else []
            )
        finally:
            if slot is not None:
                slot.release()
        if wants_followups:
            # The response waited for them (even when none came back), so the total does.
            timings = replace(timings, total_ms=(time.monotonic() - run.started) * 1000)
        # The prompt is the one captured from generation, not rebuilt here: a second
        # build_grounded_messages call would agree today but is a re-derivation that
        # can drift, and it could never reproduce the citation-retry continuation that
        # actually produced the answer when grounded.citation_retry_used is set.
        self._store_finished_trace(
            run,
            effective_settings,
            phase,
            status=trace_status(status),
            selected=selected,
            context_budget_dropped_ids=fitted.dropped_point_ids,
            prompt_messages=grounded.prompt_messages or None,
            answer=grounded.answer,
            cited_source_numbers=[source.source_number for source in grounded.sources],
            timings=timings,
            citation_retry_used=grounded.citation_retry_used,
            citation_backfilled=grounded.citation_backfilled,
            unsupported_claims=grounded.unsupported_claims,
            prompt_cached=self._prompt_cached(run),
            prompt_cut=was_prompt_cut(),
            usage=usage,
            answer_status=status,
            truncated=truncated,
            dropped_history_messages=fitted.dropped_history_messages,
        )
        return replace(
            grounded,
            timings=timings,
            trace_id=run.trace_id,
            answer_status=status,
            top_rerank_score=_top_rerank_score(phase),
            weak_matches=phase.below_min_score,
            truncated=truncated,
            passages_left_out=_picked_left_out(run.narrow, selected),
            followups=followups,
            usage=usage,
            cost_usd=cost_usd,
            prompt_cached=self._prompt_cached(run),
            prompt_cut=was_prompt_cut(),
        )

    def _log_hit(
        self,
        hit: CachedAnswer,
        question: str,
        mode: TraceMode,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        overrides: AnswerOverrides | None,
    ) -> None:
        log_served(
            self._trace_store,
            served_query(
                hit,
                question=question,
                mode=mode,
                workspace=workspace_of(self._settings),
                filenames=filenames,
                tags=tags,
                narrowing=narrowing_record((overrides or AnswerOverrides()).narrow),
            ),
        )

    def _answer_generator(self, run: AnswerRun, settings: AppSettings) -> ChatGenerator:
        """The generator for ``run``'s answer, citation retry and follow-ups: through the
        prompt cache (PROMPT_CACHE_SIZE) unless it is off or this is a Regenerate."""

        cache = shared_prompt_cache(settings) if run.use_memo else None
        run.generator = (
            CachingGenerator(self._generator, cache) if cache is not None else self._generator
        )
        return run.generator

    @staticmethod
    def _prompt_cached(run: AnswerRun) -> bool:
        return isinstance(run.generator, CachingGenerator) and run.generator.answer_reused

    def _start_run(
        self,
        question: str,
        overrides: AnswerOverrides | None,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        mode: TraceMode,
    ) -> AnswerRun:
        return AnswerRun(
            question=question,
            overrides=overrides or AnswerOverrides(),
            filenames=filenames,
            tags=tags,
            mode=mode,
            trace_id=str(uuid.uuid4()) if self._trace_store is not None else None,
            started=time.monotonic(),
        )

    def _fit_prompt(
        self, run: AnswerRun, settings: AppSettings, phase: RetrievalPhase
    ) -> tuple[str, FittedPrompt]:
        """The question as the answering prompt puts it, and what of the passages and
        history fits the window."""

        prompt_question = _with_interpretation(run.question.strip(), run.condensed_question)
        fitted = fit_prompt_to_context_window(
            prompt_question,
            phase.context_chunks[: int(settings.max_context_chunks)],
            _clip_earlier_answers(run.history, settings),
            settings,
            weak_matches=phase.below_min_score,
        )
        run.phase, run.fitted = phase, fitted
        return prompt_question, fitted

    def _answer_stream_uncached(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
        tags: Sequence[str] | None = None,
        should_stop: Callable[[], bool] | None = None,
        *,
        use_memo: bool = True,
    ) -> Generator[StreamEvent]:
        """Stream a grounded answer as SSE-ready events: sources, deltas, then done.

        Sources are emitted immediately after rerank (before the LLM call starts) so
        the UI can render citations while the answer is still generating. ``done``
        carries the final answer, the citation-filtered source list, and timings.
        ``history``, when given, drives the same condense step as ``answer``.

        In three steps — ``_stream_retrieve`` (condense, expansion, retrieval),
        ``_stream_generate`` (the streamed answer), ``_stream_finish`` (citations, the
        trace, ``done``, follow-ups) — sharing one ``AnswerRun``. Only this method
        records a trace for an error or an early end, and releases the model slot.
        """

        run = self._start_run(question, overrides, filenames, tags, "stream")
        run.use_memo = use_memo
        try:
            # Everything this answer calls the LLM for, from the condense on (the
            # follow-ups come after `done` and aren't in its usage).
            begin_usage()
            retrieved = yield from self._stream_retrieve(run, history, should_stop)
            settings, phase, condense_ms, prompt_question, fitted = retrieved
            if not fitted.context_chunks:
                yield from self._stream_no_context(run, settings, phase, condense_ms)
                return
            generated = yield from self._stream_generate(
                run, settings, phase, prompt_question, fitted, should_stop
            )
            if generated is None:
                return  # the finally below records the trace as closed early
            yield from self._stream_finish(
                run, settings, phase, fitted, condense_ms, generated, should_stop
            )
        except QuestionAbandoned:
            return  # the finally below records the trace as closed early
        except Exception as exc:
            if not run.finalized:
                self._store_error_trace(run, public_error_message(exc))
                run.finalized = True
            raise
        finally:
            if not run.finalized:
                # Reached via GeneratorExit (the client disconnected, or the generator
                # was otherwise closed mid-stream without completing or raising an
                # `Exception`) — `finally` re-raises it automatically once this block
                # returns, so no explicit `raise` is needed here — or via an early
                # return once `should_stop` reported the client gone.
                self._store_error_trace(run, "Stream closed before completion.")
                run.finalized = True
            if run.slot is not None:
                run.slot.release()

    def _stream_retrieve(
        self,
        run: AnswerRun,
        history: Sequence[ChatMessage] | None,
        should_stop: Callable[[], bool] | None,
    ) -> Generator[StreamEvent, None, tuple[AppSettings, RetrievalPhase, float, str, FittedPrompt]]:
        """Settings, history, the condense call, opt-in query expansion and retrieval,
        with their stage frames; what the answer is written from."""

        effective_settings = run.settings = self._effective_settings(run.overrides)
        run.history = self._truncate_history(history, effective_settings)
        # Announced before the call, and only when it will actually happen — an
        # unconditional frame would report a stage that never ran for the majority of
        # questions, which is worse than no frame. The rewrite is an LLM call: it waits
        # its turn at a local model like the answer does (announced, and given up if
        # the reader leaves), and gives the place back for retrieval.
        condense_slot: GenerationSlot | None = None
        try:
            if self._will_condense(run.question, run.history, effective_settings, run.use_memo):
                condense_slot = yield from wait_for_model(effective_settings, should_stop)
                yield {"type": "stage", "stage": "condensing"}
            retrieval_query, condense_ms, run.condensed_question, run.condense_note = (
                self._condense_query(run.question, run.history, effective_settings, run.use_memo)
            )
        finally:
            if condense_slot is not None:
                condense_slot.release()
        # Query expansion (opt-in) is an LLM call too: it waits its turn at a local
        # model like the condense call, and gives the place back for retrieval.
        expansion: tuple[list[str], float] | None = None
        if effective_settings.query_expansion_enabled and retrieval_query.strip():
            expansion_slot = yield from wait_for_model(effective_settings, should_stop)
            try:
                yield {"type": "stage", "stage": "searching"}
                expansion = self._expand_query(retrieval_query, effective_settings)
            finally:
                expansion_slot.release()
        else:
            yield {"type": "stage", "stage": "searching"}
        phase = self._retrieve_and_rerank(
            retrieval_query,
            effective_settings,
            run.filenames,
            run.tags,
            should_stop,
            narrow=run.narrow,
            also_search=run.question if run.condensed_question else None,
            expansion=expansion,
        )
        prompt_question, fitted = self._fit_prompt(run, effective_settings, phase)
        return effective_settings, phase, condense_ms, prompt_question, fitted

    def _stream_no_context(
        self, run: AnswerRun, settings: AppSettings, phase: RetrievalPhase, condense_ms: float
    ) -> Iterator[StreamEvent]:
        """Nothing to answer from: the canned answer, with no LLM call."""

        yield {"type": "sources", "sources": [], "trace_id": run.trace_id}
        timings = self._phase_timings(phase, total_start=run.started, condense_ms=condense_ms)
        # The condense and query-expansion calls were still made.
        usage = usage_totals()
        self._store_finished_trace(
            run,
            settings,
            phase,
            status="insufficient_context",
            selected=[],
            prompt_messages=None,
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            cited_source_numbers=[],
            timings=timings,
            usage=usage,
            answer_status="no_context",
        )
        run.finalized = True
        yield {
            "type": "done",
            "answer": INSUFFICIENT_CONTEXT_ANSWER,
            "sources": [],
            "timings": timings_dict(timings),
            "trace_id": run.trace_id,
            "answer_status": "no_context",
            "top_rerank_score": _top_rerank_score(phase),
            "rerank_min_score": float(settings.rerank_min_score),
            "passages_left_out": _picked_left_out(run.narrow, []),
            "usage": usage_dict(usage, usage_cost_usd(settings, usage)),
        }

    def _stream_generate(
        self,
        run: AnswerRun,
        settings: AppSettings,
        phase: RetrievalPhase,
        prompt_question: str,
        fitted: FittedPrompt,
        should_stop: Callable[[], bool] | None,
    ) -> Generator[StreamEvent, None, _Streamed | None]:
        """The sources frame, the wait for the model, and the streamed answer; None when
        the reader left before generation."""

        selected = fitted.context_chunks
        all_sources = source_citations(selected)
        yield {
            "type": "sources",
            "sources": [_citation_dict(source) for source in all_sources],
            "trace_id": run.trace_id,
        }
        messages = answer_messages(
            prompt_question,
            selected,
            fitted.history or None,
            settings,
            weak_matches=phase.below_min_score,
        )
        if should_stop is not None and should_stop():
            return None
        generator = self._answer_generator(run, settings)
        # An answer the prompt cache holds needs neither the model nor a wait for it.
        if not _reply_known(generator, messages, settings):
            # Held until the answer, its citation retry and follow-ups are done, so
            # another question's call can't slip in between them; released by the
            # caller. Waiting inside Ollama ran this request's timeout; waiting here
            # doesn't.
            run.slot = yield from wait_for_model(settings, should_stop)
            reading: StageEvent = {
                "type": "stage",
                "stage": "reading",
                "prompt_tokens": estimate_prompt_tokens(messages),
            }
            eta_ms = expected_prefill_ms(messages, settings)
            if eta_ms is not None:
                reading["eta_ms"] = round(eta_ms)
            yield reading
        generate_start = time.monotonic()
        first_token_ms = 0.0
        reset_finish_reason()
        for delta in generator.stream(messages, settings):
            if not run.parts:
                first_token_ms = (time.monotonic() - run.started) * 1000
            run.parts.append(delta)
            yield {"type": "delta", "text": delta}
        generate_ms = (time.monotonic() - generate_start) * 1000
        answer = "".join(run.parts).strip()
        if not answer:
            raise GenerationError("Generation provider returned an empty answer.")
        return _Streamed(messages, all_sources, answer, generate_ms, first_token_ms)

    def _stream_finish(
        self,
        run: AnswerRun,
        settings: AppSettings,
        phase: RetrievalPhase,
        fitted: FittedPrompt,
        condense_ms: float,
        generated: _Streamed,
        should_stop: Callable[[], bool] | None,
    ) -> Iterator[StreamEvent]:
        """Citations (retry or backfill), the trace, ``done``, then any follow-ups."""

        selected = fitted.context_chunks
        # Streaming can't retry before the deltas already sent, and buffering would
        # destroy the streaming UX for the common case. Instead, if the streamed answer
        # carries no citations, run one non-streaming retry and swap the corrected
        # answer/sources into the `done` event — the frontend replaces the streamed
        # content with `done.answer`, so it snaps to the cited version.
        # finalize_citations is the same helper generate_grounded_answer (sync path)
        # uses, so the retry-trigger condition can't drift between response modes.
        # The retry is a whole unstreamed generation after the last delta; without
        # this frame the UI sits on its streaming label with nothing moving.
        if will_call_llm_for_citations(
            generated.answer,
            generated.sources,
            settings,
            weak_matches=phase.below_min_score,
            context_chunks=selected,
        ):
            yield {"type": "stage", "stage": "verifying_citations"}
        # Spans only now: the sources frame went out before there was an answer.
        outcome, cited = finish_answer(
            generated.messages,
            generated.answer,
            selected,
            run.generator or self._generator,
            settings,
            weak_matches=phase.below_min_score,
        )
        answer = outcome.answer
        # The last call made the answer: the citation retry's when it fired.
        truncated = stopped_at_token_limit()
        usage = usage_totals()
        timings = self._phase_timings(
            phase,
            total_start=run.started,
            condense_ms=condense_ms,
            generate_ms=generated.generate_ms,
            citation_retry_ms=outcome.retry_ms,
            first_token_ms=generated.first_token_ms,
        )
        status = answer_status(answer, len(cited), len(selected))
        self._store_finished_trace(
            run,
            settings,
            phase,
            status=trace_status(status),
            selected=selected,
            context_budget_dropped_ids=fitted.dropped_point_ids,
            # outcome.prompt_messages, not the `messages` that were streamed: when the
            # citation retry fired, the streamed answer was replaced by the retry's, and
            # the retry's continuation is what produced the text stored just below.
            prompt_messages=list(outcome.prompt_messages),
            answer=answer,
            cited_source_numbers=[source.source_number for source in cited],
            timings=timings,
            citation_retry_used=outcome.retry_used,
            citation_backfilled=outcome.backfilled,
            unsupported_claims=outcome.unsupported_claims,
            prompt_cached=self._prompt_cached(run),
            prompt_cut=was_prompt_cut(),
            usage=usage,
            answer_status=status,
            truncated=truncated,
            dropped_history_messages=fitted.dropped_history_messages,
        )
        run.finalized = True
        yield {
            "type": "done",
            "answer": answer,
            "sources": [_citation_dict(source) for source in cited],
            "timings": timings_dict(timings),
            "trace_id": run.trace_id,
            "answer_status": status,
            "top_rerank_score": _top_rerank_score(phase),
            "rerank_min_score": float(settings.rerank_min_score),
            "weak_matches": phase.below_min_score,
            "truncated": truncated,
            "citation_backfilled": outcome.backfilled,
            "unsupported_claims": outcome.unsupported_claims,
            "prompt_cached": self._prompt_cached(run),
            "prompt_cut": was_prompt_cut(),
            "passages_left_out": _picked_left_out(run.narrow, selected),
            "usage": usage_dict(usage, usage_cost_usd(settings, usage)),
        }
        if (
            settings.followups_enabled
            and status == "answered"
            and not (should_stop is not None and should_stop())
        ):
            followups = generate_followups(
                run.question.strip(),
                answer,
                _cited_chunks(cited, selected),
                run.generator or self._generator,
                settings,
            )
            if followups:
                yield {"type": "followups", "questions": followups}
