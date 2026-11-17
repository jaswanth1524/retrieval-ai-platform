"""The events an answer stream yields (``RagPipeline.answer_stream``), moved out of
``api/pipeline.py``; it re-exports them.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


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
    unsupported_numbers: NotRequired[int]
    # A follow-up rewritten into a standalone question: what was searched for.
    retrieval_query: NotRequired[str]
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
