"""Builders turning pipeline and trace objects into response models."""

from __future__ import annotations

import dataclasses
import logging

from api.generation import ChatMessage, StageTimings
from api.retrieval import payload_optional_int
from api.schemas import (
    DocumentChunkResponse,
    HistoryMessageRequest,
    PromptMessageResponse,
    TimingsResponse,
    TraceCandidateResponse,
    TraceConfigResponse,
    TraceDetailResponse,
    TraceSummaryResponse,
    UsageResponse,
)
from api.tracing import QueryTrace, TraceSummary

logger = logging.getLogger(__name__)


def timings_response(timings: StageTimings | None) -> TimingsResponse | None:
    """Build the response's timings field from a ``StageTimings`` (or None)."""

    if timings is None:
        return None
    return TimingsResponse(**dataclasses.asdict(timings))


def history_messages(history: list[HistoryMessageRequest] | None) -> list[ChatMessage] | None:
    """Convert the request's history schema into pipeline-layer chat messages."""

    if not history:
        return None
    return [{"role": message.role, "content": message.content} for message in history]


def content_chunk(payload: dict[str, object]) -> DocumentChunkResponse:
    """Build one source-viewer chunk from a stored Qdrant payload, tolerating gaps."""

    page = payload_optional_int(payload, "page")
    return DocumentChunkResponse(
        chunk_id=str(payload.get("chunk_id", "")),
        page=page if page is not None else 0,
        section=str(payload.get("section", "")),
        text=str(payload.get("text", "")),
        chunk_ordinal=payload_optional_int(payload, "chunk_ordinal"),
        page_end=payload_optional_int(payload, "page_end"),
    )


def trace_summary_response(summary: TraceSummary) -> TraceSummaryResponse:
    """One row of the trace list."""

    return TraceSummaryResponse.model_validate(dataclasses.asdict(summary))


def usage_response(
    usage: tuple[int, int] | None, cost_usd: float | None = None
) -> UsageResponse | None:
    if usage is None:
        return None
    return UsageResponse(prompt_tokens=usage[0], completion_tokens=usage[1], cost_usd=cost_usd)


def trace_detail_response(trace: QueryTrace) -> TraceDetailResponse:
    """Build the full trace detail response from a stored trace."""

    return TraceDetailResponse(
        trace_id=trace.trace_id,
        created_at=trace.created_at,
        question=trace.question,
        mode=trace.mode,
        status=trace.status,
        # Field for field (a field added to TraceConfig/TraceCandidate shows up here
        # once the response model has it; copying by hand, one was easy to miss).
        config=(
            TraceConfigResponse.model_validate(dataclasses.asdict(trace.config))
            if trace.config is not None
            else None
        ),
        candidates=[
            TraceCandidateResponse.model_validate(dataclasses.asdict(candidate))
            for candidate in trace.candidates
        ],
        prompt_messages=(
            [
                PromptMessageResponse(role=message["role"], content=message["content"])
                for message in trace.prompt_messages
            ]
            if trace.prompt_messages is not None
            else None
        ),
        answer=trace.answer,
        cited_source_numbers=trace.cited_source_numbers,
        timings=TimingsResponse(**trace.timings) if trace.timings is not None else None,
        error=trace.error,
        condensed_question=trace.condensed_question,
        condense_note=trace.condense_note,
        weak_matches=trace.weak_matches,
        history_message_count=trace.history_message_count,
        query_variants=trace.query_variants,
        citation_retry_used=trace.citation_retry_used,
        citation_backfilled=trace.citation_backfilled,
        request_id=trace.request_id,
        usage=usage_response(trace.usage, trace.cost_usd),
        answer_status=trace.answer_status,
        truncated=trace.truncated,
        dropped_history_messages=trace.dropped_history_messages,
        kind=trace.kind,
    )


def log_question(*, provider: str | None, source_count: int, timings: dict[str, float]) -> None:
    """Emit one structured log line per answered question for latency observability."""

    logger.info(
        "question answered provider=%s sources=%d condense_ms=%.1f embed_ms=%.1f "
        "search_ms=%.1f rerank_ms=%.1f rerank_wait_ms=%.1f context_expansion_ms=%.1f "
        "first_token_ms=%.1f generate_ms=%.1f citation_retry_ms=%.1f total_ms=%.1f",
        provider or "default",
        source_count,
        timings.get("condense_ms", 0.0),
        timings["embed_ms"],
        timings["search_ms"],
        timings["rerank_ms"],
        timings.get("rerank_wait_ms", 0.0),
        timings.get("context_expansion_ms", 0.0),
        timings.get("first_token_ms", 0.0),
        timings["generate_ms"],
        timings.get("citation_retry_ms", 0.0),
        timings["total_ms"],
    )
