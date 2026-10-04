"""Builders turning pipeline and trace objects into response models."""

from __future__ import annotations

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
)
from api.tracing import QueryTrace

logger = logging.getLogger(__name__)


def timings_response(timings: StageTimings | None) -> TimingsResponse | None:
    """Build the response's timings field from a ``StageTimings`` (or None)."""

    if timings is None:
        return None
    return TimingsResponse(
        embed_ms=timings.embed_ms,
        search_ms=timings.search_ms,
        rerank_ms=timings.rerank_ms,
        generate_ms=timings.generate_ms,
        total_ms=timings.total_ms,
        condense_ms=timings.condense_ms,
        query_expansion_ms=timings.query_expansion_ms,
        context_expansion_ms=timings.context_expansion_ms,
        citation_retry_ms=timings.citation_retry_ms,
    )


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
    )


def trace_summary_response(trace: QueryTrace) -> TraceSummaryResponse:
    """Build one row of the trace list from a stored trace."""

    return TraceSummaryResponse(
        trace_id=trace.trace_id,
        created_at=trace.created_at,
        question=trace.question,
        mode=trace.mode,
        status=trace.status,
        llm_provider=trace.config.llm_provider if trace.config is not None else None,
        total_ms=trace.timings.get("total_ms") if trace.timings is not None else None,
        candidate_count=len(trace.candidates),
        kept_count=sum(1 for candidate in trace.candidates if candidate.kept),
    )


def trace_detail_response(trace: QueryTrace) -> TraceDetailResponse:
    """Build the full trace detail response from a stored trace."""

    return TraceDetailResponse(
        trace_id=trace.trace_id,
        created_at=trace.created_at,
        question=trace.question,
        mode=trace.mode,
        status=trace.status,
        config=(
            TraceConfigResponse(
                llm_provider=trace.config.llm_provider,
                rerank_top_k=trace.config.rerank_top_k,
                max_context_chunks=trace.config.max_context_chunks,
                llm_temperature=trace.config.llm_temperature,
                rerank_min_score=trace.config.rerank_min_score,
                fused_top_n=trace.config.fused_top_n,
                filenames=trace.config.filenames,
                tags=trace.config.tags,
            )
            if trace.config is not None
            else None
        ),
        candidates=[
            TraceCandidateResponse(
                point_id=candidate.point_id,
                filename=candidate.filename,
                page=candidate.page,
                section=candidate.section,
                chunk_id=candidate.chunk_id,
                retrieval_score=candidate.retrieval_score,
                rerank_score=candidate.rerank_score,
                kept=candidate.kept,
                drop_reason=candidate.drop_reason,
                selected_for_context=candidate.selected_for_context,
                neighbor_expanded=candidate.neighbor_expanded,
            )
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
        history_message_count=trace.history_message_count,
        query_variants=trace.query_variants,
        citation_retry_used=trace.citation_retry_used,
    )


def log_question(*, provider: str | None, source_count: int, timings: dict[str, float]) -> None:
    """Emit one structured log line per answered question for latency observability."""

    logger.info(
        "question answered provider=%s sources=%d condense_ms=%.1f embed_ms=%.1f "
        "search_ms=%.1f rerank_ms=%.1f context_expansion_ms=%.1f generate_ms=%.1f "
        "citation_retry_ms=%.1f total_ms=%.1f",
        provider or "default",
        source_count,
        timings.get("condense_ms", 0.0),
        timings["embed_ms"],
        timings["search_ms"],
        timings["rerank_ms"],
        timings.get("context_expansion_ms", 0.0),
        timings["generate_ms"],
        timings.get("citation_retry_ms", 0.0),
        timings["total_ms"],
    )
