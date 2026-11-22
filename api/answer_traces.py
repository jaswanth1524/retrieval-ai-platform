"""Building and storing an answer's trace (``GET /traces/{id}``): the lowest layer of
``RagPipeline``, below ``api/pipeline_core.py``. Moved out of ``api/pipeline.py``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from api.answer_run import (
    AnswerRun,
    RetrievalPhase,
    narrowing_record,
    timings_dict,
    trace_model_name,
)
from api.generation import (
    ChatMessage,
    StageTimings,
)
from api.llm import (
    usage_cost_usd,
    usage_totals,
)
from api.prompting import (
    FittedPrompt,
)
from api.repository import PassageFilter
from api.request_id import current_request_id
from api.reranking import (
    RerankedChunk,
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


class TraceRecording:
    """The trace-building methods ``RagPipeline`` inherits."""

    _settings: AppSettings
    _trace_store: TraceSink | None

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
        prompt_messages: list[ChatMessage] | None = None,
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
            prompt_messages=prompt_messages,
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
        unsupported_numbers: int = 0,
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
            unsupported_numbers=unsupported_numbers,
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
                # Only with an answer begun: what a Continue of it is written from.
                prompt_messages=run.prompt_messages if run.parts else None,
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
