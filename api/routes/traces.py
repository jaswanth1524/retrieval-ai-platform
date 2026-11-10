"""Query trace routes."""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from collections.abc import Sequence
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import StringConstraints

from api.answer_cache import AnswerCache
from api.audit import audit
from api.dependencies import (
    get_answer_cache,
    require_api_key,
    require_full_key,
)
from api.pipeline import RagPipeline, RetrievalPhase, filter_from_record
from api.query_log import LoggedQuery
from api.relevance import UNANSWERED_STATUSES, TraceDiff, score_report, trace_diff
from api.reranking import RerankedChunk
from api.routes.deps import (
    QueryLogDep,
    QuestionSlotsDep,
    RagPipelineDep,
    SettingsDep,
    TraceStoreDep,
)
from api.routes.questions import question_slot
from api.routes.responses import trace_detail_response, trace_summary_response
from api.schemas import (
    REQUEST_FILENAME_MAX_CHARS,
    AnswerCacheClearResponse,
    AnswerCacheResponse,
    CachedQuestionResponse,
    QueryLogEntryResponse,
    QueryLogResponse,
    QueryRecheckItemResponse,
    QueryRecheckResponse,
    ScoreReportResponse,
    TraceCandidateChangeResponse,
    TraceCandidateResponse,
    TraceDetailResponse,
    TraceDiffResponse,
    TraceListResponse,
    TraceRerunRequest,
    TraceRerunResponse,
    TraceSettingChangeResponse,
    TraceWhatIfItemResponse,
    TraceWhatIfRequest,
    TraceWhatIfResponse,
)
from api.settings import AppSettings
from api.streaming import ClosingStreamingResponse
from api.tracing import QueryTrace, TraceNotFoundError, TraceStore, TraceSummary
from api.workspaces import workspace_of

logger = logging.getLogger(__name__)

# The most rows one GET /query-log returns.
QUERY_LOG_LIST_MAX = 1000
# The most unanswered questions one POST /query-log/recheck searches again.
RECHECK_MAX = 100
# One re-check at a time: each searches up to RECHECK_MAX questions in a row.
_recheck_running = threading.Lock()


def _own_trace(trace_store: TraceStore, settings: AppSettings, trace_id: str) -> QueryTrace:
    trace = trace_store.get(trace_id)
    # Another workspace's trace is "not found" here, the same as an unknown id.
    if trace is None or trace.workspace != workspace_of(settings):
        raise TraceNotFoundError(f"No trace found with id '{trace_id}'.")
    return trace


# A score this much above the logged one counts as the documents having moved on.
RECHECK_SCORE_GAIN = 0.02


def _recheck_rows(
    rows: Sequence[LoggedQuery], pipeline: RagPipeline, budget_seconds: float
) -> tuple[list[QueryRecheckItemResponse], bool]:
    """Search each unanswered question again (what was searched for it: a follow-up's
    rewrite), over what it was asked over (its documents, tags and narrowing, when the
    log has them; the whole workspace for an older row). Summaries are left out: their
    "question" is a task line. Each question is on its own (an error is that item's),
    and the whole stops after ``budget_seconds`` (0 = no limit): a proxy times a long
    request out."""

    deadline = time.monotonic() + budget_seconds if budget_seconds > 0 else None
    items: list[QueryRecheckItemResponse] = []
    seen: set[str] = set()
    for row in rows:
        if row.kind == "summary":
            continue
        query = row.retrieval_query or row.question
        # Asked several times (over the same scope): searched once.
        folded = " ".join(query.split()).casefold() + json.dumps(row.scope, sort_keys=True)
        if folded in seen or not query.strip():
            continue
        if deadline is not None and time.monotonic() >= deadline:
            return items, True
        seen.add(folded)
        scope = row.scope or {}
        try:
            narrowing = scope.get("narrowing")
            phase = pipeline.retrieve(
                query,
                filenames=scope.get("filenames") or None,
                tags=scope.get("tags") or None,
                for_generation=False,
                narrow=filter_from_record(narrowing) if isinstance(narrowing, dict) else None,
            )
        except Exception as exc:  # noqa: BLE001 - one question's failure is its own
            logger.warning("Re-checking %r failed: %s", query[:80], exc)
            items.append(
                QueryRecheckItemResponse(
                    question=row.question,
                    status=row.status,
                    covered_now=False,
                    error="The search failed; see the server log.",
                )
            )
            continue
        top = phase.scored_chunks[0] if phase.scored_chunks else None
        items.append(
            QueryRecheckItemResponse(
                question=row.question,
                status=row.status,
                covered_now=_covered_now(row, phase, top),
                top_score=top.rerank_score if top is not None else None,
                top_filename=top.filename if top is not None else None,
                top_chunk_id=top.chunk_id if top is not None else None,
            )
        )
    return items, False


def _covered_now(row: LoggedQuery, phase: RetrievalPhase, top: RerankedChunk | None) -> bool:
    """A passage clears the bar now, and something changed since it went unanswered:
    a model that declined over passages that already cleared the bar would decline
    again, so the same top passage at the same score is not "covered now"."""

    if top is None or not phase.context_chunks or phase.below_min_score:
        return False
    if row.top_score is None:
        return True
    if top.rerank_score > row.top_score + RECHECK_SCORE_GAIN:
        return True
    return row.top_filename is not None and top.filename != row.top_filename


def _what_if_rows(
    summaries: Sequence[TraceSummary],
    trace_store: TraceStore,
    pipeline: RagPipeline,
    update: dict[str, float | int],
    budget_seconds: float,
) -> tuple[list[TraceWhatIfItemResponse], bool]:
    """Each traced question's retrieval run again under ``update`` (as
    ``rerun_retrieval`` does: its rewrite, scope and narrowing); each on its own, and
    the whole stops after ``budget_seconds`` (0 = no limit)."""

    deadline = time.monotonic() + budget_seconds if budget_seconds > 0 else None
    items: list[TraceWhatIfItemResponse] = []
    for summary in summaries:
        if deadline is not None and time.monotonic() >= deadline:
            return items, True
        trace = trace_store.get(summary.trace_id)
        if trace is None or trace.config is None:
            continue
        answered_before = summary.answer_status not in UNANSWERED_STATUSES
        try:
            phase, _ = pipeline.rerun_retrieval(trace, update)
        except Exception as exc:  # noqa: BLE001 - one question's failure is its own
            logger.warning("What-if for trace %s failed: %s", summary.trace_id, exc)
            items.append(
                TraceWhatIfItemResponse(
                    trace_id=summary.trace_id,
                    question=summary.question,
                    answered_before=answered_before,
                    would_answer=False,
                    error="The search failed; see the server log.",
                )
            )
            continue
        top = phase.scored_chunks[0] if phase.scored_chunks else None
        items.append(
            TraceWhatIfItemResponse(
                trace_id=summary.trace_id,
                question=summary.question,
                answered_before=answered_before,
                # Passages cleared the bar (as POST /traces/{id}/rerun says it): the
                # weak-match fallback's closest passages don't count — a model told
                # they match weakly mostly declines.
                would_answer=bool(phase.context_chunks) and not phase.below_min_score,
                weak_matches=phase.below_min_score,
                top_score=top.rerank_score if top is not None else None,
            )
        )
    return items, False


def _diff_response(diff: TraceDiff) -> TraceDiffResponse:
    return TraceDiffResponse(
        question_a=diff.question_a,
        question_b=diff.question_b,
        same_answer=diff.same_answer,
        settings=[TraceSettingChangeResponse(name=c.name, a=c.a, b=c.b) for c in diff.settings],
        candidates=[TraceCandidateChangeResponse(**dataclasses.asdict(c)) for c in diff.candidates],
    )


def register(app: FastAPI) -> None:
    """Register the trace routes."""

    # What was asked, newest first (QUERY_LOG_ENABLED). Operator data — what users
    # typed — so the full key only, and only the request's workspace's.
    @app.get(
        "/query-log",
        response_model=QueryLogResponse,
        dependencies=[Depends(require_full_key)],
        responses={404: {"description": "The query log is off (QUERY_LOG_ENABLED)."}},
    )
    def list_query_log(
        query_log: QueryLogDep,
        settings: SettingsDep,
        limit: Annotated[int, Query(ge=1, le=QUERY_LOG_LIST_MAX)] = 200,
        uncovered: bool = False,
        filename: Annotated[str | None, Query(max_length=REQUEST_FILENAME_MAX_CHARS)] = None,
    ) -> QueryLogResponse:
        if query_log is None:
            raise HTTPException(status_code=404, detail="The query log is not enabled.")
        rows = query_log.recent(
            workspace_of(settings), limit=limit, uncovered=uncovered, filename=filename
        )
        return QueryLogResponse(
            queries=[
                QueryLogEntryResponse(
                    created_at=row.created_at,
                    question=row.question,
                    status=row.status,
                    top_score=row.top_score,
                    mode=row.mode,
                    trace_id=row.trace_id,
                    filenames=list(row.filenames),
                    kind=row.kind,
                    cached=row.cached,
                )
                for row in rows
            ]
        )

    # The questions that went unanswered, searched again now (after an upload, say):
    # retrieval only, no LLM call. One at a time, taking a question slot while it runs;
    # the full key, as the query log itself.
    @app.post(
        "/query-log/recheck",
        response_model=QueryRecheckResponse,
        dependencies=[Depends(require_full_key)],
        responses={
            404: {"description": "The query log is off (QUERY_LOG_ENABLED)."},
            409: {"description": "A re-check is already running."},
            429: {"description": "Every question slot is busy."},
        },
    )
    def recheck_query_log(
        query_log: QueryLogDep,
        settings: SettingsDep,
        pipeline: RagPipelineDep,
        slots: QuestionSlotsDep,
        limit: Annotated[int, Query(ge=1, le=RECHECK_MAX)] = 20,
    ) -> QueryRecheckResponse:
        if query_log is None:
            raise HTTPException(status_code=404, detail="The query log is not enabled.")
        if not _recheck_running.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A re-check is already running.")
        try:
            release = question_slot(slots, count_metric=False)
            try:
                rows = query_log.recent(workspace_of(settings), limit=limit, uncovered=True)
                items, stopped_early = _recheck_rows(
                    rows, pipeline, float(settings.recheck_time_budget_seconds)
                )
            finally:
                release()
        finally:
            _recheck_running.release()
        return QueryRecheckResponse(
            checked=len(items),
            covered_now=sum(item.covered_now for item in items),
            items=items,
            stopped_early=stopped_early,
        )

    # The best rerank score of each traced question, answered against not: what a
    # RERANK_MIN_SCORE for this corpus would be. Scores and counts only (no text), so
    # the read key, as the trace list. Before /traces/{trace_id}.
    @app.get(
        "/traces/score-report",
        response_model=ScoreReportResponse,
        dependencies=[Depends(require_api_key)],
    )
    def trace_score_report(
        trace_store: TraceStoreDep,
        settings: SettingsDep,
        since: Annotated[
            float | None, Query(ge=0, description="Only traces asked since (epoch seconds).")
        ] = None,
        bar: Annotated[
            float | None,
            Query(ge=0, le=1, description="Only traces answered under this RERANK_MIN_SCORE."),
        ] = None,
    ) -> ScoreReportResponse:
        # Every archived trace, not the list's newest hundred: 20 a side is the floor
        # for a suggestion, and a thin sample makes a noisy one.
        report = score_report(
            trace_store.list_summaries(
                workspace_of(settings), limit=int(settings.trace_store_max_rows)
            ),
            float(settings.rerank_min_score),
            reranker_model=settings.reranker_model,
            since=since,
            bar=bar,
        )
        return ScoreReportResponse(**dataclasses.asdict(report))

    # Every trace of the request's workspace as JSON Lines, oldest first: all archived
    # ones with TRACE_STORE_PATH, else the in-memory ring's. Prompts quote documents and
    # what users asked, so the full key only (and rate-limited like /export). Registered
    # before /traces/{trace_id}, which would otherwise take "export" as an id.
    @app.get(
        "/traces/export",
        dependencies=[Depends(require_full_key)],
        response_class=StreamingResponse,
        responses={200: {"content": {"application/x-ndjson": {}}}},
    )
    def export_traces(trace_store: TraceStoreDep, settings: SettingsDep) -> StreamingResponse:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        return ClosingStreamingResponse(
            trace_store.export_lines(workspace_of(settings)),
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="docrag-traces-{stamp}.jsonl"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/traces", response_model=TraceListResponse, dependencies=[Depends(require_api_key)])
    def list_traces(
        trace_store: TraceStoreDep,
        settings: SettingsDep,
        answer_status: Annotated[
            list[Annotated[str, StringConstraints(max_length=40)]] | None,
            Query(
                max_length=10,
                description="Only traces with one of these answer statuses (repeatable): "
                "answered, uncited, insufficient_context, no_context.",
            ),
        ] = None,
    ) -> TraceListResponse:
        # Only the request's workspace's (X-Workspace): a trace quotes its documents.
        return TraceListResponse(
            traces=[
                trace_summary_response(summary)
                for summary in trace_store.list_summaries(
                    workspace_of(settings),
                    frozenset(answer_status) if answer_status else None,
                )
            ]
        )

    @app.get(
        "/traces/{trace_id}",
        response_model=TraceDetailResponse,
        dependencies=[Depends(require_api_key)],
    )
    def get_trace(
        trace_id: str, trace_store: TraceStoreDep, settings: SettingsDep
    ) -> TraceDetailResponse:
        return trace_detail_response(_own_trace(trace_store, settings, trace_id))

    # This workspace's cached answers (ANSWER_CACHE_SIZE): what was asked and how old the
    # answer is. Clearing them is the way out after a bad answer or a provider change,
    # short of waiting out ANSWER_CACHE_TTL_SECONDS. Questions users asked: full key.
    @app.get(
        "/answer-cache",
        response_model=AnswerCacheResponse,
        dependencies=[Depends(require_full_key)],
    )
    def answer_cache_entries(
        settings: SettingsDep,
        cache: Annotated[AnswerCache | None, Depends(get_answer_cache)],
    ) -> AnswerCacheResponse:
        if cache is None:
            return AnswerCacheResponse(
                enabled=False,
                entries=[],
                total_entries=0,
                max_entries=0,
                ttl_seconds=0.0,
                in_flight=0,
            )
        return AnswerCacheResponse(
            enabled=True,
            entries=[
                CachedQuestionResponse(question=question, age_seconds=round(age, 1))
                for question, age in cache.entries_for(settings.qdrant_collection)
            ],
            total_entries=len(cache),
            max_entries=cache.max_entries,
            ttl_seconds=cache.ttl_seconds,
            in_flight=len(cache.inflight),
        )

    @app.delete(
        "/answer-cache",
        response_model=AnswerCacheClearResponse,
        dependencies=[Depends(require_full_key)],
    )
    def clear_answer_cache(
        settings: SettingsDep,
        cache: Annotated[AnswerCache | None, Depends(get_answer_cache)],
    ) -> AnswerCacheClearResponse:
        removed = cache.drop_collection(settings.qdrant_collection) if cache is not None else 0
        audit("answer_cache.clear", workspace=workspace_of(settings), removed=removed)
        return AnswerCacheClearResponse(removed=removed)

    # A traced question searched again under other ranking settings — "would a lower
    # bar have answered it?" — compared with the trace. Retrieval only: no LLM call,
    # nothing stored. Takes a question slot, as a search does; the read key, as traces.
    @app.post(
        "/traces/{trace_id}/rerun",
        response_model=TraceRerunResponse,
        dependencies=[Depends(require_api_key)],
        responses={429: {"description": "Every question slot is busy."}},
    )
    def rerun_trace(
        trace_id: str,
        request: TraceRerunRequest,
        trace_store: TraceStoreDep,
        settings: SettingsDep,
        pipeline: RagPipelineDep,
        slots: QuestionSlotsDep,
    ) -> TraceRerunResponse:
        original = _own_trace(trace_store, settings, trace_id)
        if original.kind != "question" or original.config is None:
            raise HTTPException(status_code=400, detail="Only a question's trace can be run again.")
        update = request.model_dump(exclude_none=True)
        if int(update.get("rerank_top_k", settings.rerank_top_k)) > int(settings.fused_top_n):
            raise HTTPException(status_code=400, detail="rerank_top_k can't exceed FUSED_TOP_N.")
        release = question_slot(slots, count_metric=False)
        try:
            phase, rerun = pipeline.rerun_retrieval(original, update)
        finally:
            release()
        return TraceRerunResponse(
            query=original.condensed_question or original.question,
            settings=update,
            would_answer=bool(phase.context_chunks) and not phase.below_min_score,
            weak_matches=phase.below_min_score,
            diff=_diff_response(trace_diff(original, rerun)),
            candidates=[
                TraceCandidateResponse.model_validate(dataclasses.asdict(candidate))
                for candidate in rerun.candidates
            ],
        )

    # The latest traced questions searched again under a proposed bar (or floor, or
    # fallback): how many would be answered — what changing the setting would do, before
    # it is changed. Retrieval only; one question slot for the whole run, the read key
    # as a single re-run.
    @app.post(
        "/traces/what-if",
        response_model=TraceWhatIfResponse,
        dependencies=[Depends(require_api_key)],
        responses={
            409: {"description": "A re-check is already running."},
            429: {"description": "Every question slot is busy."},
        },
    )
    def what_if(
        request: TraceWhatIfRequest,
        trace_store: TraceStoreDep,
        settings: SettingsDep,
        pipeline: RagPipelineDep,
        slots: QuestionSlotsDep,
    ) -> TraceWhatIfResponse:
        update = request.model_dump(exclude_none=True, exclude={"limit", "only_unanswered"})
        statuses = UNANSWERED_STATUSES if request.only_unanswered else None
        summaries = [
            summary
            for summary in trace_store.list_summaries(
                workspace_of(settings), statuses, limit=int(settings.trace_store_max_rows)
            )
            if summary.kind == "question" and summary.gated and summary.answer_status
        ][: request.limit]
        if not _recheck_running.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A re-check is already running.")
        try:
            release = question_slot(slots, count_metric=False)
            try:
                items, stopped_early = _what_if_rows(
                    summaries,
                    trace_store,
                    pipeline,
                    update,
                    float(settings.recheck_time_budget_seconds),
                )
            finally:
                release()
        finally:
            _recheck_running.release()
        counted = [item for item in items if item.error is None]
        gained = [item for item in counted if item.would_answer and not item.answered_before]
        lost = [item for item in counted if item.answered_before and not item.would_answer]
        return TraceWhatIfResponse(
            settings=update,
            checked=len(counted),
            would_answer=sum(item.would_answer for item in counted),
            newly_answered=len(gained),
            newly_unanswered=len(lost),
            items=items,
            stopped_early=stopped_early,
        )

    # Why two runs of a question went differently: the settings that differ, and the
    # candidates whose rank, selection or drop reason moved.
    @app.get(
        "/traces/{trace_id}/diff/{other_id}",
        response_model=TraceDiffResponse,
        dependencies=[Depends(require_api_key)],
    )
    def diff_traces(
        trace_id: str, other_id: str, trace_store: TraceStoreDep, settings: SettingsDep
    ) -> TraceDiffResponse:
        return _diff_response(
            trace_diff(
                _own_trace(trace_store, settings, trace_id),
                _own_trace(trace_store, settings, other_id),
            )
        )
