"""Query trace routes."""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import StringConstraints

from api.dependencies import (
    require_api_key,
    require_full_key,
)
from api.relevance import score_report, trace_diff
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
    QueryLogEntryResponse,
    QueryLogResponse,
    QueryRecheckItemResponse,
    QueryRecheckResponse,
    ScoreReportResponse,
    TraceCandidateChangeResponse,
    TraceDetailResponse,
    TraceDiffResponse,
    TraceListResponse,
    TraceSettingChangeResponse,
)
from api.settings import AppSettings
from api.streaming import ClosingStreamingResponse
from api.tracing import QueryTrace, TraceNotFoundError, TraceStore
from api.workspaces import workspace_of

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
                items: list[QueryRecheckItemResponse] = []
                seen: set[str] = set()
                for row in rows:
                    # Asked several times: searched once.
                    folded = " ".join(row.question.split()).casefold()
                    if folded in seen or not row.question.strip():
                        continue
                    seen.add(folded)
                    phase = pipeline.retrieve(row.question, for_generation=False)
                    top = phase.scored_chunks[0] if phase.scored_chunks else None
                    items.append(
                        QueryRecheckItemResponse(
                            question=row.question,
                            status=row.status,
                            covered_now=bool(phase.context_chunks) and not phase.below_min_score,
                            top_score=top.rerank_score if top is not None else None,
                            top_filename=top.filename if top is not None else None,
                            top_chunk_id=top.chunk_id if top is not None else None,
                        )
                    )
            finally:
                release()
        finally:
            _recheck_running.release()
        return QueryRecheckResponse(
            checked=len(items),
            covered_now=sum(item.covered_now for item in items),
            items=items,
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
        trace_store: TraceStoreDep, settings: SettingsDep
    ) -> ScoreReportResponse:
        report = score_report(
            trace_store.list_summaries(workspace_of(settings)), float(settings.rerank_min_score)
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
        diff = trace_diff(
            _own_trace(trace_store, settings, trace_id),
            _own_trace(trace_store, settings, other_id),
        )
        return TraceDiffResponse(
            question_a=diff.question_a,
            question_b=diff.question_b,
            same_answer=diff.same_answer,
            settings=[TraceSettingChangeResponse(name=c.name, a=c.a, b=c.b) for c in diff.settings],
            candidates=[
                TraceCandidateChangeResponse(**dataclasses.asdict(c)) for c in diff.candidates
            ],
        )
