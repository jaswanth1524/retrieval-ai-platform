"""Query trace routes."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import StringConstraints

from api.dependencies import (
    require_api_key,
    require_full_key,
)
from api.routes.deps import QueryLogDep, SettingsDep, TraceStoreDep
from api.routes.responses import trace_detail_response, trace_summary_response
from api.schemas import (
    QueryLogEntryResponse,
    QueryLogResponse,
    TraceDetailResponse,
    TraceListResponse,
)
from api.streaming import ClosingStreamingResponse
from api.tracing import TraceNotFoundError
from api.workspaces import workspace_of

# The most rows one GET /query-log returns.
QUERY_LOG_LIST_MAX = 1000


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
    ) -> QueryLogResponse:
        if query_log is None:
            raise HTTPException(status_code=404, detail="The query log is not enabled.")
        rows = query_log.recent(workspace_of(settings), limit=limit, uncovered=uncovered)
        return QueryLogResponse(
            queries=[
                QueryLogEntryResponse(
                    created_at=row.created_at,
                    question=row.question,
                    status=row.status,
                    top_score=row.top_score,
                    mode=row.mode,
                    trace_id=row.trace_id,
                )
                for row in rows
            ]
        )

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
        trace = trace_store.get(trace_id)
        # Another workspace's trace is "not found" here, the same as an unknown id.
        if trace is None or trace.workspace != workspace_of(settings):
            raise TraceNotFoundError(f"No trace found with id '{trace_id}'.")
        return trace_detail_response(trace)
