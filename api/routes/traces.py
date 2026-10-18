"""Query trace routes."""

from __future__ import annotations

from fastapi import Depends, FastAPI

from api.dependencies import (
    require_api_key,
)
from api.routes.deps import SettingsDep, TraceStoreDep
from api.routes.responses import trace_detail_response, trace_summary_response
from api.schemas import (
    TraceDetailResponse,
    TraceListResponse,
)
from api.tracing import TraceNotFoundError
from api.workspaces import workspace_of


def register(app: FastAPI) -> None:
    """Register the trace routes."""

    @app.get("/traces", response_model=TraceListResponse, dependencies=[Depends(require_api_key)])
    def list_traces(trace_store: TraceStoreDep, settings: SettingsDep) -> TraceListResponse:
        # Only the request's workspace's (X-Workspace): a trace quotes its documents.
        return TraceListResponse(
            traces=[
                trace_summary_response(trace)
                for trace in trace_store.list_traces(workspace_of(settings))
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
