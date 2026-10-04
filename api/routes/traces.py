"""Query trace routes."""

from __future__ import annotations

from fastapi import Depends, FastAPI

from api.dependencies import (
    require_api_key,
)
from api.routes.deps import TraceStoreDep
from api.routes.responses import trace_detail_response, trace_summary_response
from api.schemas import (
    TraceDetailResponse,
    TraceListResponse,
)
from api.tracing import TraceNotFoundError


def register(app: FastAPI) -> None:
    """Register the trace routes."""

    @app.get("/traces", response_model=TraceListResponse, dependencies=[Depends(require_api_key)])
    def list_traces(trace_store: TraceStoreDep) -> TraceListResponse:
        return TraceListResponse(
            traces=[trace_summary_response(trace) for trace in trace_store.list_traces()]
        )

    @app.get(
        "/traces/{trace_id}",
        response_model=TraceDetailResponse,
        dependencies=[Depends(require_api_key)],
    )
    def get_trace(trace_id: str, trace_store: TraceStoreDep) -> TraceDetailResponse:
        trace = trace_store.get(trace_id)
        if trace is None:
            raise TraceNotFoundError(f"No trace found with id '{trace_id}'.")
        return trace_detail_response(trace)
