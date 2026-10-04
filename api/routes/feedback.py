"""Answer feedback routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query

from api.dependencies import (
    require_api_key,
    require_full_key,
)
from api.routes.deps import FeedbackStoreDep
from api.schemas import (
    FEEDBACK_LIST_MAX_LIMIT,
    FeedbackItemResponse,
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
)


def register(app: FastAPI) -> None:
    """Register the feedback routes."""

    # Off by default (AppSettings.feedback_enabled) — a deployment that hasn't
    # opted in gets the same 404 shape as any other absent route, not a distinct
    # "feature unavailable" response.
    @app.post(
        "/feedback",
        response_model=FeedbackResponse,
        status_code=201,
        dependencies=[Depends(require_api_key)],
    )
    def submit_feedback(
        request: FeedbackRequest, feedback_store: FeedbackStoreDep
    ) -> FeedbackResponse:
        if feedback_store is None:
            raise HTTPException(status_code=404, detail="Feedback capture is not enabled.")
        feedback = feedback_store.add(
            trace_id=request.trace_id,
            question=request.question,
            answer_excerpt=request.answer_excerpt,
            cited_filenames=request.cited_filenames,
            rating=request.rating,
            citation_source_number=request.citation_source_number,
        )
        return FeedbackResponse(id=feedback.id)

    # The read side of the same feature: without it, captured ratings are reachable
    # only by opening the sqlite file by hand, which defeats the point of capturing
    # them (thumbs-down turns are meant to become negative retrieval examples).
    @app.get(
        "/feedback",
        response_model=FeedbackListResponse,
        dependencies=[Depends(require_full_key)],
    )
    def list_feedback(
        feedback_store: FeedbackStoreDep,
        limit: Annotated[int, Query(ge=1, le=FEEDBACK_LIST_MAX_LIMIT)] = 100,
    ) -> FeedbackListResponse:
        if feedback_store is None:
            raise HTTPException(status_code=404, detail="Feedback capture is not enabled.")
        return FeedbackListResponse(
            feedback=[
                FeedbackItemResponse(
                    id=item.id,
                    trace_id=item.trace_id,
                    question=item.question,
                    answer_excerpt=item.answer_excerpt,
                    cited_filenames=item.cited_filenames,
                    rating=item.rating,
                    citation_source_number=item.citation_source_number,
                    created_at=item.created_at,
                )
                for item in feedback_store.list_recent(limit)
            ]
        )
