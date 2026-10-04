"""Search route: ranked passages for a query, with no LLM call."""

from __future__ import annotations

from fastapi import Depends, FastAPI

from api.dependencies import require_api_key
from api.pipeline import AnswerOverrides
from api.routes.deps import QuestionSlotsDep, RagPipelineDep
from api.routes.questions import question_slot
from api.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
    SearchTimingsResponse,
)


def register(app: FastAPI) -> None:
    """Register the search route."""

    # The same hybrid search, RRF (k=60) and cross-encoder rerank a question uses
    # (RagPipeline.retrieve, which the eval harness also scores), so "find in my
    # documents" works even when the LLM is down. Read key allowed, like asking. It
    # takes a question slot: the rerank is the stage that bounds memory.
    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
    def search(
        request: SearchRequest, pipeline: RagPipelineDep, slots: QuestionSlotsDep
    ) -> SearchResponse:
        release = question_slot(slots)
        try:
            phase = pipeline.retrieve(
                request.query,
                AnswerOverrides(rerank_top_k=request.rerank_top_k),
                filenames=request.filenames,
                tags=request.tags,
                for_generation=False,
            )
        finally:
            release()
        return SearchResponse(
            results=[
                SearchResultResponse(
                    rank=rank,
                    filename=chunk.filename,
                    page=chunk.page,
                    section=chunk.section,
                    chunk_id=chunk.chunk_id,
                    chunk_ordinal=chunk.chunk_ordinal,
                    text=chunk.text,
                    retrieval_score=chunk.retrieval_score,
                    rerank_score=chunk.rerank_score,
                )
                for rank, chunk in enumerate(phase.context_chunks, start=1)
            ],
            timings=SearchTimingsResponse(
                embed_ms=phase.embed_ms, search_ms=phase.search_ms, rerank_ms=phase.rerank_ms
            ),
        )
