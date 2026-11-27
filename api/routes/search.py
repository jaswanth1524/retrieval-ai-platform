"""Search route: ranked passages for a query, with no LLM call."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from api.citations import query_spans
from api.dependencies import require_api_key
from api.explain import explain
from api.parsers.common import clean_text
from api.pipeline import AnswerOverrides
from api.reranking import RerankedChunk
from api.routes.deps import QuestionSlotsDep, RagPipelineDep
from api.routes.questions import (
    _watch_disconnect,
    check_question_text,
    passage_narrowing,
    question_slot,
)
from api.schemas import (
    CitationSpanResponse,
    RecheckRequest,
    RecheckResponse,
    RecheckSourceResponse,
    SearchExplanationResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
    SearchTimingsResponse,
    VerifyRequest,
    VerifyResponse,
    VerifySentenceResponse,
)
from api.verify import SentenceCheck, VerifyAbandoned, recheck, verify_text

# Changed documents a recheck searches one by one; past this many, one search over all
# of them per sentence (each search is an embed, a hybrid search and a rerank).
RECHECK_SEARCHED_ONE_BY_ONE = 4


def _search_within(pipeline: Any, filenames: list[str]) -> Any:
    """For each changed document, its passages closest to a sentence (the passage an
    answer cited reads differently now). Each search is kept for the sentence: every
    source in a changed document used to search again, up to RECHECK_MAX_SOURCES
    searches and reranks for every sentence under one slot."""

    found: dict[tuple[str | None, str], list[RerankedChunk]] = {}
    together = len(filenames) > RECHECK_SEARCHED_ONE_BY_ONE

    def closest(scope: str | None, sentence: str) -> list[RerankedChunk]:
        key = (scope, sentence)
        if key not in found:
            found[key] = list(
                pipeline.retrieve(
                    sentence,
                    AnswerOverrides(),
                    filenames=filenames if scope is None else [scope],
                    for_generation=False,
                ).context_chunks
            )
        return found[key]

    def within(filename: str) -> Any:
        def search(sentence: str) -> list[RerankedChunk]:
            hits = closest(None if together else filename, sentence)
            return [chunk for chunk in hits if chunk.filename == filename][:3]

        return search

    return within


def _sentence_response(check: SentenceCheck) -> VerifySentenceResponse:
    chunk = check.chunk
    return VerifySentenceResponse(
        sentence=check.sentence,
        status=check.status,
        filename=chunk.filename if chunk else None,
        page=chunk.page if chunk else None,
        section=chunk.section if chunk else None,
        chunk_id=chunk.chunk_id if chunk else None,
        quote=check.quote,
        rerank_score=chunk.rerank_score if chunk else None,
        missing_numbers=list(check.missing_numbers),
    )


def register(app: FastAPI) -> None:
    """Register the search and verify routes."""

    # A text (a draft answer, an email) checked against the documents a sentence at a
    # time: the passage and words that support each, or that none does. The search a
    # question uses and the span and number checks its citations get; no LLM call. Read
    # key allowed, as for /search; one question slot for the whole check.
    @app.post(
        "/questions/verify",
        response_model=VerifyResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def verify(
        request: VerifyRequest,
        http_request: Request,
        pipeline: RagPipelineDep,
        slots: QuestionSlotsDep,
    ) -> VerifyResponse:
        check_question_text(request.text)
        if request.filenames or request.tags:

            def scope_empty() -> bool:
                return pipeline.scope_matches_nothing(request.filenames, request.tags)

            if await run_in_threadpool(scope_empty):
                return VerifyResponse(sentences=[], empty_scope=True)
        release = question_slot(slots, count_metric=False)
        # A client that leaves stops the check before the next sentence's search,
        # freeing its slot (up to 20 searches and reranks otherwise ran for nobody).
        stop = threading.Event()
        watcher = asyncio.create_task(_watch_disconnect(http_request, stop))

        def search(sentence: str) -> list[RerankedChunk]:
            return list(
                pipeline.retrieve(
                    sentence,
                    AnswerOverrides(),
                    filenames=request.filenames,
                    tags=request.tags,
                    for_generation=False,
                ).context_chunks
            )

        try:
            checks, more = await run_in_threadpool(
                verify_text,
                clean_text(request.text),
                search,
                stop.is_set,
                pipeline.neighbour_text,
            )
        except VerifyAbandoned:
            raise HTTPException(status_code=499, detail="The client left.") from None
        finally:
            watcher.cancel()
            release()
        return VerifyResponse(
            sentences=[
                VerifySentenceResponse(
                    sentence=check.sentence,
                    status=check.status,
                    filename=check.chunk.filename if check.chunk else None,
                    page=check.chunk.page if check.chunk else None,
                    section=check.chunk.section if check.chunk else None,
                    chunk_id=check.chunk.chunk_id if check.chunk else None,
                    quote=check.quote,
                    rerank_score=check.chunk.rerank_score if check.chunk else None,
                    missing_numbers=list(check.missing_numbers),
                )
                for check in checks
            ],
            more=more,
        )

    # A saved answer against the documents now: is each passage it cited still there as
    # it was, and does each sentence still hold (a changed document is searched for the
    # sentence instead)? No LLM. Read key allowed; one question slot.
    @app.post(
        "/questions/recheck",
        response_model=RecheckResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def recheck_answer(
        request: RecheckRequest,
        http_request: Request,
        pipeline: RagPipelineDep,
        slots: QuestionSlotsDep,
    ) -> RecheckResponse:
        release = question_slot(slots, count_metric=False)
        stop = threading.Event()
        watcher = asyncio.create_task(_watch_disconnect(http_request, stop))

        def check() -> tuple[list[RecheckSourceResponse], list[SentenceCheck], bool]:
            states: list[RecheckSourceResponse] = []
            cited: dict[int, Any] = {}
            by_file: dict[str, list[str]] = {}
            for source in request.sources:
                by_file.setdefault(source.filename, []).append(source.chunk_id)
            now = {name: pipeline.passages_now(name, ids) for name, ids in by_file.items()}
            # Only documents a cited passage went from: every indexed one counted made
            # an answer citing five documents search them all together once one changed,
            # and the others crowded out its passages.
            changed = sorted(
                name
                for name, (indexed, found) in now.items()
                if indexed and any(chunk_id not in found for chunk_id in by_file[name])
            )
            within = _search_within(pipeline, changed)
            for source in request.sources:
                indexed, found = now[source.filename]
                passage = found.get(source.chunk_id)
                state = "unchanged" if passage else "changed" if indexed else "deleted"
                states.append(
                    RecheckSourceResponse(
                        source_number=source.source_number,
                        filename=source.filename,
                        chunk_id=source.chunk_id,
                        state=state,
                    )
                )
                # A number given twice keeps its first passage (it names one source).
                if source.source_number in cited:
                    continue
                if passage is not None:
                    cited[source.source_number] = [passage]
                elif indexed:
                    cited[source.source_number] = within(source.filename)
                else:
                    cited[source.source_number] = []
            checks, more = recheck(
                clean_text(request.answer), cited, pipeline.neighbour_text, stop.is_set
            )
            return states, checks, more

        try:
            states, checks, more = await run_in_threadpool(check)
        except VerifyAbandoned:
            raise HTTPException(status_code=499, detail="The client left.") from None
        finally:
            watcher.cancel()
            release()
        sentences = [_sentence_response(check) for check in checks]
        return RecheckResponse(
            sources=states,
            sentences=sentences,
            # Sentences past the checked ones are unknown: the answer isn't said to hold.
            still_holds=not more
            and all(s.state != "deleted" for s in states)
            and all(s.status not in ("unsupported", "numbers_differ") for s in sentences),
            more=more,
        )

    # The same hybrid search, RRF (k=60) and cross-encoder rerank a question uses
    # (RagPipeline.retrieve, which the eval harness also scores), so "find in my
    # documents" works even when the LLM is down. Read key allowed, like asking. It
    # takes a question slot: the rerank is the stage that bounds memory.
    @app.post(
        "/search",
        response_model=SearchResponse,
        dependencies=[Depends(require_api_key)],
        responses={404: {"description": "No passage has the like_chunk_id given."}},
    )
    def search(
        request: SearchRequest, pipeline: RagPipelineDep, slots: QuestionSlotsDep
    ) -> SearchResponse:
        release = question_slot(slots, count_metric=False)
        try:
            overrides = AnswerOverrides(rerank_top_k=request.rerank_top_k)
            if request.like_chunk_id is not None:
                similar = pipeline.retrieve_similar(
                    request.like_chunk_id,
                    overrides,
                    filenames=request.filenames,
                    tags=request.tags,
                    narrow=passage_narrowing(request),
                )
                if similar is None:
                    raise HTTPException(
                        status_code=404, detail=f"No passage with id '{request.like_chunk_id}'."
                    )
                phase = similar
            else:
                phase = pipeline.retrieve(
                    request.query,
                    overrides,
                    filenames=request.filenames,
                    tags=request.tags,
                    for_generation=False,
                    narrow=passage_narrowing(request),
                    explain=request.explain,
                )
        finally:
            release()
        # Highlighted as it was searched: retrieval cleans the query first (ligatures,
        # soft hyphens), so the raw one found the passage but marked nothing in it. A
        # passage search has no query words to mark.
        query_text = clean_text(request.query) if request.like_chunk_id is None else ""
        explanation = None
        if request.explain and (request.expect_chunk_id or request.expect_filename):
            found = explain(
                phase,
                pipeline.effective_settings(overrides),
                chunk_id=request.expect_chunk_id,
                filename=request.expect_filename,
                narrowed=bool(request.filenames or request.tags)
                or passage_narrowing(request) is not None,
            )
            explanation = SearchExplanationResponse(**dataclasses.asdict(found))
        ranks = phase.leg_ranks if request.explain else {}
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
                    page_end=chunk.page_end,
                    dense_rank=ranks.get(chunk.point_id, [None, None])[0],
                    sparse_rank=ranks.get(chunk.point_id, [None, None])[1],
                    spans=[
                        CitationSpanResponse(start=span.start, end=span.end, quote=span.quote)
                        for span in query_spans(query_text, chunk.text, chunk.section)
                    ],
                )
                for rank, chunk in enumerate(phase.context_chunks, start=1)
            ],
            timings=SearchTimingsResponse(
                embed_ms=phase.embed_ms, search_ms=phase.search_ms, rerank_ms=phase.rerank_ms
            ),
            weak_matches=phase.below_min_score,
            explanation=explanation,
        )
