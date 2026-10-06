"""Question routes: one-shot and streamed answers."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from api.admission import QuestionSlots, Release
from api.dependencies import (
    require_api_key,
)
from api.errors import public_error_message, question_error_type
from api.generation import GenerationConfigError
from api.metrics import (
    NO_ERROR_TYPE,
    answer_cache_hits_total,
    observe_question_timings,
    questions_total,
)
from api.pipeline import AnswerOverrides, timings_dict
from api.routes.deps import QuestionSlotsDep, RagPipelineDep, SettingsDep
from api.routes.responses import history_messages, log_question, timings_response
from api.schemas import (
    CitationResponse,
    QuestionRequest,
    QuestionResponse,
)
from api.settings import AppSettings
from api.sse import (
    SSE_HEADERS,
    SSE_KEEPALIVE_FRAME,
    SSE_KEEPALIVE_SECONDS,
    StreamLease,
    sse_event,
    with_keepalive,
)

logger = logging.getLogger(__name__)


QUESTIONS_BUSY_ERROR = (
    "The server is answering as many questions as it can right now. Try again in a moment."
)


def question_slot(slots: QuestionSlots, *, count_metric: bool = True) -> Release:
    """A question slot, or 429. ``count_metric=False`` for a search, which isn't a
    question and stays out of ``questions_total``."""

    release = slots.try_acquire()
    if release is None:
        if count_metric:
            questions_total.labels(outcome="error", error_type="busy").inc()
        raise HTTPException(
            status_code=429, detail=QUESTIONS_BUSY_ERROR, headers={"Retry-After": "5"}
        )
    return release


REQUEST_PROVIDERS = ("ollama", "openai", "openai_compatible")


def allowed_request_providers(settings: AppSettings) -> list[str]:
    """Providers a question may pick per request: the allowlist plus the default."""

    allowed = settings.allowed_request_providers
    if allowed is None:
        return list(REQUEST_PROVIDERS)
    default = settings.llm_provider.lower().strip()
    return [name for name in REQUEST_PROVIDERS if name in allowed or name == default]


def check_request_provider(provider: str | None, settings: AppSettings) -> None:
    """400 for a per-request provider the operator hasn't allowed — before a question
    slot is taken, so a refused request costs nothing."""

    if provider is None or provider in allowed_request_providers(settings):
        return
    raise GenerationConfigError(
        f"LLM provider '{provider}' is not allowed on this server (ALLOWED_REQUEST_PROVIDERS)."
    )


def register(app: FastAPI) -> None:
    """Register the question routes."""

    @app.post(
        "/questions", response_model=QuestionResponse, dependencies=[Depends(require_api_key)]
    )
    def answer_question(
        request: QuestionRequest,
        pipeline: RagPipelineDep,
        slots: QuestionSlotsDep,
        settings: SettingsDep,
    ) -> QuestionResponse:
        check_request_provider(request.llm_provider, settings)
        release = question_slot(slots)
        try:
            grounded = pipeline.answer(
                request.question,
                AnswerOverrides(
                    llm_provider=request.llm_provider,
                    rerank_top_k=request.rerank_top_k,
                    max_context_chunks=request.max_context_chunks,
                    llm_temperature=request.llm_temperature,
                ),
                filenames=request.filenames,
                history=history_messages(request.history),
                tags=request.tags,
                use_cache=request.use_cache,
            )
        except Exception as exc:
            questions_total.labels(outcome="error", error_type=question_error_type(exc)).inc()
            raise
        finally:
            release()
        questions_total.labels(outcome="ok", error_type=NO_ERROR_TYPE).inc()
        if grounded.cached:
            answer_cache_hits_total.inc()
        elif grounded.timings is not None:
            timings = timings_dict(grounded.timings)
            observe_question_timings(timings)
            log_question(
                provider=request.llm_provider,
                source_count=len(grounded.sources),
                timings=timings,
            )
        return QuestionResponse(
            answer=grounded.answer,
            sources=[
                CitationResponse(
                    source_number=source.source_number,
                    filename=source.filename,
                    page=source.page,
                    section=source.section,
                    chunk_id=source.chunk_id,
                    text=source.text,
                )
                for source in grounded.sources
            ],
            timings=timings_response(grounded.timings),
            trace_id=grounded.trace_id,
            cached=grounded.cached,
        )

    @app.post("/questions/stream", dependencies=[Depends(require_api_key)])
    def answer_question_stream(
        request: QuestionRequest,
        pipeline: RagPipelineDep,
        slots: QuestionSlotsDep,
        settings: SettingsDep,
    ) -> StreamingResponse:
        check_request_provider(request.llm_provider, settings)
        # Taken here, before the 200 is committed, so a full server can still say 429.
        lease = StreamLease(question_slot(slots))
        stop = threading.Event()
        overrides = AnswerOverrides(
            llm_provider=request.llm_provider,
            rerank_top_k=request.rerank_top_k,
            max_context_chunks=request.max_context_chunks,
            llm_temperature=request.llm_temperature,
        )

        def event_source() -> Iterator[str]:
            # The try/except below turns a mid-stream domain error into a clean SSE
            # `error` event instead of a raw 500 the client would otherwise never see
            # — the HTTP status is already committed by the time an error can occur
            # here (partway through an already-started streamed response).
            try:
                for event in with_keepalive(
                    pipeline.answer_stream(
                        request.question,
                        overrides,
                        filenames=request.filenames,
                        history=history_messages(request.history),
                        tags=request.tags,
                        use_cache=request.use_cache,
                        should_stop=stop.is_set,
                    ),
                    SSE_KEEPALIVE_SECONDS,
                    stop=stop,
                    on_start=lease.hand_off,
                    on_finish=lease.release,
                ):
                    if event is None:
                        yield SSE_KEEPALIVE_FRAME
                        continue
                    if event["type"] == "done" and event.get("cached"):
                        questions_total.labels(outcome="ok", error_type=NO_ERROR_TYPE).inc()
                        answer_cache_hits_total.inc()
                    elif event["type"] == "done":
                        questions_total.labels(outcome="ok", error_type=NO_ERROR_TYPE).inc()
                        observe_question_timings(event["timings"])
                        log_question(
                            provider=request.llm_provider,
                            source_count=len(event["sources"]),
                            timings=event["timings"],
                        )
                    yield sse_event(dict(event))
            except Exception as exc:
                questions_total.labels(outcome="error", error_type=question_error_type(exc)).inc()
                logger.warning("Streamed question failed: %s", exc, exc_info=True)
                yield sse_event({"type": "error", "detail": public_error_message(exc)})
            finally:
                lease.release_if_not_handed_off()

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
            # Also after a disconnect before the first frame, which never runs the
            # generator (and so never its finally). The release is idempotent.
            background=BackgroundTask(lease.release_if_not_handed_off),
        )
