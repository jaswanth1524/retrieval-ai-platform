"""Question routes: one-shot and streamed answers."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict
from typing import Any, NamedTuple, get_args

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from api.admission import QuestionSlots, Release, SlotKind
from api.answer_cache import CachedAnswer
from api.batch_jobs import BATCH_YIELD_SECONDS, BatchJob, batch_job_store, run_batch_job
from api.citations import SourceCitation, citation_from_dict
from api.dependencies import (
    require_api_key,
)
from api.document_names import stored_filename
from api.errors import public_error_message, question_error_type
from api.generation import GenerationConfigError, GroundedAnswer
from api.llm import ServerGenerationConfigError, completion_model_and_kwargs
from api.metrics import (
    NO_ERROR_TYPE,
    answer_cache_hits_total,
    answers_total,
    answers_truncated_total,
    observe_question_timings,
    questions_total,
    summaries_total,
)
from api.pipeline import (
    AnswerOverrides,
    ContinueRefused,
    DoneEvent,
    QuestionAbandoned,
    RagPipeline,
    StreamEvent,
    check_rerank_top_k,
    timings_dict,
)
from api.repository import PassageFilter
from api.reranking import RerankedChunk
from api.routes.deps import (
    DocumentSummarizerDep,
    QuestionSlotsDep,
    RagPipelineDep,
    SettingsDep,
    TraceStoreDep,
)
from api.routes.responses import (
    history_messages,
    log_question,
    timings_response,
    usage_response,
)
from api.schemas import (
    AnswerCheckResponse,
    BatchAnswerResponse,
    BatchJobResponse,
    BatchJobStartedResponse,
    BatchQuestionRequest,
    BatchQuestionResponse,
    CitationResponse,
    ContinueRequest,
    PassageNarrowing,
    QuestionRequest,
    QuestionResponse,
    RequestProvider,
    SummaryRequest,
    SummaryResponse,
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
from api.streaming import ClosingStreamingResponse
from api.summarize import SummaryRequest as PipelineSummaryRequest
from api.verify import self_check

logger = logging.getLogger(__name__)


QUESTIONS_BUSY_ERROR = (
    "The server is answering as many questions as it can right now. Try again in a moment."
)


SUMMARIES_BUSY_ERROR = (
    "The server is running as many summaries as it allows at once "
    "(MAX_CONCURRENT_SUMMARIES, within MAX_CONCURRENT_QUESTIONS). Try again in a moment."
)


def question_slot(
    slots: QuestionSlots, *, count_metric: bool = True, kind: SlotKind = "question"
) -> Release:
    """A question slot, or 429. ``count_metric=False`` for a search, which isn't a
    question and stays out of ``questions_total``. ``kind="summary"`` also takes one of
    the summaries' own slots."""

    release = slots.try_acquire(kind)
    if release is None:
        if count_metric:
            questions_total.labels(outcome="error", error_type="busy").inc()
        detail = SUMMARIES_BUSY_ERROR if kind == "summary" else QUESTIONS_BUSY_ERROR
        raise HTTPException(status_code=429, detail=detail, headers={"Retry-After": "5"})
    return release


def _no_slot() -> None:
    return None


# How long a request finding every slot busy waits for an identical run to show up.
IDENTICAL_RUN_SETTLE_SECONDS = 1.0


class Admission(NamedTuple):
    """How a question got in: its release (its slot, any slot taken later, the
    followers of a run it claimed), and — when it needn't run the question — the cached
    answer to replay or the identical run to wait for, or the run it now leads.
    ``take_slot`` gets a follower a slot of its own when the run it waited for left no
    answer; the release frees that one too."""

    release: Release
    cached: CachedAnswer | None = None
    follow: threading.Event | None = None
    claimed: threading.Event | None = None
    take_slot: Callable[[], bool] | None = None


Claim = Callable[[], "tuple[threading.Event, bool, Callable[[], None]] | None"]


class _Held:
    """The releases a request holds — its slot or follower place, and a slot taken
    late (a follower whose leader left no answer) — freed together, once."""

    def __init__(self, slots: QuestionSlots, *releases: Release) -> None:
        self._slots = slots
        self._releases = list(releases)
        self._lock = threading.Lock()
        self._closed = False

    def take_slot(self) -> bool:
        release = self._slots.try_acquire("question")
        if release is None:
            return False
        with self._lock:
            if not self._closed:
                self._releases.append(release)
                return True
        # Released already (the request ended while its run went on): nothing would
        # ever free this one.
        release()
        return False

    def release(self) -> None:
        with self._lock:
            self._closed = True
            releases, self._releases = self._releases, []
        for release in reversed(releases):
            release()


def _following(slots: QuestionSlots, event: threading.Event) -> Admission:
    """Admitted to wait for the identical run ``event`` — within the cap on waiting
    requests (each holds a request thread), else 429."""

    place = slots.try_follow()
    if place is None:
        questions_total.labels(outcome="error", error_type="busy").inc()
        raise HTTPException(
            status_code=429, detail=QUESTIONS_BUSY_ERROR, headers={"Retry-After": "5"}
        )
    held = _Held(slots, place)
    return Admission(held.release, follow=event, take_slot=held.take_slot)


def question_slot_unless_cached(
    slots: QuestionSlots,
    cached: Callable[[], CachedAnswer | None],
    running: Callable[[], threading.Event | None] | None = None,
    claim: Claim = lambda: None,
) -> Admission:
    """Admit a question. A stored answer is replayed as found (so it can't expire into
    a full run), and a question identical to one being answered waits for its answer —
    neither needs a slot, so a full server doesn't turn them away. Otherwise a slot
    (or 429), and the run is claimed at once: an identical question sent at the same
    moment finds it registered and waits instead of taking a slot or a 429. The claim
    is released with the slot, however the request ends (the pipeline finishes it
    earlier, once the answer is stored). ``running`` is None when the question can't
    join a run (no cache key: history, Regenerate): a full server then answers 429 at
    once instead of waiting for a run that can't appear."""

    release = slots.try_acquire("question")
    if release is None:
        hit = cached()
        if hit is not None:
            return Admission(_no_slot, cached=hit)
        if running is not None:
            # A run claimed a moment ago may still be registering: wait briefly for it.
            deadline = time.monotonic() + IDENTICAL_RUN_SETTLE_SECONDS
            while True:
                event = running()
                if event is not None:
                    return _following(slots, event)
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        return Admission(question_slot(slots))
    try:
        claimed = claim()
        if claimed is None:
            hit = cached()
            if hit is not None:
                release()
                return Admission(_no_slot, cached=hit)
            return Admission(release)
        event, leading, finish = claimed
        if not leading:
            # Someone is answering it already: wait for that, and free the slot.
            release()
            return _following(slots, event)
        # Claimed first, then looked up: a leader that stored its answer and finished
        # between a lookup and the claim would otherwise have been run again.
        try:
            hit = cached()
        except BaseException:
            finish()
            raise
        if hit is not None:
            finish()
            release()
            return Admission(_no_slot, cached=hit)
    except BaseException:
        release()
        raise
    slot_release = release

    def release_both() -> None:
        try:
            finish()
        finally:
            slot_release()

    return Admission(release_both, claimed=event)


def joinable_run(
    pipeline: RagPipeline, lookup: tuple[Any, ...], use_cache: bool
) -> Callable[[], threading.Event | None] | None:
    """How admission finds an identical run of this question — None when it can't have
    one (Regenerate, or no cache key: history, the cache off)."""

    if not use_cache or not pipeline.joinable(*lookup):
        return None
    return lambda: pipeline.identical_run(*lookup)


def answer_checks(
    settings: AppSettings,
    status: str,
    answer: str,
    sources: Sequence[SourceCitation],
    widen: Callable[[RerankedChunk], str] | None = None,
    read: Callable[[Sequence[SourceCitation]], Mapping[str, RerankedChunk]] | None = None,
) -> list[dict[str, Any]] | None:
    """ANSWER_SELF_CHECK: the answer's sentences against the passages it cites (no LLM,
    no search) — on cache hits and continued answers too, worked out from the answer and
    its sources alone. None when it is off or there is nothing to check."""

    if not settings.answer_self_check or status != "answered" or not sources:
        return None
    try:
        whole = read(sources) if read is not None else None
        checks = self_check(answer, sources, _remembered(widen) if widen else None, whole)
    except Exception:  # noqa: BLE001 - advice only; the answer stands without it
        logger.warning("The answer's self-check failed.", exc_info=True)
        return None
    return [
        {
            "sentence": check.sentence,
            "status": check.status,
            "source_number": _source_number(check.chunk, sources),
            "quote": check.quote,
            "missing_numbers": list(check.missing_numbers),
        }
        for check in checks
    ]


def _remembered(widen: Callable[[RerankedChunk], str]) -> Callable[[RerankedChunk], str]:
    """``widen`` read once per passage (each is a Qdrant read)."""

    seen: dict[str, str] = {}

    def once(chunk: RerankedChunk) -> str:
        if chunk.chunk_id not in seen:
            seen[chunk.chunk_id] = widen(chunk)
        return seen[chunk.chunk_id]

    return once


def _source_number(chunk: RerankedChunk | None, sources: Sequence[SourceCitation]) -> int | None:
    if chunk is None:
        return None
    return next((s.source_number for s in sources if s.chunk_id == chunk.chunk_id), None)


def with_answer_checks(
    settings: AppSettings,
    event: DoneEvent,
    pipeline: RagPipeline | None = None,
    overrides: AnswerOverrides | None = None,
) -> None:
    """``answer_checks`` onto a stream's ``done`` frame (it goes out right after)."""

    checks = answer_checks(
        settings,
        str(event.get("answer_status", "answered")),
        event["answer"],
        [citation_from_dict(dict(source)) for source in event["sources"]],
        pipeline.widener(overrides or AnswerOverrides()) if pipeline is not None else None,
        pipeline.cited_passages if pipeline is not None else None,
    )
    if checks is not None:
        event["verification"] = checks


def check_question_text(question: str) -> None:
    """400 for a blank question, before a slot is taken: it used to take one (and,
    with history, a condense call) before the pipeline refused it, and the stream
    refused it only after its 200, in an error frame."""

    if not question.strip():
        questions_total.labels(outcome="error", error_type="invalid_request").inc()
        raise HTTPException(status_code=400, detail="Query text is required.")


# How often the sync question route checks whether its client is still there.
_DISCONNECT_POLL_SECONDS = 0.5


async def _watch_disconnect(request: Request, stop: threading.Event) -> None:
    """Set ``stop`` once ``request``'s client has gone."""

    while not stop.is_set():
        if await request.is_disconnected():
            stop.set()
            return
        await asyncio.sleep(_DISCONNECT_POLL_SECONDS)


def count_answer(status: str, truncated: bool) -> None:
    """One answer returned (generated or from the cache), by grounding status."""

    answers_total.labels(status=status).inc()
    if truncated:
        answers_truncated_total.inc()


REQUEST_PROVIDERS: tuple[str, ...] = get_args(RequestProvider)


def allowed_request_providers(settings: AppSettings) -> list[str]:
    """Providers a question may pick per request: the allowlist plus the default."""

    allowed = settings.allowed_request_providers
    if allowed is None:
        return list(REQUEST_PROVIDERS)
    default = settings.llm_provider.lower().strip()
    return [name for name in REQUEST_PROVIDERS if name in allowed or name == default]


# Anthropic's API refuses a temperature above this (OpenAI's go to 2).
ANTHROPIC_TEMPERATURE_MAX = 1.0


def check_request_provider(
    provider: str | None, settings: AppSettings, temperature: float | None = None
) -> None:
    """400 for a per-request provider the operator hasn't allowed or hasn't configured
    (``openai`` with no OPENAI_API_KEY), or a temperature the effective provider refuses
    — before a question slot is taken, so a refused request costs nothing. Unchecked, it
    ran embedding, search and the rerank first and failed only at generation, after the
    stream had already sent its sources."""

    effective = (provider or settings.llm_provider).lower().strip()
    effective_temperature = temperature if temperature is not None else settings.llm_temperature
    if effective == "anthropic" and float(effective_temperature) > ANTHROPIC_TEMPERATURE_MAX:
        raise GenerationConfigError(
            f"Anthropic models take a temperature of at most {ANTHROPIC_TEMPERATURE_MAX:g}; "
            f"got {float(effective_temperature):g}."
        )
    if provider is None:
        # The server's default: checked here too, so a missing key costs no search and
        # rerank, and is reported as the server's fault (503), not the request's.
        try:
            completion_model_and_kwargs(settings)
        except GenerationConfigError as exc:
            raise ServerGenerationConfigError(str(exc)) from exc
        return
    if provider not in allowed_request_providers(settings):
        raise GenerationConfigError(
            f"LLM provider '{provider}' is not allowed on this server (ALLOWED_REQUEST_PROVIDERS)."
        )
    completion_model_and_kwargs(settings.model_copy(update={"llm_provider": provider}))


def summary_request(request: SummaryRequest) -> PipelineSummaryRequest:
    """The pipeline's request: names checked like any document path parameter."""

    narrowed = request.page_min is not None or request.page_max is not None or request.section
    low, high = request.page_min, request.page_max
    if low is not None and high is not None and low > high:
        low, high = high, low
    return PipelineSummaryRequest(
        mode=request.mode,
        filenames=tuple(stored_filename(name) for name in request.filenames),
        instruction=(request.instruction or "").strip() or None,
        llm_provider=request.llm_provider,
        llm_temperature=request.llm_temperature,
        narrow=(
            PassageFilter(page_min=low, page_max=high, section=request.section)
            if narrowed
            else None
        ),
    )


def passage_narrowing(request: PassageNarrowing) -> PassageFilter | None:
    """A request's page/section/passage/exclusion narrowing, or None when it has none.

    Lists are sets here, sorted and deduplicated: the narrowing is part of the answer
    cache and in-flight keys, and the same passages left out in another click order
    missed both.
    """

    narrow = PassageFilter(
        page_min=request.page_min,
        page_max=request.page_max,
        section=request.section,
        exclude_filenames=tuple(sorted(set(request.exclude_filenames or ()))),
        chunk_ids=tuple(sorted(set(request.chunk_ids or ()))),
        exclude_chunk_ids=tuple(sorted(set(request.exclude_chunk_ids or ()))),
        uploaded_after=request.uploaded_after,
        uploaded_before=request.uploaded_before,
    )
    return None if narrow == PassageFilter() else narrow


def question_overrides(request: QuestionRequest) -> AnswerOverrides:
    """Everything a question changes about how it is answered, for the pipeline."""

    return AnswerOverrides(
        llm_provider=request.llm_provider,
        rerank_top_k=request.rerank_top_k,
        max_context_chunks=request.max_context_chunks,
        llm_temperature=request.llm_temperature,
        answer_format=request.answer_format,
        answer_language=request.answer_language,
        narrow=passage_narrowing(request),
        passages_only=request.passages_only,
        strong_matches_only=request.strong_matches_only,
    )


def sse_response(
    events: Iterator[StreamEvent],
    lease: StreamLease,
    stop: threading.Event,
    *,
    on_done: Callable[[DoneEvent], None],
    on_error: Callable[[Exception], None],
    what: str,
) -> ClosingStreamingResponse:
    """A question's or summary's events as an SSE response, with keepalives.

    A mid-stream error becomes an SSE ``error`` event instead of a raw 500 the client
    would never see: the status is committed by the time one can happen. Closing (not a
    plain StreamingResponse): a client that leaves mid-answer closes the frames at once,
    which sets ``stop`` and stops the pipeline thread before its next stage, instead of
    whenever the generator was collected.
    """

    def frames() -> Iterator[str]:
        keepalive = with_keepalive(
            events,
            SSE_KEEPALIVE_SECONDS,
            stop=stop,
            on_start=lease.hand_off,
            on_finish=lease.release,
        )
        try:
            for event in keepalive:
                if event is None:
                    yield SSE_KEEPALIVE_FRAME
                    continue
                if event["type"] == "done":
                    on_done(event)
                yield sse_event(dict(event))
        except Exception as exc:
            on_error(exc)
            logger.warning("Streamed %s failed: %s", what, exc, exc_info=True)
            yield sse_event({"type": "error", "detail": public_error_message(exc)})
        finally:
            # Sets `stop` when this generator is closed early (the client left).
            keepalive.close()
            lease.release_if_not_handed_off()

    return ClosingStreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
        # Stops the pipeline as soon as the client leaves (see the class).
        on_disconnect=stop.set,
        # Also after a disconnect before the first frame, which never runs the
        # generator (and so never its finally). The release is idempotent.
        background=BackgroundTask(lease.release_if_not_handed_off),
    )


router = APIRouter()


@router.post("/questions", response_model=QuestionResponse, dependencies=[Depends(require_api_key)])
async def answer_question(
    request: QuestionRequest,
    http_request: Request,
    pipeline: RagPipelineDep,
    slots: QuestionSlotsDep,
    settings: SettingsDep,
) -> QuestionResponse:
    check_request_provider(request.llm_provider, settings, request.llm_temperature)
    check_rerank_top_k(request.rerank_top_k, settings)
    check_question_text(request.question)
    lookup = (
        request.question,
        question_overrides(request),
        request.filenames,
        request.tags,
        history_messages(request.history),
    )
    # In the threadpool: on a full server it may wait a moment for an identical run.
    admitted = await run_in_threadpool(
        question_slot_unless_cached,
        slots,
        lambda: pipeline.cached_answer(*lookup) if request.use_cache else None,
        joinable_run(pipeline, lookup, request.use_cache),
        lambda: pipeline.claim_run(*lookup) if request.use_cache else None,
    )
    release, hit = admitted.release, admitted.cached
    # A client that leaves (a CLI timeout, a closed connection) stops the pipeline
    # before its next expensive stage, as on the stream: the slot, the local model
    # and the LLM call were otherwise held to the end for nobody.
    stop = threading.Event()
    watcher = asyncio.create_task(_watch_disconnect(http_request, stop))
    try:
        grounded = await run_in_threadpool(
            lambda: pipeline.answer(
                request.question,
                question_overrides(request),
                filenames=request.filenames,
                history=history_messages(request.history),
                tags=request.tags,
                use_cache=request.use_cache,
                should_stop=stop.is_set,
                cached=hit,
                follow=admitted.follow,
                claimed=admitted.claimed,
                take_slot=admitted.take_slot,
            )
        )
    except QuestionAbandoned:
        questions_total.labels(outcome="error", error_type="abandoned").inc()
        # Nobody reads it; a status is still needed for the access log.
        raise HTTPException(status_code=499, detail="The client left.") from None
    except Exception as exc:
        questions_total.labels(outcome="error", error_type=question_error_type(exc)).inc()
        raise
    finally:
        watcher.cancel()
        release()
    # Off the event loop: the self-check reads the cited passages from Qdrant.
    return await run_in_threadpool(answered_response, grounded, request, settings, pipeline)


def answered_response(
    grounded: GroundedAnswer,
    request: QuestionRequest,
    settings: AppSettings,
    pipeline: RagPipeline,
) -> QuestionResponse:
    """A sync answer's metrics, log line and response body (one question of a batch
    too)."""

    questions_total.labels(outcome="ok", error_type=NO_ERROR_TYPE).inc()
    count_answer(grounded.answer_status, grounded.truncated)
    if grounded.cached:
        answer_cache_hits_total.inc()
    elif grounded.timings is not None:
        timings = timings_dict(grounded.timings)
        observe_question_timings(timings, prompt_cached=grounded.prompt_cached)
        log_question(
            provider=request.llm_provider,
            source_count=len(grounded.sources),
            timings=timings,
        )
    return QuestionResponse(
        answer=grounded.answer,
        sources=[CitationResponse(**asdict(source)) for source in grounded.sources],
        timings=timings_response(grounded.timings),
        trace_id=grounded.trace_id,
        cached=grounded.cached,
        answer_status=grounded.answer_status,
        top_rerank_score=grounded.top_rerank_score,
        rerank_min_score=float(settings.rerank_min_score),
        weak_matches=grounded.weak_matches,
        truncated=grounded.truncated,
        citation_backfilled=grounded.citation_backfilled,
        unsupported_claims=grounded.unsupported_claims,
        unsupported_numbers=grounded.unsupported_numbers,
        prompt_cached=grounded.prompt_cached,
        prompt_cut=grounded.prompt_cut,
        retrieval_query=grounded.retrieval_query,
        passages_left_out=grounded.passages_left_out,
        followups=grounded.followups,
        usage=usage_response(grounded.usage, grounded.cost_usd),
        verification=[
            AnswerCheckResponse(**check)
            for check in answer_checks(
                settings,
                grounded.answer_status,
                grounded.answer,
                grounded.sources,
                pipeline.widener(question_overrides(request)),
                pipeline.cited_passages,
            )
            or []
        ]
        or None,
    )


# Several questions over the same scope, answered one after another under one question
# slot (a checklist over a contract, a set of audit questions): each is answered as
# /questions answers it, and one that fails doesn't stop the rest.
@router.post(
    "/questions/batch",
    response_model=BatchQuestionResponse,
    dependencies=[Depends(require_api_key)],
)
async def answer_questions(
    request: BatchQuestionRequest,
    http_request: Request,
    pipeline: RagPipelineDep,
    slots: QuestionSlotsDep,
    settings: SettingsDep,
) -> BatchQuestionResponse:
    check_request_provider(request.llm_provider, settings, request.llm_temperature)
    check_rerank_top_k(request.rerank_top_k, settings)
    for question in request.questions:
        check_question_text(question)
    overrides = question_overrides(request)
    release = question_slot(slots)
    stop = threading.Event()
    watcher = asyncio.create_task(_watch_disconnect(http_request, stop))
    answers: list[BatchAnswerResponse] = []
    try:
        for question in request.questions:
            if stop.is_set():
                raise HTTPException(status_code=499, detail="The client left.")
            one = request.model_copy(update={"question": question, "questions": [question]})
            try:
                grounded = await run_in_threadpool(
                    functools.partial(
                        pipeline.answer,
                        question,
                        overrides,
                        filenames=request.filenames,
                        tags=request.tags,
                        use_cache=request.use_cache,
                        should_stop=stop.is_set,
                    )
                )
            except QuestionAbandoned:
                raise HTTPException(status_code=499, detail="The client left.") from None
            except Exception as exc:  # noqa: BLE001 - reported on this question; the rest go on
                questions_total.labels(outcome="error", error_type=question_error_type(exc)).inc()
                answers.append(
                    BatchAnswerResponse(question=question, error=public_error_message(exc))
                )
                continue
            answers.append(
                BatchAnswerResponse(
                    question=question,
                    answer=await run_in_threadpool(
                        answered_response, grounded, one, settings, pipeline
                    ),
                )
            )
    finally:
        watcher.cancel()
        release()
    return BatchQuestionResponse(answers=answers)


# The same batch on a thread of its own: answers kept as they come, read with GET
# (again after a dropped connection), each question waiting for a slot rather than
# failing with a 429. Read key allowed, like asking.
@router.post(
    "/questions/batch/jobs",
    status_code=202,
    response_model=BatchJobStartedResponse,
    dependencies=[Depends(require_api_key)],
    responses={409: {"description": "As many batches as allowed are running already."}},
)
def start_batch_job(
    request: BatchQuestionRequest,
    pipeline: RagPipelineDep,
    slots: QuestionSlotsDep,
    settings: SettingsDep,
) -> BatchJobStartedResponse:
    check_request_provider(request.llm_provider, settings, request.llm_temperature)
    check_rerank_top_k(request.rerank_top_k, settings)
    for question in request.questions:
        check_question_text(question)
    overrides = question_overrides(request)
    job = batch_job_store().add(str(settings.qdrant_collection), list(request.questions))
    if job is None:
        raise HTTPException(
            status_code=409,
            detail="As many batches as allowed are running; wait for one to finish.",
        )

    def answer(question: str, should_stop: Callable[[], bool]) -> BatchAnswerResponse:
        grounded = pipeline.answer(
            question,
            overrides,
            filenames=request.filenames,
            tags=request.tags,
            use_cache=request.use_cache,
            should_stop=should_stop,
        )
        one = request.model_copy(update={"question": question, "questions": [question]})
        return BatchAnswerResponse(
            question=question, answer=answered_response(grounded, one, settings, pipeline)
        )

    def failed(question: str, exc: Exception) -> BatchAnswerResponse:
        questions_total.labels(outcome="error", error_type=question_error_type(exc)).inc()
        return BatchAnswerResponse(question=question, error=public_error_message(exc))

    try:
        _start_batch_thread(
            job,
            lambda: slots.try_acquire("batch"),
            answer,
            failed,
            BATCH_YIELD_SECONDS,
        )
    except BaseException:
        # Never started: ended, so it doesn't hold one of the running places for good.
        job.state, job.error, job.finished_at = "failed", "it couldn't be started", time.time()
        raise
    return BatchJobStartedResponse(batch_id=job.id, total=len(job.questions))


def _start_batch_thread(job: BatchJob, *args: Any) -> None:
    # In the request's context, so its traces and log lines carry its request id.
    threading.Thread(
        target=contextvars.copy_context().run,
        args=(run_batch_job, job, *args),
        name=f"docrag-batch-{job.id[:8]}",
        daemon=True,
    ).start()


def _batch_job(batch_id: str, settings: AppSettings) -> BatchJob:
    job = batch_job_store().get(batch_id, str(settings.qdrant_collection))
    if job is None:
        raise HTTPException(status_code=404, detail="No such batch here (or it was forgotten).")
    return job


def _batch_job_response(job: BatchJob) -> BatchJobResponse:
    # The state before the answers: read after them, a job could finish in between and
    # read as "done" with its last answer missing.
    state, finished_at, error = job.state, job.finished_at, job.error
    answers = list(job.answers)
    return BatchJobResponse(
        batch_id=job.id,
        state=state,
        total=len(job.questions),
        done=len(answers),
        answers=answers,
        created_at=job.created_at,
        finished_at=finished_at,
        error=error,
    )


@router.get(
    "/questions/batch/jobs/{batch_id}",
    response_model=BatchJobResponse,
    dependencies=[Depends(require_api_key)],
    responses={404: {"description": "No such batch in this workspace."}},
)
def batch_job_status(batch_id: str, settings: SettingsDep) -> BatchJobResponse:
    return _batch_job_response(_batch_job(batch_id, settings))


@router.post(
    "/questions/batch/jobs/{batch_id}/cancel",
    response_model=BatchJobResponse,
    dependencies=[Depends(require_api_key)],
    responses={404: {"description": "No such batch in this workspace."}},
)
def cancel_batch_job(batch_id: str, settings: SettingsDep) -> BatchJobResponse:
    """Stop before the next question (the one being answered finishes)."""

    job = _batch_job(batch_id, settings)
    job.cancel.set()
    return _batch_job_response(job)


@router.post(
    "/questions/continue",
    dependencies=[Depends(require_api_key)],
    responses={
        404: {"description": "No such trace here, or traces are off."},
        409: {"description": "That answer wasn't cut off, or its passages changed."},
    },
)
def continue_answer_stream(
    request: ContinueRequest,
    pipeline: RagPipelineDep,
    slots: QuestionSlotsDep,
    settings: SettingsDep,
    traces: TraceStoreDep,
) -> ClosingStreamingResponse:
    """The rest of an answer cut off at the token limit, streamed like
    /questions/stream (sources, deltas, done with the whole answer)."""

    try:
        plan = pipeline.continuable(request.trace_id, traces.get(request.trace_id), settings)
    except ContinueRefused as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    # The answer is continued with the provider it was written with: checked now, as a
    # question's is, so a provider since refused or left without its key is a 400/503
    # before the 200 — not an error frame after the sources.
    config = plan.trace.config
    if config is None:
        check_request_provider(None, settings)
    else:
        provider = config.llm_provider
        check_request_provider(
            None if provider == settings.llm_provider else provider,
            settings,
            config.llm_temperature,
        )
    lease = StreamLease(question_slot(slots))
    stop = threading.Event()

    def on_done(event: DoneEvent) -> None:
        count_answer(event.get("answer_status", "answered"), event.get("truncated", False))
        questions_total.labels(outcome="ok", error_type=NO_ERROR_TYPE).inc()
        # As the answer being continued had its prompt.
        alone = config is not None and config.context_neighbor_radius == 0
        with_answer_checks(settings, event, pipeline, AnswerOverrides(passages_only=alone))

    def on_error(exc: Exception) -> None:
        questions_total.labels(outcome="error", error_type=question_error_type(exc)).inc()

    return sse_response(
        pipeline.continue_stream(plan, settings, should_stop=stop.is_set),
        lease,
        stop,
        on_done=on_done,
        on_error=on_error,
        what="continued answer",
    )


@router.post("/questions/stream", dependencies=[Depends(require_api_key)])
def answer_question_stream(
    request: QuestionRequest,
    pipeline: RagPipelineDep,
    slots: QuestionSlotsDep,
    settings: SettingsDep,
) -> ClosingStreamingResponse:
    check_request_provider(request.llm_provider, settings, request.llm_temperature)
    check_rerank_top_k(request.rerank_top_k, settings)
    check_question_text(request.question)
    # Taken here, before the 200 is committed, so a full server can still say 429.
    overrides = question_overrides(request)
    lookup = (
        request.question,
        overrides,
        request.filenames,
        request.tags,
        history_messages(request.history),
    )
    admitted = question_slot_unless_cached(
        slots,
        lambda: pipeline.cached_answer(*lookup) if request.use_cache else None,
        joinable_run(pipeline, lookup, request.use_cache),
        lambda: pipeline.claim_run(*lookup) if request.use_cache else None,
    )
    release, hit = admitted.release, admitted.cached
    lease = StreamLease(release)
    stop = threading.Event()

    def on_done(event: DoneEvent) -> None:
        count_answer(event.get("answer_status", "answered"), event.get("truncated", False))
        questions_total.labels(outcome="ok", error_type=NO_ERROR_TYPE).inc()
        with_answer_checks(settings, event, pipeline, overrides)
        if event.get("cached"):
            answer_cache_hits_total.inc()
            return
        observe_question_timings(event["timings"], prompt_cached=bool(event.get("prompt_cached")))
        log_question(
            provider=request.llm_provider,
            source_count=len(event["sources"]),
            timings=event["timings"],
        )

    def on_error(exc: Exception) -> None:
        questions_total.labels(outcome="error", error_type=question_error_type(exc)).inc()

    return sse_response(
        pipeline.answer_stream(
            request.question,
            overrides,
            filenames=request.filenames,
            history=history_messages(request.history),
            tags=request.tags,
            use_cache=request.use_cache,
            should_stop=stop.is_set,
            cached=hit,
            follow=admitted.follow,
            claimed=admitted.claimed,
            take_slot=admitted.take_slot,
        ),
        lease,
        stop,
        on_done=on_done,
        on_error=on_error,
        what="question",
    )


# Whole-document tasks: the same admission, provider check, key guard, rate limit
# and answer frames as a question, over every passage of the named documents
# instead of the retrieved few (api/summarize.py).
@router.post(
    "/questions/summary",
    response_model=SummaryResponse,
    dependencies=[Depends(require_api_key)],
)
async def summarize_documents(
    request: SummaryRequest,
    http_request: Request,
    summarizer: DocumentSummarizerDep,
    slots: QuestionSlotsDep,
    settings: SettingsDep,
) -> SummaryResponse:
    check_request_provider(request.llm_provider, settings, request.llm_temperature)
    task = summary_request(request)
    await run_in_threadpool(summarizer.check_documents, task.filenames, task.narrow)
    release = question_slot(slots, count_metric=False, kind="summary")
    # As /questions: a client that leaves stops the summary before its next part,
    # instead of every part running (and both slots held) for nobody.
    stop = threading.Event()
    watcher = asyncio.create_task(_watch_disconnect(http_request, stop))
    try:
        result = await run_in_threadpool(
            lambda: summarizer.summarize(task, use_cache=request.use_cache, should_stop=stop.is_set)
        )
    except QuestionAbandoned:
        summaries_total.labels(mode=request.mode, outcome="error").inc()
        raise HTTPException(status_code=499, detail="The client left.") from None
    except Exception:
        summaries_total.labels(mode=request.mode, outcome="error").inc()
        raise
    finally:
        watcher.cancel()
        release()
    summaries_total.labels(mode=request.mode, outcome="ok").inc()
    count_answer(result.grounded.answer_status, result.grounded.truncated)
    if result.grounded.cached:
        answer_cache_hits_total.inc()
    grounded = result.grounded
    return SummaryResponse(
        answer=grounded.answer,
        sources=[CitationResponse(**asdict(source)) for source in grounded.sources],
        timings=timings_response(grounded.timings),
        trace_id=grounded.trace_id,
        cached=grounded.cached,
        answer_status=grounded.answer_status,
        truncated=grounded.truncated,
        usage=usage_response(grounded.usage, grounded.cost_usd),
        citation_backfilled=grounded.citation_backfilled,
        unsupported_claims=grounded.unsupported_claims,
        unsupported_numbers=grounded.unsupported_numbers,
        prompt_cached=grounded.prompt_cached,
        prompt_cut=grounded.prompt_cut,
        chunks_read=result.chunks_read,
        chunks_total=result.chunks_total,
        parts_skipped=result.parts_skipped,
        notes_truncated=result.notes_truncated,
    )


@router.post("/questions/summary/stream", dependencies=[Depends(require_api_key)])
def summarize_documents_stream(
    request: SummaryRequest,
    summarizer: DocumentSummarizerDep,
    slots: QuestionSlotsDep,
    settings: SettingsDep,
) -> ClosingStreamingResponse:
    check_request_provider(request.llm_provider, settings, request.llm_temperature)
    task = summary_request(request)
    # Before the 200: an unknown or oversized document is a plain 404 / 400.
    summarizer.check_documents(task.filenames, task.narrow)
    lease = StreamLease(question_slot(slots, count_metric=False, kind="summary"))
    stop = threading.Event()

    def on_done(event: DoneEvent) -> None:
        summaries_total.labels(mode=request.mode, outcome="ok").inc()
        count_answer(event.get("answer_status", "answered"), event.get("truncated", False))
        if event.get("cached"):
            answer_cache_hits_total.inc()

    return sse_response(
        summarizer.summarize_stream(task, use_cache=request.use_cache, should_stop=stop.is_set),
        lease,
        stop,
        on_done=on_done,
        on_error=lambda exc: summaries_total.labels(mode=request.mode, outcome="error").inc(),
        what="summary",
    )


def register(app: FastAPI) -> None:
    """Register the question routes."""

    app.include_router(router)
