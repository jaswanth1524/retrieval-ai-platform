"""The streamed answer (``POST /questions/stream``) and continuing a cut-off one: the
top layer of ``RagPipeline``, over ``api/pipeline_core.py``.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import replace

from api.answer_cache import CachedAnswer
from api.answer_events import (
    DoneEvent,
    StageEvent,
    StreamEvent,
)
from api.answer_run import (
    AnswerOverrides,
    AnswerRun,
    RetrievalPhase,
    _citation_dict,
    _cited_chunks,
    _int_or_none,
    _picked_left_out,
    _reply_known,
    _Streamed,
    _top_rerank_score,
    replay_cached,
    timings_dict,
    trace_status,
    usage_dict,
)
from api.citations import (
    answer_status,
)
from api.context_expansion import expand_with_neighbors
from api.continuation import (
    ContinuePlan,
    ContinueRefused,
    base_prompt,
    continuation_messages,
    drop_restart,
    join_continuation,
)
from api.errors import public_error_message
from api.generation import (
    ChatGenerator,
    ChatMessage,
    GenerationError,
    StageTimings,
    answer_messages,
    finish_answer,
    generate_followups,
    insufficient_context_answer,
    source_citations,
    will_call_llm_for_citations,
)
from api.llm import (
    GenerationSlot,
    begin_usage,
    expected_prefill_ms,
    reset_finish_reason,
    stopped_at_token_limit,
    usage_cost_usd,
    usage_totals,
    was_prompt_cut,
)
from api.pipeline_core import PipelineCore
from api.prompting import (
    FittedPrompt,
    estimate_prompt_tokens,
)
from api.repository import PassageFilter, VectorRepository
from api.request_id import current_request_id
from api.reranking import (
    RerankedChunk,
)
from api.run_coordination import QuestionAbandoned, cached_run, follow_only, wait_for_model
from api.settings import AppSettings
from api.tracing import (
    QueryTrace,
)
from api.workspaces import workspace_of


class StreamingAnswers(PipelineCore):
    """The streamed answer and its continuation (see ``RagPipeline``)."""

    def answer_stream(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
        *,
        tags: Sequence[str] | None = None,
        use_cache: bool = True,
        should_stop: Callable[[], bool] | None = None,
        cached: CachedAnswer | None = None,
        follow: threading.Event | None = None,
        claimed: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        """``_answer_stream_uncached``, replayed from the answer cache on a hit
        (``cached``: the hit already looked up, replayed as it is).

        ``should_stop`` (the client has gone) is polled between stages, so an
        abandoned question stops before the rerank or the LLM call instead of
        finishing work nobody will read.
        """

        started = time.monotonic()

        def replay(hit: CachedAnswer) -> Iterator[StreamEvent]:
            return replay_cached(
                hit,
                started,
                top_rerank_score=hit.top_rerank_score,
                # In the cache key: the bar it was answered under is today's.
                rerank_min_score=float(self._settings.rerank_min_score),
                weak_matches=hit.weak_matches,
            )

        if cached is not None:
            self._log_hit(cached, question, "stream", filenames, tags, overrides)
            yield from replay(cached)
            return
        key = self._cache_key(question, overrides, filenames, tags, history)
        if follow is not None and key is not None and self._answer_cache is not None:
            # Admitted without a slot to wait for the identical run: its answer, or the
            # error frame of a full server (NoSlotForRun).
            followed = yield from follow_only(self._answer_cache, key, follow, should_stop)
            if followed is None:
                return
            self._log_hit(followed, question, "stream", filenames, tags, overrides)
            yield from replay(followed)
            return
        yield from cached_run(
            self._answer_cache,
            key,
            use_cache=use_cache,
            should_stop=should_stop,
            replay=replay,
            run=lambda: self._answer_stream_uncached(
                question, overrides, filenames, history, tags, should_stop, use_memo=use_cache
            ),
            # A cut-off answer isn't kept: asked again, it may come out whole.
            # Nor one whose prompt the provider cut short (its grounding rules gone).
            keep=lambda done: not done.get("truncated") and not done.get("prompt_cut"),
            on_hit=lambda hit: self._log_hit(hit, question, "stream", filenames, tags, overrides),
            claimed=claimed,
        )

    def continuable(
        self, trace_id: str, trace: QueryTrace | None, settings: AppSettings
    ) -> ContinuePlan:
        """What continuing the cut-off answer of trace ``trace_id`` (``trace``, as the
        trace store has it) needs: its prompt, its answer so far and its passages in
        source order. ``ContinueRefused`` (with the HTTP status to give) when it can't
        be continued — checked before the question slot and the 200."""

        if self._trace_store is None:
            raise ContinueRefused(
                404, "Traces are off (TRACE_ENABLED), so answers can't be continued."
            )
        if trace is None or trace.workspace != workspace_of(settings):
            raise ContinueRefused(404, f"No trace found with id '{trace_id}'.")
        if trace.kind != "question":
            # A summary's trace holds no passages to check citations against.
            raise ContinueRefused(409, "Only a question's answer can be continued.")
        # Cut off at the output limit, or stopped (or broken off) part-way: either way an
        # answer begun from a prompt the trace kept.
        cut_short = trace.truncated or trace.status == "error"
        if not cut_short or not trace.answer or not trace.prompt_messages:
            raise ContinueRefused(409, "That answer wasn't cut off; there is nothing to continue.")
        prompt_sources = sorted(
            (c for c in trace.candidates if c.source_number is not None),
            key=lambda c: c.source_number or 0,
        )
        wanted: dict[str, list[str]] = {}
        for candidate in prompt_sources:
            wanted.setdefault(candidate.filename, []).append(candidate.chunk_id)
        payloads: dict[str, dict[str, object]] = {}
        for filename, chunk_ids in wanted.items():
            for payload in self._repository.chunks_for_filename(
                settings, filename, PassageFilter(chunk_ids=tuple(chunk_ids))
            ):
                payloads[str(payload.get("chunk_id"))] = payload
        chunks: list[RerankedChunk] = []
        for candidate in prompt_sources:
            found = payloads.get(candidate.chunk_id)
            if found is None:
                # Numbered sources the answer cites by number: one gone shifts them all.
                raise ContinueRefused(
                    409,
                    "A passage that answer was written from has changed or been deleted "
                    "since; ask the question again instead.",
                )
            chunks.append(
                RerankedChunk(
                    point_id=candidate.point_id,
                    filename=candidate.filename,
                    page=candidate.page,
                    section=candidate.section,
                    chunk_id=candidate.chunk_id,
                    text=str(found.get("text", "")),
                    retrieval_score=candidate.retrieval_score,
                    rerank_score=candidate.rerank_score or 0.0,
                    page_end=_int_or_none(found.get("page_end")),
                    chunk_ordinal=_int_or_none(found.get("chunk_ordinal")),
                )
            )
        return ContinuePlan(trace, _with_neighbour_text(chunks, trace, self._repository, settings))

    def continue_stream(
        self,
        plan: ContinuePlan,
        settings: AppSettings,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[StreamEvent]:
        """The rest of a cut-off answer: the model is shown its prompt and its answer so
        far, asked to go on, and the two are joined. Citations and spans are worked out
        again over the whole answer (the sources are the same, in the same order). A new
        trace is stored; nothing is cached."""

        trace = plan.trace
        assert trace.answer is not None and trace.prompt_messages is not None
        started = time.monotonic()
        config = trace.config
        if config is not None:
            settings = settings.model_copy(
                update={
                    "llm_provider": config.llm_provider,
                    "llm_temperature": config.llm_temperature,
                }
            )
        begin_usage()
        base = base_prompt(trace.prompt_messages)
        messages = continuation_messages(base, trace.answer, settings)
        all_sources = source_citations(plan.chunks)
        new_trace_id = str(uuid.uuid4())
        yield {
            "type": "sources",
            "sources": [_citation_dict(source) for source in all_sources],
            "trace_id": new_trace_id,
        }
        try:
            slot = yield from wait_for_model(settings, should_stop)
        except QuestionAbandoned:
            return  # the client left while waiting for the model: nothing was generated
        if should_stop is not None and should_stop():
            slot.release()
            return
        try:
            yield {
                "type": "stage",
                "stage": "reading",
                "prompt_tokens": estimate_prompt_tokens(messages),
            }
            generate_start = time.monotonic()
            reset_finish_reason()
            parts: list[str] = []
            for delta in self._generator.stream(messages, settings):
                parts.append(delta)
                yield {"type": "delta", "text": delta}
            generate_ms = (time.monotonic() - generate_start) * 1000
            truncated = stopped_at_token_limit()
            answer = join_continuation(
                trace.answer,
                drop_restart(trace.answer, "".join(parts)),
                prefilled=messages[-1]["role"] == "assistant",
            )
            outcome, cited = finish_answer(
                base,
                answer,
                plan.chunks,
                self._generator,
                settings,
                weak_matches=trace.weak_matches,
            )
        finally:
            slot.release()
        status = answer_status(outcome.answer, len(cited), len(plan.chunks))
        usage = usage_totals()
        cited_numbers = {source.source_number for source in cited}
        timings = StageTimings(
            embed_ms=0.0,
            search_ms=0.0,
            rerank_ms=0.0,
            generate_ms=generate_ms,
            total_ms=(time.monotonic() - started) * 1000,
        )
        if self._trace_store is not None:
            self._trace_store.add(
                replace(
                    trace,
                    trace_id=new_trace_id,
                    created_at=time.time(),
                    request_id=current_request_id(),
                    mode="stream",
                    continues=trace.trace_id,
                    candidates=[
                        replace(c, cited=c.source_number in cited_numbers) for c in trace.candidates
                    ],
                    prompt_messages=messages,
                    answer=outcome.answer,
                    cited_source_numbers=sorted(cited_numbers),
                    timings=timings_dict(timings),
                    status=trace_status(status),
                    answer_status=status,
                    truncated=truncated,
                    citation_retry_used=outcome.retry_used,
                    citation_backfilled=outcome.backfilled,
                    unsupported_claims=outcome.unsupported_claims,
                    unsupported_numbers=outcome.unsupported_numbers,
                    prompt_cached=False,
                    prompt_cut=was_prompt_cut(),
                    usage=usage,
                    cost_usd=usage_cost_usd(settings, usage),
                    error=None,
                )
            )
        yield {
            "type": "done",
            "answer": outcome.answer,
            "sources": [_citation_dict(source) for source in cited],
            "timings": timings_dict(timings),
            "trace_id": new_trace_id,
            "answer_status": status,
            "top_rerank_score": max(
                (c.rerank_score for c in trace.candidates if c.rerank_score is not None),
                default=None,
            ),
            "rerank_min_score": float(settings.rerank_min_score),
            "weak_matches": trace.weak_matches,
            "truncated": truncated,
            "citation_backfilled": outcome.backfilled,
            "unsupported_claims": outcome.unsupported_claims,
            "unsupported_numbers": outcome.unsupported_numbers,
            "prompt_cut": was_prompt_cut(),
            "usage": usage_dict(usage, usage_cost_usd(settings, usage)),
        }

    def _answer_stream_uncached(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
        tags: Sequence[str] | None = None,
        should_stop: Callable[[], bool] | None = None,
        *,
        use_memo: bool = True,
    ) -> Generator[StreamEvent]:
        """Stream a grounded answer as SSE-ready events: sources, deltas, then done.

        Sources are emitted immediately after rerank (before the LLM call starts) so
        the UI can render citations while the answer is still generating. ``done``
        carries the final answer, the citation-filtered source list, and timings.
        ``history``, when given, drives the same condense step as ``answer``.

        In three steps — ``_stream_retrieve`` (condense, expansion, retrieval),
        ``_stream_generate`` (the streamed answer), ``_stream_finish`` (citations, the
        trace, ``done``, follow-ups) — sharing one ``AnswerRun``. Only this method
        records a trace for an error or an early end, and releases the model slot.
        """

        run = self._start_run(question, overrides, filenames, tags, "stream")
        run.use_memo = use_memo
        try:
            # Everything this answer calls the LLM for, from the condense on (the
            # follow-ups come after `done` and aren't in its usage).
            begin_usage()
            retrieved = yield from self._stream_retrieve(run, history, should_stop)
            settings, phase, condense_ms, prompt_question, fitted = retrieved
            if not fitted.context_chunks:
                yield from self._stream_no_context(run, settings, phase, condense_ms)
                return
            generated = yield from self._stream_generate(
                run, settings, phase, prompt_question, fitted, should_stop
            )
            if generated is None:
                return  # the finally below records the trace as closed early
            yield from self._stream_finish(
                run, settings, phase, fitted, condense_ms, generated, should_stop
            )
        except QuestionAbandoned:
            return  # the finally below records the trace as closed early
        except Exception as exc:
            if not run.finalized:
                self._store_error_trace(run, public_error_message(exc))
                run.finalized = True
            raise
        finally:
            if not run.finalized:
                # Reached via GeneratorExit (the client disconnected, or the generator
                # was otherwise closed mid-stream without completing or raising an
                # `Exception`) — `finally` re-raises it automatically once this block
                # returns, so no explicit `raise` is needed here — or via an early
                # return once `should_stop` reported the client gone.
                self._store_error_trace(run, "Stream closed before completion.")
                run.finalized = True
            if run.slot is not None:
                run.slot.release()

    def _stream_retrieve(
        self,
        run: AnswerRun,
        history: Sequence[ChatMessage] | None,
        should_stop: Callable[[], bool] | None,
    ) -> Generator[StreamEvent, None, tuple[AppSettings, RetrievalPhase, float, str, FittedPrompt]]:
        """Settings, history, the condense call, opt-in query expansion and retrieval,
        with their stage frames; what the answer is written from."""

        effective_settings = run.settings = self._effective_settings(run.overrides)
        run.history = self._truncate_history(history, effective_settings)
        # Announced before the call, and only when it will actually happen — an
        # unconditional frame would report a stage that never ran for the majority of
        # questions, which is worse than no frame. The rewrite is an LLM call: it waits
        # its turn at a local model like the answer does (announced, and given up if
        # the reader leaves), and gives the place back for retrieval.
        condense_slot: GenerationSlot | None = None
        try:
            if self._will_condense(run.question, run.history, effective_settings, run.use_memo):
                condense_slot = yield from wait_for_model(effective_settings, should_stop)
                yield {"type": "stage", "stage": "condensing"}
            retrieval_query, condense_ms, run.condensed_question, run.condense_note = (
                self._condense_query(run.question, run.history, effective_settings, run.use_memo)
            )
        finally:
            if condense_slot is not None:
                condense_slot.release()
        # Query expansion (opt-in) is an LLM call too: it waits its turn at a local
        # model like the condense call, and gives the place back for retrieval.
        expansion: tuple[list[str], float] | None = None
        if effective_settings.query_expansion_enabled and retrieval_query.strip():
            expansion_slot = yield from wait_for_model(effective_settings, should_stop)
            try:
                yield {"type": "stage", "stage": "searching"}
                expansion = self._expand_query(retrieval_query, effective_settings)
            finally:
                expansion_slot.release()
        else:
            yield {"type": "stage", "stage": "searching"}
        phase = self._retrieve_and_rerank(
            retrieval_query,
            effective_settings,
            run.filenames,
            run.tags,
            should_stop,
            narrow=run.narrow,
            also_search=run.question if run.condensed_question else None,
            expansion=expansion,
        )
        prompt_question, fitted = self._fit_prompt(run, effective_settings, phase)
        return effective_settings, phase, condense_ms, prompt_question, fitted

    def _stream_no_context(
        self, run: AnswerRun, settings: AppSettings, phase: RetrievalPhase, condense_ms: float
    ) -> Iterator[StreamEvent]:
        """Nothing to answer from: the canned answer, with no LLM call."""

        yield {"type": "sources", "sources": [], "trace_id": run.trace_id}
        timings = self._phase_timings(phase, total_start=run.started, condense_ms=condense_ms)
        # The condense and query-expansion calls were still made.
        usage = usage_totals()
        status = self._scoped_status("no_context", run, settings)
        self._store_finished_trace(
            run,
            settings,
            phase,
            status="insufficient_context",
            selected=[],
            prompt_messages=None,
            answer=insufficient_context_answer(settings),
            cited_source_numbers=[],
            timings=timings,
            usage=usage,
            answer_status=status,
        )
        run.finalized = True
        yield {
            "type": "done",
            "answer": insufficient_context_answer(settings),
            "sources": [],
            "timings": timings_dict(timings),
            "trace_id": run.trace_id,
            "answer_status": status,
            "top_rerank_score": _top_rerank_score(phase),
            "rerank_min_score": float(settings.rerank_min_score),
            "passages_left_out": _picked_left_out(run.narrow, []),
            "usage": usage_dict(usage, usage_cost_usd(settings, usage)),
        }

    def _stream_generate(
        self,
        run: AnswerRun,
        settings: AppSettings,
        phase: RetrievalPhase,
        prompt_question: str,
        fitted: FittedPrompt,
        should_stop: Callable[[], bool] | None,
    ) -> Generator[StreamEvent, None, _Streamed | None]:
        """The sources frame, the wait for the model, and the streamed answer; None when
        the reader left before generation."""

        selected = fitted.context_chunks
        all_sources = source_citations(selected)
        yield {
            "type": "sources",
            "sources": [_citation_dict(source) for source in all_sources],
            "trace_id": run.trace_id,
        }
        messages = answer_messages(
            prompt_question,
            selected,
            fitted.history or None,
            settings,
            weak_matches=phase.below_min_score,
        )
        run.prompt_messages = list(messages)
        if should_stop is not None and should_stop():
            return None
        generator = self._answer_generator(run, settings)
        # An answer the prompt cache holds needs neither the model nor a wait for it.
        if not _reply_known(generator, messages, settings):
            # Held until the answer, its citation retry and follow-ups are done, so
            # another question's call can't slip in between them; released by the
            # caller. Waiting inside Ollama ran this request's timeout; waiting here
            # doesn't.
            run.slot = yield from wait_for_model(settings, should_stop)
            reading: StageEvent = {
                "type": "stage",
                "stage": "reading",
                "prompt_tokens": estimate_prompt_tokens(messages),
            }
            eta_ms = expected_prefill_ms(messages, settings)
            if eta_ms is not None:
                reading["eta_ms"] = round(eta_ms)
            yield reading
        generate_start = time.monotonic()
        first_token_ms = 0.0
        reset_finish_reason()
        for delta in generator.stream(messages, settings):
            if not run.parts:
                first_token_ms = (time.monotonic() - run.started) * 1000
            run.parts.append(delta)
            yield {"type": "delta", "text": delta}
        yield from self._auto_continue(run, settings, generator, messages, should_stop)
        generate_ms = (time.monotonic() - generate_start) * 1000
        answer = "".join(run.parts).strip()
        if not answer:
            raise GenerationError("Generation provider returned an empty answer.")
        return _Streamed(messages, all_sources, answer, generate_ms, first_token_ms)

    def _auto_continue(
        self,
        run: AnswerRun,
        settings: AppSettings,
        generator: ChatGenerator,
        messages: Sequence[ChatMessage],
        should_stop: Callable[[], bool] | None,
    ) -> Iterator[StreamEvent]:
        """AUTO_CONTINUE_MAX: an answer cut off at the output limit is continued in the
        same stream, as a Continue would, up to that many times. A provider that takes a
        prefill streams the rest live; one asked in words is read whole first, so a
        restart from the top can be dropped before anything is sent."""

        for _ in range(int(settings.auto_continue_max)):
            if not run.parts or not stopped_at_token_limit():
                return
            if should_stop is not None and should_stop():
                return
            so_far = "".join(run.parts)
            follow = continuation_messages(messages, so_far, settings)
            prefilled = follow[-1]["role"] == "assistant"
            yield {"type": "stage", "stage": "continuing"}
            reset_finish_reason()
            if prefilled:
                first = True
                for delta in generator.stream(follow, settings):
                    if first and so_far[-1:].isspace():
                        # The prefill was sent without its trailing space; the model
                        # writes its own (join_continuation's rule).
                        delta = delta.lstrip()
                    first = False
                    if delta:
                        run.parts.append(delta)
                        yield {"type": "delta", "text": delta}
                continue
            more = drop_restart(so_far, "".join(generator.stream(follow, settings)))
            if not more.strip():
                return
            addition = join_continuation(so_far, more)[len(so_far) :]
            run.parts.append(addition)
            yield {"type": "delta", "text": addition}

    def _stream_finish(
        self,
        run: AnswerRun,
        settings: AppSettings,
        phase: RetrievalPhase,
        fitted: FittedPrompt,
        condense_ms: float,
        generated: _Streamed,
        should_stop: Callable[[], bool] | None,
    ) -> Iterator[StreamEvent]:
        """Citations (retry or backfill), the trace, ``done``, then any follow-ups."""

        selected = fitted.context_chunks
        # Streaming can't retry before the deltas already sent, and buffering would
        # destroy the streaming UX for the common case. Instead, if the streamed answer
        # carries no citations, run one non-streaming retry and swap the corrected
        # answer/sources into the `done` event — the frontend replaces the streamed
        # content with `done.answer`, so it snaps to the cited version.
        # finalize_citations is the same helper generate_grounded_answer (sync path)
        # uses, so the retry-trigger condition can't drift between response modes.
        # The retry is a whole unstreamed generation after the last delta; without
        # this frame the UI sits on its streaming label with nothing moving.
        if will_call_llm_for_citations(
            generated.answer,
            generated.sources,
            settings,
            weak_matches=phase.below_min_score,
            context_chunks=selected,
        ):
            yield {"type": "stage", "stage": "verifying_citations"}
        # Spans only now: the sources frame went out before there was an answer.
        outcome, cited = finish_answer(
            generated.messages,
            generated.answer,
            selected,
            run.generator or self._generator,
            settings,
            weak_matches=phase.below_min_score,
        )
        answer = outcome.answer
        # The last call made the answer: the citation retry's when it fired.
        truncated = stopped_at_token_limit()
        usage = usage_totals()
        timings = self._phase_timings(
            phase,
            total_start=run.started,
            condense_ms=condense_ms,
            generate_ms=generated.generate_ms,
            citation_retry_ms=outcome.retry_ms,
            first_token_ms=generated.first_token_ms,
        )
        status = answer_status(answer, len(cited), len(selected))
        self._store_finished_trace(
            run,
            settings,
            phase,
            status=trace_status(status),
            selected=selected,
            context_budget_dropped_ids=fitted.dropped_point_ids,
            # outcome.prompt_messages, not the `messages` that were streamed: when the
            # citation retry fired, the streamed answer was replaced by the retry's, and
            # the retry's continuation is what produced the text stored just below.
            prompt_messages=list(outcome.prompt_messages),
            answer=answer,
            cited_source_numbers=[source.source_number for source in cited],
            timings=timings,
            citation_retry_used=outcome.retry_used,
            citation_backfilled=outcome.backfilled,
            unsupported_claims=outcome.unsupported_claims,
            unsupported_numbers=outcome.unsupported_numbers,
            prompt_cached=self._prompt_cached(run),
            prompt_cut=was_prompt_cut(),
            usage=usage,
            answer_status=status,
            truncated=truncated,
            dropped_history_messages=fitted.dropped_history_messages,
        )
        run.finalized = True
        done: DoneEvent = {
            "type": "done",
            "answer": answer,
            "sources": [_citation_dict(source) for source in cited],
            "timings": timings_dict(timings),
            "trace_id": run.trace_id,
            "answer_status": status,
            "top_rerank_score": _top_rerank_score(phase),
            "rerank_min_score": float(settings.rerank_min_score),
            "weak_matches": phase.below_min_score,
            "truncated": truncated,
            "citation_backfilled": outcome.backfilled,
            "unsupported_claims": outcome.unsupported_claims,
            "unsupported_numbers": outcome.unsupported_numbers,
            "prompt_cached": self._prompt_cached(run),
            "prompt_cut": was_prompt_cut(),
            "passages_left_out": _picked_left_out(run.narrow, selected),
            "usage": usage_dict(usage, usage_cost_usd(settings, usage)),
        }
        if run.condensed_question:
            done["retrieval_query"] = run.condensed_question
        yield done
        if (
            settings.followups_enabled
            and status == "answered"
            and not (should_stop is not None and should_stop())
        ):
            followups = generate_followups(
                run.question.strip(),
                answer,
                _cited_chunks(cited, selected),
                run.generator or self._generator,
                settings,
            )
            if followups:
                yield {"type": "followups", "questions": followups}


def _with_neighbour_text(
    chunks: list[RerankedChunk],
    trace: QueryTrace,
    repository: VectorRepository,
    settings: AppSettings,
) -> list[RerankedChunk]:
    """The passages with the neighbour text the first answer's prompt gave them: the
    citation, number and backfill checks over the continued answer read it, so a number
    the model took from a neighbour isn't flagged as unsupported. Only passages the
    trace marks ``neighbor_expanded`` get it back."""

    expanded_ids = {c.point_id for c in trace.candidates if c.neighbor_expanded}
    if not expanded_ids:
        return chunks
    try:
        widened = expand_with_neighbors(chunks, repository, settings)
    except Exception:  # noqa: BLE001 - the checks then read the passage alone
        return chunks
    by_id = {chunk.point_id: chunk for chunk in widened}
    return [
        replace(chunk, expanded_text=by_id[chunk.point_id].expanded_text)
        if chunk.point_id in expanded_ids and chunk.point_id in by_id
        else chunk
        for chunk in chunks
    ]
