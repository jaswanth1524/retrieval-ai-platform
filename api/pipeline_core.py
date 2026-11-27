"""``RagPipeline`` below its streaming path: construction, settings, retrieval and
reranking, the condense step, the non-streamed answer, re-runs and similar-passage
search. ``api/answer_stream.py`` adds the stream on top; ``api/pipeline.py`` defines
``RagPipeline`` itself.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import replace
from typing import Any

from api.answer_cache import (
    AnswerCache,
    CachedAnswer,
    CacheKey,
    answer_cache_key,
)
from api.answer_run import (
    AnswerOverrides,
    AnswerRun,
    RetrievalPhase,
    _cached_timings,
    _cited_chunks,
    _picked_left_out,
    _recorded_settings,
    _reply_known,
    _top_rerank_score,
    check_rerank_top_k,
    filter_from_record,
    narrowing_record,
    served_query,
    trace_status,
)
from api.answer_traces import TraceRecording
from api.citations import (
    SourceCitation,
    answer_status,
    opens_with_a_refusal,
    strip_citation_markers,
)
from api.context_expansion import (
    _TAG_LOOKUP_EXECUTOR,
    _map_concurrently,
    expand_with_neighbors,
)
from api.continuation import AutoContinuing
from api.conversation_history import (
    CondenseMemo,
    _clip_earlier_answers,
    _condense_skip_reason,
    _from_first_question,
    _with_interpretation,
    _without_unanswered_exchanges,
    condense_memo,
)
from api.documents import contextual_text
from api.embeddings import EmbeddedText
from api.errors import public_error_message
from api.generation import (
    ChatGenerator,
    ChatMessage,
    GenerationConfigError,
    GenerationError,
    GroundedAnswer,
    answer_messages,
    fit_prompt_to_context_window,
    generate_followups,
    generate_grounded_answer,
    generate_query_variants,
)
from api.llm import (
    GenerationSlot,
    begin_usage,
    reset_finish_reason,
    stopped_at_token_limit,
    usage_cost_usd,
    usage_totals,
    was_degraded,
    was_prompt_cut,
)
from api.passage_selection import (
    _select_passages,
    without_recency_for_the_past,
)
from api.prompt_cache import CachingGenerator, shared_prompt_cache
from api.prompting import (
    FittedPrompt,
    condense_question_detailed,
    decompose_question,
    might_be_compound,
)
from api.qdrant_schema import check_dense_dimension
from api.query_log import log_served
from api.repository import LegRanks, PassageFilter, VectorRepository, reciprocal_rank_fusion
from api.request_id import submit_in_context
from api.reranking import (
    RerankedChunk,
    Reranker,
    rerank_candidates_detailed,
)
from api.retrieval import (
    QueryEmbeddingProvider,
    RetrievalConfigError,
    RetrievalError,
    RetrievedChunk,
    embed_query,
    points_to_chunks,
)
from api.run_coordination import (
    QuestionAbandoned,
    drain,
    follow_only,
    lead_or_join,
    wait_for_model,
)
from api.settings import AppSettings
from api.tracing import (
    QueryTrace,
    TraceMode,
    TraceSink,
    build_trace_candidates,
)
from api.workspaces import workspace_of

# Logs as api.pipeline, where this code lived, so existing log filters still match.
logger = logging.getLogger("api.pipeline")


# Bracketed numbers up to this are taken for an earlier answer's citations.
HISTORY_MARKER_MAX = 20


class PipelineCore(TraceRecording):
    """Everything in ``RagPipeline`` but the streamed answer (see ``RagPipeline``)."""

    def __init__(
        self,
        repository: VectorRepository,
        embedding_provider: QueryEmbeddingProvider,
        reranker: Reranker,
        generator: ChatGenerator,
        settings: AppSettings,
        trace_store: TraceSink | None = None,
        answer_cache: AnswerCache | None = None,
    ) -> None:
        self._repository = repository
        # None disables answer caching (ANSWER_CACHE_SIZE=0).
        self._answer_cache = answer_cache
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._generator = generator
        self._settings = settings
        # None disables trace capture entirely (the default) — every trace-related
        # call below is skipped rather than merely producing an empty trace, so
        # tracing costs nothing when the operator has turned it off.
        self._trace_store = trace_store

    def effective_settings(self, overrides: AnswerOverrides) -> AppSettings:
        """The settings a question with ``overrides`` runs under (a search's explanation
        names the bar and cuts it was given)."""

        return self._effective_settings(overrides)

    def _effective_settings(self, overrides: AnswerOverrides) -> AppSettings:
        """Fold per-request overrides into a settings copy without mutating the base."""

        settings = self._settings
        check_rerank_top_k(overrides.rerank_top_k, settings)

        update: dict[str, Any] = {
            key: value
            for key, value in {
                "llm_provider": overrides.llm_provider,
                "rerank_top_k": overrides.rerank_top_k,
                "max_context_chunks": overrides.max_context_chunks,
                "llm_temperature": overrides.llm_temperature,
                "answer_format": overrides.answer_format,
                "answer_language": overrides.answer_language,
            }.items()
            if value is not None
        }
        if overrides.passages_only:
            update["context_neighbor_radius"] = 0
        if overrides.strong_matches_only:
            update["rerank_fallback_count"] = 0
        return settings.model_copy(update=update) if update else settings

    def _truncate_history(
        self, history: Sequence[ChatMessage] | None, effective_settings: AppSettings
    ) -> list[ChatMessage]:
        """Keep only the most recent history messages, per server-side config.

        Earlier answers lose their ``[n]`` markers: those numbered the previous turn's
        sources, and this turn's prompt numbers its own from 1 again, so a model
        continuing the conversation could copy a stale ``[2]`` onto an unrelated source.
        An exchange whose answer was "not enough information" is left out (see
        ``_without_unanswered_exchanges``).
        """

        if not history:
            return []
        cap = int(effective_settings.conversation_max_history_messages)
        kept = _without_unanswered_exchanges(list(history)[-cap:] if cap > 0 else [])
        # An odd cap (or a client's odd-length history) could leave an answer first,
        # with its question cut away: the model read an answer to nothing, and
        # Anthropic's API refuses a conversation that doesn't open with the user.
        kept = _from_first_question(kept)
        # The most sources an earlier answer could cite (generously: a workspace or a
        # request may have allowed more): "[2024]" past it is the answer's own text.
        most_sources = max(
            HISTORY_MARKER_MAX,
            int(effective_settings.max_context_chunks),
            int(effective_settings.rerank_top_k),
        )
        return [
            {
                "role": "assistant",
                # A bare "[1]" answer still holds its place, so the turns keep alternating.
                "content": strip_citation_markers(message["content"], most_sources)
                or "(answer omitted)",
            }
            if message["role"] == "assistant"
            else message
            for message in kept
        ]

    def _will_condense(
        self,
        question: str,
        truncated_history: Sequence[ChatMessage],
        effective_settings: AppSettings,
        use_memo: bool = True,
    ) -> bool:
        """Whether ``_condense_query`` will actually make its LLM call.

        Split out purely so ``answer_stream`` can announce the ``condensing`` stage
        *before* the call blocks it, without restating the gate — two copies of this
        condition would drift the first time either side changed.
        """

        return (
            bool(truncated_history)
            and _condense_skip_reason(question, effective_settings) is None
            and self._remembered_rewrite(question, truncated_history, effective_settings, use_memo)
            is None
        )

    @staticmethod
    def _remembered_rewrite(
        question: str,
        truncated_history: Sequence[ChatMessage],
        settings: AppSettings,
        use_memo: bool,
    ) -> str | None:
        if not use_memo or int(settings.condense_cache_size) <= 0:
            return None
        return condense_memo.get(CondenseMemo.key(question, truncated_history, settings))

    def _condense_query(
        self,
        question: str,
        truncated_history: Sequence[ChatMessage],
        effective_settings: AppSettings,
        use_memo: bool = True,
    ) -> tuple[str, float, str | None, str | None]:
        """Rewrite a follow-up into a standalone retrieval query, when history exists.

        ``truncated_history`` is expected to already be capped (see
        ``_truncate_history``) — the same truncated list is what generation sees.
        Returns ``(retrieval_query, condense_ms, condensed_question, note)`` — the third
        element is the trace-visible rewrite, or None when condense was skipped
        (no history, or disabled) or failed/returned nothing usable (falls back to the
        raw question); ``note`` says why for the trace. A condense failure must never
        fail the whole request — retrieval just proceeds on the raw question, exactly
        as it did before this feature.
        """

        if not truncated_history:
            return question, 0.0, None, None
        skip_reason = _condense_skip_reason(question, effective_settings)
        if skip_reason is not None:
            return question, 0.0, None, skip_reason
        # The same call made before: its rewrite, with no LLM call (Regenerate, which
        # passes use_memo=False, asks again — the earlier rewrite may be the problem).
        remembered = self._remembered_rewrite(
            question, truncated_history, effective_settings, use_memo
        )
        if remembered is not None:
            return remembered, 0.0, remembered, "remembered"

        condense_start = time.monotonic()
        try:
            condensed = condense_question_detailed(
                question, truncated_history, self._generator, effective_settings
            )
        except GenerationError as exc:
            logger.warning("Condense step failed, falling back to raw question: %s", exc)
            return question, (time.monotonic() - condense_start) * 1000, None, "failed"
        condense_ms = (time.monotonic() - condense_start) * 1000
        if (
            condensed.text
            and not condensed.text.rstrip().endswith("?")
            and opens_with_a_refusal(condensed.text)
        ):
            # The model answered from the history ("The documents do not contain ...")
            # instead of rewriting; searched for, it matches nothing. A question that
            # merely uses the words ("What if a receipt does not contain a date?") is
            # a rewrite.
            condensed = replace(condensed, text="", note="rejected: not a question")
        if not condensed.text:
            logger.warning(
                "Condense step returned nothing usable (%s), falling back to raw question.",
                condensed.note,
            )
            return question, condense_ms, None, condensed.note
        condense_memo.put(
            CondenseMemo.key(question, truncated_history, effective_settings),
            condensed.text,
            int(effective_settings.condense_cache_size),
        )
        return condensed.text, condense_ms, condensed.text, condensed.note

    def retrieval_query(self, question: str, history: Sequence[ChatMessage] | None = None) -> str:
        """The query a question is retrieved with: a follow-up rewritten into a
        standalone one, as ``answer`` does (the eval harness scores this)."""

        settings = self._effective_settings(AnswerOverrides())
        truncated = self._truncate_history(history, settings)
        query, _, _, _ = self._condense_query(question.strip(), truncated, settings)
        return query

    def _scoped_status(self, status: str, run: AnswerRun, settings: AppSettings) -> str:
        """``empty_scope`` for a question with nothing to answer from because its scope
        (``filenames``, ``tags``) names no indexed document — a typo'd name, a tag no
        document carries — rather than ``no_context``, which reads as "the documents
        don't cover it". Else ``status``."""

        if status != "no_context" or not (run.filenames or run.tags):
            return status
        if self.scope_matches_nothing(run.filenames, run.tags, settings):
            return "empty_scope"
        return status

    def scope_matches_nothing(
        self,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        settings: AppSettings | None = None,
    ) -> bool:
        """Whether a scope (``filenames``, ``tags``) names no indexed document — a
        typo'd name, a tag no document carries. False when there is no scope, or when
        the index can't be read (the plain outcome is still right then)."""

        settings = settings or self._settings
        try:
            if filenames:
                indexed = self._repository.filename_metadata(settings)
                if not any(name in indexed for name in filenames):
                    return True
            if tags:
                wanted = {" ".join(tag.split()).casefold() for tag in tags}
                stored = {tag.casefold() for tag in self._repository.stored_tags(settings)}
                if not wanted & stored:
                    return True
        except Exception:  # noqa: BLE001 - the plain status is still right
            return False
        return False

    def passages_now(
        self, filename: str, chunk_ids: Sequence[str]
    ) -> tuple[bool, dict[str, RerankedChunk]]:
        """Whether ``filename`` is indexed now, and which of ``chunk_ids`` it still holds
        (a chunk id is its text's hash: one present reads as it did)."""

        if filename not in self._repository.filename_metadata(self._settings):
            return False, {}
        found: dict[str, RerankedChunk] = {}
        for payload in self._repository.chunks_for_filename(
            self._settings, filename, PassageFilter(chunk_ids=tuple(chunk_ids))
        ):
            chunk_id = str(payload.get("chunk_id"))
            ordinal = payload.get("chunk_ordinal")
            found[chunk_id] = RerankedChunk(
                point_id=chunk_id,
                filename=filename,
                page=int(payload.get("page") or 1),  # type: ignore[call-overload]
                section=str(payload.get("section") or ""),
                chunk_id=chunk_id,
                text=str(payload.get("text") or ""),
                retrieval_score=0.0,
                rerank_score=0.0,
                chunk_ordinal=ordinal if isinstance(ordinal, int) else None,
            )
        return True, found

    def cited_passages(self, sources: Sequence[SourceCitation]) -> dict[str, RerankedChunk]:
        """The whole passages ``sources`` cite, by chunk id, as indexed now (one read per
        document). A source's ``text`` is only a 200-character excerpt: the self-check
        held answers against that, so a claim backed past its first lines read as
        unsupported. A passage that can't be read is left out (its excerpt stands in)."""

        wanted: dict[str, list[str]] = {}
        for source in sources:
            wanted.setdefault(source.filename, []).append(source.chunk_id)
        found: dict[str, RerankedChunk] = {}
        for filename, chunk_ids in wanted.items():
            try:
                payloads = self._repository.chunks_for_filename(
                    self._settings,
                    filename,
                    PassageFilter(chunk_ids=tuple(dict.fromkeys(chunk_ids))),
                )
            except Exception:  # noqa: BLE001 - the excerpt is still a fair reading
                logger.warning("Couldn't read the cited passages of %r.", filename, exc_info=True)
                continue
            for payload in payloads:
                chunk = _stored_chunk(payload, filename)
                if chunk is not None:
                    found[chunk.chunk_id] = chunk
        return found

    def widener(self, overrides: AnswerOverrides) -> Callable[[RerankedChunk], str] | None:
        """``neighbour_text`` as a question with ``overrides`` had its prompt: none with
        no neighbouring text (``passages_only``), none from outside its narrowing."""

        if overrides.passages_only or int(self._settings.context_neighbor_radius) <= 0:
            return None
        return lambda chunk: self.neighbour_text(chunk, overrides.narrow)

    def neighbour_text(self, chunk: RerankedChunk, narrow: PassageFilter | None = None) -> str:
        """``chunk``'s text with its neighbours (as a question's prompt would have it),
        or its own text when they can't be read."""

        try:
            if chunk.chunk_ordinal is None:
                # From an answer's sources (no position): read it off the passage.
                found = self._repository.chunks_for_filename(
                    self._settings, chunk.filename, PassageFilter(chunk_ids=(chunk.chunk_id,))
                )
                ordinal = found[0].get("chunk_ordinal") if found else None
                if not isinstance(ordinal, int):
                    return chunk.text
                chunk = replace(chunk, chunk_ordinal=ordinal)
            widened = expand_with_neighbors(
                [chunk], self._repository, self._settings, narrow=narrow
            )
        except Exception:  # noqa: BLE001 - the passage alone is still a fair reading
            return chunk.text
        return (widened[0].expanded_text if widened else None) or chunk.text

    def _stored_tag_variants(self, tags: Sequence[str], settings: AppSettings) -> list[str]:
        """The stored spellings of ``tags``, matched the way tags are de-duplicated.

        Tags are stored as first typed ("Finance") and de-duplicated case-insensitively,
        but the payload filter matches exactly, so asking for "finance" found nothing.
        Tags no document carries are kept as given: they still match nothing, where an
        empty list would have dropped the filter and searched every document.
        """

        wanted = {" ".join(tag.split()).casefold() for tag in tags}
        stored = self._repository.stored_tags(settings)
        return sorted(tag for tag in stored if tag.casefold() in wanted) or list(tags)

    def _expand_context(
        self,
        kept: Sequence[RerankedChunk],
        settings: AppSettings,
        for_generation: bool,
        narrow: PassageFilter | None = None,
    ) -> tuple[list[RerankedChunk], float]:
        """The kept passages with neighbour text for the prompt, and the time taken.

        Timed apart from rerank: neighbour expansion is a Qdrant round trip, not model
        scoring, and folding it into rerank_ms hid it from every latency breakdown.
        Only the max_context_chunks prefix reaches the prompt, so only it is expanded;
        the tail stays in the list unexpanded because traces report it as kept.
        """

        if not for_generation:
            return list(kept), 0.0
        start = time.monotonic()
        limit = int(settings.max_context_chunks)
        expanded = [
            # Neighbours stay inside the question's narrowing: a passage left out, or one
            # outside its pages or section, doesn't come back as neighbour text.
            *expand_with_neighbors(kept[:limit], self._repository, settings, narrow=narrow),
            *kept[limit:],
        ]
        return expanded, (time.monotonic() - start) * 1000

    def _expand_query(self, query: str, settings: AppSettings) -> tuple[list[str], float]:
        """QUERY_EXPANSION_ENABLED's variant phrasings of ``query`` and the time taken."""

        start = time.monotonic()
        variants = generate_query_variants(
            query.strip(), int(settings.query_expansion_count), self._generator, settings
        )
        return variants, (time.monotonic() - start) * 1000

    def _decompose(self, query: str, settings: AppSettings) -> tuple[list[str], float]:
        """QUERY_DECOMPOSITION_ENABLED's standalone parts of ``query`` (none when it is
        one question) and the time taken."""

        start = time.monotonic()
        parts = decompose_question(
            query.strip(), int(settings.query_decomposition_max_parts), self._generator, settings
        )
        return parts, (time.monotonic() - start) * 1000

    def _decomposition(self, query: str, settings: AppSettings) -> tuple[list[str], float] | None:
        """Whether this question gets decomposed at all (an LLM call it must wait for at a
        local model): only with the setting on and a question that looks compound."""

        if not settings.query_decomposition_enabled or not might_be_compound(query):
            return None
        return self._decompose(query, settings)

    def _search(
        self,
        queries: Sequence[str],
        settings: AppSettings,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        tag_variants: Future[list[str]] | None,
        narrow: PassageFilter | None,
        seed: EmbeddedText | None = None,
        leg_ranks: LegRanks | None = None,
    ) -> tuple[list[RetrievedChunk], float, float]:
        """Every query embedded and hybrid-searched side by side, the result lists
        RRF-fused when there are several; the candidates, embed_ms and search_ms.
        With ``seed`` (a stored passage's own vectors) that is the one query, as is.
        ``leg_ranks`` is filled for the trace (``VectorRepository.hybrid_search``)."""

        embed_start = time.monotonic()
        query_embeddings = (
            [seed]
            if seed is not None
            else _map_concurrently(
                lambda query: embed_query(query, self._embedding_provider), queries
            )
        )
        embed_ms = (time.monotonic() - embed_start) * 1000
        check_dense_dimension(len(query_embeddings[0].dense), settings)
        if tag_variants is not None:
            tags = tag_variants.result()

        search_start = time.monotonic()
        # Passed only when set, so a repository double without the parameters still works.
        extra: dict[str, Any] = {"narrow": narrow} if narrow is not None else {}
        if leg_ranks is not None:
            extra["leg_ranks"] = leg_ranks
        result_lists = _map_concurrently(
            lambda embedding: self._repository.hybrid_search(
                settings, embedding, filenames, tags, **extra
            ),
            query_embeddings,
        )
        points = (
            result_lists[0]
            if len(result_lists) == 1
            else reciprocal_rank_fusion(
                result_lists,
                k=int(settings.rrf_k),
                limit=int(settings.fused_top_n),
            )
        )
        candidates = points_to_chunks(points)
        search_ms = (time.monotonic() - search_start) * 1000
        return candidates, embed_ms, search_ms

    def _retrieve_and_rerank(
        self,
        question: str,
        effective_settings: AppSettings,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None = None,
        should_stop: Callable[[], bool] | None = None,
        *,
        for_generation: bool = True,
        narrow: PassageFilter | None = None,
        also_search: str | None = None,
        expansion: tuple[list[str], float] | None = None,
        seed: EmbeddedText | None = None,
        traced: bool | None = None,
        collect_leg_ranks: bool = False,
        parts: tuple[list[str], float] | None = None,
        expand_neighbours: bool = True,
    ) -> RetrievalPhase:
        """Run retrieve -> rerank -> neighbor-expand.

        ``parts`` (QUERY_DECOMPOSITION_ENABLED): a compound question's standalone parts,
        each retrieved and reranked on its own (``_retrieve_parts``).

        ``traced`` says whether a trace will show this retrieval (default: when it
        feeds an answer): only then are each candidate's leg ranks collected, which
        can cost Qdrant one more query (TRACE_LEG_RANKS).

        ``expansion`` is ``_expand_query``'s result when the caller already ran it (the
        stream does, so it can wait for the local model with a frame first).

        ``also_search`` is a second query searched alongside ``question`` and fused
        with it (the raw follow-up next to its condensed rewrite); the rerank scores
        against ``question`` alone.

        ``for_generation=False`` (search without an answer) skips the two stages that
        only serve the prompt: LLM query expansion and neighbour expansion.

        Embedding and search are separate steps so each can be timed for
        observability.
        ``should_stop`` is checked before the rerank, the costliest stage, and raises
        ``QuestionAbandoned`` when the client has gone.
        """

        normalized_query = question.strip()
        if not normalized_query:
            raise RetrievalError("Query text is required.")
        picked = narrow is not None and bool(narrow.chunk_ids)
        if (
            parts is not None
            and len(parts[0]) >= 2
            and for_generation
            and not picked
            and seed is None
        ):
            return self._retrieve_parts(
                parts,
                effective_settings,
                filenames,
                tags,
                should_stop,
                narrow=narrow,
                traced=traced,
                collect_leg_ranks=collect_leg_ranks,
            )
        self._repository.ensure_ready(effective_settings)
        # Looked up alongside embedding (and, for a sync answer, query expansion), which
        # don't depend on it; only the search below needs the stored spellings.
        tag_variants = (
            submit_in_context(
                _TAG_LOOKUP_EXECUTOR,
                functools.partial(self._stored_tag_variants, tags, effective_settings),
            )
            if tags
            else None
        )

        # Multi-query expansion (opt-in): rewrite into variant phrasings, retrieve each,
        # and RRF-fuse the result lists with the same pinned rrf_k before one rerank.
        # Left un-timed (query_expansion_ms stays 0.0) when disabled, like condense_ms.
        variants: list[str] = []
        query_expansion_ms = 0.0
        if expansion is not None:
            variants, query_expansion_ms = expansion
        elif effective_settings.query_expansion_enabled and for_generation:
            variants, query_expansion_ms = self._expand_query(normalized_query, effective_settings)
        queries = [normalized_query, *variants]
        # The raw follow-up searched next to its rewrite: a rewrite that lost or changed
        # what the user asked no longer takes the raw question's matches with it.
        extra_query = (also_search or "").strip()
        if extra_query and extra_query.casefold() not in {q.casefold() for q in queries}:
            queries.insert(1, extra_query)

        # Which leg found each candidate, for its trace (none is kept without one).
        traced = for_generation if traced is None else traced
        leg_ranks: LegRanks | None = (
            {} if (traced and self._trace_store is not None) or collect_leg_ranks else None
        )
        candidates, embed_ms, search_ms = self._search(
            queries, effective_settings, filenames, tags, tag_variants, narrow, seed, leg_ranks
        )
        if should_stop is not None and should_stop():
            raise QuestionAbandoned

        rerank_start = time.monotonic()
        outcome = rerank_candidates_detailed(
            query=question,
            candidates=candidates,
            reranker=self._reranker,
            settings=effective_settings,
        )
        # Picked passages (chunk_ids) are the context exactly: their neighbours were not
        # picked, and the question was asked about these alone.
        picked_only = for_generation and narrow is not None and bool(narrow.chunk_ids)
        selection = _select_passages(
            outcome,
            without_recency_for_the_past(effective_settings, question, extra_query),
            for_generation=for_generation,
            picked_only=picked_only,
        )
        diverse_outcome = selection.outcome
        rerank_ms = (time.monotonic() - rerank_start) * 1000
        # Read on this thread right after the call: the reranker keeps it per thread.
        rerank_wait_ms = float(getattr(self._reranker, "last_wait_ms", 0.0))
        expanded, context_expansion_ms = self._expand_context(
            diverse_outcome.kept,
            effective_settings,
            for_generation and not picked_only and expand_neighbours,
            narrow=narrow,
        )

        return RetrievalPhase(
            context_chunks=expanded,
            fused_candidates=candidates,
            scored_chunks=outcome.scored,
            embed_ms=embed_ms,
            search_ms=search_ms,
            rerank_ms=rerank_ms,
            rerank_wait_ms=rerank_wait_ms,
            query_expansion_ms=query_expansion_ms,
            context_expansion_ms=context_expansion_ms,
            diversity_dropped_ids=frozenset(selection.diversity_dropped),
            relative_cut_ids=frozenset(selection.relative_cut_ids),
            query_variants=queries[1:],
            below_min_score=selection.below_min_score,
            leg_ranks=leg_ranks or {},
        )

    def _retrieve_parts(
        self,
        parts: tuple[list[str], float],
        effective_settings: AppSettings,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        should_stop: Callable[[], bool] | None,
        *,
        narrow: PassageFilter | None,
        traced: bool | None,
        collect_leg_ranks: bool,
    ) -> RetrievalPhase:
        """A compound question's parts each retrieved and reranked on its own (scored
        against the whole question, a part's passages fell under the bar: the cross-
        encoder spreads one passage's relevance over every topic asked), then their kept
        passages taken in turn — each part's best first — so the prompt's first
        ``max_context_chunks`` cover every part. Neighbour expansion runs once, over the
        merged list."""

        questions, decompose_ms = parts
        phases = [
            self._retrieve_and_rerank(
                part,
                effective_settings,
                filenames,
                tags,
                should_stop,
                narrow=narrow,
                # Selected as for an answer (fallback, relative cut); no expansion call
                # per part, and neighbours added once, over the merged list.
                expansion=([], 0.0),
                traced=traced,
                collect_leg_ranks=collect_leg_ranks,
                expand_neighbours=False,
            )
            for part in questions
        ]
        merged: list[RerankedChunk] = []
        seen: set[str] = set()
        for rank in range(max((len(phase.context_chunks) for phase in phases), default=0)):
            for phase in phases:
                if rank < len(phase.context_chunks):
                    chunk = phase.context_chunks[rank]
                    if chunk.point_id not in seen:
                        seen.add(chunk.point_id)
                        merged.append(chunk)
        merged = merged[: int(effective_settings.rerank_top_k)]
        expanded, context_expansion_ms = self._expand_context(
            merged, effective_settings, True, narrow=narrow
        )

        def unique[T](items: Sequence[T], key: Callable[[T], str]) -> list[T]:
            out: list[T] = []
            keys: set[str] = set()
            for item in items:
                if key(item) not in keys:
                    keys.add(key(item))
                    out.append(item)
            return out

        leg_ranks: dict[str, list[int | None]] = {}
        for phase in phases:
            for point, ranks in phase.leg_ranks.items():
                leg_ranks.setdefault(point, ranks)
        return RetrievalPhase(
            context_chunks=expanded,
            fused_candidates=unique(
                [c for phase in phases for c in phase.fused_candidates], lambda c: c.point_id
            ),
            scored_chunks=sorted(
                unique(
                    sorted(
                        (c for phase in phases for c in phase.scored_chunks),
                        key=lambda c: -c.rerank_score,
                    ),
                    lambda c: c.point_id,
                ),
                key=lambda c: -c.rerank_score,
            ),
            embed_ms=sum(phase.embed_ms for phase in phases),
            search_ms=sum(phase.search_ms for phase in phases),
            rerank_ms=sum(phase.rerank_ms for phase in phases),
            rerank_wait_ms=sum(phase.rerank_wait_ms for phase in phases),
            # The decomposition is a query rewrite, as the expansion is.
            query_expansion_ms=decompose_ms,
            context_expansion_ms=context_expansion_ms,
            diversity_dropped_ids=frozenset().union(*(p.diversity_dropped_ids for p in phases)),
            relative_cut_ids=frozenset().union(*(p.relative_cut_ids for p in phases)),
            query_variants=list(questions),
            # Weak only when no part found a passage clearing the bar.
            below_min_score=all(phase.below_min_score for phase in phases),
            leg_ranks=leg_ranks,
        )

    def retrieve(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        *,
        for_generation: bool = True,
        narrow: PassageFilter | None = None,
        also_search: str | None = None,
        explain: bool = False,
    ) -> RetrievalPhase:
        """Retrieval exactly as ``answer`` runs it — query expansion, hybrid search,
        rerank, diversity filter, neighbour expansion — without generating. The eval
        harness scores this, so it measures the pipeline users actually get.
        ``for_generation=False`` is ``POST /search``: the same hybrid search, RRF and
        rerank, minus the LLM query expansion and the neighbour expansion that only
        feed a prompt. ``explain`` (``POST /search``'s): each candidate's rank on the
        dense and the BM25 search is recorded too (``phase.leg_ranks``)."""

        effective_settings = self._effective_settings(overrides or AnswerOverrides())
        if explain:
            # With Qdrant's own fusion the leg ranks cost one more query; asked for here.
            effective_settings = effective_settings.model_copy(update={"trace_leg_ranks": True})
        # QUERY_DECOMPOSITION_ENABLED, as ``answer`` does it (an LLM call): the eval
        # measured retrieval without it, so turning it on moved nothing it scored.
        # ``POST /search`` (``for_generation=False``) makes no LLM call.
        decomposition: tuple[list[str], float] | None = None
        if (
            for_generation
            and effective_settings.query_decomposition_enabled
            and might_be_compound(question)
            and not (narrow is not None and narrow.chunk_ids)
        ):
            slot = drain(wait_for_model(effective_settings, None))
            try:
                decomposition = self._decomposition(question, effective_settings)
            finally:
                slot.release()
        return self._retrieve_and_rerank(
            question,
            effective_settings,
            filenames,
            tags,
            for_generation=for_generation,
            narrow=narrow,
            also_search=also_search,
            # Search, re-check and the eval harness keep no trace of it.
            traced=False,
            collect_leg_ranks=explain,
            parts=decomposition,
        )

    def rerun_retrieval(
        self, trace: QueryTrace, settings_update: Mapping[str, Any]
    ) -> tuple[RetrievalPhase, QueryTrace]:
        """``trace``'s question searched again — what it searched (a follow-up's
        rewrite, next to the question as asked), in its scope and narrowing — under the
        settings it was asked with (``_RERUN_FROM_TRACE``; the server's where the trace
        has none) with ``settings_update`` (ranking settings only), and the
        trace that retrieval would have left: ``POST /traces/{id}/rerun``. No LLM call:
        query expansion is left out (its variants came from the model)."""

        config = trace.config
        settings = self._effective_settings(AnswerOverrides()).model_copy(
            update={
                **_recorded_settings(config),
                **settings_update,
                "query_expansion_enabled": False,
            }
        )
        record = config.narrowing if config is not None else None
        narrow = filter_from_record(record) if record else None
        query = trace.condensed_question or trace.question
        phase = self._retrieve_and_rerank(
            query,
            settings,
            config.filenames if config is not None else None,
            config.tags if config is not None else None,
            for_generation=True,
            narrow=narrow,
            also_search=trace.question if query != trace.question else None,
        )
        # What reached the prompt, fitted as the answer was (without its history, which
        # a trace doesn't keep): passages the window left out are left out again.
        fitted = fit_prompt_to_context_window(
            query,
            phase.context_chunks[: int(settings.max_context_chunks)],
            None,
            settings,
            weak_matches=phase.below_min_score,
        )
        selected = fitted.context_chunks
        rerun = dataclasses.replace(
            trace,
            config=self._trace_config(
                settings,
                config.filenames if config is not None else None,
                config.tags if config is not None else None,
                narrow,
            ),
            candidates=build_trace_candidates(
                phase.fused_candidates,
                phase.scored_chunks,
                phase.context_chunks,
                selected,
                min_score=float(settings.rerank_min_score),
                diversity_dropped_ids=phase.diversity_dropped_ids,
                relative_cut_ids=phase.relative_cut_ids,
                context_budget_dropped_ids=fitted.dropped_point_ids,
                leg_ranks=phase.leg_ranks,
            ),
            answer=None,
            weak_matches=phase.below_min_score,
        )
        return phase, rerun

    def retrieve_similar(
        self,
        chunk_id: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        *,
        narrow: PassageFilter | None = None,
    ) -> RetrievalPhase | None:
        """``POST /search`` from a passage ("more like this"), or None when no passage
        has ``chunk_id``.

        The same hybrid search, RRF and rerank as a search: both legs are queried with
        the passage's stored dense and sparse vectors, and the cross-encoder scores
        each candidate against the passage as it was embedded. The passage itself is
        left out; the request's narrowing still applies.
        """

        effective_settings = self._effective_settings(overrides or AnswerOverrides())
        seed = self._repository.seed_passage(effective_settings, chunk_id)
        if seed is None:
            return None
        without_seed = dataclasses.replace(
            narrow or PassageFilter(),
            exclude_chunk_ids=tuple(
                sorted({*(narrow.exclude_chunk_ids if narrow else ()), chunk_id})
            ),
        )
        return self._retrieve_and_rerank(
            contextual_text(seed.filename, seed.section, seed.text),
            effective_settings,
            filenames,
            tags,
            for_generation=False,
            narrow=without_seed,
            seed=seed.embedding,
        )

    def _cache_key(
        self,
        question: str,
        overrides: AnswerOverrides | None,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        history: Sequence[ChatMessage] | None,
    ) -> CacheKey | None:
        """The answer-cache key for this request, or None when it must not be cached."""

        if self._answer_cache is None or history or not question.strip():
            return None
        try:
            overrides = overrides or AnswerOverrides()
            settings = self._effective_settings(overrides)
            return answer_cache_key(question, settings, filenames, tags, overrides.narrow)
        except (RetrievalConfigError, GenerationConfigError):
            # The uncached path raises it with its proper handling, error trace
            # included (a provider without its key failed here, before the trace).
            return None

    def cached_answer(
        self,
        question: str,
        overrides: AnswerOverrides | None,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        history: Sequence[ChatMessage] | None,
    ) -> CachedAnswer | None:
        """The answer cache's answer to this question now, if it has one: a full server
        still replays it (no question slot is needed for that), passed back as
        ``answer(cached=)`` so it can't expire between the check and the replay."""

        key = self._cache_key(question, overrides, filenames, tags, history)
        if key is None or self._answer_cache is None:
            return None
        return self._answer_cache.get(key)

    def joinable(
        self,
        question: str,
        overrides: AnswerOverrides | None,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        history: Sequence[ChatMessage] | None,
    ) -> bool:
        """Whether this question could join (or lead) an identical run: it has a cache
        key and there is a cache. Admission skips waiting for a run otherwise."""

        if self._answer_cache is None:
            return False
        return self._cache_key(question, overrides, filenames, tags, history) is not None

    def identical_run(
        self,
        question: str,
        overrides: AnswerOverrides | None,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        history: Sequence[ChatMessage] | None,
    ) -> threading.Event | None:
        """The event of an identical question being answered right now, if there is
        one: a full server lets a request wait for its answer without a slot
        (``answer(follow=)``)."""

        key = self._cache_key(question, overrides, filenames, tags, history)
        if key is None or self._answer_cache is None:
            return None
        return self._answer_cache.inflight.running(key)

    def claim_run(
        self,
        question: str,
        overrides: AnswerOverrides | None,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        history: Sequence[ChatMessage] | None,
    ) -> tuple[threading.Event, bool, Callable[[], None]] | None:
        """Claim this question's run as it is admitted: (its event, whether the caller
        now leads it, a finish that releases its followers — idempotent, so the
        request's own end can call it however the run went). None when it isn't
        cached. Claimed at admission, an identical question sent at the same moment
        finds the run already registered and waits for it, never a 429."""

        key = self._cache_key(question, overrides, filenames, tags, history)
        cache = self._answer_cache
        if key is None or cache is None:
            return None
        event, leading = cache.inflight.claim(key)
        return event, leading, (lambda: cache.inflight.finish(key, event))

    def answer(
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
        take_slot: Callable[[], bool] | None = None,
    ) -> GroundedAnswer:
        """``_answer_uncached``, served from the answer cache when it has this question
        (``cached``: the hit already looked up, replayed as it is).

        ``use_cache=False`` (Regenerate) skips the lookup but still stores the new
        answer, so the next plain ask gets the regenerated one. ``should_stop`` (the
        client has gone) is checked between stages, as on the stream: ``QuestionAbandoned``
        is raised instead of finishing work nobody will read.
        """

        started = time.monotonic()
        if cached is not None:
            self._log_hit(cached, question, "sync", filenames, tags, overrides)
            return cached.as_grounded(_cached_timings(started))
        key = self._cache_key(question, overrides, filenames, tags, history)
        if follow is not None and key is not None and self._answer_cache is not None:
            # Admitted without a slot to wait for the identical run: its answer, or — when
            # it left none — a slot of this request's own to run it (else a 429).
            followed = drain(follow_only(self._answer_cache, key, follow, should_stop, take_slot))
            if followed is None:
                raise QuestionAbandoned
            if isinstance(followed, CachedAnswer):
                self._log_hit(followed, question, "sync", filenames, tags, overrides)
                return followed.as_grounded(_cached_timings(started))
            # This request leads the run now (it claimed it and took a slot).
            claimed = followed
        # ``claimed``: this request leads the run already (claimed as it was admitted).
        lead: threading.Event | None = claimed
        if claimed is None and key is not None and use_cache and self._answer_cache is not None:
            hit = self._answer_cache.get(key)
            if hit is None:
                joined = drain(lead_or_join(self._answer_cache, key, should_stop))
                if joined is None:
                    raise QuestionAbandoned
                if isinstance(joined, CachedAnswer):
                    hit = joined
                elif joined is not None:
                    lead = joined
            if hit is not None:
                self._log_hit(hit, question, "sync", filenames, tags, overrides)
                return hit.as_grounded(_cached_timings(started))
        failure: BaseException | None = None
        try:
            grounded = self._answer_uncached(
                question, overrides, filenames, history, tags, should_stop, use_memo=use_cache
            )
            # A cut-off answer isn't kept: asked again, it may come out whole.
            if (
                key is not None
                and self._answer_cache is not None
                and not was_degraded()
                and not grounded.truncated
                and not grounded.prompt_cut
            ):
                self._answer_cache.put(key, CachedAnswer.from_grounded(grounded))
        except BaseException as exc:
            failure = exc
            raise
        finally:
            # After the store, so followers find the answer; on a failure they find
            # none and ask themselves — unless asking again would fail the same way.
            if lead is not None and key is not None and self._answer_cache is not None:
                self._answer_cache.inflight.finish(key, lead, failure)
        return grounded

    def _answer_uncached(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
        tags: Sequence[str] | None = None,
        should_stop: Callable[[], bool] | None = None,
        *,
        use_memo: bool = True,
    ) -> GroundedAnswer:
        """Retrieve, rerank, and generate a grounded, cited answer.

        ``overrides`` are folded into a single ``model_copy`` of the shared settings
        (``_effective_settings``) — the singleton is never mutated, so concurrent
        requests with different overrides stay isolated. The dense/sparse retrieval
        limits, ``fused_top_n``, and ``rrf_k`` always come from the base settings
        (embeddings and hybrid fusion stay fixed regardless). ``filenames``, when given,
        restricts retrieval to those documents. ``history``, when given, drives one
        extra condense call that rewrites ``question`` into a standalone retrieval
        query — the server itself stores no conversation state.
        """

        run = self._start_run(question, overrides, filenames, tags, "sync")
        run.use_memo = use_memo
        # Held from the answer through its citation retry and follow-ups, as the stream
        # does, so another question's call can't slip in between them at a local model.
        slot: GenerationSlot | None = None
        # Every LLM call this question makes counts: the condense, query expansion,
        # the answer, its citation retry and the follow-ups.
        begin_usage()

        try:
            effective_settings = run.settings = self._effective_settings(run.overrides)
            run.history = self._truncate_history(history, effective_settings)
            # The condense and query-expansion calls wait their turn at a local model
            # like the answer does, and give the place back for retrieval (as the
            # stream does): sent unguarded, they queued inside Ollama with their
            # timeout running behind another question's generation.
            condense_slot: GenerationSlot | None = None
            try:
                if self._will_condense(question, run.history, effective_settings, run.use_memo):
                    condense_slot = drain(wait_for_model(effective_settings, should_stop))
                retrieval_query, condense_ms, run.condensed_question, run.condense_note = (
                    self._condense_query(question, run.history, effective_settings, run.use_memo)
                )
            finally:
                if condense_slot is not None:
                    condense_slot.release()
            expansion: tuple[list[str], float] | None = None
            if effective_settings.query_expansion_enabled and retrieval_query.strip():
                expansion_slot = drain(wait_for_model(effective_settings, should_stop))
                try:
                    expansion = self._expand_query(retrieval_query, effective_settings)
                finally:
                    expansion_slot.release()
            # A compound question's parts (QUERY_DECOMPOSITION_ENABLED): an LLM call too.
            decomposition: tuple[list[str], float] | None = None
            if (
                effective_settings.query_decomposition_enabled
                and might_be_compound(retrieval_query)
                and not (run.narrow is not None and run.narrow.chunk_ids)
            ):
                decompose_slot = drain(wait_for_model(effective_settings, should_stop))
                try:
                    decomposition = self._decomposition(retrieval_query, effective_settings)
                finally:
                    decompose_slot.release()
            phase = self._retrieve_and_rerank(
                retrieval_query,
                effective_settings,
                filenames,
                tags,
                should_stop,
                narrow=run.narrow,
                also_search=question if run.condensed_question else None,
                expansion=expansion,
                parts=decomposition,
            )
            prompt_question, fitted = self._fit_prompt(run, effective_settings, phase)
            selected = fitted.context_chunks
            generator = self._answer_generator(run, effective_settings)

            # Nothing to answer from: the canned answer makes no LLM call, so it must not
            # queue behind another question's generation at a local model — nor must an
            # answer the prompt cache already holds.
            if selected and not _reply_known(
                generator,
                answer_messages(
                    prompt_question.strip(),
                    list(selected[: int(effective_settings.max_context_chunks)]),
                    fitted.history or None,
                    effective_settings,
                    weak_matches=phase.below_min_score,
                ),
                effective_settings,
            ):
                slot = drain(wait_for_model(effective_settings, should_stop))
                if should_stop is not None and should_stop():
                    raise QuestionAbandoned
            generate_start = time.monotonic()
            reset_finish_reason()
            grounded = generate_grounded_answer(
                query=prompt_question,
                context_chunks=selected,
                # AUTO_CONTINUE_MAX, as on the stream: a cut-off answer is continued.
                generator=(
                    AutoContinuing(generator, int(effective_settings.auto_continue_max))
                    if effective_settings.auto_continue_max
                    else generator
                ),
                settings=effective_settings,
                history=fitted.history or None,
                weak_matches=phase.below_min_score,
            )
            # generate_grounded_answer runs the zero-citation retry internally, so the
            # block just timed covers two LLM calls when it fired. Split it back out so
            # generate_ms means the same thing here as it does on the streaming path.
            generate_ms = (time.monotonic() - generate_start) * 1000 - grounded.citation_retry_ms
            # The last call made the answer: the citation retry's when it fired.
            truncated = stopped_at_token_limit()
            timings = self._phase_timings(
                phase,
                total_start=run.started,
                condense_ms=condense_ms,
                generate_ms=generate_ms,
                citation_retry_ms=grounded.citation_retry_ms,
            )
        except Exception as exc:
            if slot is not None:
                slot.release()
            # Traces are readable with the read-only key: the same rule as a response
            # body, so no internal detail.
            self._store_error_trace(run, public_error_message(exc))
            raise

        # The answer's usage: everything up to it (condense, expansion, the answer and
        # its citation retry), as the stream's `done` reports it. Follow-up tokens are
        # still counted in the token and cost metrics.
        usage = usage_totals()
        cost_usd = usage_cost_usd(effective_settings, usage)
        try:
            status = self._scoped_status(
                answer_status(grounded.answer, len(grounded.sources), len(selected)),
                run,
                effective_settings,
            )
            wants_followups = effective_settings.followups_enabled and status == "answered"
            followups = (
                generate_followups(
                    question.strip(),
                    grounded.answer,
                    _cited_chunks(grounded.sources, selected),
                    generator,
                    effective_settings,
                )
                if wants_followups
                else []
            )
        finally:
            if slot is not None:
                slot.release()
        if wants_followups:
            # The response waited for them (even when none came back), so the total does.
            timings = replace(timings, total_ms=(time.monotonic() - run.started) * 1000)
        # The prompt is the one captured from generation, not rebuilt here: a second
        # build_grounded_messages call would agree today but is a re-derivation that
        # can drift, and it could never reproduce the citation-retry continuation that
        # actually produced the answer when grounded.citation_retry_used is set.
        self._store_finished_trace(
            run,
            effective_settings,
            phase,
            status=trace_status(status),
            selected=selected,
            context_budget_dropped_ids=fitted.dropped_point_ids,
            prompt_messages=grounded.prompt_messages or None,
            answer=grounded.answer,
            cited_source_numbers=[source.source_number for source in grounded.sources],
            timings=timings,
            citation_retry_used=grounded.citation_retry_used,
            citation_backfilled=grounded.citation_backfilled,
            unsupported_claims=grounded.unsupported_claims,
            unsupported_numbers=grounded.unsupported_numbers,
            prompt_cached=self._prompt_cached(run),
            prompt_cut=was_prompt_cut(),
            usage=usage,
            answer_status=status,
            truncated=truncated,
            dropped_history_messages=fitted.dropped_history_messages,
        )
        return replace(
            grounded,
            timings=timings,
            trace_id=run.trace_id,
            answer_status=status,
            top_rerank_score=_top_rerank_score(phase),
            weak_matches=phase.below_min_score,
            truncated=truncated,
            passages_left_out=_picked_left_out(run.narrow, selected),
            followups=followups,
            usage=usage,
            cost_usd=cost_usd,
            prompt_cached=self._prompt_cached(run),
            prompt_cut=was_prompt_cut(),
            retrieval_query=run.condensed_question or None,
        )

    def _log_hit(
        self,
        hit: CachedAnswer,
        question: str,
        mode: TraceMode,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        overrides: AnswerOverrides | None,
    ) -> None:
        log_served(
            self._trace_store,
            served_query(
                hit,
                question=question,
                mode=mode,
                workspace=workspace_of(self._settings),
                filenames=filenames,
                tags=tags,
                narrowing=narrowing_record((overrides or AnswerOverrides()).narrow),
            ),
        )

    def _answer_generator(self, run: AnswerRun, settings: AppSettings) -> ChatGenerator:
        """The generator for ``run``'s answer, citation retry and follow-ups: through the
        prompt cache (PROMPT_CACHE_SIZE) unless it is off. A Regenerate reuses nothing
        but stores its reply over the one it replaces."""

        cache = shared_prompt_cache(settings)
        run.generator = (
            CachingGenerator(self._generator, cache, lookup=run.use_memo)
            if cache is not None
            else self._generator
        )
        return run.generator

    @staticmethod
    def _prompt_cached(run: AnswerRun) -> bool:
        return isinstance(run.generator, CachingGenerator) and run.generator.answer_reused

    def _start_run(
        self,
        question: str,
        overrides: AnswerOverrides | None,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        mode: TraceMode,
    ) -> AnswerRun:
        return AnswerRun(
            question=question,
            overrides=overrides or AnswerOverrides(),
            filenames=filenames,
            tags=tags,
            mode=mode,
            trace_id=str(uuid.uuid4()) if self._trace_store is not None else None,
            started=time.monotonic(),
        )

    def _fit_prompt(
        self, run: AnswerRun, settings: AppSettings, phase: RetrievalPhase
    ) -> tuple[str, FittedPrompt]:
        """The question as the answering prompt puts it, and what of the passages and
        history fits the window."""

        prompt_question = _with_interpretation(run.question.strip(), run.condensed_question)
        fitted = fit_prompt_to_context_window(
            prompt_question,
            phase.context_chunks[: int(settings.max_context_chunks)],
            _clip_earlier_answers(run.history, settings),
            settings,
            weak_matches=phase.below_min_score,
        )
        run.phase, run.fitted = phase, fitted
        return prompt_question, fitted


def _stored_chunk(payload: Mapping[str, object], filename: str) -> RerankedChunk | None:
    """A stored point's payload as a passage to check an answer against."""

    text, chunk_id = payload.get("text"), payload.get("chunk_id")
    if not isinstance(text, str) or not isinstance(chunk_id, str):
        return None

    def number(key: str) -> int | None:
        value = payload.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    section = payload.get("section")
    return RerankedChunk(
        point_id=chunk_id,
        filename=filename,
        page=number("page") or 0,
        section=section if isinstance(section, str) else "",
        chunk_id=chunk_id,
        text=text,
        retrieval_score=0.0,
        rerank_score=0.0,
        chunk_ordinal=number("chunk_ordinal"),
        page_end=number("page_end"),
    )
