"""In-process cache of finished answers, so a repeated question skips the whole pipeline.

A question costs retrieval, reranking and an LLM call — tens of seconds on the local
default. The same question against the same corpus and settings gets the same grounded
answer (temperature 0 by default), so it is served from here instead.

Keys include a *corpus generation* that every ingest (even a failed one), delete and
tag change bumps, so an answer is never served from before the documents it was drawn from
changed. Entries also expire after a TTL, which bounds staleness across processes
(the generation is per process; a second API worker's writes don't bump this one's).
Questions with conversation history are never cached: their retrieval query depends
on the history.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any

from api.citations import citation_from_dict
from api.corpus import changed_since, corpus_generation
from api.generation import (
    GroundedAnswer,
    SourceCitation,
    StageTimings,
    completion_model_and_kwargs,
)
from api.settings import AppSettings


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    sources: tuple[SourceCitation, ...]
    # The trace of the run that produced the answer, so the inspector can still show
    # how it was retrieved.
    trace_id: str | None
    answer_status: str = "answered"
    top_rerank_score: float | None = None
    truncated: bool = False
    weak_matches: bool = False
    citation_backfilled: bool = False
    passages_left_out: int = 0
    # Summaries only: (passages read, passages in the documents).
    coverage: tuple[int, int] | None = None
    followups: tuple[str, ...] = ()

    # The three ways an answer is stored and served — from the sync answer, from the
    # stream's (or a summary's) `done` frame, and back as a sync answer — in one place,
    # so a field added to one can't be left out of another.

    @classmethod
    def from_grounded(cls, grounded: GroundedAnswer) -> CachedAnswer:
        return cls(
            grounded.answer,
            tuple(grounded.sources),
            grounded.trace_id,
            grounded.answer_status,
            grounded.top_rerank_score,
            grounded.truncated,
            weak_matches=grounded.weak_matches,
            citation_backfilled=grounded.citation_backfilled,
            passages_left_out=grounded.passages_left_out,
            followups=tuple(grounded.followups),
        )

    @classmethod
    def from_done(cls, done: Mapping[str, Any], followups: Sequence[str] = ()) -> CachedAnswer:
        """From a ``done`` frame; a summary's (it has ``chunks_read``) keeps its coverage."""

        summary = "chunks_read" in done or "chunks_total" in done
        return cls(
            done["answer"],
            tuple(citation_from_dict(source) for source in done["sources"]),
            done["trace_id"],
            done.get("answer_status", "answered"),
            None if summary else done.get("top_rerank_score"),
            done.get("truncated", False),
            weak_matches=done.get("weak_matches", False),
            citation_backfilled=done.get("citation_backfilled", False),
            passages_left_out=done.get("passages_left_out", 0),
            coverage=(done.get("chunks_read", 0), done.get("chunks_total", 0)) if summary else None,
            followups=tuple(followups),
        )

    def as_grounded(self, timings: StageTimings) -> GroundedAnswer:
        """The sync answer this entry serves, marked cached."""

        return GroundedAnswer(
            answer=self.answer,
            sources=list(self.sources),
            timings=timings,
            trace_id=self.trace_id,
            cached=True,
            answer_status=self.answer_status,
            top_rerank_score=self.top_rerank_score,
            weak_matches=self.weak_matches,
            truncated=self.truncated,
            citation_backfilled=self.citation_backfilled,
            passages_left_out=self.passages_left_out,
            followups=list(self.followups),
        )


CacheKey = tuple[object, ...]


class ScopedGeneration:
    """The corpus generation in the key of an answer drawn from named documents only.

    Equal (and hashed) by collection and documents alone, so a lookup finds the entry
    whatever has changed since; ``AnswerCache.get`` then serves it only if none of
    those documents changed after ``generation`` — read when the question was asked,
    before retrieval — so an upload elsewhere in the corpus (a watched folder, a URL
    refresh) no longer empties every scoped entry. When the change log can't say
    (``changed_since`` is None), it is a miss.
    """

    def __init__(self, collection: str, filenames: frozenset[str], generation: int) -> None:
        self.collection = collection
        self.filenames = filenames
        self.generation = generation

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ScopedGeneration)
            and self.collection == other.collection
            and self.filenames == other.filenames
        )

    def __hash__(self) -> int:
        return hash((self.collection, self.filenames))

    def unchanged(self) -> bool:
        changed = changed_since(self.generation, self.collection)
        return changed is not None and not (changed & self.filenames)


def corpus_state(
    settings: AppSettings, filenames: Sequence[str] | None, tags: Sequence[str] | None = None
) -> int | ScopedGeneration:
    """The part of a cache key that says which corpus an answer was drawn from: for
    named documents only (no tags, which any upload could match), a
    ``ScopedGeneration``; else the collection's generation, which every change moves."""

    collection = settings.qdrant_collection
    if filenames and not tags:
        return ScopedGeneration(collection, frozenset(filenames), corpus_generation(collection))
    return corpus_generation(collection)


def answer_cache_key(
    question: str,
    settings: AppSettings,
    filenames: Sequence[str] | None,
    tags: Sequence[str] | None = None,
    narrow: Hashable | None = None,
) -> CacheKey:
    """Everything the answer depends on that can differ between two requests.

    ``narrow`` is the question's passage narrowing (``repository.PassageFilter``, a
    frozen dataclass): a question asked over pages 3-4 is another answer.
    """

    model, _ = completion_model_and_kwargs(settings)
    return (
        " ".join(question.split()).casefold(),
        tuple(sorted(set(filenames or ()))),
        # Question tags match stored tags case-insensitively, so "Finance" and
        # "finance" are one answer.
        tuple(sorted({tag.casefold() for tag in tags or ()})),
        settings.llm_provider.lower().strip(),
        model,
        # The workspace: the same question over another corpus is another answer.
        settings.qdrant_collection,
        int(settings.rerank_top_k),
        int(settings.max_context_chunks),
        float(settings.llm_temperature),
        settings.answer_language,
        settings.answer_format,
        # PASSAGES_ONLY / STRONG_MATCHES_ONLY: a stricter answer is another answer.
        int(settings.context_neighbor_radius),
        int(settings.rerank_fallback_count),
        # Settings a workspace can change (WorkspaceSettingsStore): a key built after a
        # change can't be served an answer ranked under the old ones.
        float(settings.rerank_min_score),
        float(settings.recency_boost_weight),
        float(settings.rerank_relative_floor),
        # An answer cached without follow-ups is replayed without them.
        bool(settings.followups_enabled),
        narrow,
        corpus_state(settings, filenames, tags),
    )


# Where the collection sits in each kind of key (``answer_cache_key``, and
# ``summarize.summary_cache_key``, whose first element is "summary").
_QUESTION_KEY_COLLECTION = 5
_SUMMARY_KEY_COLLECTION = 4


def _is_summary_key(key: CacheKey) -> bool:
    # A question's second element is its filenames tuple, a summary's its mode.
    return len(key) > _SUMMARY_KEY_COLLECTION and key[0] == "summary" and isinstance(key[1], str)


def key_collection(key: CacheKey) -> object:
    """The collection (workspace) a cache key was built for."""

    index = _SUMMARY_KEY_COLLECTION if _is_summary_key(key) else _QUESTION_KEY_COLLECTION
    return key[index] if len(key) > index else None


def key_label(key: CacheKey) -> str:
    """What a cache key was asked: the question, or a summary's mode, documents and
    task ("compare: a.pdf, b.pdf — the costs")."""

    if not _is_summary_key(key):
        return str(key[0])
    mode, instruction, filenames = key[1], key[2], key[3]
    names = ", ".join(str(name) for name in filenames) if isinstance(filenames, tuple) else ""
    label = f"{mode}: {names}" if names else str(mode)
    return f"{label} — {instruction}" if instruction else label


class InflightRuns:
    """The questions (and summaries) being answered right now, by cache key.

    Two identical questions arriving together each ran retrieval, the rerank and the
    LLM; at a local model the second also queued behind the first for the whole
    generation. Now the first *leads*: the others *follow*, waiting until the leader's
    answer is in the cache and replaying it. A leader that fails or is abandoned leaves
    no answer, and a follower then runs the question itself — no hand-over machinery.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._runs: dict[CacheKey, Event] = {}

    def claim(self, key: CacheKey) -> tuple[Event, bool]:
        """The event of ``key``'s run, and whether the caller now leads it."""

        with self._lock:
            event = self._runs.get(key)
            if event is not None:
                return event, False
            event = Event()
            self._runs[key] = event
            return event, True

    def finish(self, key: CacheKey, event: Event) -> None:
        """Release ``key``'s followers (idempotent). Called right after the leader's
        answer is cached, and again from its ``finally`` as a backstop."""

        event.set()
        with self._lock:
            if self._runs.get(key) is event:
                del self._runs[key]

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)


class AnswerCache:
    """Thread-safe LRU with a TTL."""

    def __init__(
        self,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = Lock()
        self._entries: OrderedDict[
            CacheKey, tuple[float, CachedAnswer, ScopedGeneration | None]
        ] = OrderedDict()
        # Shared with the cache: a follower replays the leader's cached answer.
        self.inflight = InflightRuns()

    def get(self, key: CacheKey) -> CachedAnswer | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, value, scope = entry
            if self._clock() - stored_at > self._ttl or (
                scope is not None and not scope.unchanged()
            ):
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return value

    def put(self, key: CacheKey, value: CachedAnswer) -> None:
        scope = key[-1] if key and isinstance(key[-1], ScopedGeneration) else None
        with self._lock:
            # Replaced, not updated: a dict keeps the first key object, whose generation
            # would then describe an older answer.
            self._entries.pop(key, None)
            self._entries[key] = (self._clock(), value, scope)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def entries_for(self, collection: str) -> list[tuple[str, float]]:
        """``collection``'s live entries, newest first: what was asked (a summary's
        mode, documents and task) and how old the answer is, in seconds."""

        now = self._clock()
        with self._lock:
            found = [
                (key_label(key), now - stored_at)
                for key, (stored_at, _value, _scope) in self._entries.items()
                if key_collection(key) == collection and now - stored_at <= self._ttl
            ]
        return list(reversed(found))

    def drop_collection(self, collection: str) -> int:
        """Forget ``collection``'s entries (DELETE /answer-cache); how many went."""

        with self._lock:
            keys = [key for key in self._entries if key_collection(key) == collection]
            for key in keys:
                del self._entries[key]
        return len(keys)
