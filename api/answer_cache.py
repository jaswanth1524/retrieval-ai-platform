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
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from threading import Event, Lock

from api.corpus import corpus_generation
from api.generation import SourceCitation, completion_model_and_kwargs
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
    # Summaries only: (passages read, passages in the documents).
    coverage: tuple[int, int] | None = None
    followups: tuple[str, ...] = ()


CacheKey = tuple[object, ...]


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
        narrow,
        corpus_generation(settings.qdrant_collection),
    )


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
        self._entries: OrderedDict[CacheKey, tuple[float, CachedAnswer]] = OrderedDict()
        # Shared with the cache: a follower replays the leader's cached answer.
        self.inflight = InflightRuns()

    def get(self, key: CacheKey) -> CachedAnswer | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if self._clock() - stored_at > self._ttl:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return value

    def put(self, key: CacheKey, value: CachedAnswer) -> None:
        with self._lock:
            self._entries[key] = (self._clock(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
