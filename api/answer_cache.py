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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Lock

from api.corpus import bump_corpus_generation, corpus_generation
from api.generation import SourceCitation, completion_model_and_kwargs
from api.settings import AppSettings

__all__ = [
    "AnswerCache",
    "CacheKey",
    "CachedAnswer",
    "answer_cache_key",
    "bump_corpus_generation",
    "corpus_generation",
]


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    sources: tuple[SourceCitation, ...]
    # The trace of the run that produced the answer, so the inspector can still show
    # how it was retrieved.
    trace_id: str | None


CacheKey = tuple[object, ...]


def answer_cache_key(
    question: str,
    settings: AppSettings,
    filenames: Sequence[str] | None,
    tags: Sequence[str] | None = None,
) -> CacheKey:
    """Everything the answer depends on that can differ between two requests."""

    model, _ = completion_model_and_kwargs(settings)
    return (
        " ".join(question.split()).casefold(),
        tuple(sorted(set(filenames or ()))),
        tuple(sorted(set(tags or ()))),
        settings.llm_provider.lower().strip(),
        model,
        int(settings.rerank_top_k),
        int(settings.max_context_chunks),
        float(settings.llm_temperature),
        corpus_generation(),
    )


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
