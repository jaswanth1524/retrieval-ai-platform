"""Replies to identical LLM calls, reused (PROMPT_CACHE_SIZE, off by default).

The answer cache (``api/answer_cache.py``) keys on the question and its scope, so every
ingest empties its unscoped entries. This one keys on what is actually sent to the
model — the messages, the model and the options that shape the reply — so a question
asked again after an unrelated upload still skips its LLM call when retrieval found the
same passages. Retrieval, reranking, citations and the trace all run as usual; only the
call is saved. A reply cut off at the token limit is never kept (asked again, it may
come out whole), and neither is an empty or failed one.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

from api.llm import (
    GenerationConfigError,
    completion_model_and_kwargs,
    effective_max_tokens,
    reset_finish_reason,
    stopped_at_token_limit,
    was_prompt_cut,
)
from api.settings import AppSettings

if TYPE_CHECKING:
    from api.generation import ChatGenerator, ChatMessage


class PromptCache:
    """Replies by prompt key, least recently used dropped past ``max_entries``, each
    kept ``ttl_seconds``."""

    def __init__(
        self,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def get(self, key: str) -> str | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, reply = entry
            if self._clock() - stored_at > self.ttl_seconds:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return reply

    def put(self, key: str, reply: str) -> None:
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = (self._clock(), reply)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def prompt_key(messages: Sequence[ChatMessage], settings: AppSettings) -> str | None:
    """A digest of everything the call sends that shapes the reply: the messages, the
    model, the provider's options (keys left out — they don't change the reply), the
    temperature and the output budget. None when the provider isn't configured."""

    try:
        model, kwargs = completion_model_and_kwargs(settings)
    except GenerationConfigError:
        return None
    options = {name: value for name, value in kwargs.items() if "key" not in name.lower()}
    payload = {
        "messages": [[message["role"], message["content"]] for message in messages],
        "model": model,
        "options": options,
        "temperature": float(settings.llm_temperature),
        "max_tokens": effective_max_tokens(settings, model),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class CachingGenerator:
    """``inner`` with its replies reused from ``cache`` (one per question being
    answered). ``answer_reused`` says whether a call was served from the cache, for the
    answer's ``prompt_cached`` flag."""

    def __init__(self, inner: ChatGenerator, cache: PromptCache) -> None:
        self.inner = inner
        self.cache = cache
        self.answer_reused = False

    def cached_reply(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str | None:
        key = prompt_key(messages, settings)
        return self.cache.get(key) if key is not None else None

    def _hit(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str | None:
        reply = self.cached_reply(messages, settings)
        if reply is not None:
            # It finished normally the first time (cut-off replies aren't kept).
            reset_finish_reason("stop")
            self.answer_reused = True
        return reply

    def _keep(self, messages: Sequence[ChatMessage], settings: AppSettings, reply: str) -> None:
        if not reply.strip() or stopped_at_token_limit() or was_prompt_cut():
            return
        key = prompt_key(messages, settings)
        if key is not None:
            self.cache.put(key, reply)

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        reply = self._hit(messages, settings)
        if reply is not None:
            return reply
        reply = self.inner.complete(messages, settings)
        self._keep(messages, settings, reply)
        return reply

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterator[str]:
        reply = self._hit(messages, settings)
        if reply is not None:
            yield reply
            return
        parts: list[str] = []
        for delta in self.inner.stream(messages, settings):
            parts.append(delta)
            yield delta
        # Only a stream read to its end: a reader that left closed it at a yield above.
        self._keep(messages, settings, "".join(parts))


_shared: PromptCache | None = None
_shared_lock = threading.Lock()


def shared_prompt_cache(settings: AppSettings) -> PromptCache | None:
    """The process's prompt cache (PROMPT_CACHE_SIZE entries, kept for
    ANSWER_CACHE_TTL_SECONDS), or None when it is off."""

    global _shared
    size = int(settings.prompt_cache_size)
    if size <= 0:
        return None
    with _shared_lock:
        if _shared is None:
            _shared = PromptCache(size, float(settings.answer_cache_ttl_seconds))
        else:
            _shared.max_entries = size
            _shared.ttl_seconds = float(settings.answer_cache_ttl_seconds)
        return _shared


def reset_shared_prompt_cache() -> None:
    global _shared
    with _shared_lock:
        _shared = None
