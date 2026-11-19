"""Conversation history for an answer: trimming it, and the condense step's memo of
follow-up rewrites. Moved out of ``api/pipeline.py``, which re-exports these names.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Sequence

from api.answer_run import trace_model_name
from api.citations import (
    cited_numbers,
    declines_to_answer,
    states_insufficiency,
)
from api.generation import (
    ChatMessage,
    needs_condense,
)
from api.prompting import (
    clip_text,
    one_line,
)
from api.settings import AppSettings


def _without_unanswered_exchanges(history: list[ChatMessage]) -> list[ChatMessage]:
    """``history`` minus every exchange whose answer was "not enough information".

    Sent back, a refusal primed the next question twice over: the condense step folded
    it into the search query ("What if they miss it?" became "What are the consequences
    if they miss the key points?") and the answering model read it as the
    conversation's own conclusion and refused again. A refusal that cites the sources
    it looked at counts too (``declines_to_answer``); an answer that says something is
    missing but gives something else from a source answered part of it, and stays.
    """

    kept: list[ChatMessage] = []
    for message in history:
        content = message["content"]
        if message["role"] == "assistant" and (
            (
                states_insufficiency(content)
                and not cited_numbers(content, _HISTORY_CITATION_CHECK_MAX)
            )
            or declines_to_answer(content)
        ):
            if kept and kept[-1]["role"] == "user":
                kept.pop()
            continue
        kept.append(message)
    return kept


def _from_first_question(history: list[ChatMessage]) -> list[ChatMessage]:
    """``history`` from its first user message on."""

    start = next(
        (index for index, message in enumerate(history) if message["role"] == "user"),
        len(history),
    )
    return history[start:]


# Only whether an earlier answer cites anything matters, so ranges are clamped small.
_HISTORY_CITATION_CHECK_MAX = 1000


def _clip_earlier_answers(history: list[ChatMessage], settings: AppSettings) -> list[ChatMessage]:
    """``history`` with each earlier answer cut to GENERATION_HISTORY_ASSISTANT_CHARS
    (0: unchanged). The answering model reads all of it before the first token."""

    limit = int(settings.generation_history_assistant_chars)
    if limit <= 0:
        return history
    return [
        {"role": "assistant", "content": clip_text(message["content"], limit)}
        if message["role"] == "assistant"
        else message
        for message in history
    ]


def _with_interpretation(question: str, condensed: str | None) -> str:
    """The question as the answering prompt shows it: a follow-up that was rewritten for
    search carries the rewrite, so a small model reading "Explain in more detail" next
    to sources about vacation days knows what "more" is about."""

    if not condensed or condensed.casefold() == question.casefold():
        return question
    return f"{question}\n(Read with the conversation as: {one_line(condensed)})"


class CondenseMemo:
    """Rewrites of follow-ups already made: Retry, an edited follow-up asked again or
    the same follow-up asked in another tab paid the condense call (2.5-7 s at a local
    model) every time. Keyed on everything the call is sent — the question, the history
    as truncated, the provider and model, the condense settings — so only an identical
    call is skipped. Only usable rewrites are kept (a failure or a rejected rewrite is
    tried again). CONDENSE_CACHE_SIZE entries, least recently used dropped; 0 = off."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def key(question: str, history: Sequence[ChatMessage], settings: AppSettings) -> str:
        parts = [
            question,
            str(settings.llm_provider),
            trace_model_name(settings) or "",
            str(settings.condense_max_tokens),
            *(f"{message['role']}:{message['content']}" for message in history),
        ]
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            found = self._entries.get(key)
            if found is not None:
                self._entries.move_to_end(key)
            return found

    def put(self, key: str, rewrite: str, size: int) -> None:
        if size <= 0:
            return
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = rewrite
            while len(self._entries) > size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


condense_memo = CondenseMemo()


def _condense_skip_reason(question: str, settings: AppSettings) -> str | None:
    """Why a question with history is not condensed, or None when it is. The one copy
    of the gate: ``_will_condense`` announces the stage from it, ``_condense_query``
    runs (or skips) the call from it."""

    if not settings.conversation_condense_enabled:
        return "disabled"
    if not needs_condense(question):
        return "standalone"
    return None
