"""Continuing an answer cut off at the output limit (``POST /questions/continue`` and
``AUTO_CONTINUE_MAX``), moved out of ``api/pipeline.py``; it re-exports these names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from api.generation import (
    ChatMessage,
)
from api.llm import ChatGenerator, reset_finish_reason, stopped_at_token_limit
from api.reranking import (
    RerankedChunk,
)
from api.settings import AppSettings
from api.tracing import (
    QueryTrace,
)

CONTINUE_INSTRUCTION = (
    "Your answer above was cut off before it finished. Continue it from exactly where it "
    "stopped: do not repeat anything already written, and keep citing the sources by "
    "their [n] numbers."
)


class ContinueRefused(Exception):
    """A cut-off answer that can't be continued, with the HTTP status that says why."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ContinuePlan:
    trace: QueryTrace
    chunks: list[RerankedChunk]


# Providers whose chat API continues a trailing assistant message (prefill) — measured
# on Ollama (qwen2.5); Anthropic documents it. Asked in words instead ("continue where
# you stopped"), qwen2.5 started its answer over every time. Hosted OpenAI ignores a
# trailing assistant message, so it is asked in words.
_PREFILL_PROVIDERS = frozenset({"ollama", "anthropic", "openai_compatible"})


def continuation_messages(
    base: Sequence[ChatMessage], answer_so_far: str, settings: AppSettings
) -> list[ChatMessage]:
    """The prompt that asks for the rest of ``answer_so_far``: the answer as a trailing
    assistant message to continue, or, where the provider can't, followed by a request
    to go on."""

    if settings.llm_provider.lower().strip() in _PREFILL_PROVIDERS:
        # Anthropic refuses a prefill ending in whitespace; the join puts it back.
        return [*base, {"role": "assistant", "content": answer_so_far.rstrip()}]
    return [
        *base,
        {"role": "assistant", "content": answer_so_far},
        {"role": "user", "content": CONTINUE_INSTRUCTION},
    ]


def base_prompt(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """A trace's prompt without what a Continue added (the answer so far, the request to
    go on), so continuing again starts from the original prompt, not a nested one."""

    trimmed = list(messages)
    if trimmed and trimmed[-1]["role"] == "user" and trimmed[-1]["content"] == CONTINUE_INSTRUCTION:
        trimmed = trimmed[:-1]
    if trimmed and trimmed[-1]["role"] == "assistant":
        trimmed = trimmed[:-1]
    return trimmed


def drop_restart(before: str, after: str) -> str:
    """``after`` without a restart: a model asked to continue that writes the answer
    again from the top has the part it repeats removed (the rest is the new text), and
    one that starts again a few words back has those words removed."""

    head = after.lstrip()
    probe = before.lstrip()[:40]
    if len(probe) < 20 or not head.startswith(probe):
        return _drop_repeated_tail(before, after)
    original = before.lstrip()
    shared = 0
    while shared < min(len(original), len(head)) and original[shared] == head[shared]:
        shared += 1
    return head[shared:]


# How far back a continuation is looked at for words it repeats: a whole list item.
_REPEAT_MAX_CHARS = 400


def _drop_repeated_tail(before: str, after: str) -> str:
    """``after`` without what it repeats from the end of ``before``. Measured on
    gpt-5-mini continuing by instruction (AUTO_CONTINUE_MAX): "… It also" went on "It
    also states …", "… expense policies (" on "policies (no …", a cut list item was
    written again from its "- ", a lone "- " was followed by "- Review …", and "…
    within" by "within two days". The repeat must start where a word starts and be two
    words or more, a whole line, or one word of four letters or more (its comma, colon
    or semicolon included) — so "… of the" followed by "the policy" keeps its "the"."""

    head = after.lstrip()
    end = before.rstrip()
    for size in range(min(len(end), _REPEAT_MAX_CHARS), 0, -1):
        tail = end[-size:]
        if tail[:1].isspace() or (size < len(end) and not end[-size - 1].isspace()):
            continue  # not where a word starts
        whole_line = size == len(end) or end[-size - 1] == "\n"
        words = tail.split()
        # One word may end in its punctuation: gpt-4o-mini cut at "… Furthermore,"
        # went on "Furthermore, software …" (pass 21).
        word = tail.rstrip(",;:")
        if not (len(words) >= 2 or whole_line or (len(word) >= 4 and word.isalpha())):
            continue
        if head.startswith(tail):
            rest = head[len(tail) :]
            # The cut text keeps its own trailing space ("- "): not a second one.
            return rest.lstrip() if before[-1:].isspace() else rest
    return after


# Markdown emphasis or code markers at a cut. One that closes ("**" then a line break,
# space, punctuation or the end) belongs to the text before it; one that opens (at the
# start or after a space) belongs to the word after it. A space between either and its
# text breaks the markup ("**Security Policy: **" shows the asterisks).
_CLOSING_MARKER_RE = re.compile(r"[*_`]+(?:$|[\s.,;:!?)\]}'\"])")
_OPENING_MARKER_RE = re.compile(r"(?:^|\s)[*_`]+$")


def join_continuation(before: str, after: str, *, prefilled: bool = False) -> str:
    """A cut-off answer and its continuation as one text. Continuing a prefill, the
    model writes its own leading space where a word ends (" can be", or "2]" inside a
    marker): the two are joined as they are. Asked in words, a space goes between them
    unless either side already has one, the continuation starts with punctuation or a
    closing Markdown marker, or the cut came just after an opening bracket, quote or
    Markdown marker."""

    if not after:
        return before
    if prefilled:
        return before.rstrip() + after
    if before[-1:].isspace() or after[:1].isspace() or after[:1] in ".,;:!?)]}'\"":
        return before + after
    if before[-1:] in "([{\u201c\u2018":  # cut just after an opening bracket or quote
        return before + after
    if _CLOSING_MARKER_RE.match(after) or _OPENING_MARKER_RE.search(before):
        return before + after
    return f"{before} {after}"


class AutoContinuing:
    """A generator whose answers cut off at the output limit are continued, as
    AUTO_CONTINUE_MAX does on the stream, up to that many times — for the sync route,
    which used to return (and the eval to measure) the cut-off answer the stream
    would have finished. ``stream`` is passed through unchanged."""

    def __init__(self, inner: ChatGenerator, limit: int) -> None:
        self._inner = inner
        self._limit = limit

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        text = self._inner.complete(messages, settings)
        for _ in range(self._limit):
            if not text or not stopped_at_token_limit():
                break
            follow = continuation_messages(messages, text, settings)
            prefilled = follow[-1]["role"] == "assistant"
            reset_finish_reason()
            more = self._inner.complete(follow, settings)
            if not prefilled:
                more = drop_restart(text, more)
            if not more.strip():
                # Still cut off where it was: the empty reply's "stop" would have
                # marked it whole (and kept it in the answer cache).
                reset_finish_reason("length")
                break
            text = join_continuation(text, more, prefilled=prefilled)
        return text

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterable[str]:
        return self._inner.stream(messages, settings)
