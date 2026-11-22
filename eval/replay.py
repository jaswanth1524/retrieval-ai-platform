"""Recorded model replies, so answer-level evals run in CI with no LLM.

``harness answer --record FILE`` wraps the real generator and writes every call's
reply (condense, answer, citation retry, summary parts) keyed by the exact messages
sent. ``--replay FILE`` answers each call from that file instead: everything around the
model — retrieval, prompt fitting, citation retry and backfill, status, spans — runs for
real, so a change to any of it shows in the answer metrics.

A call the file doesn't hold (the prompt changed: a retrieval or prompt change, or a
new question) is a miss: the generator raises, the question becomes an error row, and
``misses`` names it. Record again after an intended change.

The int8 reranker's scores move slightly with the CPU (and the batch they are scored
in), enough to reorder two passages or swap the last one in a prompt — measured: a
different batch size alone changed the prompt of 1 to 10 questions per set. So each
reply is also kept under a looser key, the messages with the passages (the ``Context:``
block) taken out; a call whose exact prompt wasn't recorded gets the reply recorded for
the same question, history and instructions (``loose_hits`` counts them). A different
question, rewrite or instruction is still a miss.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from api.llm import GenerationError, reset_finish_reason, stopped_at_token_limit
from api.settings import AppSettings

if TYPE_CHECKING:
    from api.generation import ChatGenerator, ChatMessage


class ReplayMissError(GenerationError):
    """The recording holds no reply for these messages."""


def reply_key(messages: Sequence[ChatMessage]) -> str:
    """The messages' identity: roles and contents, nothing about the model."""

    plain = [[str(m.get("role", "")), str(m.get("content", ""))] for m in messages]
    return hashlib.sha256(json.dumps(plain, ensure_ascii=False).encode("utf-8")).hexdigest()


_CONTEXT_BLOCK_RE = re.compile(r"\nContext:\n.*?\n\nInstructions:\n", re.DOTALL)


def loose_key(messages: Sequence[ChatMessage]) -> str:
    """``reply_key`` without the passages each user message quotes."""

    plain = [
        [
            str(m.get("role", "")),
            _CONTEXT_BLOCK_RE.sub("\nInstructions:\n", str(m.get("content", ""))),
        ]
        for m in messages
    ]
    return hashlib.sha256(json.dumps(plain, ensure_ascii=False).encode("utf-8")).hexdigest()


def _last_user(messages: Sequence[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return " ".join(str(message.get("content", "")).split())[-160:]
    return ""


class RecordingGenerator:
    """``inner``'s replies, each also appended to ``path`` (one JSON object a line)."""

    def __init__(self, inner: ChatGenerator, path: Path) -> None:
        self.inner = inner
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def _record(self, messages: Sequence[ChatMessage], reply: str) -> None:
        entry = {
            "key": reply_key(messages),
            "loose": loose_key(messages),
            "finish": "length" if stopped_at_token_limit() else "stop",
            "reply": reply,
            # For a person reading the file: which call this was.
            "asked": _last_user(messages),
        }
        with self.path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        reply = self.inner.complete(messages, settings)
        self._record(messages, reply)
        return reply

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterator[str]:
        parts: list[str] = []
        for delta in self.inner.stream(messages, settings):
            parts.append(delta)
            yield delta
        self._record(messages, "".join(parts))


class ReplayGenerator:
    """Replies from a recording; a call it doesn't hold raises ``ReplayMissError``."""

    def __init__(self, path: Path) -> None:
        self._replies: dict[str, tuple[str, str]] = {}
        self._loose: dict[str, tuple[str, str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            reply = (str(entry["reply"]), str(entry.get("finish")))
            # The last recording of a call wins (a re-recorded question).
            self._replies[str(entry["key"])] = reply
            if entry.get("loose"):
                self._loose[str(entry["loose"])] = reply
        self.misses: list[str] = []
        self.loose_hits = 0

    def __len__(self) -> int:
        return len(self._replies)

    def _reply(self, messages: Sequence[ChatMessage]) -> str:
        found = self._replies.get(reply_key(messages))
        if found is None:
            found = self._loose.get(loose_key(messages))
            if found is not None:
                self.loose_hits += 1
        if found is None:
            asked = _last_user(messages)
            self.misses.append(asked)
            raise ReplayMissError(f"No recorded reply for this call ({asked[-80:]!r}).")
        reply, finish = found
        reset_finish_reason(finish)
        return reply

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        return self._reply(messages)

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterator[str]:
        yield self._reply(messages)


# Answer metrics a replayed run is held to: higher is better for these...
HIGHER_IS_BETTER = (
    "fact_hit_rate",
    "answered_rate",
    "cited_relevant_rate",
    "refused_unanswerable_rate",
    "cited_chunk_precision",
)
# ...lower for these.
LOWER_IS_BETTER = (
    "uncited_rate",
    "error_rate",
    "excluded_rate",
    "unsupported_numbers_mean",
    "unsupported_claims_mean",
    "truncated_rate",
    "prompt_cut_rate",
)

# Recorded to explain a drift, never a reason to refuse the comparison (CI is amd64,
# a laptop arm64; a --server run's address and release).
_NOTED_ONLY = frozenset({"platform_machine", "server", "server_version"})


def answer_shape_problems(current: dict[str, object], baseline: dict[str, object]) -> list[str]:
    """Reasons an answer baseline can't judge this run: a setting both recorded with
    different values (another model, ``--limit``, a changed ``max_context_chunks``), as
    the retrieval baselines' ``baseline_shape_problems``. A key only one side has (a
    baseline from before it was recorded) is no reason."""

    return [
        f"{key} differs: {current[key]!r} vs baseline {baseline[key]!r} "
        "(re-record the baseline if the change is intended)"
        for key in sorted(set(current) & set(baseline) - _NOTED_ONLY)
        if current[key] != baseline[key]
    ]


def answer_regressions(
    metrics: dict[str, float], baseline: dict[str, float], tolerance: float
) -> list[str]:
    """Each metric that moved the wrong way by more than ``tolerance`` from the
    baseline's, or that the baseline has and this run lacks."""

    problems = []
    for name in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER):
        if name not in baseline:
            continue
        if name not in metrics:
            problems.append(f"{name}: in the baseline, missing from this run")
            continue
        drop = baseline[name] - metrics[name]
        if name in LOWER_IS_BETTER:
            drop = -drop
        if drop > tolerance + 1e-9:
            problems.append(f"{name}: {metrics[name]:.3f} vs baseline {baseline[name]:.3f}")
    return problems
