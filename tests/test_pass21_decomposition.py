"""Pass 21: compound-question decomposition (QUERY_DECOMPOSITION_ENABLED)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches, get_generator, get_reranker
from api.prompting import DECOMPOSITION_SYSTEM_PROMPT, decompose_question, might_be_compound
from api.settings import AppSettings
from tests.factories import make_test_settings
from tests.routes_http.support import hermetic_app, upload_and_wait
from tests.routes_http.support import make_settings as route_settings

PARTS = (
    "What does the leave policy say?\n"
    "What does the expense policy say?\n"
    "What does the equipment policy say?"
)
_WORD = re.compile(r"[a-z]+")
_STOP = {"what", "does", "do", "the", "say", "and", "a", "of"}


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class _Overlap:
    """A cross-encoder stand-in that, like the real one, scores a passage lower the
    more topics the question spreads over: the share of the question's words in it."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        asked = _words(query)
        # Logits, as a cross-encoder gives (normalized to [0, 1] afterwards).
        return [10 * len(asked & _words(d)) / max(len(asked), 1) - 7 for d in documents]


class _Model:
    def __init__(self) -> None:
        self.decompositions = 0

    def complete(self, messages: Sequence[Any], settings: AppSettings) -> str:
        if messages and messages[0]["content"] == DECOMPOSITION_SYSTEM_PROMPT:
            self.decompositions += 1
            return PARTS
        return "From the documents [1]."

    def stream(self, messages: Sequence[Any], settings: AppSettings) -> Any:
        yield self.complete(messages, settings)


def test_only_questions_that_look_compound_are_sent_to_be_split() -> None:
    assert might_be_compound("What do the leave and expense policies say?")
    assert might_be_compound("Compare the 2019 travel policy with the current one.")
    assert not might_be_compound("How many vacation days do new staff get?")


@pytest.mark.parametrize(
    ("reply", "parts"),
    [
        (PARTS, PARTS.splitlines()),
        ("SINGLE", []),
        ("Only one line?", []),
        (
            "1. What about leave?\n2. What about expenses?",
            ["What about leave?", "What about expenses?"],
        ),
    ],
)
def test_the_split_is_read_from_the_models_reply(reply: str, parts: list[str]) -> None:
    class _Says:
        def complete(self, messages: Sequence[Any], settings: AppSettings) -> str:
            return reply

        def stream(self, messages: Sequence[Any], settings: AppSettings) -> Any:
            yield reply

    got = decompose_question(
        "What do the leave and expense policies say?", 3, _Says(), make_test_settings()
    )
    assert got == parts


def _ask(enabled: bool, stream: bool = False) -> tuple[list[str], _Model, Any]:
    clear_dependency_caches()
    settings = route_settings(
        query_decomposition_enabled=enabled,
        rerank_min_score=0.5,
        rerank_fallback_count=0,
        citation_retry_enabled=False,
        followups_enabled=False,
        chunk_size_tokens=200,
        max_context_chunks=3,
    )
    model = _Model()
    app = hermetic_app(settings, QdrantClient(":memory:"))
    app.dependency_overrides[get_generator] = lambda: model
    app.dependency_overrides[get_reranker] = lambda: _Overlap()
    with TestClient(app) as client:
        for name, text in (
            ("leave.txt", b"Leave\nLeave policy: staff get 28 days."),
            ("expense.txt", b"Expenses\nExpense policy: submit receipts within 30 days."),
            ("equipment.txt", b"Equipment\nEquipment policy: laptops every 4 years."),
        ):
            assert upload_and_wait(client, name, text)["state"] == "done"
        question = "What do the leave policy, the expense policy and the equipment policy say?"
        if stream:
            from tests.test_pass17_answers import _sse_events

            events = _sse_events(client.post("/questions/stream", json={"question": question}).text)
            done = next(e for e in events if e["type"] == "done")
            trace = client.get(f"/traces/{done['trace_id']}").json()
            sources = [
                s["filename"] for s in next(e for e in events if e["type"] == "sources")["sources"]
            ]
        else:
            answer = client.post("/questions", json={"question": question}).json()
            trace = client.get(f"/traces/{answer['trace_id']}").json()
            sources = [s["filename"] for s in trace["candidates"] if s.get("source_number")]
    clear_dependency_caches()
    return sources, model, trace


def test_without_decomposition_a_three_topic_question_misses_a_topic() -> None:
    sources, model, _ = _ask(enabled=False)
    assert model.decompositions == 0
    assert sorted(sources) != ["equipment.txt", "expense.txt", "leave.txt"]


@pytest.mark.parametrize("stream", [False, True])
def test_with_decomposition_each_part_brings_its_own_passages(stream: bool) -> None:
    sources, model, trace = _ask(enabled=True, stream=stream)
    assert model.decompositions == 1
    assert sorted(set(sources)) == ["equipment.txt", "expense.txt", "leave.txt"]
    assert trace["query_variants"] == PARTS.splitlines()


def test_a_reply_that_copies_the_examples_keeps_only_its_own_parts() -> None:
    class _Copies:
        def complete(self, messages: Sequence[Any], settings: AppSettings) -> str:
            # The examples are turns of their own, not part of the question.
            assert "laptops" not in messages[-1]["content"]
            return (
                "What does the travel policy say about laptops?\n"
                "What does the security policy say about laptops?\n"
                "How many vacation days do staff get?\n"
                "How soon must receipts be submitted?"
            )

        def stream(self, messages: Sequence[Any], settings: AppSettings) -> Any:
            yield ""

    got = decompose_question(
        "How many vacation days do staff get, and how soon must receipts be submitted?",
        3,
        _Copies(),
        make_test_settings(),
    )
    assert got == ["How many vacation days do staff get?", "How soon must receipts be submitted?"]
