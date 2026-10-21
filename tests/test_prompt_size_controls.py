"""The opt-in prompt-size controls: PROMPT_TOKEN_BUDGET, RERANK_RELATIVE_FLOOR and
GENERATION_HISTORY_ASSISTANT_CHARS. All off by default."""

from __future__ import annotations

from typing import Any

from api.llm import ChatMessage
from api.pipeline import _clip_earlier_answers
from api.prompting import (
    build_grounded_messages,
    estimate_prompt_tokens,
    fit_prompt_to_context_window,
)
from api.reranking import RerankedChunk, relative_cut
from api.retrieval import RetrievedChunk
from api.tracing import build_trace_candidates
from tests.factories import make_test_settings


def chunk(number: int, score: float, expanded: str | None = None) -> RerankedChunk:
    return RerankedChunk(
        point_id=f"p{number}",
        filename="guide.md",
        page=1,
        section="Body",
        chunk_id=f"c{number}",
        text=f"passage {number} " + "word " * 40,
        retrieval_score=0.5,
        rerank_score=score,
        expanded_text=expanded,
    )


def ollama(**overrides: Any) -> Any:
    return make_test_settings(llm_provider="ollama", **overrides)


NEIGHBOURS = "neighbour " * 300


def test_no_budget_changes_nothing() -> None:
    chunks = [chunk(1, 0.9, NEIGHBOURS), chunk(2, 0.8, NEIGHBOURS)]
    fitted = fit_prompt_to_context_window("q", chunks, None, ollama())
    assert fitted.context_chunks == chunks


def test_a_budget_drops_neighbour_text_from_the_bottom_before_any_passage() -> None:
    chunks = [chunk(1, 0.9, NEIGHBOURS), chunk(2, 0.8, NEIGHBOURS), chunk(3, 0.7, NEIGHBOURS)]
    without_last = [*chunks[:2], chunk(3, 0.7)]
    budget = estimate_prompt_tokens(build_grounded_messages("q", without_last))
    fitted = fit_prompt_to_context_window("q", chunks, None, ollama(prompt_token_budget=budget))
    assert [c.chunk_id for c in fitted.context_chunks] == ["c1", "c2", "c3"]
    assert [c.expanded_text is not None for c in fitted.context_chunks] == [True, True, False]
    assert not fitted.dropped_point_ids


def test_a_tight_budget_then_drops_passages_then_history() -> None:
    chunks = [chunk(1, 0.9, NEIGHBOURS), chunk(2, 0.8, NEIGHBOURS)]
    history: list[ChatMessage] = [
        {"role": "user", "content": "earlier " * 50},
        {"role": "assistant", "content": "answer " * 50},
    ]
    one_bare = estimate_prompt_tokens(build_grounded_messages("q", [chunk(1, 0.9)], history))
    fitted = fit_prompt_to_context_window(
        "q", chunks, history, ollama(prompt_token_budget=one_bare)
    )
    assert [c.chunk_id for c in fitted.context_chunks] == ["c1"]
    assert fitted.context_chunks[0].expanded_text is None
    assert fitted.dropped_point_ids == {"p2"}
    assert fitted.history == history
    tighter = fit_prompt_to_context_window(
        "q", chunks, history, ollama(prompt_token_budget=one_bare - 10)
    )
    assert tighter.dropped_history_messages >= 1


def test_the_relative_cut_keeps_passages_near_the_best_and_always_two() -> None:
    scored = [chunk(n, s) for n, s in enumerate([0.95, 0.9, 0.5, 0.3, 0.1, 0.1, 0.1], 1)]
    kept = scored[:4]
    # median 0.3, top 0.95: 0.5 is (0.5-0.3)/0.65 = 0.31 of the way, 0.3 is 0.
    result, cut = relative_cut(kept, scored, 0.25)
    assert [c.chunk_id for c in result] == ["c1", "c2", "c3"]
    assert cut == {"p4"}
    result, cut = relative_cut(kept, scored, 0.99)
    assert [c.chunk_id for c in result] == ["c1", "c2"]  # never below two
    assert relative_cut(kept, scored, 0.0) == (kept, set())
    # Every score equal: nothing to tell apart.
    flat = [chunk(n, 0.5) for n in range(1, 5)]
    assert relative_cut(flat, flat, 0.5) == (flat, set())


def test_a_relative_cut_is_traced_as_such() -> None:
    scored = [chunk(1, 0.9), chunk(2, 0.2)]
    fused = [
        RetrievedChunk(c.point_id, c.filename, c.page, c.section, c.chunk_id, c.text, 0.5)
        for c in scored
    ]
    candidates = build_trace_candidates(
        fused,
        scored,
        [scored[0]],
        [scored[0]],
        min_score=0.15,
        relative_cut_ids={"p2"},
    )
    assert [c.drop_reason for c in candidates] == [None, "relative_cut"]


def test_earlier_answers_are_clipped_only_when_asked() -> None:
    history: list[ChatMessage] = [
        {"role": "user", "content": "q " * 300},
        {"role": "assistant", "content": "answer " * 300},
    ]
    assert _clip_earlier_answers(history, ollama()) == history
    clipped = _clip_earlier_answers(history, ollama(generation_history_assistant_chars=100))
    assert clipped[0] == history[0]  # questions are kept whole
    assert len(clipped[1]["content"]) <= 101
    assert clipped[1]["content"].endswith("…")
