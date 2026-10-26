"""Uncited answers: how they are labeled, when citations are added, and how
(backfilled from the sources, or by the LLM retry)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from api.citations import (
    answer_status,
    backfill_citations,
    declines_to_answer,
    finalize_citations,
    only_declines,
    source_citations,
    will_call_llm_for_citations,
    will_retry_citations,
)
from api.generation import generate_grounded_answer
from api.llm import ChatMessage
from api.reranking import RerankedChunk
from api.settings import AppSettings
from tests.factories import make_test_settings

FLIGHTS = "Flights must be booked at least fourteen days ahead. Economy class is the default."
MEALS = "The daily meal allowance abroad is sixty euros. Receipts are not required for meals."


def chunk(number: int, text: str) -> RerankedChunk:
    return RerankedChunk(
        point_id=f"p{number}",
        filename=f"doc{number}.md",
        page=1,
        section="Policy",
        chunk_id=f"c{number}",
        text=text,
        retrieval_score=0.2,
        rerank_score=0.9,
    )


CHUNKS = [chunk(1, FLIGHTS), chunk(2, MEALS)]


class CountingGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.calls.append(list(messages))
        return self.answer

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Any:
        yield self.complete(messages, settings)


def settings_for(provider: str, **overrides: Any) -> AppSettings:
    extra: dict[str, Any] = {"openai_api_key": "k"} if provider == "openai" else {}
    return make_test_settings(llm_provider=provider, **extra, **overrides)


def test_an_uncited_refusal_is_insufficient_context_and_a_partial_answer_is_not() -> None:
    assert answer_status("The documents do not mention fuel cards.", 0, 3) == (
        "insufficient_context"
    )
    assert only_declines("The provided policy does not specify a fuel card limit.")
    partial = "The documents do not mention fuel cards. Mileage is 30 cents per kilometre."
    assert not only_declines(partial)
    assert answer_status(partial, 0, 3) == "uncited"
    assert answer_status("Mileage is 30 cents per kilometre.", 0, 3) == "uncited"


def test_an_answer_with_a_caveat_after_a_dash_is_an_answer() -> None:
    answer = (
        "The document states parents receive sixteen weeks of paid parental leave (which "
        "can be split into two blocks) — it does not specify the number of days [1]."
    )
    assert not declines_to_answer(answer)
    assert answer_status(answer, 1, 3) == "answered"
    caveat = (
        "The policy states mileage for a private car is paid at thirty cents per "
        "kilometre, but the document does not specify the currency [1]."
    )
    assert answer_status(caveat, 1, 3) == "answered"
    # The refusal itself still reads as one, dash or not.
    refusal = "The documents do not contain enough information — they cover travel only [1]."
    assert declines_to_answer(refusal)
    assert declines_to_answer("The context covers travel, but does not mention dividends [1].")
    assert declines_to_answer("However, the documents do not mention dividends [1].")


def test_does_not_include_is_a_refusal() -> None:
    answer = (
        "The provided document does not include any details about how far ahead flights "
        "must be booked [1]."
    )
    assert declines_to_answer(answer)


def test_backfill_cites_the_sentences_a_source_states() -> None:
    answer = (
        "Flights must be booked at least 14 days ahead. The daily meal allowance abroad "
        "is sixty euros."
    )
    assert backfill_citations(answer, CHUNKS) == (
        "Flights must be booked at least 14 days ahead [1]. The daily meal allowance "
        "abroad is sixty euros [2]."
    )


def test_backfill_leaves_unmatched_and_declining_sentences_alone() -> None:
    answer = "Flights must be booked at least fourteen days ahead. Hotels are not covered."
    assert backfill_citations(answer, CHUNKS) == (
        "Flights must be booked at least fourteen days ahead [1]. Hotels are not covered."
    )
    assert backfill_citations("Contractors get sick leave from day one.", CHUNKS) is None
    assert (
        backfill_citations("The documents do not mention the daily meal allowance.", CHUNKS) is None
    )
    # An answer that already cites is never touched.
    assert backfill_citations("Flights are booked fourteen days ahead [2].", CHUNKS) is None


UNCITED = "Flights must be booked at least fourteen days ahead."


def test_auto_backfills_on_a_local_model_without_a_second_call() -> None:
    settings = settings_for("ollama")
    generator = CountingGenerator(UNCITED)
    answer = generate_grounded_answer("How far ahead?", CHUNKS, generator, settings)
    assert len(generator.calls) == 1
    assert answer.citation_backfilled
    assert not answer.citation_retry_used
    assert answer.answer == "Flights must be booked at least fourteen days ahead [1]."
    assert [source.chunk_id for source in answer.sources] == ["c1"]
    # The spans are attached as for a cited answer.
    assert answer.sources[0].spans


def test_auto_falls_back_to_the_llm_retry_when_nothing_matches() -> None:
    settings = settings_for("ollama")
    generator = CountingGenerator("Contractors get sick leave from day one.")
    sources = source_citations(CHUNKS)
    assert will_call_llm_for_citations(generator.answer, sources, settings, context_chunks=CHUNKS)
    answer = generate_grounded_answer("Sick leave?", CHUNKS, generator, settings)
    assert len(generator.calls) == 2
    assert not answer.citation_backfilled


@pytest.mark.parametrize(
    ("provider", "mode", "calls", "backfilled"),
    [
        ("openai", "auto", 2, False),  # hosted: the LLM retry, as before
        ("ollama", "llm", 2, False),
        ("openai", "backfill", 1, True),
        ("ollama", "off", 1, False),
    ],
)
def test_the_citation_mode_picks_how_citations_are_added(
    provider: str, mode: str, calls: int, backfilled: bool
) -> None:
    settings = settings_for(provider, citation_retry_mode=mode)
    generator = CountingGenerator(UNCITED)
    answer = generate_grounded_answer("How far ahead?", CHUNKS, generator, settings)
    assert len(generator.calls) == calls
    assert answer.citation_backfilled is backfilled


def test_citation_retry_disabled_turns_off_backfill_too() -> None:
    settings = settings_for("ollama", citation_retry_enabled=False)
    generator = CountingGenerator(UNCITED)
    answer = generate_grounded_answer("How far ahead?", CHUNKS, generator, settings)
    assert len(generator.calls) == 1
    assert not answer.citation_backfilled
    assert answer.sources == []


def test_a_weak_match_refusal_gets_no_citation_retry() -> None:
    refusal = "The provided documents do not mention dental cover in any way."
    sources = source_citations(CHUNKS)
    settings = settings_for("openai")
    # A "doesn't mention" refusal isn't in the narrow retry hints, so normally it retries.
    assert will_retry_citations(refusal, sources, settings)
    assert not will_retry_citations(refusal, sources, settings, weak_matches=True)
    generator = CountingGenerator(refusal)
    outcome = finalize_citations(
        [{"role": "user", "content": "q"}],
        refusal,
        sources,
        generator,
        settings,
        weak_matches=True,
    )
    assert generator.calls == []
    assert outcome.answer == refusal
    assert answer_status(outcome.answer, len(outcome.cited), len(sources)) == (
        "insufficient_context"
    )


def test_a_refusal_that_describes_the_sections_it_had_is_still_a_refusal() -> None:
    answer = (
        "The provided documents do not state how many sick days contractors get. The "
        "available sections cover invoicing, remote work location, and equipment use only [1]."
    )
    assert declines_to_answer(answer)


def test_a_marker_only_in_an_echoed_source_header_announces_the_retry() -> None:
    # qwen2.5 appends "[1] filename=…" lines; they are stripped before the citation
    # check, so the announcement must not count their markers either.
    settings = settings_for("ollama", citation_retry_mode="llm")
    answer = "Contractors get sick leave from day one.\n[1] filename=doc1.md; page=1"
    sources = source_citations(CHUNKS)
    assert will_call_llm_for_citations(answer, sources, settings, context_chunks=CHUNKS)
    generator = CountingGenerator("Contractors get sick leave from day one [1].")
    outcome = finalize_citations([], answer, sources, generator, settings, context_chunks=CHUNKS)
    assert outcome.retry_used and len(generator.calls) == 1


def test_a_bulleted_answer_is_backfilled_one_bullet_at_a_time() -> None:
    """Each line of a bulleted answer (ANSWER_FORMAT=bullets) is its own sentence: a
    bullet ending with no period isn't glued to the next one and cited as one."""

    answer = (
        "- Flights are booked at least fourteen days ahead\n"
        "- The daily meal allowance abroad is sixty euros"
    )
    assert backfill_citations(answer, CHUNKS) == (
        "- Flights are booked at least fourteen days ahead [1]\n"
        "- The daily meal allowance abroad is sixty euros [2]"
    )
