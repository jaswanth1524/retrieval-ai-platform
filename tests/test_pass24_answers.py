"""Pass 24: the self-check's second look by meaning (SELF_CHECK_MEANING_BAR)."""

from __future__ import annotations

from collections.abc import Sequence

from api.citations import SourceCitation
from api.verify import self_check

_MEALS = SourceCitation(
    source_number=1,
    filename="travel_policy.md",
    page=1,
    section="Meals",
    chunk_id="c1",
    text="Travellers receive a daily meal allowance of sixty euros while abroad.",
)
_PARAPHRASE = "You get about 60 EUR per day for food on trips [1]."


def _scorer(score: float, calls: list[tuple[str, list[str]]]):  # type: ignore[no-untyped-def]
    def meaning(claim: str, passages: Sequence[str]) -> list[float]:
        calls.append((claim, list(passages)))
        return [score for _ in passages]

    return meaning


def test_a_paraphrase_the_words_miss_is_supported_by_meaning() -> None:
    assert [c.status for c in self_check(_PARAPHRASE, [_MEALS])] == ["unsupported"]
    calls: list[tuple[str, list[str]]] = []
    checks = self_check(_PARAPHRASE, [_MEALS], meaning=_scorer(0.9, calls), meaning_bar=0.5)
    assert [c.status for c in checks] == ["supported"]
    assert checks[0].chunk is not None and checks[0].chunk.chunk_id == "c1"
    # Scored without its citation marker, against the passage it cites.
    assert calls == [("You get about 60 EUR per day for food on trips .", [_MEALS.text])]


def test_below_the_bar_or_off_it_stays_unsupported() -> None:
    calls: list[tuple[str, list[str]]] = []
    low = self_check(_PARAPHRASE, [_MEALS], meaning=_scorer(0.2, calls), meaning_bar=0.5)
    assert [c.status for c in low] == ["unsupported"]
    off = self_check(_PARAPHRASE, [_MEALS], meaning=_scorer(0.9, calls), meaning_bar=0.0)
    assert [c.status for c in off] == ["unsupported"]
    assert len(calls) == 1


def test_meaning_never_overrules_different_numbers_or_a_missing_source() -> None:
    calls: list[tuple[str, list[str]]] = []
    meaning = _scorer(0.99, calls)
    wrong = "Travellers receive a daily meal allowance of seventy euros while abroad [1]."
    checks = self_check(wrong, [_MEALS], meaning=meaning, meaning_bar=0.5)
    assert [c.status for c in checks] == ["numbers_differ"]
    missing = self_check("Hotels are booked by the desk [7].", [_MEALS], meaning=meaning)
    assert [c.status for c in missing] == ["unsupported"]
    assert calls == []


def test_the_untuned_floor_warning_names_a_measured_bar(caplog) -> None:  # type: ignore[no-untyped-def]
    import logging

    from api.settings import AppSettings

    with caplog.at_level(logging.WARNING, logger="api.settings"):
        AppSettings(_env_file=None, reranker_model="Xenova/ms-marco-MiniLM-L-6-v2")  # type: ignore[call-arg]
        AppSettings(_env_file=None, reranker_model="BAAI/bge-reranker-base")  # type: ignore[call-arg]
    messages = [record.getMessage() for record in caplog.records]
    assert any("best bar was 0.004" in message for message in messages)
    assert any("calibrate" in message and "bge-reranker-base" in message for message in messages)
