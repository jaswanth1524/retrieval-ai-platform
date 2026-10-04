from __future__ import annotations

from typing import Any

from api.answer_cache import (
    AnswerCache,
    CachedAnswer,
    answer_cache_key,
    bump_corpus_generation,
)
from api.settings import AppSettings
from tests.factories import make_test_settings


def _settings(**overrides: Any) -> AppSettings:
    return make_test_settings(**overrides)


def _answer(text: str) -> CachedAnswer:
    return CachedAnswer(answer=text, sources=(), trace_id=None)


def test_the_key_ignores_case_whitespace_and_scope_order() -> None:
    settings = _settings()

    assert answer_cache_key("What is  DocRAG?", settings, ["b.md", "a.md"]) == answer_cache_key(
        "what is docrag?", settings, ["a.md", "b.md"]
    )


def test_the_key_changes_with_anything_that_changes_the_answer() -> None:
    base = answer_cache_key("q", _settings(), None)

    assert answer_cache_key("q", _settings(), ["a.md"]) != base
    assert answer_cache_key("q", _settings(), None, ["legal"]) != base
    assert answer_cache_key("q", _settings(llm_temperature=0.7), None) != base
    assert answer_cache_key("q", _settings(llm_model="other"), None) != base
    assert answer_cache_key("q", _settings(max_context_chunks=3), None) != base


def test_a_corpus_change_invalidates_every_key() -> None:
    settings = _settings()
    before = answer_cache_key("q", settings, None)

    bump_corpus_generation()

    assert answer_cache_key("q", settings, None) != before


def test_least_recently_used_entries_are_evicted_past_the_cap() -> None:
    cache = AnswerCache(max_entries=2, ttl_seconds=60)
    cache.put(("a",), _answer("A"))
    cache.put(("b",), _answer("B"))
    assert cache.get(("a",)) is not None  # "a" is now the most recently used

    cache.put(("c",), _answer("C"))

    assert cache.get(("b",)) is None
    assert cache.get(("a",)) is not None
    assert len(cache) == 2


def test_entries_expire_after_the_ttl() -> None:
    now = [0.0]
    cache = AnswerCache(max_entries=4, ttl_seconds=10, clock=lambda: now[0])
    cache.put(("a",), _answer("A"))

    now[0] = 9.0
    assert cache.get(("a",)) is not None
    now[0] = 20.0
    assert cache.get(("a",)) is None
