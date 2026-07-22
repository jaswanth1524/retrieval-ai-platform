from __future__ import annotations

import pytest

from api.chunking import HeuristicTokenCounter, HFTokenCounter, make_token_counter


def test_heuristic_counter_scales_with_word_count() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count("") == 0
    assert counter.count("one") == 2  # ceil(1 * 1.6)
    assert counter.count("one two three") == 5  # ceil(3 * 1.6)


def test_make_token_counter_returns_hf_counter() -> None:
    assert isinstance(make_token_counter("BAAI/bge-small-en-v1.5"), HFTokenCounter)


def test_hf_counter_degrades_to_heuristic_when_tokenizer_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = HFTokenCounter("does-not-exist/model")

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("no network")

    # Force the lazy tokenizer load to fail; the counter must fall back, not raise.
    monkeypatch.setattr("tokenizers.Tokenizer.from_pretrained", _boom)

    result = counter.count("one two three")
    assert result == HeuristicTokenCounter().count("one two three")
    # Subsequent calls stay on the heuristic without retrying the failed load.
    assert counter.count("four five") == HeuristicTokenCounter().count("four five")
