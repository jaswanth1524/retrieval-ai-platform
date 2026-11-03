from __future__ import annotations

import pytest

from api.chunking import HeuristicTokenCounter, HFTokenCounter, make_token_counter
from tests.factories import make_test_settings


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

    assert counter.mode == "unknown"
    result = counter.count("one two three")
    assert result == HeuristicTokenCounter().count("one two three")
    assert counter.mode == "heuristic"
    # Subsequent calls stay on the heuristic without retrying the failed load.
    assert counter.count("four five") == HeuristicTokenCounter().count("four five")


def test_hf_counter_retries_the_tokenizer_after_a_while(monkeypatch: pytest.MonkeyPatch) -> None:
    """One failed fetch (offline at boot) degraded chunk sizing for the process's life."""

    import api.chunking as chunking

    class FakeEncoding:
        ids = [1, 2]

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> FakeEncoding:
            return FakeEncoding()

    attempts: list[int] = []

    def flaky(*args: object, **kwargs: object) -> object:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("no network")
        return FakeTokenizer()

    clock = [100.0]
    monkeypatch.setattr("tokenizers.Tokenizer.from_pretrained", flaky)
    monkeypatch.setattr(chunking.time, "monotonic", lambda: clock[0])
    counter = HFTokenCounter("model")

    counter.count("one two three")
    clock[0] += 299
    counter.count("one two three")
    assert (len(attempts), counter.mode) == (1, "heuristic")

    clock[0] += 2
    assert counter.count("one two three") == 2
    assert (len(attempts), counter.mode) == (2, "hf")


class CharTokenCounter:
    """One token per character: CJK and unspaced runs are where this matters."""

    special_tokens_per_sequence = 0

    def count(self, text: str) -> int:
        return len(text)


def test_cjk_sentences_end_at_full_width_punctuation() -> None:
    from api.chunking import _SENTENCE_SPLIT_RE

    parts = [
        part for part in _SENTENCE_SPLIT_RE.split("第一句。第二句！第三句？ Next one. Done") if part
    ]

    assert parts == ["第一句。", "第二句！", "第三句？", "Next one.", "Done"]


def test_an_over_budget_word_is_cut_into_budget_sized_pieces() -> None:
    from api.chunking import _split_oversized

    blob = "QUJD" * 100  # 400 characters, no whitespace: a base64 blob

    pieces = _split_oversized(f"data {blob} end", 50, CharTokenCounter())

    assert all(tokens <= 50 for _, tokens in pieces)
    assert "".join(text.replace(" ", "") for text, _ in pieces) == f"data{blob}end"


def test_every_chunk_of_unspaced_cjk_fits_the_budget() -> None:
    from api.documents import DocumentSection, chunk_sections

    settings = make_test_settings(chunk_size_tokens=60, chunk_overlap_tokens=0)
    text = "这是一个没有空格的很长的中文段落" * 30  # 480 characters, no sentence end
    section = DocumentSection(filename="zh.txt", page=1, section="正文", text=text)

    chunks = chunk_sections([section], settings, CharTokenCounter())

    assert len(chunks) > 1
    prefix = len("zh.txt › 正文\n")
    assert all(len(chunk.text) + prefix <= 60 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def _split_long_word_before(word: str, budget: int, token_counter) -> list[tuple[str, int]]:  # type: ignore[no-untyped-def]
    """The pass-13 implementation, kept here as the reference for ordinary words."""

    pieces: list[tuple[str, int]] = []
    rest = word
    while rest:
        total = token_counter.count(rest)
        if total <= budget:
            pieces.append((rest, total))
            break
        length = max(1, len(rest) * budget // total)
        piece_tokens = token_counter.count(rest[:length])
        while length > 1 and piece_tokens > budget:
            length = max(1, length * budget // max(piece_tokens, 1) - 1)
            piece_tokens = token_counter.count(rest[:length])
        pieces.append((rest[:length], piece_tokens))
        rest = rest[length:]
    return pieces


class _ThirdsCounter:
    """A subword-like counter: a token per three characters, and counts what it read."""

    def __init__(self) -> None:
        self.read = 0

    def count(self, text: str) -> int:
        self.read += len(text)
        return -(-len(text) // 3)


def test_long_words_within_the_window_split_exactly_as_before() -> None:
    import base64
    import random

    from api.chunking import _SPLIT_WINDOW_CHARS_PER_TOKEN, _split_long_word

    generator = random.Random(14)
    for budget in (5, 32, 250):
        for _ in range(30):
            size = generator.randint(budget, budget * _SPLIT_WINDOW_CHARS_PER_TOKEN)
            word = base64.b64encode(generator.randbytes(size))[:size].decode()
            for counter in (HeuristicTokenCounter(), _ThirdsCounter()):
                assert _split_long_word(word, budget, counter) == _split_long_word_before(
                    word, budget, counter
                )


def test_a_multi_megabyte_run_is_split_in_linear_time() -> None:
    import base64
    import random

    from api.chunking import _split_long_word

    class CountingHeuristic(HeuristicTokenCounter):
        def __init__(self) -> None:
            self.read = 0

        def count(self, text: str) -> int:
            self.read += len(text)
            return super().count(text)

    word = base64.b64encode(random.Random(1).randbytes(4_000_000)).decode()  # ~5.3 MB
    # The work is bounded, not timed (a timed test fails on a loaded machine): the
    # whole remainder used to be re-read for every piece — thousands of times the
    # input, where now each piece reads at most a few windows.
    for counter in (_ThirdsCounter(), CountingHeuristic()):
        pieces = _split_long_word(word, 250, counter)
        assert "".join(piece for piece, _ in pieces) == word
        assert all(tokens <= 250 for _, tokens in pieces)
        assert counter.read < 100 * len(word)
