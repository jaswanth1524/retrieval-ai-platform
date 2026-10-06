"""Token counting for chunk sizing.

Chunking must respect the dense embedding model's subword-token limit (bge-small hard-
truncates at 512), but loading the real tokenizer needs a (network) model fetch. This
module exposes a ``TokenCounter`` seam so ingestion counts real subword tokens when the
tokenizer is available and degrades to a word-based heuristic when it isn't — ingest
never hard-fails just because the tokenizer couldn't be fetched (offline build, etc.).
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, replace
from threading import Lock
from typing import Literal, Protocol

from api.parsers.common import DocumentSection

logger = logging.getLogger(__name__)

# English words expand to ~1.3-1.8 subword tokens each; 1.6 is a deliberately slight
# over-estimate so the heuristic errs toward smaller (safe) chunks, never overflow.
_HEURISTIC_TOKENS_PER_WORD = 1.6


class TokenCounter(Protocol):
    """Counts model tokens for a piece of text, excluding special tokens.

    Implementations may expose ``special_tokens_per_sequence`` (tokens the model adds
    around each input); chunking reserves that many once per chunk, 0 if absent.
    """

    def count(self, text: str) -> int: ...


# [CLS] + [SEP]: what bge (and BERT-family models generally) wrap around each input.
_BERT_SPECIAL_TOKENS = 2


TokenCounterMode = Literal["hf", "heuristic", "unknown"]

# How long a failed tokenizer fetch keeps chunking on the heuristic before it is tried
# again. One failure (offline at boot) used to degrade the process for its lifetime.
_TOKENIZER_RETRY_SECONDS = 300.0


class HeuristicTokenCounter:
    """Word-count-based token estimate; no model, always available."""

    special_tokens_per_sequence = _BERT_SPECIAL_TOKENS
    mode: TokenCounterMode = "heuristic"

    def count(self, text: str) -> int:
        words = len(text.split())
        return math.ceil(words * _HEURISTIC_TOKENS_PER_WORD)


class HFTokenCounter:
    """Counts subword tokens with the dense model's own tokenizer, lazy-loaded once.

    The tokenizer is fetched on the first ``count`` call, not at construction, so
    building the counter is free and a fetch failure degrades to the heuristic (logged)
    rather than raising — a document must still ingest without network access. The
    fetch is retried after ``_TOKENIZER_RETRY_SECONDS``, and ``mode`` says which counter
    sized the chunks, so documents chunked by the heuristic can be found and re-indexed.
    """

    special_tokens_per_sequence = _BERT_SPECIAL_TOKENS

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._tokenizer: object | None = None
        self._fallback = HeuristicTokenCounter()
        self._degraded_at: float | None = None
        self._load_lock = Lock()

    @property
    def mode(self) -> TokenCounterMode:
        """Which counter ``count`` uses now; reading it never triggers a load."""

        if self._tokenizer is not None:
            return "hf"
        return "heuristic" if self._degraded_at is not None else "unknown"

    def count(self, text: str) -> int:
        if self._tokenizer is None and not self._load():
            return self._fallback.count(text)
        # Without special tokens: [CLS]/[SEP] wrap the whole embedded string once, and
        # chunking reserves them once in its budget. Counting them per call added 2 to
        # every sentence (and every word, when an oversized sentence is split word by
        # word), leaving windows well short of the budget they were sized for.
        encode = self._tokenizer.encode(  # type: ignore[union-attr]
            text, add_special_tokens=False
        )
        return len(encode.ids)

    def _load(self) -> bool:
        with self._load_lock:
            if self._tokenizer is not None:
                return True
            if (
                self._degraded_at is not None
                and time.monotonic() - self._degraded_at < _TOKENIZER_RETRY_SECONDS
            ):
                return False
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_pretrained(self._model_name)
            except Exception as exc:  # noqa: BLE001 - any failure must degrade, not crash ingest
                self._degraded_at = time.monotonic()
                logger.warning(
                    "Could not load tokenizer for %s (%s); counting tokens for chunk sizing "
                    "with a heuristic, retrying in %.0f s.",
                    self._model_name,
                    exc,
                    _TOKENIZER_RETRY_SECONDS,
                )
                return False
            if self._degraded_at is not None:
                logger.info("Loaded tokenizer for %s after an earlier failure.", self._model_name)
            return True


def make_token_counter(model_name: str) -> TokenCounter:
    """Build the token counter for a dense embedding model name."""

    return HFTokenCounter(model_name)


# Sentence boundaries: whitespace after .!? that precedes a capital/digit (optionally
# behind an opening quote/bracket), OR a blank line. Windows prefer to break here rather
# than mid-sentence. Not linguistically perfect (abbreviations, decimals) — a pragmatic
# heuristic that keeps most chunk edges on real sentence ends.
# Latin sentence ends need a following space and capital; CJK full-width ones (。！？)
# end a sentence on their own — CJK has no spaces, so without them a whole Chinese or
# Japanese paragraph was one "sentence".
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|(?<=[。！？])\s*|\n{2,}")


@dataclass(frozen=True)
class _Unit:
    """One sentence (or forced fragment of an oversized sentence) with its metadata."""

    text: str
    tokens: int
    page: int
    section: str


def _sentence_units(
    group: list[DocumentSection], budget: int, token_counter: TokenCounter
) -> list[_Unit]:
    """Flatten a group's sections into sentence units, splitting oversized sentences."""

    units: list[_Unit] = []
    for section in group:
        for raw_sentence in _SENTENCE_SPLIT_RE.split(section.text):
            # Collapse intra-sentence whitespace (newlines, runs of spaces) to single
            # spaces so chunk text is clean and stable regardless of source formatting.
            sentence = " ".join(raw_sentence.split())
            if not sentence:
                continue
            for piece, tokens in _split_oversized(sentence, budget, token_counter):
                units.append(
                    _Unit(
                        text=piece,
                        tokens=tokens,
                        page=section.page,
                        section=section.section,
                    )
                )
    return units


def _split_oversized(
    sentence: str, budget: int, token_counter: TokenCounter
) -> list[tuple[str, int]]:
    """Greedily pack a single sentence's words into <=budget-token pieces.

    A sentence within budget is returned unchanged. A "word" longer than budget — a
    base64 blob, a long URL, a run of unspaced CJK — is cut by characters (see
    ``_split_long_word``); it used to be emitted whole and silently truncated by the
    embedder at 512 tokens. Each piece comes back with its token count so the caller
    doesn't tokenize every sentence a second time.
    """

    sentence_tokens = token_counter.count(sentence)
    if sentence_tokens <= budget:
        return [(sentence, sentence_tokens)]

    # Count each word once and accumulate, rather than re-tokenizing the whole running
    # string per word — that was O(words^2) tokenizer work on exactly the inputs that
    # reach this path (CSV row blocks, wide tables), which are the longest ones.
    # bge's WordPiece tokenizer pre-splits on whitespace, so (without special tokens)
    # the per-word sum equals the joined piece's real count; for any tokenizer that
    # merges across spaces it can only over-estimate, which errs toward smaller pieces.
    pieces: list[tuple[str, int]] = []
    current: list[str] = []
    current_tokens = 0
    words: list[tuple[str, int]] = []
    for word in sentence.split():
        word_tokens = token_counter.count(word)
        if word_tokens > budget:
            words.extend(_split_long_word(word, budget, token_counter))
        else:
            words.append((word, word_tokens))
    for word, word_tokens in words:
        if current and current_tokens + word_tokens > budget:
            pieces.append((" ".join(current), current_tokens))
            current = [word]
            current_tokens = word_tokens
        else:
            current.append(word)
            current_tokens += word_tokens
    if current:
        pieces.append((" ".join(current), current_tokens))
    return pieces


def _split_long_word(word: str, budget: int, token_counter: TokenCounter) -> list[tuple[str, int]]:
    """Cut one over-budget run of characters into pieces of at most ``budget`` tokens."""

    pieces: list[tuple[str, int]] = []
    rest = word
    while rest:
        total = token_counter.count(rest)
        if total <= budget:
            pieces.append((rest, total))
            break
        # Start from the proportional guess and shrink until it fits; each piece takes
        # at least one character, so this always advances.
        length = max(1, len(rest) * budget // total)
        piece_tokens = token_counter.count(rest[:length])
        while length > 1 and piece_tokens > budget:
            length = max(1, length * budget // max(piece_tokens, 1) - 1)
            piece_tokens = token_counter.count(rest[:length])
        pieces.append((rest[:length], piece_tokens))
        rest = rest[length:]
    return pieces


def _window_units(units: list[_Unit], budget: int, overlap: int) -> list[list[_Unit]]:
    """Slide a token-bounded window over units with token-bounded overlap.

    Each window holds as many whole units as fit in ``budget`` (always at least one,
    guaranteeing forward progress). The next window starts by stepping back over the
    trailing units whose token sum is within ``overlap`` — but never so far that the
    window fails to advance by at least one unit.
    """

    windows: list[list[_Unit]] = []
    start = 0
    total = len(units)
    while start < total:
        end = start
        window_tokens = 0
        while end < total and (end == start or window_tokens + units[end].tokens <= budget):
            window_tokens += units[end].tokens
            end += 1
        windows.append(units[start:end])
        if end >= total:
            break

        overlap_tokens = 0
        next_start = end
        while next_start > start + 1 and overlap_tokens + units[next_start - 1].tokens <= overlap:
            next_start -= 1
            overlap_tokens += units[next_start].tokens
        start = next_start

    return windows


def _group_sections_for_windowing(
    sections: list[DocumentSection],
) -> list[list[DocumentSection]]:
    """Group sections into runs that should be windowed as one continuous stream.

    Consecutive same-filename ``physical`` sections merge into one group (chunk
    overlap crosses the boundary between them); each ``semantic`` section is always
    its own singleton group (windowed alone, exactly as before this change).
    """

    groups: list[list[DocumentSection]] = []
    for section in sections:
        if (
            section.boundary == "physical"
            and groups
            and groups[-1][-1].boundary == "physical"
            and groups[-1][-1].filename == section.filename
        ):
            groups[-1].append(section)
        else:
            groups.append([section])
    return groups


def merge_tiny_semantic_sections(
    sections: list[DocumentSection],
    min_words: int,
) -> list[DocumentSection]:
    """Fold undersized Markdown heading sections into a neighbor before windowing.

    Only operates on ``boundary == "semantic"`` runs (Markdown headings) — physical
    sections don't need this, cross-boundary windowing already absorbs short
    paragraphs/pages into their neighbors.
    """

    result: list[DocumentSection] = []
    index = 0
    while index < len(sections):
        if sections[index].boundary != "semantic":
            result.append(sections[index])
            index += 1
            continue
        run_start = index
        while index < len(sections) and sections[index].boundary == "semantic":
            index += 1
        result.extend(_merge_semantic_run(sections[run_start:index], min_words))
    return result


def _merge_semantic_run(
    run: list[DocumentSection],
    min_words: int,
) -> list[DocumentSection]:
    """Fold sections under ``min_words`` forward into the next section in the run.

    A trailing run of tiny sections with no bigger section after them folds
    backward into the last merged section instead (or, if the whole run is tiny,
    collapses into one section under the last section's label).
    """

    merged: list[DocumentSection] = []
    pending: list[str] = []
    for section in run:
        if len(section.text.split()) < min_words:
            pending.append(section.text)
            continue
        text = "\n\n".join([*pending, section.text]) if pending else section.text
        pending = []
        merged.append(replace(section, text=text))
    if pending:
        if merged:
            merged[-1] = replace(merged[-1], text=merged[-1].text + "\n\n" + "\n\n".join(pending))
        else:
            merged.append(replace(run[-1], text="\n\n".join(pending)))
    return merged
