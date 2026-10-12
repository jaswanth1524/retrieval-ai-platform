"""Exact citation spans: which passage of a cited chunk an answer's claim rests on."""

from __future__ import annotations

from dataclasses import asdict

from api.chunking import sentence_spans
from api.citations import (
    MAX_SPANS_PER_SOURCE,
    CitationSpan,
    attach_spans,
    citation_from_dict,
    source_citations,
)
from api.reranking import RerankedChunk


def chunk(text: str, chunk_id: str = "c") -> RerankedChunk:
    return RerankedChunk(
        point_id=f"p-{chunk_id}",
        filename="policy.md",
        page=1,
        section="Leave",
        chunk_id=chunk_id,
        text=text,
        retrieval_score=0.1,
        rerank_score=0.9,
    )


def spans_for(answer: str, *texts: str) -> list[tuple[CitationSpan, ...]]:
    chunks = [chunk(text, f"c{index}") for index, text in enumerate(texts)]
    cited = attach_spans(answer, source_citations(chunks), chunks)
    return [source.spans for source in cited]


POLICY = (
    "The policy applies to all staff. Full-time employees accrue twenty vacation days "
    "per year. Unused days expire in March. Requests go through the HR portal."
)


def test_the_sentence_a_claim_rests_on_is_quoted_with_its_offsets() -> None:
    [spans] = spans_for("Full-time staff get twenty vacation days a year [1].", POLICY)

    assert [span.quote for span in spans] == [
        "Full-time employees accrue twenty vacation days per year."
    ]
    assert POLICY[spans[0].start : spans[0].end] == spans[0].quote


def test_a_marker_after_the_period_belongs_to_the_sentence_before_it() -> None:
    [spans] = spans_for("Unused days expire in March. [1] That is all.", POLICY)

    assert [span.quote for span in spans] == ["Unused days expire in March."]


def test_a_claim_sharing_too_little_points_at_nothing() -> None:
    assert spans_for("The weather is nice today [1].", POLICY) == [()]
    # One shared word is a coincidence, not a source.
    assert spans_for("Portal maintenance happens weekly [1].", POLICY) == [()]


def test_a_claim_spanning_two_sentences_quotes_both() -> None:
    [spans] = spans_for(
        "Vacation days accrue each year and unused ones expire in March [1].", POLICY
    )

    assert [span.quote for span in spans] == [
        "Full-time employees accrue twenty vacation days per year. Unused days expire in March."
    ]


def test_list_and_range_markers_reach_every_source_they_name() -> None:
    other = "Contractors accrue no vacation days. They are paid hourly."
    first, second = spans_for("Vacation days accrue differently by contract [1-2].", POLICY, other)

    assert first and second
    assert "accrue" in first[0].quote and "accrue" in second[0].quote


def test_offsets_count_code_points_past_astral_characters() -> None:
    text = "🌴 Holidays: 😀 employees accrue twenty vacation days per year. Other things."
    [spans] = spans_for("Employees accrue twenty vacation days [1].", text)

    assert text[spans[0].start : spans[0].end] == spans[0].quote
    assert spans[0].quote.endswith("twenty vacation days per year.")


def test_spans_never_overlap_and_are_capped_per_source() -> None:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    text = " ".join(f"Rule {word} covers room {word}." for word in words)
    answer = " ".join(f"Rule {word} covers room {word} [1]." for word in words)
    [spans] = spans_for(answer, text)

    assert len(spans) == MAX_SPANS_PER_SOURCE
    ordered = sorted((span.start, span.end) for span in spans)
    assert all(
        end <= next_start for (_, end), (next_start, _) in zip(ordered, ordered[1:], strict=False)
    )


def test_an_answer_without_markers_gets_no_spans() -> None:
    assert spans_for("Full-time employees accrue twenty vacation days.", POLICY) == [()]


def test_a_citation_survives_its_dict_form() -> None:
    chunks = [chunk(POLICY)]
    [cited] = attach_spans(
        "Full-time staff get twenty vacation days [1].", source_citations(chunks), chunks
    )

    assert cited.spans
    assert citation_from_dict(asdict(cited)) == cited


def test_sentence_spans_trim_whitespace_and_follow_the_chunker() -> None:
    text = "  First one. Second one!\n\nThird 一。二。 "

    assert [text[start:end] for start, end in sentence_spans(text)] == [
        "First one.",
        "Second one!",
        "Third 一。",
        "二。",
    ]


def test_a_flattened_markdown_heading_ends_the_sentence_before_it() -> None:
    text = (
        "The company reviews the policy every year. ## Receipts Submit receipts in the "
        "expense tool within thirty days."
    )
    [spans] = spans_for("Receipts must be submitted in the expense tool within 30 days [1].", text)

    assert [span.quote for span in spans] == [
        "## Receipts Submit receipts in the expense tool within thirty days."
    ]


def test_numbers_match_whether_written_as_digits_or_words() -> None:
    text = (
        "The 2019 daily meal allowance abroad was forty-five euros. "
        "Hotels were capped at one hundred and eighty euros per night."
    )
    [spans] = spans_for("Abroad the allowance was 45 euros a day [1].", text)
    assert [span.quote for span in spans] == [
        "The 2019 daily meal allowance abroad was forty-five euros."
    ]
    [spans] = spans_for("Hotels were capped at 180 euros [1].", text)
    assert [span.quote for span in spans] == [
        "Hotels were capped at one hundred and eighty euros per night."
    ]


def test_spelled_numbers_become_digits() -> None:
    from api.citations import _words

    assert _words("Twenty-eight days, one hundred and five, two thousand twenty, zero") == [
        "28",
        "days",
        "105",
        "2020",
        "0",
    ]
