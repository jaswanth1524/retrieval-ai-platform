"""Exact citation spans: which passage of a cited chunk an answer's claim rests on."""

from __future__ import annotations

from dataclasses import asdict, replace

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

    # The heading's marker stays off the quote; a heading met mid-chunk (this chunk's
    # section is "Leave") keeps its title, as nothing says where the title ends.
    assert [span.quote for span in spans] == [
        "Receipts Submit receipts in the expense tool within thirty days."
    ]
    assert text[spans[0].start : spans[0].end] == spans[0].quote


def _spans_with_section(answer: str, text: str, section: str) -> tuple[CitationSpan, ...]:
    source = replace(chunk(text), section=section)
    [cited] = attach_spans(answer, source_citations([source]), [source])
    return cited.spans


def test_the_chunks_own_heading_stays_off_the_quote() -> None:
    text = "## Receipts Submit receipts in the expense tool within thirty days. Keep copies."
    spans = _spans_with_section(
        "Receipts go in the expense tool within 30 days [1].", text, "Receipts"
    )
    assert [span.quote for span in spans] == [
        "Submit receipts in the expense tool within thirty days."
    ]
    assert text[spans[0].start : spans[0].end] == spans[0].quote


def test_a_setext_heading_stays_off_the_quote() -> None:
    text = "Receipts ======== Submit receipts in the expense tool within thirty days."
    spans = _spans_with_section(
        "Receipts go in the expense tool within 30 days [1].", text, "Receipts"
    )
    assert [span.quote for span in spans] == [
        "Submit receipts in the expense tool within thirty days."
    ]


def test_a_sentence_starting_with_the_section_word_is_quoted_whole() -> None:
    text = "Receipts are submitted in the expense tool within thirty days."
    spans = _spans_with_section(
        "Receipts go in the expense tool within 30 days [1].", text, "Receipts"
    )
    assert [span.quote for span in spans] == [text]


def test_a_quote_never_joins_the_sentence_before_a_heading() -> None:
    text = (
        "Expense claims use the expense tool. ## Receipts Receipts use the expense tool "
        "within thirty days."
    )
    [spans] = spans_for("Claims and receipts use the expense tool within 30 days [1].", text)
    assert all("Expense claims use" not in span.quote or "##" not in span.quote for span in spans)
    assert all("##" not in span.quote for span in spans)


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


def test_a_data_table_citation_quotes_its_row_not_the_whole_chunk() -> None:
    """Table rows are joined by line breaks in the chunk text (chunking.join_units), so a
    claim about one row of a CSV without full stops quotes that row. Measured on a
    40-row CSV at the default chunk size: mean quote 1044 characters before, 55 after."""

    from api.chunking import HeuristicTokenCounter
    from api.documents import chunk_sections, parse_document_bytes
    from tests.factories import make_test_settings

    settings = make_test_settings()
    rows = [f"person{i},team{i % 4},city{i % 7},{2010 + i % 9}" for i in range(40)]
    csv = ("name,team,office,start\n" + "\n".join(rows)).encode()
    chunks = chunk_sections(
        parse_document_bytes("staff.csv", csv, settings), settings, HeuristicTokenCounter()
    )
    target = next(c for c in chunks if "name: person7;" in c.text)
    assert "\nname: person8;" in target.text
    [spans] = spans_for("person7 is on team3 in the city0 office, since 2017 [1].", target.text)

    assert [span.quote for span in spans] == [
        "name: person7; team: team3; office: city0; start: 2017"
    ]


def test_units_are_joined_by_spaces_and_rows_by_line_breaks() -> None:
    from api.chunking import _Unit, join_units

    window = [
        _Unit("Intro text.", 2, 1, "S"),
        _Unit("More text.", 2, 1, "S"),
        _Unit("name: a", 2, 1, "S", row=2),
        _Unit("| b | c |", 2, 1, "S"),
    ]
    assert join_units(window) == "Intro text. More text.\nname: a\n| b | c |"


def test_query_spans_mark_the_best_sentences_in_text_order() -> None:
    from api.citations import MAX_QUERY_SPANS, query_spans

    text = (
        "Parking is free. Passes for parking are issued at reception. "
        "Reception issues parking passes and visitor passes daily. The cafe opens at nine."
    )
    spans = query_spans("parking passes reception", text)
    assert len(spans) == MAX_QUERY_SPANS == 2
    assert [span.quote for span in spans] == [
        "Passes for parking are issued at reception.",
        "Reception issues parking passes and visitor passes daily.",
    ]
    assert all(text[span.start : span.end] == span.quote for span in spans)
    # One word is enough for a one- or two-word query; stopwords alone never are.
    assert [span.quote for span in query_spans("cafe", text)] == ["The cafe opens at nine."]
    assert query_spans("the of and", text) == ()


def test_past_heading_moves_only_past_the_chunks_own_heading() -> None:
    from api.citations import _past_heading

    text = "## Leave\nAnnual leave is twenty days."
    assert text[_past_heading(text, 0, len(text), "Leave") :] == "Annual leave is twenty days."
    # A label that only starts a longer word is not the heading.
    other = "Leaves fall in autumn."
    assert _past_heading(other, 0, len(other), "Leave") == 0
    # A span that is nothing but the heading keeps its title rather than becoming empty.
    bare = "## Leave"
    assert bare[_past_heading(bare, 0, len(bare), "Leave") :] == "Leave"
    setext = "Leave\n=====\nAnnual leave is twenty days."
    assert setext[_past_heading(setext, 0, len(setext), "Leave") :].startswith("Annual")
