"""CHUNKER_VERSION 5: whole table rows, honest row labels, page ranges, page-bounded
merging of tiny sections, and chunk ids that survive edits elsewhere in a document."""

from __future__ import annotations

from api.chunking import HeuristicTokenCounter
from api.documents import CHUNKER_VERSION, DocumentSection, chunk_sections, parse_document_bytes
from tests.factories import make_test_settings
from tests.samples import xlsx_bytes


def _chunks(name: str, content: bytes, **settings: object):  # type: ignore[no-untyped-def]
    config = make_test_settings(**settings)
    return chunk_sections(
        parse_document_bytes(name, content, config), config, HeuristicTokenCounter()
    )


def test_the_chunker_version_is_at_least_5() -> None:
    assert CHUNKER_VERSION >= 5


def test_csv_chunks_hold_whole_rows_labelled_with_the_rows_they_hold() -> None:
    rows = "\n".join(
        f"row{n},Some longer cell text for row {n}. With a full stop inside" for n in range(1, 41)
    )
    chunks = _chunks(
        "t.csv", f"name,notes\n{rows}\n".encode(), chunk_size_tokens=80, chunk_overlap_tokens=0
    )

    assert len(chunks) > 1
    for chunk in chunks:
        # Every chunk starts at the start of a row and ends at the end of one.
        assert chunk.text.startswith("name: row")
        assert chunk.text.endswith("With a full stop inside")
        first = int(chunk.text.split("name: row")[1].split(";")[0])
        last = int(chunk.text.rsplit("name: row", 1)[1].split(";")[0])
        # Spreadsheet rows: data row n is line n + 1.
        expected = f"Row {first + 1}" if first == last else f"Rows {first + 1}-{last + 1}"
        assert chunk.section == expected


def test_xlsx_chunk_labels_name_the_sheet_and_their_rows() -> None:
    sheet = [["Item", "Note"]] + [[f"i{n}", "word " * 12] for n in range(1, 21)]
    chunks = _chunks(
        "b.xlsx", xlsx_bytes({"Costs": sheet}), chunk_size_tokens=80, chunk_overlap_tokens=0
    )

    assert len(chunks) > 1
    assert all(chunk.section.startswith("Costs · Row") for chunk in chunks)
    assert chunks[0].section.startswith("Costs · Rows 2-")


def test_markdown_table_rows_are_never_split() -> None:
    header = "| Name | Notes |\n| --- | --- |\n"
    rows = "".join(f"| item {n} | {'detail ' * 8}end. |\n" for n in range(1, 25))
    chunks = _chunks(
        "t.md", f"# Table\n\n{header}{rows}".encode(), chunk_size_tokens=60, chunk_overlap_tokens=0
    )

    body = [chunk for chunk in chunks if "item" in chunk.text]
    assert len(body) > 1
    for chunk in body:
        assert chunk.text.rstrip().endswith("end. |")


def test_a_chunk_across_a_page_break_records_its_last_page() -> None:
    sections = [
        DocumentSection(filename="r.pdf", page=1, section="P1", text="Alpha starts here."),
        DocumentSection(filename="r.pdf", page=2, section="P2", text="Beta continues there."),
    ]
    settings = make_test_settings(chunk_size_tokens=200, chunk_overlap_tokens=0)

    [chunk] = chunk_sections(sections, settings, HeuristicTokenCounter())

    assert (chunk.page, chunk.page_end) == (1, 2)
    assert chunk.to_payload()["page_end"] == 2


def test_a_chunk_on_one_page_has_no_last_page() -> None:
    sections = [DocumentSection(filename="r.pdf", page=3, section="P3", text="Alone.")]
    [chunk] = chunk_sections(sections, make_test_settings(), HeuristicTokenCounter())

    assert chunk.page_end is None
    assert "page_end" not in chunk.to_payload()


def test_tiny_sections_are_not_merged_across_pages() -> None:
    """A short EPUB chapter (a chapter is a page) was cited as the next chapter."""

    sections = [
        DocumentSection(
            filename="b.epub", page=1, section="Prologue", text="Short.", boundary="semantic"
        ),
        DocumentSection(
            filename="b.epub",
            page=2,
            section="Chapter 1",
            text="word " * 60,
            boundary="semantic",
        ),
    ]
    settings = make_test_settings(min_section_words=40, chunk_size_tokens=200)

    chunks = chunk_sections(sections, settings, HeuristicTokenCounter())

    assert [(chunk.page, chunk.section) for chunk in chunks] == [(1, "Prologue"), (2, "Chapter 1")]


def test_chunk_ids_survive_an_insert_earlier_in_the_document() -> None:
    """With the position in the id, every later chunk got a new one after an edit, and
    citations saved in a conversation no longer found their passage."""

    def document(intro: str) -> bytes:
        sections = [f"# Part {n}\n\n{intro if n == 1 else ''}" for n in range(1, 6)]
        body = "\n\n".join(
            section + " ".join(f"Sentence {n}.{m} about part {n}." for m in range(12))
            for n, section in enumerate(sections, start=1)
        )
        return body.encode()

    settings = {"chunk_size_tokens": 60, "chunk_overlap_tokens": 0, "min_section_words": 0}
    before = _chunks("d.md", document(""), **settings)
    after = _chunks("d.md", document("A paragraph added at the top, long enough. " * 6), **settings)

    ids_before = {chunk.text: chunk.chunk_id for chunk in before if chunk.section != "Part 1"}
    ids_after = {chunk.text: chunk.chunk_id for chunk in after if chunk.section != "Part 1"}
    assert ids_before and ids_before == ids_after
    # The later chunks did move (the insert added a chunk before them): only their
    # position changed, and the id no longer depends on it.
    assert [c.ordinal for c in after if c.section == "Part 2"] == [
        c.ordinal + 1 for c in before if c.section == "Part 2"
    ]


def test_identical_chunks_still_get_distinct_ids() -> None:
    text = "\n\n".join(["Same boilerplate line here."] * 4)
    chunks = _chunks("d.txt", text.encode(), chunk_size_tokens=12, chunk_overlap_tokens=0)

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
