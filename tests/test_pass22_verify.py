"""Pass 22: verification fixes (CJK sentences, citations of missing sources)."""

from __future__ import annotations

from api.citations import SourceCitation
from api.reranking import RerankedChunk
from api.verify import check_sentence, self_check, sentences_of

_ZH = "这里是公司介绍。员工每年有28天年假，需要经理书面批准。办公室在北京。"


def _chunk(text: str) -> RerankedChunk:
    return RerankedChunk("a", "h.txt", 1, "", "a", text, 0.0, 0.0)


def test_chinese_sentences_are_split_and_checked() -> None:
    assert sentences_of("员工每年有28天年假。需要经理批准[1]。月亮绕着地球转。") == [
        "员工每年有28天年假。",
        "需要经理批准[1]。",
        "月亮绕着地球转。",
    ]
    passage = _chunk(_ZH)
    supported = check_sentence("员工每年有28天年假[1]。", lambda _: [passage])
    assert supported.status == "supported" and supported.quote is not None
    assert "28" in supported.quote
    differs = check_sentence("员工每年有35天年假[1]。", lambda _: [passage])
    assert differs.status == "numbers_differ" and differs.missing_numbers == ("35",)
    assert check_sentence("月亮绕着地球转动很安静。", lambda _: [passage]).status == "unsupported"


def test_english_sentences_are_checked_as_before() -> None:
    passage = _chunk("Staff take annual leave of 28 days after asking their manager.")
    assert check_sentence("Staff take 28 days of annual leave.", lambda _: [passage]).status == (
        "supported"
    )
    assert sentences_of("One claim here. Another one there.") == [
        "One claim here.",
        "Another one there.",
    ]


def test_a_sentence_citing_a_source_the_answer_lacks_is_unsupported() -> None:
    source = SourceCitation(
        source_number=1,
        filename="h.txt",
        page=1,
        section="",
        chunk_id="a",
        text="Staff take annual leave of 28 days after asking their manager.",
    )
    checks = self_check("Staff take annual leave of 28 days [9].", [source])
    assert [c.status for c in checks] == ["unsupported"]


def test_a_chinese_sentence_keeps_its_citation_marker() -> None:
    assert sentences_of("员工每年有28天假期。[1]病假最多10天。[1, 2]「好」。下一句") == [
        "员工每年有28天假期。[1]",
        "病假最多10天。[1, 2]",
        "「好」。",
        "下一句",
    ]
