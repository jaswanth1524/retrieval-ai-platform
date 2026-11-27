"""Pass 23: self-check warnings gpt-5-mini drew on correct answers, and the CLI's
account of an answer that changed after streaming."""

from __future__ import annotations

import argparse
import io
from typing import Any

from api.citations import SourceCitation
from api.verify import check_sentence, self_check


def _source(n: int, filename: str, section: str, text: str) -> SourceCitation:
    return SourceCitation(
        source_number=n, filename=filename, page=1, section=section, chunk_id=f"c{n}", text=text
    )


_CURRENT = _source(
    1,
    "travel_policy.md",
    "Booking trips",
    "Book flights through the approved agency at least fourteen days ahead.",
)
_ARCHIVED = _source(
    2,
    "travel_policy_2019_archived.md",
    "Status",
    "Archived. This version was replaced by the current travel policy and is kept for "
    "audit purposes only. Employees booked flights through the former travel desk.",
)


def test_a_year_the_cited_documents_name_is_not_a_differing_number() -> None:
    answer = "The archived 2019 policy refers to the former travel desk for booking flights [2]."
    assert [c.status for c in self_check(answer, [_CURRENT, _ARCHIVED])] == ["supported"]
    # A figure neither the passage nor its name states is still flagged.
    wrong = "The archived 2021 policy refers to the former travel desk for booking flights [2]."
    checks = self_check(wrong, [_CURRENT, _ARCHIVED])
    assert [(c.status, c.missing_numbers) for c in checks] == [("numbers_differ", ("2021",))]


def test_a_sentence_citing_two_passages_may_take_its_numbers_from_either() -> None:
    answer = (
        "The current policy says to book flights through the approved agency at least "
        "fourteen days ahead [1]; the archived 2019 policy refers to a former travel desk [2]."
    )
    assert [c.status for c in self_check(answer, [_CURRENT, _ARCHIVED])] == ["supported"]
    # One passage cited: its numbers must be there.
    alone = "Book flights through the approved agency at least fourteen days ahead in 2019 [1]."
    assert [c.status for c in self_check(alone, [_CURRENT, _ARCHIVED])] == ["numbers_differ"]


def test_a_line_introducing_a_list_claims_nothing() -> None:
    answer = (
        "There are two values in the provided documents:\n"
        "- Book flights through the approved agency at least fourteen days ahead [1]."
    )
    assert [c.status for c in self_check(answer, [_CURRENT, _ARCHIVED])] == [
        "no_claim",
        "supported",
    ]
    assert check_sentence("规定如下：", lambda _: []).status == "no_claim"
    # A claim with a colon inside it is still checked.
    inside = "Booking rule: flights go through the moon agency forty days ahead [1]."
    assert [c.status for c in self_check(inside, [_CURRENT])] != ["no_claim"]


def _continue_args() -> argparse.Namespace:
    return argparse.Namespace(json=False, trace_id="t1")


class _Continues:
    def __init__(self, new: str, whole: str) -> None:
        self.new, self.whole = new, whole

    def continue_answer(self, trace_id: str, on_delta: Any = None) -> Any:
        from cli.client import StreamResult

        if on_delta is not None:
            on_delta(self.new)
        return StreamResult({"answer": self.whole, "truncated": False}, [])


def test_docrag_continue_shows_the_whole_answer_when_the_join_changed_it() -> None:
    from cli.question_commands import cmd_continue

    out = io.StringIO()
    # Streamed as the model wrote it; saved with its marker read as a citation.
    whole = "1. Book early [1].\n2. 假期为28天[1]。"
    assert cmd_continue(_Continues("2. 假期为28天【1】。", whole), _continue_args(), out) == 0  # type: ignore[arg-type]
    assert out.getvalue().endswith("[the whole answer as saved]\n" + whole + "\n")
    plain = io.StringIO()
    cmd_continue(_Continues(" receipts [1].", "1. Keep receipts [1]."), _continue_args(), plain)  # type: ignore[arg-type]
    assert "as saved" not in plain.getvalue()


def test_a_sentence_closed_inside_brackets_or_quotes_ends_there() -> None:
    from api.verify import sentences_of

    # As gpt-5-mini wrote it: one aside in brackets, then the next sentence.
    text = (
        '(The archived 2019 policy refers to a "former travel desk" [2].) You can ask '
        "the policy owner which agency is approved [1]."
    )
    assert sentences_of(text) == [
        '(The archived 2019 policy refers to a "former travel desk" [2].)',
        "You can ask the policy owner which agency is approved [1].",
    ]
    assert sentences_of('He said "book early." Then he left.') == [
        'He said "book early."',
        "Then he left.",
    ]
    # Not a sentence end: brackets and quotes with no stop before them stay joined.
    assert sentences_of("Book flights (fourteen days ahead) through the agency [1].") == [
        "Book flights (fourteen days ahead) through the agency [1]."
    ]


def test_a_number_written_with_a_typographic_hyphen_is_one_number() -> None:
    from api.citations import _content_words

    # gpt-5-mini writes "forty‑five" with a non-breaking hyphen (U+2011).
    assert _content_words("forty‑five euros") == _content_words("forty-five euros")
    archived = _source(
        1,
        "travel_policy_2019_archived.md",
        "Per diem",
        "The 2019 daily meal allowance abroad was forty-five euros.",
    )
    answer = "(The 2019 archived version specified forty‑five euros daily allowance) [1]."
    assert [c.status for c in self_check(answer, [archived])] == ["supported"]
