"""Pass 23: full-width citation markers (a model answering in Chinese or Japanese)."""

from __future__ import annotations

from typing import Any

import pytest

from api.citations import normalize_citation_markers
from api.dependencies import clear_dependency_caches
from tests.routes_http.support import upload_and_wait
from tests.test_pass17_answers import _sse_events
from tests.test_pass21_features import _DOC, _client


@pytest.mark.parametrize(
    ("written", "read"),
    [
        ("假期为28天【1】。", "假期为28天[1]。"),
        ("假期为28天［１］。", "假期为28天[1]。"),
        ("假期为28天[１]。", "假期为28天[1]。"),
        ("见【1，2】和【 3 】", "见[1,2]和[3]"),
        ("见【１、３－４】", "见[1,3-4]"),
        ("见【1】【2】", "见[1][2]"),
        # Not markers, or already plain: left as written.
        ("【注意】请假需要批准。【2024】年度", "【注意】请假需要批准。【2024】年度"),
        (
            "Leave is 28 days [1, 2] and [3-4]. See [ 5 ].",
            "Leave is 28 days [1, 2] and [3-4]. See [ 5 ].",
        ),
        ("【1] mismatched ［2】", "【1] mismatched ［2】"),
    ],
)
def test_full_width_markers_are_read_as_citations(written: str, read: str) -> None:
    assert normalize_citation_markers(written) == read
    assert normalize_citation_markers(read) == read


_ANSWER = "Staff take annual leave of 28 days after asking their manager【1】."
_PLAIN = "Staff take annual leave of 28 days after asking their manager[1]."


def test_an_answer_cited_with_full_width_brackets_has_its_sources_and_self_check() -> None:
    with _client(_ANSWER) as client:
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        sync = client.post("/questions", json={"question": "leave?"}).json()
        events = _sse_events(
            client.post("/questions/stream", json={"question": "leave?", "use_cache": False}).text
        )
        done: dict[str, Any] = next(e for e in events if e["type"] == "done")
        summary = client.post(
            "/questions/summary", json={"mode": "summarize", "filenames": ["leave.txt"]}
        ).json()
        for found in (sync, done):
            assert found["answer"] == _PLAIN, found["answer"]
            assert found["answer_status"] == "answered" and found["sources"]
            assert [c["status"] for c in found["verification"]] == ["supported"]
            assert not found.get("citation_backfilled")
        assert summary["answer"] == _PLAIN and summary["sources"], summary
        assert summary["answer_status"] == "answered"
    clear_dependency_caches()


def test_a_continuation_restarting_with_full_width_markers_isnt_repeated() -> None:
    from api.continuation import AutoContinuing
    from api.llm import reset_finish_reason
    from tests.factories import make_test_settings

    first = "Staff take annual leave of 28 days [1]. Furthermore, unused days"
    again = "Staff take annual leave of 28 days 【1】. Furthermore, unused days carry over."

    class _Restarts:
        calls = 0

        def complete(self, messages: Any, settings: Any) -> str:
            type(self).calls += 1
            reset_finish_reason("length" if type(self).calls == 1 else "stop")
            return first if type(self).calls == 1 else again

        def stream(self, messages: Any, settings: Any) -> Any:
            yield ""

    text = AutoContinuing(_Restarts(), 1).complete(
        # OpenAI takes no prefill: asked in words, it may restart from the top.
        [{"role": "user", "content": "leave?"}],
        make_test_settings(llm_provider="openai", openai_api_key="x"),
    )
    assert text.count("Staff take annual leave") == 1 and "carry over" in text, text
    reset_finish_reason()


@pytest.mark.parametrize(
    ("written", "read"),
    [
        # qwen2.5 (live): part of the source line put inside the marker.
        (
            "Staff receive twenty-eight vacation days per year [1, section=Vacation].",
            "Staff receive twenty-eight vacation days per year [1].",
        ),
        ("Book early [2; page=1; section=Booking trips].", "Book early [2]."),
        ("Book early [1, 2, section=Booking].", "Book early [1, 2]."),
        # Anything else with "=" in brackets is the answer's own text.
        ("Set [x=1] in the config.", "Set [x=1] in the config."),
    ],
)
def test_a_marker_carrying_part_of_its_source_line_is_cut_to_the_marker(
    written: str, read: str
) -> None:
    from api.citations import strip_echoed_source_headers

    assert strip_echoed_source_headers(written) == read


def test_such_an_answer_is_cited_and_checked() -> None:
    from api.citations import SourceCitation
    from api.verify import self_check

    source = SourceCitation(
        source_number=1,
        filename="leave_policy.md",
        page=1,
        section="Vacation",
        chunk_id="c1",
        text="Full-time staff receive twenty-eight vacation days per year.",
    )
    with _client(
        "Full-time staff receive twenty-eight vacation days per year [1, section=Vacation]."
    ) as client:
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        found = client.post("/questions", json={"question": "leave?"}).json()
        assert found["answer"].endswith("per year [1].") and found["sources"], found["answer"]
    clear_dependency_caches()
    assert [
        c.status for c in self_check("Staff get twenty-eight vacation days per year [1].", [source])
    ] == ["supported"]
