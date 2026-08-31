from __future__ import annotations

from pathlib import Path

from api.feedback import FeedbackStore


def test_add_returns_a_feedback_record_with_a_generated_id(tmp_path: Path) -> None:
    store = FeedbackStore(str(tmp_path / "feedback.db"))

    feedback = store.add(
        trace_id="trace-1",
        question="What is DocRAG?",
        answer_excerpt="A self-hostable document Q&A system.",
        cited_filenames=["docrag.md"],
        rating="up",
        citation_source_number=None,
    )

    assert feedback.id
    assert feedback.trace_id == "trace-1"
    assert feedback.question == "What is DocRAG?"
    assert feedback.rating == "up"
    assert feedback.citation_source_number is None
    assert feedback.created_at > 0


def test_add_accepts_a_null_trace_id(tmp_path: Path) -> None:
    # trace_id can be absent entirely — it's a best-effort join key, not a required
    # foreign key (see the module docstring).
    store = FeedbackStore(str(tmp_path / "feedback.db"))

    feedback = store.add(
        trace_id=None,
        question="q",
        answer_excerpt="a",
        cited_filenames=[],
        rating="up",
        citation_source_number=None,
    )

    assert feedback.trace_id is None


def test_list_recent_returns_newest_first(tmp_path: Path) -> None:
    store = FeedbackStore(str(tmp_path / "feedback.db"))
    first = store.add(
        trace_id=None,
        question="q1",
        answer_excerpt="a1",
        cited_filenames=[],
        rating="up",
        citation_source_number=None,
    )
    second = store.add(
        trace_id=None,
        question="q2",
        answer_excerpt="a2",
        cited_filenames=[],
        rating="down",
        citation_source_number=2,
    )

    recent = store.list_recent()

    assert [f.id for f in recent] == [second.id, first.id]


def test_list_recent_round_trips_cited_filenames_and_citation_source_number(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(str(tmp_path / "feedback.db"))
    store.add(
        trace_id="t1",
        question="q",
        answer_excerpt="a",
        cited_filenames=["a.pdf", "b.pdf"],
        rating="down",
        citation_source_number=3,
    )

    recent = store.list_recent()

    assert recent[0].cited_filenames == ["a.pdf", "b.pdf"]
    assert recent[0].citation_source_number == 3


def test_list_recent_respects_the_limit(tmp_path: Path) -> None:
    store = FeedbackStore(str(tmp_path / "feedback.db"))
    for i in range(5):
        store.add(
            trace_id=None,
            question=f"q{i}",
            answer_excerpt="a",
            cited_filenames=[],
            rating="up",
            citation_source_number=None,
        )

    assert len(store.list_recent(limit=2)) == 2


def test_survives_a_process_restart(tmp_path: Path) -> None:
    db_path = str(tmp_path / "feedback.db")
    first_process_store = FeedbackStore(db_path)
    feedback = first_process_store.add(
        trace_id="t1",
        question="q",
        answer_excerpt="a",
        cited_filenames=["doc.md"],
        rating="up",
        citation_source_number=None,
    )
    del first_process_store

    second_process_store = FeedbackStore(db_path)
    recent = second_process_store.list_recent()

    assert len(recent) == 1
    assert recent[0].id == feedback.id
    assert recent[0].cited_filenames == ["doc.md"]
