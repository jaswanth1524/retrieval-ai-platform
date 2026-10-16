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


def test_feedback_store_prunes_the_oldest_rows_past_its_cap(tmp_path: Path) -> None:
    store = FeedbackStore(str(tmp_path / "feedback.db"), max_rows=3)

    for index in range(5):
        store.add(
            trace_id=None,
            question=f"q{index}",
            answer_excerpt="a",
            cited_filenames=[],
            rating="up",
            citation_source_number=None,
        )

    assert [row.question for row in store.list_recent()] == ["q4", "q3", "q2"]


def test_rows_restored_from_a_backup_are_ordered_and_pruned_by_their_age(
    tmp_path: Path,
) -> None:
    """Restored rows are inserted last; ordered by insert they listed as the newest and
    were the last to be pruned, however old they were."""

    from api.feedback import Feedback

    store = FeedbackStore(str(tmp_path / "feedback.db"), max_rows=3)
    for index in range(2):
        store.add(
            trace_id=None,
            question=f"live{index}",
            answer_excerpt="a",
            cited_filenames=[],
            rating="up",
            citation_source_number=None,
        )
    old = [
        Feedback(
            id=f"old{index}",
            trace_id=None,
            question=f"old{index}",
            answer_excerpt="a",
            cited_filenames=[],
            rating="down",
            citation_source_number=None,
            created_at=1_000.0 + index,
        )
        for index in range(2)
    ]

    assert store.import_rows(old) == 2

    # Four rows against a cap of three: the oldest restored one goes.
    assert [row.question for row in store.list_recent()] == ["live1", "live0", "old1"]


def _rate(store: FeedbackStore, question: str, workspace: str = "") -> None:
    store.add(
        trace_id=None,
        question=question,
        answer_excerpt="a",
        cited_filenames=[],
        rating="up",
        citation_source_number=None,
        workspace=workspace,
    )


def test_feedback_is_kept_per_workspace(tmp_path: Path) -> None:
    store = FeedbackStore(str(tmp_path / "feedback.db"))
    _rate(store, "default q")
    _rate(store, "legal q", workspace="legal")

    assert [row.question for row in store.list_recent(10, workspace="")] == ["default q"]
    assert [row.question for row in store.list_recent(10, workspace="legal")] == ["legal q"]
    assert {row.question for row in store.list_recent(10)} == {"default q", "legal q"}
    page, _ = store.rows_oldest_first(None, 10, workspace="legal")
    assert [(row.question, row.workspace) for row in page] == [("legal q", "legal")]
    assert len(store.rows_oldest_first(None, 10)[0]) == 2


def test_restored_rows_join_the_workspace_they_are_restored_into(tmp_path: Path) -> None:
    source = FeedbackStore(str(tmp_path / "a.db"))
    _rate(source, "q")
    rows, _ = source.rows_oldest_first(None, 10)
    target = FeedbackStore(str(tmp_path / "b.db"))

    assert target.import_rows(rows, workspace="legal") == 1
    assert [row.workspace for row in target.list_recent(10)] == ["legal"]


def test_a_file_from_before_workspaces_gets_the_column_in_place(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "feedback.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE feedback (id TEXT PRIMARY KEY, trace_id TEXT, "
            "question TEXT NOT NULL, answer_excerpt TEXT NOT NULL, "
            "cited_filenames_json TEXT NOT NULL, rating TEXT NOT NULL, "
            "citation_source_number INTEGER, created_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO feedback VALUES ('old', NULL, 'old q', 'a', '[]', 'down', NULL, 1.0)"
        )
    conn.close()

    store = FeedbackStore(str(path))
    _rate(store, "new q", workspace="legal")

    assert [row.question for row in store.list_recent(10, workspace="")] == ["old q"]
    assert [row.question for row in store.list_recent(10, workspace="legal")] == ["new q"]
    FeedbackStore(str(path))  # opening it again doesn't try to add the column twice


def test_the_same_rating_can_be_restored_into_two_workspaces_but_only_once_into_each(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(str(tmp_path / "feedback.db"))
    _rate(store, "q", workspace="legal")
    rows, _ = store.rows_oldest_first(None, 10)

    assert store.import_rows(rows, workspace="archive") == 1
    assert store.import_rows(rows, workspace="archive") == 0
    assert store.import_rows(rows, workspace="legal") == 0
    assert len(store.list_recent(10)) == 2
