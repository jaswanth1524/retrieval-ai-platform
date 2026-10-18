from __future__ import annotations

import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest

from api.feedback import FeedbackStore
from eval.feedback_dataset import (
    FeedbackDataError,
    FeedbackRow,
    build_dataset,
    rows_from_export,
    rows_from_store,
)
from eval.harness import load_retrieval_dataset, main


def _row(
    question: str,
    rating: str,
    cited: tuple[str, ...] = ("a.pdf",),
    at: float = 1.0,
    citation: int | None = None,
) -> FeedbackRow:
    return FeedbackRow(
        question=question,
        answer_excerpt="An answer [1].",
        cited_filenames=cited,
        rating=rating,
        citation_source_number=citation,
        created_at=at,
    )


def test_thumbs_up_becomes_a_label_and_thumbs_down_a_question_to_review() -> None:
    dataset = build_dataset(
        [
            _row("What is alpha?", "up", ("b.pdf", "a.pdf", "a.pdf"), at=1),
            _row("Why beta?", "down", ("c.pdf",), at=2),
            _row("Which source?", "up", at=3, citation=2),
            _row("Uncited?", "up", (), at=4),
        ]
    )

    assert dataset.examples == [
        {
            "question": "What is alpha?",
            "relevant_filenames": ["a.pdf", "b.pdf"],
            "relevant_chunk_ids": [],
            "source": "feedback",
        }
    ]
    assert [row["question"] for row in dataset.review] == ["Why beta?"]
    assert dataset.review[0]["relevant_filenames"] == []
    assert dataset.review[0]["cited_filenames"] == ["c.pdf"]
    assert (dataset.ratings_read, dataset.citation_ratings, dataset.uncited) == (4, 1, 1)


def test_the_latest_rating_of_a_question_wins_whatever_its_spacing_or_case() -> None:
    dataset = build_dataset(
        [
            _row("What is alpha?", "up", at=1),
            _row("what  is ALPHA?", "down", at=5),
            _row("Gamma?", "down", at=1),
            _row("gamma?", "up", ("g.pdf",), at=2),
        ]
    )

    assert [row["question"] for row in dataset.review] == ["what is ALPHA?"]
    assert [row["question"] for row in dataset.examples] == ["gamma?"]


def _store_with_ratings(path: Path) -> None:
    store = FeedbackStore(str(path))
    store.add(
        trace_id=None,
        question="What is alpha?",
        answer_excerpt="Alpha [1].",
        cited_filenames=["a.pdf"],
        rating="up",
        citation_source_number=None,
    )
    store.add(
        trace_id="t",
        question="Why beta?",
        answer_excerpt="Beta [1].",
        cited_filenames=["b.pdf"],
        rating="down",
        citation_source_number=None,
    )


def test_ratings_are_read_from_the_store_and_from_an_export(tmp_path: Path) -> None:
    store_path = tmp_path / "feedback.sqlite3"
    _store_with_ratings(store_path)
    from_store = list(rows_from_store(store_path))
    assert [(row.question, row.rating) for row in from_store] == [
        ("What is alpha?", "up"),
        ("Why beta?", "down"),
    ]

    export = tmp_path / "backup.zip"
    rows = FeedbackStore(str(store_path)).list_recent()
    with zipfile.ZipFile(export, "w") as archive:
        archive.writestr(
            "feedback.jsonl",
            "".join(json.dumps(asdict(row)) + "\n" for row in reversed(rows)) + "not json\n",
        )
    assert [row.question for row in rows_from_export(export)] == ["What is alpha?", "Why beta?"]


def test_unreadable_sources_are_named(tmp_path: Path) -> None:
    with pytest.raises(FeedbackDataError, match="No feedback store"):
        list(rows_from_store(tmp_path / "missing.sqlite3"))
    not_zip = tmp_path / "x.zip"
    not_zip.write_text("nope")
    with pytest.raises(FeedbackDataError, match="not a readable export zip"):
        list(rows_from_export(not_zip))
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("manifest.json", "{}")
    with pytest.raises(FeedbackDataError, match="no feedback.jsonl"):
        list(rows_from_export(empty))


def test_the_cli_writes_a_dataset_the_harness_loads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store_path = tmp_path / "feedback.sqlite3"
    _store_with_ratings(store_path)
    output = tmp_path / "from_feedback.jsonl"
    review = tmp_path / "review.jsonl"

    code = main(
        [
            "from-feedback",
            "--store",
            str(store_path),
            "--output",
            str(output),
            "--review",
            str(review),
        ]
    )

    assert code == 0
    examples = load_retrieval_dataset(output)
    assert [example.question for example in examples] == ["What is alpha?"]
    assert examples[0].relevant_filenames == frozenset({"a.pdf"})
    assert json.loads(review.read_text())["question"] == "Why beta?"
    assert "1 labeled examples, 1 rated-down questions to review" in capsys.readouterr().out


def test_the_cli_fails_when_nothing_can_be_labeled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store_path = tmp_path / "feedback.sqlite3"
    FeedbackStore(str(store_path)).add(
        trace_id=None,
        question="Why beta?",
        answer_excerpt="",
        cited_filenames=[],
        rating="down",
        citation_source_number=None,
    )
    output = tmp_path / "out.jsonl"

    assert main(["from-feedback", "--store", str(store_path), "--output", str(output)]) == 2
    assert not output.exists()
    assert "no thumbs-up answer" in capsys.readouterr().err
    assert main(["from-feedback", "--store", str(tmp_path / "nope"), "--output", str(output)]) == 2


def test_the_store_can_be_read_for_one_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store_path = tmp_path / "feedback.sqlite3"
    store = FeedbackStore(str(store_path))
    for question, workspace in (("Default q?", ""), ("Legal q?", "legal")):
        store.add(
            trace_id=None,
            question=question,
            answer_excerpt="An answer [1].",
            cited_filenames=["a.pdf"],
            rating="up",
            citation_source_number=None,
            workspace=workspace,
        )

    assert [row.question for row in rows_from_store(store_path, "legal")] == ["Legal q?"]
    assert [row.question for row in rows_from_store(store_path, "")] == ["Default q?"]
    assert len(list(rows_from_store(store_path))) == 2

    output = tmp_path / "legal.jsonl"
    code = main(
        [
            "from-feedback",
            "--store",
            str(store_path),
            "--workspace",
            "legal",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert [example.question for example in load_retrieval_dataset(output)] == ["Legal q?"]


def test_the_cli_reads_the_default_workspace_unless_told_otherwise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store_path = tmp_path / "feedback.sqlite3"
    store = FeedbackStore(str(store_path))
    for question, workspace in (("Default q?", ""), ("Legal q?", "legal")):
        store.add(
            trace_id=None,
            question=question,
            answer_excerpt="An answer [1].",
            cited_filenames=["a.pdf"],
            rating="up",
            citation_source_number=None,
            workspace=workspace,
        )
    output = tmp_path / "out.jsonl"
    store_args = ["from-feedback", "--store", str(store_path), "--output", str(output)]

    assert main(store_args) == 0
    assert [example.question for example in load_retrieval_dataset(output)] == ["Default q?"]
    assert main([*store_args, "--all-workspaces"]) == 0
    assert len(load_retrieval_dataset(output)) == 2

    export = tmp_path / "backup.zip"
    with zipfile.ZipFile(export, "w") as archive:
        archive.writestr("feedback.jsonl", "")
    capsys.readouterr()
    for scope in (["--workspace", "legal"], ["--all-workspaces"]):
        args = ["from-feedback", "--export", str(export), "--output", str(output), *scope]
        assert main(args) == 2
        assert "apply to --store only" in capsys.readouterr().err


def test_only_question_ratings_are_labels(tmp_path: Path) -> None:
    store_path = tmp_path / "feedback.sqlite3"
    store = FeedbackStore(str(store_path))
    for question, kind in (
        ("What is alpha?", "question"),
        ('Summarize the document "a.pdf".', "summary"),
        ("How is beta used?", "follow_up"),
    ):
        store.add(
            trace_id=None,
            question=question,
            answer_excerpt="An answer [1].",
            cited_filenames=["a.pdf"],
            rating="up",
            citation_source_number=None,
            kind=kind,  # type: ignore[arg-type]
        )
    dataset = build_dataset(rows_from_store(store_path, ""))

    assert [row["question"] for row in dataset.examples] == ["What is alpha?"]
    assert dataset.not_questions == 2
    assert [row["source"] for row in dataset.review] == [
        "feedback (a rated summary, not a question: left out of the dataset)",
        "feedback (a rated follow-up, not a question: left out of the dataset)",
    ]

    # Rows from before kinds: a summary's or comparison's task text gives it away.
    legacy = build_dataset(
        [
            _row('Summarize the document "a.pdf".', "up", at=1),
            _row('Compare the documents "a.pdf" and "b.pdf".', "up", at=2),
            _row("What is gamma?", "up", at=3),
        ]
    )
    assert [row["question"] for row in legacy.examples] == ["What is gamma?"]
    assert legacy.not_questions == 2


def test_the_store_is_read_without_writing_to_it(tmp_path: Path) -> None:
    import os
    import sqlite3

    store_path = tmp_path / "feedback.sqlite3"
    with sqlite3.connect(store_path) as conn:
        # A file from before workspaces and kinds: reading must not migrate it.
        conn.execute(
            "CREATE TABLE feedback (id TEXT PRIMARY KEY, trace_id TEXT, "
            "question TEXT NOT NULL, answer_excerpt TEXT NOT NULL, "
            "cited_filenames_json TEXT NOT NULL, rating TEXT NOT NULL, "
            "citation_source_number INTEGER, created_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO feedback VALUES ('1', NULL, 'Old q?', 'a', '[\"a.pdf\"]', 'up', NULL, 1)"
        )
    conn.close()
    os.chmod(store_path, 0o444)
    os.chmod(tmp_path, 0o555)  # no journal or WAL file could be created either
    try:
        assert [row.question for row in rows_from_store(store_path, "")] == ["Old q?"]
        assert list(rows_from_store(store_path, "legal")) == []
    finally:
        os.chmod(tmp_path, 0o755)
        os.chmod(store_path, 0o644)
    with sqlite3.connect(store_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
    conn.close()
    assert "workspace" not in columns and "kind" not in columns

    garbage = tmp_path / "garbage.sqlite3"
    garbage.write_bytes(b"not a database at all" * 100)
    with pytest.raises(FeedbackDataError, match="not a readable feedback store"):
        list(rows_from_store(garbage))
    empty = tmp_path / "empty.sqlite3"
    sqlite3.connect(empty).close()
    empty.write_bytes(b"")  # a valid, empty database file
    with pytest.raises(FeedbackDataError, match="no feedback table"):
        list(rows_from_store(empty))
