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
