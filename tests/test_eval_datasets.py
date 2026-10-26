"""The labeled eval datasets must stay true to the fixture corpora they are run against.

A label naming a file that isn't in its corpus, or a passage its file no longer contains,
only shows up when someone runs the live harness — this catches it in the unit suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from api.documents import parse_document_bytes
from eval.harness import load_retrieval_dataset

EVAL = Path(__file__).resolve().parents[1] / "eval"

PAIRS = [
    ("retrieval_eval.sample.jsonl", "corpus"),
    ("handbook_eval.jsonl", "corpus_handbook"),
    ("multilingual_eval.jsonl", "corpus_multilingual"),
    ("unanswerable_eval.jsonl", "corpus_handbook"),
    ("followup_eval.jsonl", "corpus_handbook"),
    ("paraphrase_eval.jsonl", "corpus_handbook"),
    ("compare_eval.jsonl", "corpus_handbook"),
    ("mixed_eval.jsonl", "corpus_mixed"),
]


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


@pytest.mark.parametrize(("dataset", "corpus"), PAIRS)
def test_labels_point_at_files_and_passages_in_the_corpus(dataset: str, corpus: str) -> None:
    examples = load_retrieval_dataset(EVAL / "datasets" / dataset)
    # As indexed: a PDF or DOCX label must hold for the text its parser reads.
    texts = {
        path.name: _normalized(
            " ".join(section.text for section in parse_document_bytes(path.name, path.read_bytes()))
        )
        for path in (EVAL / corpus).iterdir()
    }

    assert examples
    for example in examples:
        missing = example.relevant_filenames - texts.keys()
        assert not missing, f"{example.question!r} names files not in {corpus}: {missing}"
        search = example.relevant_filenames or texts.keys()
        for passage in example.relevant_passages:
            assert any(_normalized(passage) in texts[name] for name in search), (
                f"{example.question!r}: {passage!r} is in none of {sorted(search)}"
            )
        if example.answer_contains:
            # At least one way of stating the fact is the corpus's own wording.
            assert any(
                _normalized(fact) in texts[name]
                for fact in example.answer_contains
                for name in search
            ), f"{example.question!r}: no answer_contains fact is in {sorted(search)}"


def test_the_multilingual_set_covers_each_language_and_cross_lingual_questions() -> None:
    examples = load_retrieval_dataset(EVAL / "datasets" / "multilingual_eval.jsonl")
    labeled = {name for example in examples for name in example.relevant_filenames}
    # Every non-archived document is asked about; the archived one is only a distractor.
    corpus = {path.name for path in (EVAL / "corpus_multilingual").iterdir()}
    assert labeled == {name for name in corpus if "archivada" not in name}
    english = [example for example in examples if example.question.isascii()]
    assert len({name for example in english for name in example.relevant_filenames}) == 3


def test_the_mixed_corpus_is_the_handbook_in_other_formats() -> None:
    """eval/make_mixed_corpus.py's output is committed: it must match the handbook."""

    from eval.make_mixed_corpus import RENDERED, renamed

    handbook = {path.name for path in (EVAL / "corpus_handbook").glob("*.md")}
    mixed = {path.name for path in (EVAL / "corpus_mixed").iterdir()}
    assert mixed == {renamed(name) for name in handbook}
    assert {name.rsplit(".", 1)[1] for name in mixed} >= {
        "pdf",
        "docx",
        "html",
        "odt",
        "pptx",
        "xlsx",
    }
    for name in handbook - set(RENDERED):
        assert (EVAL / "corpus_mixed" / name).read_bytes() == (
            EVAL / "corpus_handbook" / name
        ).read_bytes()
    handbook_questions = load_retrieval_dataset(EVAL / "datasets" / "handbook_eval.jsonl")
    mixed_questions = load_retrieval_dataset(EVAL / "datasets" / "mixed_eval.jsonl")
    assert [e.question for e in mixed_questions] == [e.question for e in handbook_questions]
