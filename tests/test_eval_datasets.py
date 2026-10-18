"""The labeled eval datasets must stay true to the fixture corpora they are run against.

A label naming a file that isn't in its corpus, or a passage its file no longer contains,
only shows up when someone runs the live harness — this catches it in the unit suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.harness import load_retrieval_dataset

EVAL = Path(__file__).resolve().parents[1] / "eval"

PAIRS = [
    ("retrieval_eval.sample.jsonl", "corpus"),
    ("handbook_eval.jsonl", "corpus_handbook"),
    ("multilingual_eval.jsonl", "corpus_multilingual"),
]


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


@pytest.mark.parametrize(("dataset", "corpus"), PAIRS)
def test_labels_point_at_files_and_passages_in_the_corpus(dataset: str, corpus: str) -> None:
    examples = load_retrieval_dataset(EVAL / "datasets" / dataset)
    texts = {path.name: _normalized(path.read_text()) for path in (EVAL / corpus).iterdir()}

    assert examples
    for example in examples:
        missing = example.relevant_filenames - texts.keys()
        assert not missing, f"{example.question!r} names files not in {corpus}: {missing}"
        search = example.relevant_filenames or texts.keys()
        for passage in example.relevant_passages:
            assert any(_normalized(passage) in texts[name] for name in search), (
                f"{example.question!r}: {passage!r} is in none of {sorted(search)}"
            )


def test_the_multilingual_set_covers_each_language_and_cross_lingual_questions() -> None:
    examples = load_retrieval_dataset(EVAL / "datasets" / "multilingual_eval.jsonl")
    labeled = {name for example in examples for name in example.relevant_filenames}
    # Every non-archived document is asked about; the archived one is only a distractor.
    corpus = {path.name for path in (EVAL / "corpus_multilingual").iterdir()}
    assert labeled == {name for name in corpus if "archivada" not in name}
    english = [example for example in examples if example.question.isascii()]
    assert len({name for example in english for name in example.relevant_filenames}) == 3
