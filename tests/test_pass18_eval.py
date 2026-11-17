"""Pass 18: recorded answer evals, excluded values, chains, summaries, calibration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from api.settings import AppSettings
from eval.calibrate import best_bar, judge
from eval.harness import (
    HarnessDataError,
    RetrievalExample,
    _example_from_mapping,
    models_table,
    run_answer_eval,
    with_model,
)
from eval.replay import RecordingGenerator, ReplayGenerator, ReplayMissError, answer_regressions
from tests.test_eval_harness import (
    AnsweringRepository,
    FakeEmbeddingProvider,
    FakeReranker,
    ScriptedGenerator,
    _point,
    make_settings,
)


class Collab:
    def __init__(self, generator: Any) -> None:
        self.repository = AnsweringRepository([_point("c1", "a.md"), _point("c2", "b.md")])
        self.embedding_provider = FakeEmbeddingProvider()
        self.reranker = FakeReranker(["c1", "c2"])
        self.generator = generator
        self.settings = make_settings()


def _example(question: str, **fields: Any) -> RetrievalExample:
    return RetrievalExample(
        question=question,
        reference=None,
        relevant_filenames=frozenset({"a.md"}),
        relevant_chunk_ids=frozenset(),
        **fields,
    )


def test_a_recorded_run_replays_with_no_model_and_a_changed_prompt_is_a_miss(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "replies.jsonl"
    examples = [_example("first?", answer_contains=("c1",)), _example("second?")]
    live = ScriptedGenerator("It is c1 [1].")
    assert (
        run_answer_eval(examples, Collab(RecordingGenerator(live, recording)), tmp_path / "a.json")
        == 0
    )  # type: ignore[arg-type]
    recorded = json.loads((tmp_path / "a.json").read_text())

    replay = ReplayGenerator(recording)
    assert len(replay) == 2
    out = tmp_path / "b.json"
    assert run_answer_eval(examples, Collab(replay), out, replay=replay) == 0  # type: ignore[arg-type]
    replayed = json.loads(out.read_text())
    assert [r["answer"] for r in replayed["examples"]] == [
        r["answer"] for r in recorded["examples"]
    ]
    assert replayed["metrics"] == recorded["metrics"] and not replay.misses

    # A new question: its prompt was never recorded.
    changed = [*examples, _example("third?")]
    replay = ReplayGenerator(recording)
    run_answer_eval(changed, Collab(replay), out, replay=replay)  # type: ignore[arg-type]
    rows = json.loads(out.read_text())["examples"]
    assert rows[2]["answer_status"] == "error" and len(replay.misses) == 1
    with pytest.raises(ReplayMissError):
        replay.complete([{"role": "user", "content": "never asked"}], make_settings())


def test_an_answer_metric_worse_than_the_baseline_is_a_regression() -> None:
    baseline = {"fact_hit_rate": 0.9, "uncited_rate": 0.1, "error_rate": 0.0}
    assert answer_regressions(baseline, baseline, 0.0) == []
    worse = {"fact_hit_rate": 0.8, "uncited_rate": 0.2, "error_rate": 0.0}
    assert [p.split(":")[0] for p in answer_regressions(worse, baseline, 0.05)] == [
        "fact_hit_rate",
        "uncited_rate",
    ]
    assert answer_regressions(worse, baseline, 0.1) == []
    assert answer_regressions({}, baseline, 0.0)[0].endswith("missing from this run")


def test_an_excluded_value_fails_the_row_and_numbers_match_whole() -> None:
    from eval.answer_eval import scored_facts, states_excluded

    assert states_excluded("Leave is 30 days [1].", ["30 days"]) is True
    assert states_excluded("Leave is 130 days [1].", ["30 days"]) is False
    assert states_excluded("It costs 3.5 euros.", ["3"]) is False
    assert states_excluded("The old policy said so.", ["old policy"]) is True
    assert states_excluded("anything", []) is None
    example = _example("leave?", answer_contains=("25 days",), answer_excludes=("30 days",))
    assert scored_facts("It was 30 days and is now 25 days.", example) == (False, True)
    assert scored_facts("It is 25 days.", example) == (True, False)


class TurnGenerator:
    """Answers each question with its own text, so the chain's history is visible."""

    def __init__(self) -> None:
        self.prompts: list[list[dict[str, str]]] = []

    def complete(self, messages: Sequence[Any], settings: AppSettings) -> str:
        self.prompts.append([dict(m) for m in messages])
        return f"c1 answer {len(self.prompts)} [1]."

    def stream(self, messages: Sequence[Any], settings: AppSettings) -> Any:
        yield self.complete(messages, settings)


def test_a_chain_is_asked_in_order_with_this_runs_own_answers_as_history(
    tmp_path: Path,
) -> None:
    generator = TurnGenerator()
    examples = [
        _example("What does c1 say?", chain="c"),
        _example("And for how long is that?", chain="c"),
    ]
    run_answer_eval(examples, Collab(generator), tmp_path / "a.json")  # type: ignore[arg-type]
    rows = json.loads((tmp_path / "a.json").read_text())["examples"]
    first_answer = rows[0]["answer"]
    # The second question's prompts carry the first question and the answer it got.
    later = [m["content"] for prompt in generator.prompts[1:] for m in prompt]
    assert any("What does c1 say?" in text for text in later)
    assert any(first_answer.split(" [")[0] in text for text in later)


def test_dataset_rows_take_excludes_chains_and_summary_tasks() -> None:
    row = _example_from_mapping(
        {
            "question": "Summarize it",
            "task": "summarize",
            "filenames": ["a.md"],
            "answer_excludes": ["30 days"],
        },
        0,
    )
    assert row.task == "summarize" and row.answer_excludes == ("30 days",)
    for bad in (
        {"question": "q", "task": "summarize"},
        {"question": "q", "task": "poem", "filenames": ["a.md"]},
        {"question": "q", "chain": "c", "history": [{"role": "user", "content": "x"}]},
        {"question": "q", "chain": " "},
    ):
        with pytest.raises(HarnessDataError):
            _example_from_mapping(bad, 0)


def test_the_best_bar_separates_answerable_from_unanswerable_questions() -> None:
    answerable = [0.42, 0.31, 0.12, 0.55]
    unanswerable = [0.02, 0.05, 0.09]
    suggested = best_bar(answerable, unanswerable)
    assert suggested is not None and suggested.wrong == 0
    assert 0.09 < suggested.bar < 0.12
    current = judge(0.15, answerable, unanswerable)
    assert (current.refused_answerable, current.answered_unanswerable) == (1, 0)
    overlapping = best_bar([0.1, 0.3], [0.2])
    assert overlapping is not None and overlapping.wrong == 1
    assert best_bar([0.3], []) is None


def test_a_model_comparison_overrides_the_provider_and_tables_the_runs(tmp_path: Path) -> None:
    settings = make_settings()
    switched = with_model(settings, "openai", "gpt-4o-mini")
    assert (switched.llm_provider, switched.openai_model) == ("openai", "gpt-4o-mini")
    assert with_model(settings, None, "qwen2.5").llm_model == "qwen2.5"
    assert with_model(settings, None, None) is settings
    run = tmp_path / "qwen.json"
    run.write_text(
        json.dumps(
            {
                "fingerprint": {"llm_provider": "ollama", "llm_model": "qwen2.5"},
                "metrics": {"fact_hit_rate": 0.75, "uncited_rate": 0.1},
                "examples": [{"seconds": 10.0, "cost_usd": None}, {"seconds": 20.0}],
            }
        )
    )
    table = models_table([run])
    assert "ollama/qwen2.5" in table and "0.75" in table and "15.0" in table


def test_a_run_becomes_its_own_baseline(tmp_path: Path) -> None:
    from eval.harness import main

    run = tmp_path / "run.json"
    run.write_text(json.dumps({"fingerprint": {"a": 1}, "metrics": {"m": 0.5}, "examples": []}))
    out = tmp_path / "baseline.json"
    assert main(["baseline-from", str(run), str(out)]) == 0
    assert json.loads(out.read_text()) == {"fingerprint": {"a": 1}, "metrics": {"m": 0.5}}


def test_a_prompt_with_other_passages_gets_the_reply_for_the_same_question(tmp_path: Path) -> None:
    from eval.replay import loose_key

    def prompt(context: str, question: str = "Leave?") -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are DocRAG."},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nContext:\n{context}\n\nInstructions:\n- Cite."
                ),
            },
        ]

    recording = tmp_path / "r.jsonl"
    recorder = RecordingGenerator(ScriptedGenerator("28 days [1]."), recording)
    recorder.complete(prompt("[1] leave.md p.1 ...\n28 days"), make_settings())  # type: ignore[arg-type]
    assert loose_key(prompt("a")) == loose_key(prompt("b"))  # type: ignore[arg-type]
    replay = ReplayGenerator(recording)
    assert replay.complete(prompt("[1] other order"), make_settings()) == "28 days [1]."  # type: ignore[arg-type]
    assert replay.loose_hits == 1
    with pytest.raises(ReplayMissError):
        replay.complete(prompt("x", question="Parental leave?"), make_settings())  # type: ignore[arg-type]
