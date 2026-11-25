"""Pass 21: eval fixes."""

from __future__ import annotations

import json
from typing import Any

import pytest


def test_an_unanswerable_question_scoped_to_nothing_counts_as_refused() -> None:
    from eval.answer_eval import AnswerRow, answer_metrics
    from eval.harness import RetrievalExample

    examples = [
        RetrievalExample(q, None, frozenset(), frozenset(), expect_no_context=True)
        for q in ("q1", "q2")
    ]
    rows = [
        AnswerRow("q1", "a", [], None, answer_status="empty_scope"),
        AnswerRow("q2", "a", [], None, answer_status="answered"),
    ]
    assert answer_metrics(examples, rows)["refused_unanswerable_rate"] == 0.5


def test_a_baseline_with_no_fingerprint_is_warned_about(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from eval.harness import _check_answer_baseline

    run = tmp_path / "run.json"
    baseline = tmp_path / "baseline.json"
    run.write_text(json.dumps({"fingerprint": {"question_count": 5}, "metrics": {}}))
    baseline.write_text(json.dumps({"metrics": {}}))
    assert _check_answer_baseline(run, baseline, 0.0) == 0
    assert "records no fingerprint" in capsys.readouterr().err


def test_every_answer_baseline_records_the_history_and_continue_settings() -> None:
    from pathlib import Path

    for path in sorted(Path("eval/baselines").glob("answers_*.json")):
        fingerprint = json.loads(path.read_text(encoding="utf-8"))["fingerprint"]
        assert fingerprint["conversation_max_history_messages"] == 12, path
        assert fingerprint["auto_continue_max"] == 0, path
