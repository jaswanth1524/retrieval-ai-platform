"""Pass 22: eval fixes."""

from __future__ import annotations

import json
from pathlib import Path


def test_errored_rows_dont_count_towards_the_answer_quality_means() -> None:
    from eval.answer_eval import AnswerRow, answer_metrics
    from eval.harness import RetrievalExample

    examples = [RetrievalExample(q, None, frozenset(), frozenset()) for q in ("q1", "q2")]
    rows = [
        AnswerRow("q1", "a", [], None, answer_status="answered", unsupported_claims=2),
        AnswerRow("q2", "", [], None, answer_status="error", error="provider down"),
    ]
    metrics = answer_metrics(examples, rows)
    assert metrics["unsupported_claims_mean"] == 2.0
    assert metrics["error_rate"] == 0.5


def test_the_fingerprint_records_query_decomposition() -> None:
    from eval.harness import settings_fingerprint
    from tests.factories import make_test_settings

    fingerprint = settings_fingerprint(
        make_test_settings(query_decomposition_enabled=True), "reranked"
    )
    assert fingerprint["query_decomposition_enabled"] is True
    assert fingerprint["query_decomposition_max_parts"] == 3
    for path in Path("eval/baselines").glob("*.json"):
        recorded = json.loads(path.read_text())["fingerprint"]
        assert recorded["query_decomposition_enabled"] is False, path


def test_a_paired_bootstrap_tells_one_question_from_a_real_drop() -> None:
    from eval.stats import answer_noise, paired_bootstrap

    same = [1.0] * 11 + [0.0]
    one_flip = [1.0] * 10 + [0.0, 0.0]
    change, low, high = paired_bootstrap(one_flip, same)
    assert round(change, 3) == -0.083 and low <= 0 <= high
    change, low, high = paired_bootstrap([0.0] * 12, [1.0] * 12)
    assert change == -1.0 and high < 0

    rows = [{"question": f"q{i}", "answer_status": "answered", "fact_hit": True} for i in range(6)]
    worse = [dict(r) for r in rows]
    worse[0]["fact_hit"] = False
    noise = answer_noise(worse, rows)
    assert noise["fact_hit_rate"][1] <= 0 <= noise["fact_hit_rate"][2]


def test_answer_metrics_are_reported_per_kind_of_question() -> None:
    from eval.answer_eval import AnswerRow, metrics_by_category, question_category
    from eval.harness import RetrievalExample

    examples = [
        RetrievalExample("q1", None, frozenset({"a.md"}), frozenset()),
        RetrievalExample("q2", None, frozenset(), frozenset(), expect_no_context=True),
    ]
    rows = [
        AnswerRow("q1", "a", ["a.md"], None, answer_status="answered"),
        AnswerRow("q2", "no", [], None, answer_status="no_context"),
    ]
    assert [question_category(e) for e in examples] == ["single", "unanswerable"]
    by_category = metrics_by_category(examples, rows)
    assert by_category["single"]["answered_rate"] == 1.0
    assert by_category["unanswerable"]["refused_unanswerable_rate"] == 1.0
    assert metrics_by_category(examples[:1], rows[:1]) == {}


def test_the_decomposed_arm_needs_an_llm_and_is_fingerprinted() -> None:
    import pytest

    from eval.harness import ARMS, rank_for_question, settings_fingerprint
    from tests.factories import make_test_settings

    assert "decomposed" in ARMS
    settings = make_test_settings()
    assert settings_fingerprint(settings, "decomposed")["query_decomposition_enabled"] is True
    with pytest.raises(RuntimeError, match="needs an LLM"):
        rank_for_question(
            "a and b?",
            repository=None,
            embedding_provider=None,  # type: ignore[arg-type]
            reranker=None,  # type: ignore[arg-type]
            settings=settings,
            arm="decomposed",
        )
