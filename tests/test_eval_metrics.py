from __future__ import annotations

from eval.metrics import (
    aggregate,
    compare_to_baseline,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    score_ranking,
)


def test_reciprocal_rank_uses_first_relevant_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1.0 / 3


def test_reciprocal_rank_zero_when_no_relevant() -> None:
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_hit_at_k_respects_cutoff() -> None:
    assert hit_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0


def test_recall_at_k_counts_distinct_found() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.5
    # duplicates in the ranking never inflate recall
    assert recall_at_k(["a", "a", "a"], {"a", "b"}, 3) == 0.5


def test_recall_at_k_zero_when_relevant_empty() -> None:
    assert recall_at_k(["a", "b"], set(), 3) == 0.0


def test_score_ranking_emits_all_requested_keys() -> None:
    scores = score_ranking(["a", "b"], {"a"}, ks=(1, 2))
    assert set(scores) == {"mrr", "hit@1", "recall@1", "hit@2", "recall@2"}
    assert scores["hit@1"] == 1.0
    assert scores["mrr"] == 1.0


def test_aggregate_means_across_rows() -> None:
    rows = [{"mrr": 1.0, "hit@1": 1.0}, {"mrr": 0.0, "hit@1": 0.0}]
    assert aggregate(rows) == {"mrr": 0.5, "hit@1": 0.5}


def test_aggregate_empty_returns_empty() -> None:
    assert aggregate([]) == {}


def test_compare_to_baseline_flags_regression_beyond_tolerance() -> None:
    regressions = compare_to_baseline({"mrr": 0.80}, {"mrr": 0.90}, max_regression=0.05)
    assert len(regressions) == 1
    assert "mrr" in regressions[0]


def test_compare_to_baseline_allows_within_tolerance() -> None:
    assert compare_to_baseline({"mrr": 0.86}, {"mrr": 0.90}, max_regression=0.05) == []


def test_compare_to_baseline_skips_missing_current_keys() -> None:
    assert compare_to_baseline({}, {"mrr": 0.90}, max_regression=0.05) == []
