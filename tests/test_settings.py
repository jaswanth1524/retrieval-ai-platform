from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.settings import AppSettings


def test_llm_temperature_rejects_below_zero() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, llm_temperature=-0.1)  # type: ignore[call-arg]


def test_llm_temperature_rejects_above_two() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, llm_temperature=2.1)  # type: ignore[call-arg]


def test_llm_temperature_accepts_boundary_values() -> None:
    assert AppSettings(_env_file=None, llm_temperature=0.0).llm_temperature == 0.0  # type: ignore[call-arg]
    assert AppSettings(_env_file=None, llm_temperature=2.0).llm_temperature == 2.0  # type: ignore[call-arg]


def test_chunk_overlap_tokens_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, chunk_overlap_tokens=-1)  # type: ignore[call-arg]


def test_llm_num_retries_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, llm_num_retries=-1)  # type: ignore[call-arg]


def test_llm_request_timeout_seconds_rejects_zero_and_below() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, llm_request_timeout_seconds=0)  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, llm_request_timeout_seconds=-5)  # type: ignore[call-arg]


def test_max_upload_bytes_rejects_non_positive() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, max_upload_bytes=0)  # type: ignore[call-arg]


def test_qdrant_dense_vector_size_rejects_non_positive() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, qdrant_dense_vector_size=0)  # type: ignore[call-arg]


def test_openai_reasoning_effort_normalizes_case_and_whitespace() -> None:
    settings = AppSettings(_env_file=None, openai_reasoning_effort="  LOW ")  # type: ignore[call-arg]
    assert settings.openai_reasoning_effort == "low"


def test_openai_reasoning_effort_accepts_empty_to_disable() -> None:
    assert AppSettings(_env_file=None, openai_reasoning_effort="").openai_reasoning_effort == ""  # type: ignore[call-arg]


def test_openai_reasoning_effort_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, openai_reasoning_effort="turbo")  # type: ignore[call-arg]


def test_reasoning_token_headroom_rejects_non_positive() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, reasoning_token_headroom=0)  # type: ignore[call-arg]


def test_warmup_models_defaults_to_false() -> None:
    assert AppSettings(_env_file=None).warmup_models is False  # type: ignore[call-arg]


def test_conversation_memory_defaults() -> None:
    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]

    assert settings.conversation_condense_enabled is True
    assert settings.conversation_max_history_messages == 12
    assert settings.condense_max_tokens == 128


def test_conversation_max_history_messages_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, conversation_max_history_messages=-1)  # type: ignore[call-arg]


def test_condense_max_tokens_rejects_non_positive() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, condense_max_tokens=0)  # type: ignore[call-arg]


def test_rerank_min_score_rejects_below_zero() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, rerank_min_score=-0.01)  # type: ignore[call-arg]


def test_rerank_min_score_rejects_above_one() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, rerank_min_score=1.01)  # type: ignore[call-arg]


def test_rerank_min_score_accepts_boundary_values() -> None:
    assert AppSettings(_env_file=None, rerank_min_score=0.0).rerank_min_score == 0.0  # type: ignore[call-arg]
    assert AppSettings(_env_file=None, rerank_min_score=1.0).rerank_min_score == 1.0  # type: ignore[call-arg]


def test_csv_rows_per_section_defaults_and_validates() -> None:
    assert AppSettings(_env_file=None).csv_rows_per_section == 50  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, csv_rows_per_section=0)  # type: ignore[call-arg]


def test_query_expansion_count_bounds() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, query_expansion_count=0)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, query_expansion_count=9)  # type: ignore[call-arg]


def test_context_diversity_defaults() -> None:
    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.context_diversity_enabled is True
    assert settings.context_diversity_max_similarity == 0.6


def test_context_diversity_max_similarity_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, context_diversity_max_similarity=0.0)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, context_diversity_max_similarity=1.5)  # type: ignore[call-arg]


def test_min_section_words_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, min_section_words=-1)  # type: ignore[call-arg]


def test_log_level_defaults_to_info() -> None:
    assert AppSettings(_env_file=None).log_level == "INFO"  # type: ignore[call-arg]


def test_log_level_normalizes_case() -> None:
    assert AppSettings(_env_file=None, log_level="debug").log_level == "DEBUG"  # type: ignore[call-arg]


def test_log_level_rejects_unknown_level() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, log_level="verbose")  # type: ignore[call-arg]


def test_app_settings_env_isolation_survives_a_leaked_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for a real bug found this session: litellm's import-time
    ``dotenv.load_dotenv()`` call copies ``.env`` values into the real process
    ``os.environ`` — ``_env_file=None`` alone does NOT protect against this, since
    pydantic-settings always reads real environment variables regardless of the
    ``env_file`` setting. ``tests/conftest.py``'s autouse fixture clears these before
    every test; this test documents and exercises the exact mechanism directly.
    """

    monkeypatch.setenv("OPENAI_API_KEY", "leaked-from-somewhere")
    # The leak IS visible to a bare AppSettings() call...
    assert AppSettings().openai_api_key == "leaked-from-somewhere"  # noqa: S106

    monkeypatch.delenv("OPENAI_API_KEY")
    # ...but once cleared (what the autouse conftest fixture does before every test),
    # it's gone — regardless of what the developer's real .env file contains.
    assert AppSettings(_env_file=None).openai_api_key is None  # type: ignore[call-arg]


def test_reranker_batch_size_actually_batches_the_default_candidate_set() -> None:
    """The cross-encoder batch must not exceed the candidates it will be given.

    Not a check that the value is any particular number — it's the relationship that
    matters. reranker_batch_size is the only thing bounding the reranker's peak memory
    (attention grows with batch x sequence^2), but it can only bound anything if it is
    smaller than the number of pairs being scored. The old default of 64 against 50
    candidates meant every pair went through in a single forward pass: 6.33 GB peak,
    which OOM-killed (exit 137) the container on a ~6 GB Docker Desktop the first time
    anyone asked a whole-corpus question. Raising it back above rerank_candidates would
    silently restore one-shot scoring, so pin the invariant rather than the constant.
    """

    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]

    assert settings.reranker_batch_size <= settings.rerank_candidates
    # rerank_candidates is itself clamped to fused_top_n at the call site, so the batch
    # has to clear that bar too — otherwise a lowered fused_top_n reintroduces the bug.
    assert settings.reranker_batch_size <= settings.fused_top_n
