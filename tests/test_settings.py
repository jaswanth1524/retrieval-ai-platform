from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from api.main import MEASURED_SAFE_RERANKER_BATCH_SIZE, warn_on_risky_reranker_config
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


def test_reranker_batch_size_default_is_within_measured_safe_peak_memory() -> None:
    """The default cross-encoder batch must stay at the measured-safe absolute value.

    Peak reranker memory is a function of reranker_batch_size *alone* (attention grows
    with batch x sequence^2): measured over 50 candidates of ~448 tokens, batch 8 peaked
    at 2.66 GB — the loaded-model baseline — while batch 50 peaked at 6.33 GB and
    OOM-killed (exit 137) the container on a ~6 GB Docker Desktop the first time anyone
    asked a whole-corpus question.

    The `<= rerank_candidates` relation asserted below is a separate and weaker rule
    (above it, batching is merely a no-op) and is deliberately NOT sufficient on its own:
    batch 50 against 50 candidates satisfies it and still peaks at 6.33 GB. The old
    default of 64 against 50 violated both at once, which is what made the relation look
    like the cause. Both are pinned here so neither can drift back.
    """

    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]

    assert settings.reranker_batch_size <= MEASURED_SAFE_RERANKER_BATCH_SIZE
    assert settings.reranker_batch_size <= settings.rerank_candidates
    # rerank_candidates is itself clamped to fused_top_n at the call site, so the batch
    # has to clear that bar too — otherwise a lowered fused_top_n reintroduces the bug.
    assert settings.reranker_batch_size <= settings.fused_top_n


def test_risky_reranker_batch_size_warns_without_refusing_to_boot(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An operator-set batch above the measured-safe peak warns but still constructs.

    Deliberately not a validator: memory headroom is deployment-specific, so a large
    host is entitled to this config. What was missing is the signal — field-by-field the
    settings look reasonable and the only symptom is exit 137 mid-question.
    """

    settings = AppSettings(  # type: ignore[call-arg]
        _env_file=None, reranker_batch_size=32, rerank_candidates=50
    )

    with caplog.at_level(logging.WARNING, logger="api.main"):
        warn_on_risky_reranker_config(settings)

    assert "RERANKER_BATCH_SIZE=32" in caplog.text
    assert "6.33 GB" in caplog.text
    # The relation holds here (32 <= 50), so only the memory warning should fire —
    # this is exactly the config the old relation-only rule called safe.
    assert "no-op" not in caplog.text


def test_batch_size_above_candidates_warns_that_batching_is_a_no_op(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = AppSettings(  # type: ignore[call-arg]
        _env_file=None, reranker_batch_size=64, rerank_candidates=50
    )

    with caplog.at_level(logging.WARNING, logger="api.main"):
        warn_on_risky_reranker_config(settings)

    # The historical incident config trips both rules.
    assert "no-op" in caplog.text
    assert "6.33 GB" in caplog.text


def test_default_reranker_config_logs_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="api.main"):
        warn_on_risky_reranker_config(AppSettings(_env_file=None))  # type: ignore[call-arg]

    assert caplog.text == ""


def test_job_store_backend_defaults_to_memory() -> None:
    assert AppSettings(_env_file=None).job_store_backend == "memory"  # type: ignore[call-arg]


def test_job_store_backend_accepts_sqlite() -> None:
    settings = AppSettings(_env_file=None, job_store_backend="sqlite")  # type: ignore[call-arg]
    assert settings.job_store_backend == "sqlite"


def test_job_store_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, job_store_backend="redis")  # type: ignore[call-arg]
