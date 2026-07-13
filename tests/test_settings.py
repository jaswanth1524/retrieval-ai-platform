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


def test_warmup_models_defaults_to_false() -> None:
    assert AppSettings(_env_file=None).warmup_models is False  # type: ignore[call-arg]


def test_rerank_min_score_rejects_below_zero() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, rerank_min_score=-0.01)  # type: ignore[call-arg]


def test_rerank_min_score_rejects_above_one() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, rerank_min_score=1.01)  # type: ignore[call-arg]


def test_rerank_min_score_accepts_boundary_values() -> None:
    assert AppSettings(_env_file=None, rerank_min_score=0.0).rerank_min_score == 0.0  # type: ignore[call-arg]
    assert AppSettings(_env_file=None, rerank_min_score=1.0).rerank_min_score == 1.0  # type: ignore[call-arg]


def test_min_section_words_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, min_section_words=-1)  # type: ignore[call-arg]


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
