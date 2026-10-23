"""Shared test fixtures.

Guards test isolation against a real developer ``.env``. Two separate leak paths exist:

1. ``AppSettings`` itself reads ``.env`` (see ``api/settings.py``'s ``env_file=".env"``) —
   ``tests/factories.py:make_test_settings`` (behind every ``make_settings()`` helper)
   passes ``_env_file=None`` to disable this.
2. ``litellm`` calls ``dotenv.load_dotenv()`` at import time (``litellm/__init__.py``),
   which copies real ``.env`` values into the actual process ``os.environ`` as a side
   effect. ``_env_file=None`` does NOT protect against this — pydantic-settings always
   reads real environment variables regardless of the ``env_file`` setting, and by the
   time any test runs, ``api.main`` (imported transitively by most test modules) has
   already imported ``litellm``, which has already polluted ``os.environ``.

This autouse fixture clears every environment variable ``AppSettings`` could read before
each test, so neither leak path can reach a test's settings even if litellm (or anything
else) has already copied ``.env`` into the process environment.

A third path: ``api.main`` (``create_app``, the lifespan) calls the cached
``get_app_settings()`` directly, which dependency overrides don't reach, and that plain
``AppSettings()`` read the developer's ``.env`` (a ``WARMUP_MODELS=true`` there loaded
the real models in every HTTP test). ``.env`` is switched off for every instance here.

LiteLLM also fetches its model price map from GitHub at import unless
``LITELLM_LOCAL_MODEL_COST_MAP`` is set: tests then depended on the network and on
that day's remote file. It is set before anything imports litellm.
"""

from __future__ import annotations

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import logging  # noqa: E402 - after the environment above, before litellm loads
from collections.abc import Generator  # noqa: E402

import pytest  # noqa: E402

from api.dependencies import get_app_settings  # noqa: E402
from api.settings import AppSettings  # noqa: E402

_SETTINGS_ENV_VARS = tuple(name.upper() for name in AppSettings.model_fields)


@pytest.fixture(autouse=True)
def _isolate_app_settings_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Clear AppSettings-relevant environment variables for the duration of each test."""

    for env_var in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    # Parse in the test process: tests patch the parsers (api.parsers.pdf.PdfReader...),
    # which a child process would not see. tests/test_parse_isolation.py covers the
    # child-process path with its own settings.
    monkeypatch.setenv("PARSE_TIMEOUT_SECONDS", "0")
    yield


@pytest.fixture(autouse=True)
def _settings_without_dotenv(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Make every ``AppSettings()`` ignore ``.env``, not only ``make_test_settings``.

    pydantic-settings reads ``model_config["env_file"]`` at construction, so this
    covers the cached ``get_app_settings()`` that ``create_app`` and the lifespan call.
    """

    monkeypatch.setitem(AppSettings.model_config, "env_file", None)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_llm_process_state() -> Generator[None]:
    """Each test starts with no measured prompt-read rate and free Ollama slots.

    Both are process-wide in ``api.llm``: a test that left a stream unclosed would
    otherwise hold the slot every later Ollama test waits on.
    """

    from api import citations, llm

    def reset() -> None:
        llm._slots.clear()
        llm._measured_rates.clear()
        llm._slow_samples.clear()
        # Memoized span matching: results depend on constants tests patch.
        citations._span_index.cache_clear()
        citations._backfill.cache_clear()
        llm._held.__dict__.clear()  # the slots this (the test's) thread holds

    reset()
    yield
    reset()


@pytest.fixture(autouse=True)
def _drop_log_handlers_added_by_the_test() -> Generator[None]:
    """Remove the ``api``/``eval`` log handlers a test attached, when it ends.

    ``create_app`` configures logging, which binds a StreamHandler to the ``sys.stderr``
    of that moment — pytest's capture stream for the running test. Left in place, every
    later test that logged wrote to that closed stream ("Logging error ... I/O operation
    on closed file" in the output).
    """

    loggers = [logging.getLogger(name) for name in ("api", "eval")]
    before = {logger.name: list(logger.handlers) for logger in loggers}
    yield
    for logger in loggers:
        for handler in logger.handlers[:]:
            if handler not in before[logger.name]:
                logger.removeHandler(handler)
                handler.close()
