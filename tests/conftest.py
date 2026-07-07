"""Shared test fixtures.

Guards test isolation against a real developer ``.env``. Two separate leak paths exist:

1. ``AppSettings`` itself reads ``.env`` (see ``api/settings.py``'s ``env_file=".env"``) —
   every ``make_settings()`` helper already passes ``_env_file=None`` to disable this.
2. ``litellm`` calls ``dotenv.load_dotenv()`` at import time (``litellm/__init__.py``),
   which copies real ``.env`` values into the actual process ``os.environ`` as a side
   effect. ``_env_file=None`` does NOT protect against this — pydantic-settings always
   reads real environment variables regardless of the ``env_file`` setting, and by the
   time any test runs, ``api.main`` (imported transitively by most test modules) has
   already imported ``litellm``, which has already polluted ``os.environ``.

This autouse fixture clears every environment variable ``AppSettings`` could read before
each test, so neither leak path can reach a test's settings even if litellm (or anything
else) has already copied ``.env`` into the process environment.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from api.settings import AppSettings

_SETTINGS_ENV_VARS = tuple(name.upper() for name in AppSettings.model_fields)


@pytest.fixture(autouse=True)
def _isolate_app_settings_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Clear AppSettings-relevant environment variables for the duration of each test."""

    for env_var in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    yield
