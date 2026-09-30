"""The app's version, from ``pyproject.toml`` — the one place it is written down.

``[tool.uv] package = false`` means the project is never installed, so
``importlib.metadata`` has nothing to report; the Docker image copies
``pyproject.toml`` next to ``api/`` exactly as the repository has it.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@lru_cache
def app_version() -> str:
    try:
        with _PYPROJECT.open("rb") as handle:
            version = tomllib.load(handle).get("project", {}).get("version")
    except (OSError, tomllib.TOMLDecodeError):
        return "0.0.0"
    return version if isinstance(version, str) else "0.0.0"
