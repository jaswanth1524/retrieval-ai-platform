"""``.env.example`` must not silently change behaviour.

Docker Compose loads ``.env.example`` as the API's ``env_file``, so every active value in
it is a real setting in the default deployment. A value that drifts from its
``AppSettings`` default (the query instruction once lost its trailing space this way)
changes behaviour for every Docker user without anyone choosing it.
"""

from __future__ import annotations

import re
from pathlib import Path

from api.settings import AppSettings

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"

# Values that are deliberately different in Docker: service hostnames inside the
# compose network.
DOCKER_ONLY = {"qdrant_url", "ollama_base_url"}


def test_env_example_values_match_the_settings_defaults() -> None:
    from_file = AppSettings(_env_file=ENV_EXAMPLE).model_dump()  # type: ignore[call-arg]
    defaults = AppSettings(_env_file=None).model_dump()  # type: ignore[call-arg]

    drifted = {
        name: (from_file[name], defaults[name])
        for name in defaults
        if name not in DOCKER_ONLY
        # An empty line (KEY=) means "unset", same as a None default.
        and not (from_file[name] in ("", None) and defaults[name] in ("", None))
        and from_file[name] != defaults[name]
    }
    assert drifted == {}


def test_env_example_values_are_unquoted_and_unpadded() -> None:
    # Compose and python-dotenv disagree on quotes and surrounding whitespace, so a value
    # that needs either can't be written the same way for both; leave it commented out.
    problems = []
    for number, line in enumerate(ENV_EXAMPLE.read_text().splitlines(), start=1):
        match = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line)
        if not match:
            continue
        value = match.group(2)
        if value != value.strip() or value[:1] in ("'", '"'):
            problems.append(f"line {number}: {line!r}")
    assert problems == []
