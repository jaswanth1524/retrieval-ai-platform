"""Settings builders and polling shared by the test modules.

Each module keeps its own ``make_settings`` defaults (they differ on purpose: chunk
sizes, limits, collection names); what they share is how settings are built and the
in-memory Qdrant shape most of them index into.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from api.settings import AppSettings


def make_test_settings(**values: Any) -> AppSettings:
    """``AppSettings`` from ``values`` and the field defaults only, never ``.env``.

    ``tests/conftest.py`` clears the environment variables; this keeps the developer's
    own ``.env`` file out as well.
    """

    return AppSettings(_env_file=None, **values)  # type: ignore[call-arg]


def in_memory_qdrant(collection: str) -> dict[str, Any]:
    """Settings for an in-process ``:memory:`` collection of 3-dimensional test vectors."""

    return {
        "qdrant_url": ":memory:",
        "qdrant_collection": collection,
        "qdrant_dense_vector_name": "dense",
        "qdrant_sparse_vector_name": "sparse",
        "qdrant_dense_vector_size": 3,
        "embedding_model_tag": "test-embedding:v1",
    }


def wait_until[T](
    probe: Callable[[], T | None],
    *,
    timeout: float = 10.0,
    message: str = "condition never became true",
) -> T:
    """Poll ``probe`` until it returns something truthy, and return that.

    Bounded by wall-clock time, not by a count of iterations: a loop of N x 10 ms
    sleeps gave up after a fraction of a second on a loaded CI runner, while a fast
    machine never needed more than a few tries.
    """

    deadline = time.monotonic() + timeout
    while True:
        value = probe()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(message)
        time.sleep(0.01)
