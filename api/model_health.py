"""Whether the models loaded at boot, for /health/ready.

The startup warmup (``api.main.run_model_warmup``) never raises: a model that fails to
load only resurfaces as an error on every question. Readiness used to report "ok"
regardless, so a pod whose reranker never loaded stayed in rotation. This records each
model's warmup result and, when one failed, warms it again in the background — at most
once per ``retry_seconds`` and only when readiness is asked — so a transient download
failure clears without a restart.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Literal

logger = logging.getLogger(__name__)

ModelState = Literal["ok", "failed", "fallback"]

WARMUP_RETRY_SECONDS = 60.0

# A failed tokenizer only means chunking counts tokens with the heuristic — the designed
# fallback, retried on its own — so it is reported but never makes readiness degraded.
_NOT_DEGRADING = frozenset({"tokenizer"})

Rewarm = Callable[[frozenset[str]], Mapping[str, ModelState]]


class ModelHealth:
    """Thread-safe record of each warmed model's state, plus the background retry."""

    def __init__(
        self,
        rewarm: Rewarm | None = None,
        *,
        retry_seconds: float = WARMUP_RETRY_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rewarm = rewarm
        self._retry_seconds = retry_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[str, ModelState] = {}
        self._last_retry: float | None = None
        self._retrying = False

    def record(self, results: Mapping[str, ModelState]) -> None:
        with self._lock:
            self._states.update(results)

    def states(self) -> dict[str, ModelState]:
        with self._lock:
            return dict(self._states)

    def degraded(self) -> bool:
        with self._lock:
            return any(
                state == "failed" and name not in _NOT_DEGRADING
                for name, state in self._states.items()
            )

    def retry_failed(self) -> threading.Thread | None:
        """Start warming the failed models again, unless one is running or the last
        attempt was under ``retry_seconds`` ago; the thread started, if any."""

        with self._lock:
            failed = frozenset(
                name
                for name, state in self._states.items()
                if state == "failed" and name not in _NOT_DEGRADING
            )
            if not failed or self._rewarm is None or self._retrying:
                return None
            now = self._clock()
            if self._last_retry is not None and now - self._last_retry < self._retry_seconds:
                return None
            self._retrying = True
            self._last_retry = now
        thread = threading.Thread(
            target=self._run, args=(failed,), name="docrag-rewarm", daemon=True
        )
        thread.start()
        return thread

    def _run(self, names: frozenset[str]) -> None:
        assert self._rewarm is not None
        results: Mapping[str, ModelState] = {}
        try:
            results = self._rewarm(names)
        except Exception:  # noqa: BLE001 - a retry failing leaves the state as it was
            logger.exception("Retrying the model warmup failed.")
        finally:
            with self._lock:
                self._retrying = False
                self._states.update(results)
        recovered = sorted(name for name, state in results.items() if state == "ok")
        if recovered:
            logger.info("Model warmup retry succeeded for: %s", ", ".join(recovered))
