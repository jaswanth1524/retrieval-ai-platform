"""A process-wide counter bumped whenever the indexed documents change.

Anything derived from the corpus — cached answers (``api.answer_cache``), the cached
document listing (``api.repository``) — records the generation it was built from and
is discarded once the counter moves: every ingest (even a failed one, which may have
written points), delete and tag change bumps it. Per process: another API process's
writes don't move it, so those caches also expire on a timer.
"""

from __future__ import annotations

from threading import Lock

_lock = Lock()
_generation = 0


def corpus_generation() -> int:
    with _lock:
        return _generation


def bump_corpus_generation() -> None:
    """Invalidate everything derived from the corpus: the indexed documents changed."""

    global _generation
    with _lock:
        _generation += 1
