"""A process-wide counter bumped whenever the indexed documents change.

Anything derived from the corpus — cached answers (``api.answer_cache``), the cached
document listing (``api.repository``) — records the generation it was built from and
is discarded once the counter moves: every ingest (even a failed one, which may have
written points), delete and tag change bumps it. Per process: another API process's
writes don't move it, so those caches also expire on a timer.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from threading import Lock

_lock = Lock()
_generation = 0
# Which documents each recent bump changed (None: not said, so anything may have), so
# the document listing can re-read just those instead of the whole collection.
_CHANGE_LOG_SIZE = 256
_changes: deque[tuple[int, frozenset[str] | None]] = deque(maxlen=_CHANGE_LOG_SIZE)


def corpus_generation() -> int:
    with _lock:
        return _generation


def bump_corpus_generation(filenames: Iterable[str] | None = None) -> None:
    """Invalidate everything derived from the corpus: the indexed documents changed.

    ``filenames`` names the documents that changed, when the caller knows; a bare call
    means anything may have.
    """

    global _generation
    with _lock:
        _generation += 1
        _changes.append((_generation, frozenset(filenames) if filenames is not None else None))


def changed_since(generation: int) -> frozenset[str] | None:
    """The documents changed after ``generation``, or None when that isn't known (a bump
    that named nothing, or one old enough to have left the log)."""

    with _lock:
        if generation == _generation:
            return frozenset()
        if generation > _generation or not _changes or _changes[0][0] > generation + 1:
            return None
        changed: set[str] = set()
        for bumped, names in _changes:
            if bumped <= generation:
                continue
            if names is None:
                return None
            changed |= names
        return frozenset(changed)
