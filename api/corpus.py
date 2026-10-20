"""Counters bumped whenever a collection's indexed documents change.

Anything derived from a corpus — cached answers (``api.answer_cache``), the cached
document listing (``api.repository``) — records the generation it was built from and
is discarded once that collection's counter moves: every ingest that wrote, delete and
tag change bumps it. One counter per Qdrant collection, i.e. per workspace: an upload
to one workspace used to empty every workspace's cached answers. Per process: another
API process's writes don't move it, so those caches also expire on a timer.

Every value comes from one process-wide sequence, so a collection that is deleted and
created again never sees a generation it had before (a cached entry from the old one
can't match). A bump that names no collection (``collection=None``) moves them all.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from threading import Lock

_lock = Lock()
# The last value handed out, by any collection.
_sequence = 0
# Each collection's current generation; one never bumped is at _floor.
_current: dict[str, int] = {}
# The last bump that named no collection: every collection is at least here.
_floor = 0
# Which documents each recent bump changed, and in which collection (None: not said,
# so anything may have), so the document listing can re-read just those instead of the
# whole collection.
_CHANGE_LOG_SIZE = 256
_changes: deque[tuple[int, str | None, frozenset[str] | None]] = deque(maxlen=_CHANGE_LOG_SIZE)


def _generation_of(collection: str | None) -> int:
    if collection is None:
        return _sequence
    return max(_current.get(collection, 0), _floor)


def corpus_generation(collection: str | None = None) -> int:
    """``collection``'s generation; with None, the latest of any (moves on every bump)."""

    with _lock:
        return _generation_of(collection)


def bump_corpus_generation(
    filenames: Iterable[str] | None = None, *, collection: str | None = None
) -> None:
    """Invalidate everything derived from ``collection`` (every collection when None):
    its indexed documents changed.

    ``filenames`` names the documents that changed, when the caller knows; None means
    anything may have.
    """

    global _sequence, _floor
    with _lock:
        _sequence += 1
        if collection is None:
            _floor = _sequence
        else:
            _current[collection] = _sequence
        names = frozenset(filenames) if filenames is not None else None
        _changes.append((_sequence, collection, names))


def changed_since(generation: int, collection: str | None = None) -> frozenset[str] | None:
    """The documents of ``collection`` changed after ``generation``, or None when that
    isn't known (a bump that named nothing, or one old enough to have left the log)."""

    with _lock:
        current = _generation_of(collection)
        if generation == current:
            return frozenset()
        if generation > current or not _changes or _changes[0][0] > generation + 1:
            return None
        changed: set[str] = set()
        for bumped, bumped_collection, names in _changes:
            if bumped <= generation:
                continue
            if collection is not None and bumped_collection not in (None, collection):
                continue
            if names is None:
                return None
            changed |= names
        return frozenset(changed)
