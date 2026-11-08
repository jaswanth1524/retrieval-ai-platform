"""The wait before trying something again, doubled each time it failed: one rule for the
folder watcher's retries, the stale re-indexer's and the URL refresher's."""

from __future__ import annotations

# Doublings past this change nothing (every cap is reached long before) and would only
# make an overflowing float of a long failure streak.
_MAX_DOUBLINGS = 30


def doubled_wait(base: float, doublings: int, cap: float) -> float:
    """``base`` doubled ``doublings`` times, at most ``cap``."""

    factor: float = 2.0 ** min(max(doublings, 0), _MAX_DOUBLINGS)
    return min(base * factor, cap)
