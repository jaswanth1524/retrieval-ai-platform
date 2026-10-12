"""Per-client rate limiting of requests that change things (``RATE_LIMIT_PER_MINUTE``).

Questions were already bounded by concurrency (``api/admission.py``), but nothing else
was: a script could post feedback, run passage searches or guess API keys as fast as
the server answered. Off by default, since a self-hosted instance usually has one
trusted user; on, each client gets a token bucket of ``per_minute`` requests that
refills continuously, and the request guard answers 429 + ``Retry-After`` when it is
empty.

A client is its address as the ASGI server reports it. Behind a reverse proxy, run
uvicorn with ``--forwarded-allow-ips`` set to the proxy so that is the real client and
not the proxy itself (otherwise every user shares one bucket).
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable

# Clients remembered at once. Past it the least recently seen is forgotten, which only
# ever gives that client a full bucket back: memory stays bounded under a flood of
# addresses.
MAX_TRACKED_CLIENTS = 10_000


class RateLimiter:
    """A token bucket per client: ``per_minute`` tokens, refilled evenly over a minute."""

    def __init__(
        self,
        per_minute: int,
        *,
        max_clients: int = MAX_TRACKED_CLIENTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = float(per_minute)
        self._refill_per_second = per_minute / 60.0
        self._max_clients = max_clients
        self._clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def take(self, client: str) -> float:
        """Spend one token for ``client``: 0.0 when allowed, else the seconds until one
        is available. Called only from the event loop, so no lock is needed."""

        now = self._clock()
        tokens, last = self._buckets.pop(client, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - last) * self._refill_per_second)
        if tokens >= 1.0:
            self._buckets[client] = (tokens - 1.0, now)
            wait = 0.0
        else:
            self._buckets[client] = (tokens, now)
            wait = (1.0 - tokens) / self._refill_per_second
        while len(self._buckets) > self._max_clients:
            self._buckets.popitem(last=False)
        return wait


def retry_after_seconds(wait: float) -> int:
    """A ``Retry-After`` value: whole seconds, at least one."""

    return max(1, math.ceil(wait))
