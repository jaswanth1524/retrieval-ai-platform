"""RATE_LIMIT_PER_MINUTE: the token bucket itself (api/rate_limit.py)."""

from __future__ import annotations

from api.rate_limit import RateLimiter, retry_after_seconds


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_client_gets_a_burst_of_the_minute_then_waits_for_refill() -> None:
    clock = Clock()
    limiter = RateLimiter(3, clock=clock)

    assert [limiter.take("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    wait = limiter.take("a")
    assert 19.9 < wait <= 20.0  # one token per 20 s at 3/minute
    assert limiter.take("b") == 0.0  # each client has its own bucket

    clock.now += 20
    assert limiter.take("a") == 0.0
    assert limiter.take("a") > 0
    clock.now += 600  # refills only up to the burst
    assert [limiter.take("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.take("a") > 0


def test_refused_requests_spend_nothing() -> None:
    clock = Clock()
    limiter = RateLimiter(1, clock=clock)
    assert limiter.take("a") == 0.0
    for _ in range(5):
        assert limiter.take("a") > 0
    clock.now += 60
    assert limiter.take("a") == 0.0


def test_memory_is_bounded_by_forgetting_the_least_recent_client() -> None:
    clock = Clock()
    limiter = RateLimiter(1, max_clients=2, clock=clock)
    limiter.take("a")
    limiter.take("b")
    limiter.take("c")  # "a" is forgotten: it gets a full bucket back, nothing worse

    assert limiter.take("a") == 0.0
    assert limiter.take("c") > 0


def test_retry_after_is_whole_seconds_and_at_least_one() -> None:
    assert retry_after_seconds(0.2) == 1
    assert retry_after_seconds(19.01) == 20
