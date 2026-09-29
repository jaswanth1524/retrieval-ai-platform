from __future__ import annotations

from typing import Any

import httpx
import pytest
from qdrant_client import QdrantClient

from api.provider_health import check_ollama_reachable, check_qdrant_reachable
from api.settings import AppSettings


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {"ollama_base_url": "http://localhost:11434"}
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


def test_check_ollama_reachable_true_on_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, timeout: float) -> httpx.Response:
        return httpx.Response(status_code=200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)

    assert check_ollama_reachable(make_settings()) is True


def test_check_ollama_reachable_false_on_non_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, timeout: float) -> httpx.Response:
        return httpx.Response(status_code=500, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)

    assert check_ollama_reachable(make_settings()) is False


def test_check_ollama_reachable_false_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, timeout: float) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    assert check_ollama_reachable(make_settings()) is False


def test_check_ollama_reachable_false_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, timeout: float) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "get", fake_get)

    assert check_ollama_reachable(make_settings()) is False


def test_check_ollama_reachable_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_urls: list[str] = []

    def fake_get(url: str, timeout: float) -> httpx.Response:
        seen_urls.append(url)
        return httpx.Response(status_code=200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)

    check_ollama_reachable(make_settings(ollama_base_url="http://localhost:11434/"))

    assert seen_urls == ["http://localhost:11434/api/tags"]


class RaisingQdrantClient:
    def get_collections(self) -> None:
        raise RuntimeError("connection refused")


def test_check_qdrant_reachable_true_for_a_live_client() -> None:
    client = QdrantClient(":memory:")

    assert check_qdrant_reachable(client) is True


def test_check_qdrant_reachable_false_on_any_exception() -> None:
    assert check_qdrant_reachable(RaisingQdrantClient()) is False  # type: ignore[arg-type]


def test_cached_ollama_check_probes_once_per_ttl_per_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """/config is unauthenticated: each call used to send its own outbound probe (up to a
    1.5s block when Ollama is down)."""

    import api.provider_health as provider_health

    provider_health.clear_ollama_reachability_cache()
    probed: list[str] = []

    def fake_get(url: str, timeout: float) -> httpx.Response:
        probed.append(url)
        return httpx.Response(status_code=200, request=httpx.Request("GET", url))

    now = [1000.0]
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(provider_health.time, "monotonic", lambda: now[0])
    check = provider_health.check_ollama_reachable_cached

    assert check(make_settings()) is True
    assert check(make_settings()) is True
    assert len(probed) == 1
    # A different base URL is its own entry.
    assert check(make_settings(ollama_base_url="http://other:11434")) is True
    assert len(probed) == 2
    # Past the TTL, Ollama is probed live again.
    now[0] += provider_health.OLLAMA_REACHABILITY_TTL_SECONDS + 0.1
    assert check(make_settings()) is True
    assert len(probed) == 3

    provider_health.clear_ollama_reachability_cache()
    assert check(make_settings()) is True
    assert len(probed) == 4
    provider_health.clear_ollama_reachability_cache()
