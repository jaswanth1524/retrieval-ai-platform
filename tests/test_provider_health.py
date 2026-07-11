from __future__ import annotations

from typing import Any

import httpx
import pytest

from api.provider_health import check_ollama_reachable
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
