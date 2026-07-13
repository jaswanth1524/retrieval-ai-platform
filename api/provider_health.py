"""Lightweight reachability checks for generation providers."""

from __future__ import annotations

import httpx

from api.settings import AppSettings

OLLAMA_REACHABILITY_TIMEOUT_SECONDS = 1.5


def check_ollama_reachable(settings: AppSettings) -> bool:
    """Return True only if Ollama responds successfully at ``ollama_base_url``.

    Best-effort UI hint, never raises — any connection failure, timeout, or non-2xx
    status is treated as unreachable.
    """

    try:
        response = httpx.get(
            f"{settings.ollama_base_url.rstrip('/')}/api/tags",
            timeout=OLLAMA_REACHABILITY_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError:
        return False
    return response.is_success
