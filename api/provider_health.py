"""Lightweight reachability checks for generation providers and the vector store."""

from __future__ import annotations

import time
from threading import Lock

import httpx
from qdrant_client import QdrantClient

from api.settings import AppSettings

OLLAMA_REACHABILITY_TIMEOUT_SECONDS = 1.5
# /config is unauthenticated and the UI calls it on every boot; without a cache each
# call was an outbound probe that could block for the full timeout when Ollama is down.
# Short enough that starting or stopping Ollama shows up within one UI refresh.
OLLAMA_REACHABILITY_TTL_SECONDS = 10.0

_ollama_cache: dict[str, tuple[float, bool]] = {}
# Held across the probe itself: concurrent misses for one URL wait for a single probe
# instead of each sending their own.
_ollama_cache_lock = Lock()


def clear_ollama_reachability_cache() -> None:
    """Forget cached probe results (tests, dependency reloads)."""

    with _ollama_cache_lock:
        _ollama_cache.clear()


def check_ollama_reachable_cached(settings: AppSettings) -> bool:
    """``check_ollama_reachable``, remembered per base URL for a few seconds."""

    base_url = settings.ollama_base_url
    with _ollama_cache_lock:
        cached = _ollama_cache.get(base_url)
        now = time.monotonic()
        if cached is not None and now - cached[0] < OLLAMA_REACHABILITY_TTL_SECONDS:
            return cached[1]
        reachable = check_ollama_reachable(settings)
        _ollama_cache[base_url] = (time.monotonic(), reachable)
        return reachable


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


def check_qdrant_reachable(client: QdrantClient) -> bool:
    """Return True only if Qdrant answers a cheap metadata call.

    Same never-raises contract as ``check_ollama_reachable`` — a readiness probe
    must never itself become a source of 500s.
    """

    try:
        client.get_collections()
    except Exception:
        return False
    return True
