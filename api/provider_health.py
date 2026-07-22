"""Lightweight reachability checks for generation providers and the vector store."""

from __future__ import annotations

import httpx
from qdrant_client import QdrantClient

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
