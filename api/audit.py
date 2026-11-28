"""One log line per action that changes or removes indexed documents.

Logged on ``api.audit`` at INFO; under ``LOG_FORMAT=json`` the fields are an ``audit``
object on the line, next to the request id. A line records which kind of key was used
("full", "read"), its name (``key_name``: a named key of API_KEYS_FILE, else the kind
again) and the client address. A key is still a shared secret, not a person's login:
the name says which key acted, as far as only one person holds it.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.audit_log import AuditLog

audit_logger = logging.getLogger("api.audit")


@dataclass(frozen=True, slots=True)
class Actor:
    # "full" / "read" (the key that was sent), or "open" when no API_KEY is set.
    key: str
    client: str
    # The key's name in API_KEYS_FILE; for API_KEY / API_READ_KEY, "full" / "read".
    name: str = ""


# Set by RequestGuardMiddleware for each guarded request; copied into the threadpool
# with the rest of the request's context. Unset (None) outside a request, i.e. the app's
# own background work such as the folder watcher.
_actor: ContextVar[Actor | None] = ContextVar("docrag_audit_actor", default=None)


def set_actor(key: str, client: str, name: str = "") -> None:
    _actor.set(Actor(key=key, client=client, name=name or key))


def actor_name() -> str:
    """The name of the key behind the current request ("system" outside one)."""

    actor = _actor.get()
    return actor.name if actor is not None else "system"


# AUDIT_LOG_PATH's store, set for the app's lifetime (main.lifespan); None: log only.
_store: AuditLog | None = None


def set_audit_store(store: AuditLog | None) -> None:
    global _store
    _store = store


def audit(event: str, *, workspace: str, **fields: object) -> None:
    """Log ``event`` (e.g. ``document.delete``) with who asked and ``fields``."""

    actor = _actor.get()
    record: dict[str, object] = {
        "event": event,
        # workspace_of gives "" for the default workspace.
        "workspace": workspace or "default",
        "key": actor.key if actor is not None else "system",
        "key_name": actor.name if actor is not None else "system",
        "client": actor.client if actor is not None else None,
        **fields,
    }
    details = " ".join(f"{name}={value!r}" for name, value in record.items() if name != "event")
    audit_logger.info("%s %s", event, details, extra={"audit": record})
    store = _store
    if store is None:
        return
    from api.audit_log import AuditEntry

    try:
        store.add(
            AuditEntry(
                created_at=time.time(),
                workspace=str(record["workspace"]),
                event=event,
                key=str(record["key"]),
                client=actor.client if actor is not None else None,
                key_name=str(record["key_name"]),
                fields=dict(fields),
            )
        )
    except Exception:  # noqa: BLE001 - the action happened; only its record failed
        audit_logger.warning("Could not store the audit event %s.", event, exc_info=True)
