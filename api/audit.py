"""One log line per action that changes or removes indexed documents.

Logged on ``api.audit`` at INFO; under ``LOG_FORMAT=json`` the fields are an ``audit``
object on the line, next to the request id. Not per-user: DocRAG has one shared key
(two with ``API_READ_KEY``), so a line records which *kind* of key was used and the
client address — enough to tell an operator's action from the folder watcher's or an
unexpected caller's, not who it was.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass

audit_logger = logging.getLogger("api.audit")


@dataclass(frozen=True, slots=True)
class Actor:
    # "full" / "read" (the key that was sent), or "open" when no API_KEY is set.
    key: str
    client: str


# Set by RequestGuardMiddleware for each guarded request; copied into the threadpool
# with the rest of the request's context. Unset (None) outside a request, i.e. the app's
# own background work such as the folder watcher.
_actor: ContextVar[Actor | None] = ContextVar("docrag_audit_actor", default=None)


def set_actor(key: str, client: str) -> None:
    _actor.set(Actor(key=key, client=client))


def audit(event: str, *, workspace: str, **fields: object) -> None:
    """Log ``event`` (e.g. ``document.delete``) with who asked and ``fields``."""

    actor = _actor.get()
    record: dict[str, object] = {
        "event": event,
        # workspace_of gives "" for the default workspace.
        "workspace": workspace or "default",
        "key": actor.key if actor is not None else "system",
        "client": actor.client if actor is not None else None,
        **fields,
    }
    details = " ".join(f"{name}={value!r}" for name, value in record.items() if name != "event")
    audit_logger.info("%s %s", event, details, extra={"audit": record})
