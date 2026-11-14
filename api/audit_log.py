"""The audit events (``api.audit``) kept in sqlite (AUDIT_LOG_PATH, off by default), so
they outlive log rotation: ``GET /audit`` and ``docrag audit`` read them back.

The newest AUDIT_LOG_MAX_ROWS are kept. Not in backups, and not removed with a
workspace — the record of a deletion has to survive it. A failed write is logged and
never fails the action it records.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from api.sqlite_store import CappedRowLog

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit (
    created_at REAL NOT NULL,
    workspace TEXT NOT NULL,
    event TEXT NOT NULL,
    key TEXT NOT NULL,
    client TEXT,
    fields_json TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class AuditEntry:
    created_at: float
    workspace: str
    event: str
    # The kind of key used ("full", "read", "open") or "system" (background work).
    key: str
    client: str | None
    fields: dict[str, Any]


class AuditLog(CappedRowLog):
    _table = "audit"

    def __init__(self, path: str, max_rows: int) -> None:
        super().__init__(path, max_rows)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS audit_workspace_created_at "
                "ON audit (workspace, created_at)"
            )

    def add(self, entry: AuditEntry) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO audit (created_at, workspace, event, key, client, fields_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.created_at,
                    entry.workspace,
                    entry.event,
                    entry.key,
                    entry.client,
                    json.dumps(entry.fields, default=str, ensure_ascii=False),
                ),
            )
            self._added_one(conn)

    def recent(
        self,
        workspace: str | None,
        *,
        event: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Newest first; only ``workspace``'s (None: every workspace), ``event``'s
        (an exact name, or a prefix ending in "." such as "document."), since ``since``."""

        where: list[str] = []
        params: list[Any] = []
        if workspace is not None:
            where.append("workspace = ?")
            params.append(workspace)
        if event:
            if event.endswith("."):
                where.append("event LIKE ? ESCAPE '\\'")
                params.append(
                    event.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                )
            else:
                where.append("event = ?")
                params.append(event)
        if since is not None:
            where.append("created_at >= ?")
            params.append(since)
        clause = f"WHERE {' AND '.join(where)} " if where else ""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT created_at, workspace, event, key, client, fields_json FROM audit "
                f"{clause}ORDER BY rowid DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        entries = []
        for created_at, workspace_name, event_name, key, client, fields_json in rows:
            try:
                fields = json.loads(fields_json)
            except ValueError:
                fields = {}
            entries.append(
                AuditEntry(
                    float(created_at),
                    str(workspace_name),
                    str(event_name),
                    str(key),
                    client,
                    fields if isinstance(fields, dict) else {},
                )
            )
        return entries
