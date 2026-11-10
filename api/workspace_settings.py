"""Relevance and answer settings per workspace (``GET``/``PUT /workspaces/{name}/settings``).

Workspaces are separate corpora, and RERANK_MIN_SCORE was measured on one: a workspace
of scanned forms and one of policy prose want different bars. These few settings can
be set per workspace; every request in it then runs with them
(``dependencies.get_request_settings`` folds them into its settings copy). Unset ones
are the server's. Kept in memory, or in WORKSPACE_SETTINGS_PATH (sqlite) to survive a
restart. A change empties that workspace's cached answers, which were given under the
old ones.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import AbstractContextManager
from pathlib import Path
from threading import Lock
from typing import Any

from api.sqlite_store import enable_wal, sqlite_transaction

# The settings a workspace may set: none of them changes what is indexed (chunking and
# models stay the server's), only how passages are picked and answers written.
WORKSPACE_SETTING_FIELDS = (
    "rerank_min_score",
    "recency_boost_weight",
    "answer_language",
    "answer_format",
    # How many closest passages answer when none clears the bar (0: refuse instead),
    # the relative floor, passages per answer, and suggested follow-up questions.
    "rerank_fallback_count",
    "rerank_relative_floor",
    "max_context_chunks",
    "followups_enabled",
)


class WorkspaceSettingsStore:
    """Each workspace's settings ("" = the default workspace), read from memory; with a
    path, written through to sqlite and loaded from it on start."""

    def __init__(self, path: str = "") -> None:
        self._path = path
        self._lock = Lock()
        self._values: dict[str, dict[str, Any]] = {}
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            enable_wal(path)
            with self._connect() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS workspace_settings ("
                    "workspace TEXT PRIMARY KEY, settings_json TEXT NOT NULL)"
                )
                for workspace, raw in conn.execute(
                    "SELECT workspace, settings_json FROM workspace_settings"
                ):
                    try:
                        loaded = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(loaded, dict):
                        self._values[str(workspace)] = _known(loaded)

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_transaction(self._path)

    def get(self, workspace: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._values.get(workspace, {}))

    def put(self, workspace: str, values: dict[str, Any]) -> None:
        """Replace ``workspace``'s settings (an empty dict: back to the server's)."""

        kept = _known(values)
        with self._lock:
            if kept:
                self._values[workspace] = kept
            else:
                self._values.pop(workspace, None)
            if self._path:
                with self._connect() as conn:
                    if kept:
                        conn.execute(
                            "INSERT OR REPLACE INTO workspace_settings VALUES (?, ?)",
                            (workspace, json.dumps(kept, sort_keys=True)),
                        )
                    else:
                        conn.execute(
                            "DELETE FROM workspace_settings WHERE workspace = ?", (workspace,)
                        )

    def delete_workspace(self, workspace: str) -> int:
        """Forget a deleted workspace's settings; how many rows went (0 or 1)."""

        had = bool(self.get(workspace))
        self.put(workspace, {})
        return int(had)


def _known(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if key in WORKSPACE_SETTING_FIELDS and value is not None
    }
