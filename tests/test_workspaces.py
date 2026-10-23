"""Workspace existence checks (api/workspaces.py)."""

from __future__ import annotations

import pytest


def test_a_workspace_seen_to_exist_survives_qdrant_being_briefly_unreachable() -> None:
    from api.workspaces import collection_for, forget_workspace, workspace_exists

    class FlakyClient:
        def __init__(self) -> None:
            self.down = False

        def collection_exists(self, name: str) -> bool:
            if self.down:
                raise ConnectionError("qdrant unreachable")
            return name.endswith("legal")

    client = FlakyClient()
    assert workspace_exists(client, "docs", "legal")  # type: ignore[arg-type]
    assert not workspace_exists(client, "docs", "hr")  # type: ignore[arg-type]
    client.down = True
    assert workspace_exists(client, "docs", "legal")  # type: ignore[arg-type]
    with pytest.raises(ConnectionError):
        workspace_exists(client, "docs", "hr")  # type: ignore[arg-type]
    # Deleted: no longer assumed to exist.
    forget_workspace(client, collection_for("docs", "legal"))  # type: ignore[arg-type]
    with pytest.raises(ConnectionError):
        workspace_exists(client, "docs", "legal")  # type: ignore[arg-type]
