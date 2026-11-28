"""Pass 24: named API keys (API_KEYS_FILE) and the workspaces each may use."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from api.audit_log import AuditLog
from api.dependencies import get_audit_log, key_grant
from api.keys import KeyFileError, key_table, parse_key_file
from api.main import app as main_app
from api.routes.deps import SettingsDep
from tests.routes_http.support import keyed_client, make_settings

ALICE, BOB, CAROL = "alice-key-1", "bob-key-2", "carol-key-3"


def _keys_file(tmp_path: Path) -> str:
    path = tmp_path / "keys.toml"
    path.write_text(
        key_table("alice", ALICE, "full", ["*"])
        + key_table("bob", BOB, "read", ["legal"])
        + key_table("carol", CAROL, "full", ["default", "hr"])
    )
    return str(path)


def test_the_file_holds_hashes_and_is_checked_strictly() -> None:
    ring = parse_key_file(key_table("bob", BOB, "read", ["legal"]))
    grant = ring.match(BOB.encode())
    assert grant is not None and (grant.name, grant.level) == ("bob", "read")
    assert grant.can_use("legal") and not grant.can_use("") and not grant.can_use("hr")
    assert ring.match(b"wrong") is None
    assert BOB not in key_table("bob", BOB, "read", ["legal"])
    for text, message in [
        ("[[key]]\nname='x'\nsha256='00'\n", "64 hex"),
        (key_table("a", "k1", "full", ["*"]) + key_table("a", "k2", "full", ["*"]), "twice"),
        (key_table("a", "k1", "full", ["*"]) + key_table("b", "k1", "full", ["*"]), "twice"),
        (key_table("full", "k1", "full", ["*"]), "reserved"),
        (key_table("a", "k1", "admin", ["*"]), "level"),  # type: ignore[arg-type]
        ("[[key]]\nname='a'\nsha256='" + "0" * 64 + "'\nworkspaces=[]\n", "workspaces"),
        ("nonsense = [", "TOML"),
    ]:
        with pytest.raises(KeyFileError, match=message):
            parse_key_file(text, reserved=("full", "read"))


def test_a_keys_file_alone_turns_the_key_check_on(tmp_path: Path) -> None:
    settings = make_settings(api_keys_file=_keys_file(tmp_path))
    assert settings.auth_required
    assert key_grant(settings, None) is None
    grant = key_grant(settings, CAROL.encode())
    assert grant is not None and grant.name == "carol"


def test_api_key_and_file_keys_work_side_by_side(tmp_path: Path) -> None:
    with keyed_client("shared", api_keys_file=_keys_file(tmp_path)) as client:
        assert client.get("/access", headers={"X-API-Key": "shared"}).json() == {
            "access": "full",
            "key_name": "full",
            "workspaces": None,
        }
        assert client.get("/access", headers={"X-API-Key": BOB}).json() == {
            "access": "read",
            "key_name": "bob",
            "workspaces": ["legal"],
        }
        assert client.get("/access", headers={"X-API-Key": "nope"}).status_code == 401


def test_a_key_is_kept_to_its_workspaces(tmp_path: Path) -> None:
    with keyed_client(
        "shared", api_keys_file=_keys_file(tmp_path), workspaces_enabled=True
    ) as client:
        for name in ("legal", "hr"):
            created = client.post("/workspaces", json={"name": name}, headers={"X-API-Key": ALICE})
            assert created.status_code == 201, created.text
        carol = {"X-API-Key": CAROL}
        bob = {"X-API-Key": BOB}
        # Carol: the default workspace and hr only.
        assert client.get("/documents", headers=carol).status_code == 200
        assert client.get("/documents", headers={**carol, "X-Workspace": "hr"}).status_code == 200
        refused = client.get("/documents", headers={**carol, "X-Workspace": "legal"})
        assert refused.status_code == 403 and "legal" in refused.json()["detail"]
        # Refused before saying whether it exists.
        unknown = client.get("/documents", headers={**carol, "X-Workspace": "nowhere"})
        assert unknown.status_code == 403
        # Bob: legal only, and read-only there.
        assert client.get("/documents", headers=bob).status_code == 403
        assert client.get("/documents", headers={**bob, "X-Workspace": "legal"}).status_code == 200
        assert client.get("/workspaces", headers=bob).json()["workspaces"] == ["legal"]
        assert client.get("/workspaces", headers=carol).json()["workspaces"] == ["hr"]
        assert client.get("/workspaces/legal/settings", headers=carol).status_code == 403
        assert client.get("/workspaces/hr/settings", headers=carol).status_code == 200
        assert client.delete("/workspaces/legal", headers=carol).status_code == 403
        # Routes over every workspace need a key for every workspace.
        assert client.post("/workspaces", json={"name": "x"}, headers=carol).status_code == 403
        assert client.get("/metrics", headers=carol).status_code == 403
        assert client.get("/admin/doctor", headers=carol).status_code == 403
        assert client.get("/metrics", headers={"X-API-Key": ALICE}).status_code == 200


def test_audit_entries_name_the_key(tmp_path: Path) -> None:
    log = AuditLog(str(tmp_path / "audit.sqlite"), 1000)
    with keyed_client(
        "shared",
        extra_overrides={get_audit_log: lambda: log},
        api_keys_file=_keys_file(tmp_path),
        workspaces_enabled=True,
    ) as client:
        assert (
            client.post(
                "/workspaces", json={"name": "legal"}, headers={"X-API-Key": ALICE}
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/workspaces", json={"name": "hr"}, headers={"X-API-Key": "shared"}
            ).status_code
            == 201
        )
        entries = client.get(
            "/admin/audit", params={"all_workspaces": True}, headers={"X-API-Key": ALICE}
        ).json()["entries"]
        assert [(e["event"], e["key"], e["key_name"]) for e in entries][:2] == [
            ("workspace.create", "full", "full"),
            ("workspace.create", "full", "alice"),
        ]
        # A key for some workspaces sees their audit, not every workspace's.
        carol = {"X-API-Key": CAROL}
        assert client.get("/admin/audit", headers=carol).status_code == 200
        assert (
            client.get("/admin/audit", params={"all_workspaces": True}, headers=carol).status_code
            == 403
        )


def test_an_older_audit_file_gains_the_key_name_column(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE audit (created_at REAL NOT NULL, workspace TEXT NOT NULL, "
            "event TEXT NOT NULL, key TEXT NOT NULL, client TEXT, fields_json TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO audit VALUES (1, 'default', 'document.delete', 'full', '', '{}')")
    entries = AuditLog(str(path), 100).recent(None)
    assert [(e.event, e.key_name) for e in entries] == [("document.delete", "")]


def test_a_changed_or_broken_keys_file_is_read_again(tmp_path: Path) -> None:
    path = tmp_path / "keys.toml"
    path.write_text(key_table("alice", ALICE, "full", ["*"]))
    settings = make_settings(api_keys_file=str(path))
    assert key_grant(settings, ALICE.encode()) is not None
    path.write_text(key_table("dave", "dave-key", "read", ["*"]))
    os.utime(path, ns=(1, 2_000_000_000_000_000_000))
    assert key_grant(settings, ALICE.encode()) is None
    assert key_grant(settings, b"dave-key") is not None
    # Broken: no key from it works, until it is fixed.
    path.write_text("[[key]]\nname = 'dave'\n")
    os.utime(path, ns=(1, 2_000_000_000_000_000_001))
    assert key_grant(settings, b"dave-key") is None
    path.unlink()
    assert key_grant(settings, b"dave-key") is None


def test_every_workspace_route_checks_the_key() -> None:
    """Each route whose settings come from X-Workspace is checked by the same
    dependency; any other route that names a workspace checks the key itself."""

    from fastapi.dependencies.models import Dependant
    from fastapi.routing import APIRoute

    from api.dependencies import get_request_settings, request_grant, require_all_workspaces

    def calls(dependant: Dependant) -> set[object]:
        found = {dependant.call}
        for sub in dependant.dependencies:
            found |= calls(sub)
        return found

    unchecked = []
    for route in main_app.routes:
        if not isinstance(route, APIRoute) or "{name}" not in route.path:
            continue
        if not route.path.startswith("/workspaces/"):
            continue
        used = calls(route.dependant)
        if not used & {get_request_settings, request_grant, require_all_workspaces}:
            unchecked.append(route.path)
    assert unchecked == []
    assert SettingsDep is not None


def test_docrag_new_key_prints_a_key_and_the_entry_that_grants_it() -> None:
    import io

    from cli.__main__ import main

    out = io.StringIO()
    assert main(["new-key", "dana", "--read", "--allow", "legal"], out=out) == 0
    text = out.getvalue()
    key = text.splitlines()[1].strip()
    entry = text[text.index("[[key]]") :]
    grant = parse_key_file(entry).match(key.encode())
    assert grant is not None
    assert (grant.name, grant.level, grant.workspaces) == ("dana", "read", frozenset({"legal"}))
    assert main(["new-key", "full"], out=io.StringIO()) == 2


def test_a_file_key_that_is_the_shared_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "keys.toml"
    path.write_text(
        key_table("eve", "shared", "read", ["legal"]) + key_table("bo", BOB, "full", ["*"])
    )
    settings = make_settings(api_key="shared", api_keys_file=str(path))
    # The file is refused whole (no key from it), the shared key still works.
    assert key_grant(settings, BOB.encode()) is None
    grant = key_grant(settings, b"shared")
    assert grant is not None and grant.name == "full"
    from api.keys import key_file_error, shared_key_hashes

    assert "API_KEY" in (key_file_error(str(path), shared_key_hashes("shared")) or "")


def test_another_workspaces_job_reads_as_not_found_for_a_limited_key(tmp_path: Path) -> None:
    with keyed_client(
        "shared", api_keys_file=_keys_file(tmp_path), workspaces_enabled=True
    ) as client:
        full = {"X-API-Key": "shared"}
        assert client.post("/workspaces", json={"name": "hr"}, headers=full).status_code == 201
        upload = client.post(
            "/documents", files={"file": ("a.txt", b"Leave is 28 days.")}, headers=full
        )
        assert upload.status_code == 202, upload.text
        job_id = upload.json()["job_id"]
        carol_hr = {"X-API-Key": CAROL, "X-Workspace": "hr"}
        assert client.get(f"/documents/jobs/{job_id}", headers=carol_hr).status_code == 404
        assert client.delete(f"/documents/jobs/{job_id}", headers=carol_hr).status_code == 404
        bob = {"X-API-Key": BOB, "X-Workspace": "legal"}
        assert client.get(f"/documents/jobs/{job_id}", headers=bob).status_code in (403, 404)
        assert client.get(f"/traces/{job_id}", headers=carol_hr).status_code == 404


def test_feedback_and_the_query_log_name_the_key(tmp_path: Path) -> None:
    from api.dependencies import get_feedback_store
    from api.feedback import FeedbackStore
    from api.query_log import LoggedQuery, QueryLogSink, QueryLogStore

    feedback = FeedbackStore(str(tmp_path / "feedback.db"))
    with keyed_client(
        "shared",
        extra_overrides={get_feedback_store: lambda: feedback},
        api_keys_file=_keys_file(tmp_path),
        feedback_enabled=True,
    ) as client:
        rated = client.post(
            "/feedback",
            json={
                "question": "Leave?",
                "answer_excerpt": "28 days",
                "cited_filenames": [],
                "rating": "up",
            },
            headers={"X-API-Key": CAROL},
        )
        assert rated.status_code == 201, rated.text
        rows = client.get("/feedback", headers={"X-API-Key": ALICE}).json()["feedback"]
        assert [row["key_name"] for row in rows] == ["carol"]
        exported = client.get("/export", headers={"X-API-Key": ALICE})
        assert exported.status_code == 200

    import io
    import json
    import zipfile

    lines = zipfile.ZipFile(io.BytesIO(exported.content)).read("feedback.jsonl").decode()
    assert json.loads(lines.splitlines()[0])["key_name"] == "carol"

    from api.audit import set_actor

    store = QueryLogStore(str(tmp_path / "queries.db"), 100)
    set_actor("read", "127.0.0.1", "bob")
    QueryLogSink(store).add_logged(
        LoggedQuery(
            created_at=1.0,
            workspace="",
            question="Leave?",
            status="ok",
            top_score=0.5,
            mode="answer",
            trace_id=None,
        )
    )
    assert [row.key_name for row in store.recent("", limit=5)] == ["bob"]


def test_an_older_feedback_or_query_log_file_opens_with_names_empty(tmp_path: Path) -> None:
    import sqlite3

    from api.feedback import FeedbackStore
    from api.query_log import QueryLogStore

    old = tmp_path / "f.db"
    with sqlite3.connect(old) as conn:
        conn.execute(
            "CREATE TABLE feedback (id TEXT PRIMARY KEY, trace_id TEXT, question TEXT NOT NULL, "
            "answer_excerpt TEXT NOT NULL, cited_filenames_json TEXT NOT NULL, "
            "rating TEXT NOT NULL, citation_source_number INTEGER, created_at REAL NOT NULL, "
            "workspace TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL DEFAULT 'question')"
        )
        conn.execute(
            "INSERT INTO feedback VALUES ('a', NULL, 'Q', 'A', '[]', 'up', NULL, 1, '', 'question')"
        )
    assert [row.key_name for row in FeedbackStore(str(old)).list_recent()] == [""]
    queries = tmp_path / "q.db"
    with sqlite3.connect(queries) as conn:
        conn.execute(
            "CREATE TABLE queries (created_at REAL NOT NULL, workspace TEXT NOT NULL DEFAULT '', "
            "question TEXT NOT NULL, status TEXT NOT NULL, top_score REAL, mode TEXT NOT NULL, "
            "trace_id TEXT)"
        )
        conn.execute("INSERT INTO queries VALUES (1, '', 'Q', 'ok', 0.5, 'answer', NULL)")
    assert [row.key_name for row in QueryLogStore(str(queries), 10).recent("", limit=5)] == [""]
