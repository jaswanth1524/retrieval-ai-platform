"""Pass 24: shared answers (SHARES_ENABLED)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from api.keys import key_table
from api.shares import ShareStore
from tests.routes_http.support import keyed_client

ALICE, BOB, CAROL = "alice-key-1", "bob-key-2", "carol-key-3"


def _turn(**extra: Any) -> dict[str, Any]:
    return {
        "question": "What is the meal allowance in Paris?",
        "answer": "It is 60 euros a day [1].",
        "sources": [
            {
                "source_number": 1,
                "filename": "travel.docx",
                "page": 1,
                "section": "Meals",
                "chunk_id": "c1",
                "text": "The allowance is 60 euros in Paris.",
                "spans": [{"start": 0, "end": 35, "quote": "The allowance is 60 euros in Paris."}],
            }
        ],
        "answer_status": "answered",
        "model": "gpt-5-mini",
        **extra,
    }


def _settings(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    keys = tmp_path / "keys.toml"
    keys.write_text(
        key_table("alice", ALICE, "full", ["*"])
        + key_table("bob", BOB, "read", ["default"])
        + key_table("carol", CAROL, "read", ["legal"])
    )
    return {
        "api_keys_file": str(keys),
        "shares_enabled": True,
        "share_store_path": str(tmp_path / "shares.db"),
        "workspaces_enabled": True,
        **extra,
    }


def test_a_share_is_made_opened_listed_and_deleted(tmp_path: Path) -> None:
    with keyed_client("shared", **_settings(tmp_path)) as client:
        bob = {"X-API-Key": BOB}
        made = client.post("/shares", json={"title": "Meals", "turns": [_turn()]}, headers=bob)
        assert made.status_code == 201, made.text
        share = made.json()
        assert (share["workspace"], share["created_by"], share["turns"]) == ("default", "bob", 1)
        assert share["expires_at"] == share["created_at"] + 30 * 86400

        opened = client.get(f"/shares/{share['id']}", headers={"X-API-Key": ALICE}).json()
        assert opened["conversation"][0]["answer"] == "It is 60 euros a day [1]."
        assert opened["conversation"][0]["sources"][0]["text"].startswith("The allowance")
        listed = client.get("/shares", headers=bob).json()["shares"]
        assert [s["id"] for s in listed] == [share["id"]]
        assert (
            client.get("/shares", params={"mine": True}, headers={"X-API-Key": ALICE}).json()[
                "shares"
            ]
            == []
        )

        # The key that shared it may delete it, read-only or not.
        assert client.delete(f"/shares/{share['id']}", headers=bob).status_code == 204
        assert client.get(f"/shares/{share['id']}", headers=bob).status_code == 404


def test_a_share_is_only_for_keys_that_may_use_its_workspace(tmp_path: Path) -> None:
    with keyed_client("shared", **_settings(tmp_path)) as client:
        made = client.post(
            "/shares", json={"title": "Meals", "turns": [_turn()]}, headers={"X-API-Key": BOB}
        ).json()
        carol = {"X-API-Key": CAROL}
        # Carol may use only "legal": the share reads as not there at all.
        refused = client.get(f"/shares/{made['id']}", headers=carol)
        assert refused.status_code == 404
        assert refused.json() == client.get("/shares/nothing-here", headers=carol).json()
        assert client.delete(f"/shares/{made['id']}", headers=carol).status_code == 404
        assert client.get(f"/shares/{made['id']}", headers={"X-API-Key": "nope"}).status_code == 401


def test_only_its_maker_or_a_full_key_deletes_a_share(tmp_path: Path) -> None:
    keys = tmp_path / "keys.toml"
    with keyed_client("shared", **_settings(tmp_path)) as client:
        keys.write_text(keys.read_text() + key_table("dana", "dana-key", "read", ["default"]))
        made = client.post(
            "/shares", json={"title": "Meals", "turns": [_turn()]}, headers={"X-API-Key": BOB}
        ).json()
        other = client.delete(f"/shares/{made['id']}", headers={"X-API-Key": "dana-key"})
        assert other.status_code == 403
        assert (
            client.delete(f"/shares/{made['id']}", headers={"X-API-Key": ALICE}).status_code == 204
        )


def test_shares_are_capped_checked_and_off_by_default(tmp_path: Path) -> None:
    with keyed_client("shared", **_settings(tmp_path, share_max_per_workspace=1)) as client:
        full = {"X-API-Key": "shared"}
        assert (
            client.post(
                "/shares", json={"title": "a", "turns": [_turn()]}, headers=full
            ).status_code
            == 201
        )
        second = client.post("/shares", json={"title": "b", "turns": [_turn()]}, headers=full)
        assert second.status_code == 409
        assert (
            client.post("/shares", json={"title": "a", "turns": []}, headers=full).status_code
            == 422
        )
    with keyed_client("shared") as client:
        off = client.post(
            "/shares", json={"title": "a", "turns": [_turn()]}, headers={"X-API-Key": "shared"}
        )
        assert off.status_code == 404
        assert client.get("/config").json()["shares_enabled"] is False


def test_a_share_larger_than_share_max_bytes_is_refused_before_it_is_read(tmp_path: Path) -> None:
    with keyed_client("shared", **_settings(tmp_path, share_max_bytes=64 * 1024)) as client:
        big = _turn(answer="x" * 100_000)
        refused = client.post(
            "/shares", json={"title": "big", "turns": [big]}, headers={"X-API-Key": "shared"}
        )
        assert refused.status_code == 413


def test_expired_shares_are_gone_and_purged(tmp_path: Path) -> None:
    now = [1000.0]
    store = ShareStore(
        str(tmp_path / "s.db"), max_per_workspace=1, ttl_seconds=60, clock=lambda: now[0]
    )
    share = store.add(workspace="", created_by="bob", title="t", turns=[_turn()])
    assert store.get(share.id) is not None
    now[0] += 61
    assert store.get(share.id) is None
    assert store.list("default") == []
    # The expired one no longer counts against the cap.
    store.add(workspace="", created_by="bob", title="t2", turns=[_turn()])


def test_deleting_a_workspace_deletes_its_shares(tmp_path: Path) -> None:
    with keyed_client("shared", **_settings(tmp_path)) as client:
        full = {"X-API-Key": "shared"}
        assert client.post("/workspaces", json={"name": "legal"}, headers=full).status_code == 201
        made = client.post(
            "/shares",
            json={"title": "Legal", "turns": [_turn()]},
            headers={**full, "X-Workspace": "legal"},
        ).json()
        assert made["workspace"] == "legal"
        assert client.get(f"/shares/{made['id']}", headers={"X-API-Key": CAROL}).status_code == 200
        assert client.delete("/workspaces/legal", headers=full).status_code == 200
        assert client.get(f"/shares/{made['id']}", headers=full).status_code == 404
