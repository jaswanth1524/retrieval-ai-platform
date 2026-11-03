"""Pass 14: what a job changed, and documents holding the same bytes."""

from __future__ import annotations

from tests.routes_http.support import ApiTestContext, upload_and_wait


def test_a_job_says_what_it_changed_and_names_a_duplicate(api_context: ApiTestContext) -> None:
    client = api_context.client
    first = upload_and_wait(client, "a.txt", b"Intro\nalpha beta gamma")
    assert first["result"]["chunks_removed"] == 0
    assert first["result"]["duplicate_of"] is None

    same = upload_and_wait(client, "b.txt", b"Intro\nalpha beta gamma")
    assert same["result"]["duplicate_of"] == "a.txt"
    assert client.get("/documents").json()["duplicate_groups"] == [["a.txt", "b.txt"]]

    changed = upload_and_wait(client, "a.txt", b"Intro\nalpha beta gamma delta")
    assert changed["state"] == "done"
    assert changed["result"]["chunks_removed"] >= 1
    assert client.get("/documents").json()["duplicate_groups"] == []
