"""Pass 21: backup drill and restoring chosen documents."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.scheduled_backup import ScheduledBackup
from tests.routes_http.support import upload_and_wait, wait_for_job
from tests.test_pass15_admin import Clock, admin  # noqa: F401 - the fixture

Admin = tuple[TestClient, ScheduledBackup, Clock, Path]


def test_a_drill_parses_the_backups_originals_and_indexes_nothing(admin: Admin) -> None:  # noqa: F811
    client, backup, _, _ = admin
    for name in ("a.txt", "b.txt", "c.txt"):
        assert upload_and_wait(client, name, f"Intro\n{name} body".encode())["state"] == "done"
    (written,) = backup.run_once()
    before = client.get("/documents").json()["filenames"]

    drill = client.post(f"/backups/{written.name}/drill", params={"sample": 0}).json()
    assert drill["ok"] is True and drill["listed"] == 3 and drill["checked"] == 3
    assert all(d["chunks"] >= 1 and d["chunks_then"] == d["chunks"] for d in drill["documents"])
    sampled = client.post(f"/backups/{written.name}/drill", params={"sample": 2}).json()
    assert sampled["checked"] == 2 and sampled["listed"] == 3
    assert client.get("/documents").json()["filenames"] == before
    assert client.post("/backups/nope.zip/drill").status_code == 404


def test_a_backup_restores_only_the_documents_asked_for(admin: Admin) -> None:  # noqa: F811
    client, backup, _, _ = admin
    for name in ("a.txt", "b.txt"):
        assert upload_and_wait(client, name, f"Intro\n{name} body".encode())["state"] == "done"
    (written,) = backup.run_once()
    for name in ("a.txt", "b.txt"):
        assert client.delete(f"/documents/{name}").status_code == 200

    restored = client.post(
        f"/backups/{written.name}/restore", params={"only": ["b.txt", "zzz.txt"]}
    ).json()
    (job,) = restored["jobs"]
    assert job["filename"] == "b.txt"
    assert "zzz.txt (not in this backup)" in restored["rejected"]
    assert wait_for_job(client, job["job_id"])["state"] == "done"
    assert client.get("/documents").json()["filenames"] == ["b.txt"]


def test_a_bundle_download_resumes_with_a_range_and_is_checked(tmp_path: Path) -> None:
    import hashlib

    import httpx

    import api.routes.migrate as migrate_routes
    from cli.client import DocragClient, DocragError

    body = bytes(range(256)) * 400
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(body)
    sha = hashlib.sha256(body).hexdigest()
    migrate_routes._keep_export("r1", bundle, sha)
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/b")
    def get_bundle() -> object:
        return migrate_routes._KeptUntilSent("r1", bundle, "b.zip", sha)

    with TestClient(app) as client:
        part = client.get("/b", headers={"Range": "bytes=0-99"})
        assert part.status_code == 206 and part.headers["X-Content-SHA256"] == sha
        assert bundle.exists()  # a range short of the end keeps it
        tail = client.get("/b", headers={"Range": f"bytes=100-{len(body) - 1}"})
        assert tail.status_code == 206 and part.content + tail.content == body
        assert not bundle.exists()  # the last byte went out

    # The client: a cut connection resumed with a Range, then the checksum checked.
    calls: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("Range"))
        if len(calls) == 1:

            def cut() -> object:
                yield body[:1000]
                raise httpx.ReadError("connection reset")

            return httpx.Response(200, headers={"X-Content-SHA256": sha}, content=cut())
        start = int(request.headers["Range"].split("=")[1].rstrip("-"))
        return httpx.Response(206, headers={"X-Content-SHA256": sha}, content=body[start:])

    http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    client_ = DocragClient("http://x", None, http=http, sleep=lambda _: None)
    target = tmp_path / "out.zip"
    assert client_.download_resumable("/b", target) == len(body)
    assert target.read_bytes() == body and calls == [None, "bytes=1000-"]

    def wrong(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"X-Content-SHA256": "0" * 64}, content=body)

    bad = DocragClient(
        "http://x",
        None,
        http=httpx.Client(base_url="http://x", transport=httpx.MockTransport(wrong)),
    )
    try:
        bad.download_resumable("/b", tmp_path / "bad.zip")
        raise AssertionError("a bundle that doesn't match its checksum was accepted")
    except DocragError as exc:
        assert "checksum" in str(exc)
    assert not (tmp_path / "bad.zip").exists()
