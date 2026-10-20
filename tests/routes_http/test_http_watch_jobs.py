"""GET /documents/jobs and the watched-folder routes (GET /watch, POST /watch/scan)."""

from __future__ import annotations

from pathlib import Path

from api.dependencies import get_ingest_job_store
from api.jobs import IngestJobStore, SqliteIngestJobStore
from api.watch import WatchRetryStatus, WatchStatus
from tests.routes_http.support import ApiTestContext, keyed_client

FULL = {"X-API-Key": "full-key"}
READ = {"X-API-Key": "read-key"}


def test_both_job_stores_list_a_workspaces_active_jobs_oldest_first(tmp_path: Path) -> None:
    for store in (
        IngestJobStore(max_retained=50),
        SqliteIngestJobStore(str(tmp_path / "jobs.sqlite3"), max_retained=50),
    ):
        first = store.create("a.txt")
        second = store.create("b.txt")
        finished = store.create("c.txt")
        store.create("other.txt", workspace="legal")
        store.update(second.id, state="embedding")
        store.update(finished.id, state="done")

        assert [job.filename for job in store.list_active("")] == ["a.txt", "b.txt"]
        assert [job.state for job in store.list_active("")] == ["queued", "embedding"]
        assert [job.filename for job in store.list_active("legal")] == ["other.txt"]
        assert first.id in {job.id for job in store.list_active("")}


def test_the_jobs_route_lists_jobs_started_by_anyone(api_context: ApiTestContext) -> None:
    store = IngestJobStore(max_retained=50)
    queued = store.create("watched.txt")
    store.create("elsewhere.txt", workspace="legal")
    api_context.app.dependency_overrides[get_ingest_job_store] = lambda: store

    body = api_context.client.get("/documents/jobs").json()

    assert [job["job_id"] for job in body["jobs"]] == [queued.id]
    job = body["jobs"][0]
    assert job["filename"] == "watched.txt" and job["state"] == "queued"
    assert job["created_at"] == queued.created_at_wall and job["unchanged"] is False
    assert api_context.client.get("/documents/jobs?state=done").status_code == 422


def _status() -> WatchStatus:
    return WatchStatus(
        folder="/data/inbox",
        last_scan_at=1_700_000_000.0,
        files=3,
        problems=["/data/inbox/x.txt: not a regular file"],
        deferred=[],
        retrying=[WatchRetryStatus("a.txt", 1, 1_700_000_030.0, "Qdrant is unavailable")],
        gave_up=["b.txt"],
    )


class FakeWatcher:
    def __init__(self) -> None:
        self.scans = 0

    def status(self) -> WatchStatus:
        return _status()

    def scan_now(self) -> None:
        self.scans += 1


def test_watch_routes_report_and_scan_with_the_right_key() -> None:
    with keyed_client("full-key", api_read_key="read-key") as client:
        watcher = FakeWatcher()
        client.app.state.watcher = watcher  # type: ignore[attr-defined]

        status = client.get("/watch", headers=READ)
        assert status.status_code == 200
        body = status.json()
        assert body["files"] == 3 and body["gave_up"] == ["b.txt"]
        assert body["retrying"] == [
            {
                "filename": "a.txt",
                "attempts": 1,
                "next_attempt_at": 1_700_000_030.0,
                "error": "Qdrant is unavailable",
            }
        ]
        assert client.get("/watch").status_code == 401
        assert client.post("/watch/scan", headers=READ).status_code == 403
        assert client.post("/watch/scan", headers=FULL).status_code == 202
        assert watcher.scans == 1
        assert client.get("/documents/jobs", headers=READ).status_code == 200


def test_watch_routes_are_404_without_a_watched_folder(api_context: ApiTestContext) -> None:
    api_context.app.state.watcher = None
    assert api_context.client.get("/watch").status_code == 404
    assert api_context.client.post("/watch/scan").status_code == 404
    assert api_context.client.get("/config").json()["watch_enabled"] is False
