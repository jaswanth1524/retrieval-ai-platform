from __future__ import annotations

from api.jobs import IngestJobStore


def test_ingest_job_store_create_returns_queued_job() -> None:
    store = IngestJobStore()

    job = store.create("guide.txt")

    assert job.filename == "guide.txt"
    assert job.state == "queued"
    assert job.chunks_done == 0
    assert job.chunks_total == 0
    assert job.error is None
    assert job.result is None


def test_ingest_job_store_get_returns_none_for_unknown_id() -> None:
    store = IngestJobStore()

    assert store.get("missing") is None


def test_ingest_job_store_update_mutates_tracked_fields() -> None:
    store = IngestJobStore()
    job = store.create("guide.txt")

    store.update(job.id, state="embedding", chunks_done=3, chunks_total=10)

    fetched = store.get(job.id)
    assert fetched is not None
    assert fetched.state == "embedding"
    assert fetched.chunks_done == 3
    assert fetched.chunks_total == 10


def test_ingest_job_store_update_is_a_noop_for_unknown_id() -> None:
    store = IngestJobStore()

    store.update("missing", state="done")  # must not raise


def test_ingest_job_store_get_returns_a_snapshot_not_the_live_object() -> None:
    """Mutating the store after a `get()` must not retroactively change the
    already-returned snapshot — callers should see a point-in-time status."""

    store = IngestJobStore()
    job = store.create("guide.txt")

    snapshot = store.get(job.id)
    store.update(job.id, state="done", chunks_done=5)

    assert snapshot is not None
    assert snapshot.state == "queued"
    assert snapshot.chunks_done == 0


def test_ingest_job_store_prunes_oldest_finished_jobs_beyond_retention_limit() -> None:
    store = IngestJobStore()
    # _MAX_RETAINED_JOBS is 50 — create one more finished job than that limit.
    job_ids = []
    for i in range(51):
        job = store.create(f"doc-{i}.txt")
        store.update(job.id, state="done")
        job_ids.append(job.id)

    # The oldest finished job should have been pruned; the newest must survive.
    assert store.get(job_ids[0]) is None
    assert store.get(job_ids[-1]) is not None
