from __future__ import annotations

from api.jobs import IngestJobStore


def test_ingest_job_store_create_returns_queued_job() -> None:
    store = IngestJobStore(max_retained=50)

    job = store.create("guide.txt")

    assert job.filename == "guide.txt"
    assert job.state == "queued"
    assert job.chunks_done == 0
    assert job.chunks_total == 0
    assert job.error is None
    assert job.result is None


def test_ingest_job_store_get_returns_none_for_unknown_id() -> None:
    store = IngestJobStore(max_retained=50)

    assert store.get("missing") is None


def test_ingest_job_store_update_mutates_tracked_fields() -> None:
    store = IngestJobStore(max_retained=50)
    job = store.create("guide.txt")

    store.update(job.id, state="embedding", chunks_done=3, chunks_total=10)

    fetched = store.get(job.id)
    assert fetched is not None
    assert fetched.state == "embedding"
    assert fetched.chunks_done == 3
    assert fetched.chunks_total == 10


def test_ingest_job_store_update_is_a_noop_for_unknown_id() -> None:
    store = IngestJobStore(max_retained=50)

    store.update("missing", state="done")  # must not raise


def test_ingest_job_store_get_returns_a_snapshot_not_the_live_object() -> None:
    """Mutating the store after a `get()` must not retroactively change the
    already-returned snapshot — callers should see a point-in-time status."""

    store = IngestJobStore(max_retained=50)
    job = store.create("guide.txt")

    snapshot = store.get(job.id)
    store.update(job.id, state="done", chunks_done=5)

    assert snapshot is not None
    assert snapshot.state == "queued"
    assert snapshot.chunks_done == 0


def test_ingest_job_store_prunes_oldest_finished_jobs_beyond_retention_limit() -> None:
    store = IngestJobStore(max_retained=50)
    # Create one more finished job than the retention limit.
    job_ids = []
    for i in range(51):
        job = store.create(f"doc-{i}.txt")
        store.update(job.id, state="done")
        job_ids.append(job.id)

    # The oldest finished job should have been pruned; the newest must survive.
    assert store.get(job_ids[0]) is None
    assert store.get(job_ids[-1]) is not None


def test_ingest_job_store_respects_a_custom_max_retained() -> None:
    store = IngestJobStore(max_retained=2)
    job_ids = []
    for i in range(3):
        job = store.create(f"doc-{i}.txt")
        store.update(job.id, state="done")
        job_ids.append(job.id)

    assert store.get(job_ids[0]) is None
    assert store.get(job_ids[1]) is not None
    assert store.get(job_ids[2]) is not None


def test_ingest_job_store_evicts_unfinished_jobs_past_the_cap() -> None:
    """The cap must hold even when nothing has finished.

    Only two executor workers run, so a burst of uploads leaves a long queue of
    non-terminal jobs. Pruning finished jobs alone left that queue unbounded — a client
    could grow the store without limit just by uploading. A client polling an evicted
    job gets the same 404 it already gets for one pruned after completion.
    """

    store = IngestJobStore(max_retained=3)
    job_ids = [store.create(f"doc-{i}.txt").id for i in range(6)]

    assert store.get(job_ids[0]) is None
    assert store.get(job_ids[2]) is None
    # Oldest-first eviction, so the three most recent survive.
    assert store.get(job_ids[3]) is not None
    assert store.get(job_ids[5]) is not None


def test_ingest_job_store_prefers_evicting_finished_jobs_over_running_ones() -> None:
    store = IngestJobStore(max_retained=2)
    finished = store.create("done.txt")
    store.update(finished.id, state="done")
    running = store.create("running.txt")
    store.update(running.id, state="embedding")
    newest = store.create("newest.txt")

    # The finished job is the one nobody is polling, so it goes first.
    assert store.get(finished.id) is None
    assert store.get(running.id) is not None
    assert store.get(newest.id) is not None
