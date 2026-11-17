"""Pass 18: recovery and scale (rebuild from originals, disk guard, listing pages)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait, wait_until


@contextmanager
def _server(
    monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> Iterator[tuple[TestClient, QdrantClient]]:
    clear_dependency_caches()
    qdrant = QdrantClient(":memory:")
    settings = make_settings(**overrides)
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    try:
        with TestClient(hermetic_app(settings, qdrant)) as client:
            yield client, qdrant
    finally:
        monkeypatch.undo()
        clear_dependency_caches()


def _lose_the_index(qdrant: QdrantClient) -> None:
    # Behind the API's back: its cached listing still shows every document.
    for collection in qdrant.get_collections().collections:
        qdrant.delete_collection(collection.name)


def _import_finished(client: TestClient, import_id: str) -> Callable[[], bool]:
    def finished() -> bool:
        status = client.get(f"/documents/imports/{import_id}").json()
        return status["finished_at"] is not None

    return finished


def test_a_lost_index_is_rebuilt_from_the_originals_with_their_tags_and_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    with _server(monkeypatch, raw_document_dir=str(raw)) as (client, qdrant):
        for index in range(3):
            upload_and_wait(client, f"d{index}.txt", f"Intro\ndocument {index}".encode())
        assert (
            client.patch("/documents/d1.txt/tags", json={"tags": ["hr", "2024"]}).status_code == 200
        )
        before = client.get("/documents").json()
        # An original kept by a release before details were: indexed as a plain upload.
        (raw / "old.txt").write_bytes(b"Intro\nan older original")

        _lose_the_index(qdrant)
        assert len(client.get("/documents").json()["filenames"]) == 3  # cached
        # The check reads the index as it is now, not the cache.
        assert len(client.get("/admin/consistency").json()["originals_without_document"]) == 4

        started = client.post("/admin/rebuild")
        assert started.status_code == 202, started.text
        assert started.json()["originals"] == 4
        wait_until(_import_finished(client, started.json()["import_id"]))
        wait_until(lambda: len(client.get("/documents").json()["filenames"]) == 4)

        after = client.get("/documents").json()
        assert after["tags"] == {"d1.txt": ["hr", "2024"]}
        for name in ("d0.txt", "d1.txt", "d2.txt"):
            assert after["uploaded_ats"][name] == before["uploaded_ats"][name]
        status = client.get(f"/documents/imports/{started.json()['import_id']}").json()
        assert (status["kind"], status["state"], status["queued"]) == ("rebuild", "done", 4)
        # Nothing left to do: a second rebuild starts nothing.
        again = client.post("/admin/rebuild").json()
        assert again == {"import_id": None, "originals": 0}


def test_a_rebuild_needs_stored_originals(monkeypatch: pytest.MonkeyPatch) -> None:
    with _server(monkeypatch) as (client, _):
        refused = client.post("/admin/rebuild")
        assert refused.status_code == 409 and "RAW_DOCUMENT_DIR" in refused.json()["detail"]


def test_details_follow_the_original_and_older_originals_get_theirs_at_a_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    with _server(monkeypatch, raw_document_dir=str(raw)) as (client, _):
        upload_and_wait(client, "a.txt", b"Intro\nalpha")
        details = raw / ".meta" / "a.txt"
        assert details.is_file()
        # As a release before details would have left it.
        details.unlink()
        from api.dependencies import get_raw_document_store

        store = get_raw_document_store()
        assert store is not None
        store.details_backfilled = False
        client.get("/documents")
        assert details.is_file()
        assert client.delete("/documents/a.txt").status_code == 200
        assert not details.exists()
        # The store's own folder is not a document: the listing and a check ignore it.
        assert client.get("/admin/consistency").json()["problems"] == 0


def test_a_nearly_full_disk_refuses_writes_before_they_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    free = {"bytes": 10 * 1024 * 1024}
    monkeypatch.setattr(
        "api.storage.shutil.disk_usage", lambda path: usage(10**12, 10**12, free["bytes"])
    )
    raw = tmp_path / "raw"
    with _server(monkeypatch, raw_document_dir=str(raw), min_free_bytes=100 * 1024 * 1024) as (
        client,
        _,
    ):
        monkeypatch.setattr(
            "api.storage.shutil.disk_usage", lambda path: usage(10**12, 10**12, free["bytes"])
        )
        refused = client.post("/documents", files={"file": ("a.txt", b"Intro\nalpha")})
        assert refused.status_code == 507
        assert "MIN_FREE_BYTES" in refused.json()["detail"] and "10 MB" in refused.json()["detail"]
        assert client.post("/import", files={"file": ("b.zip", b"PK")}).status_code == 507
        assert client.get("/documents").json()["filenames"] == []
        free["bytes"] = 10**9
        upload_and_wait(client, "a.txt", b"Intro\nalpha")


def test_a_document_past_the_passage_limit_fails_and_a_preview_says_so_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Intro\n" + " ".join(f"Sentence number {n} says something new." for n in range(400))
    from api import ingestion

    ingestion._embed_rate = None  # whatever an earlier test's ingest measured
    with _server(
        monkeypatch, max_chunks_per_document=3, chunk_size_tokens=32, chunk_overlap_tokens=0
    ) as (client, _):
        preview = client.post("/documents/preview", files={"file": ("long.txt", text.encode())})
        assert preview.status_code == 200, preview.text
        body = preview.json()
        assert body["chunk_count"] > 3 and body["over_chunk_limit"] is True
        assert body["estimated_seconds"] is None  # nothing indexed yet to measure by
        accepted = client.post("/documents", files={"file": ("long.txt", text.encode())}).json()

        def failed() -> bool:
            return client.get(f"/documents/jobs/{accepted['job_id']}").json()["state"] == "failed"

        wait_until(failed)
        job = client.get(f"/documents/jobs/{accepted['job_id']}").json()
        assert "MAX_CHUNKS_PER_DOCUMENT" in job["error"]
        short = client.post("/documents/preview", files={"file": ("s.txt", b"Intro\nshort")})
        assert short.json()["over_chunk_limit"] is False


def test_a_previews_time_estimate_follows_the_measured_embedding_speed() -> None:
    from api import ingestion

    ingestion._embed_rate = None
    ingestion._note_embed_rate(3, 1.0)  # too small to say anything
    assert ingestion.measured_embed_rate() is None
    ingestion._note_embed_rate(100, 10.0)
    assert ingestion.measured_embed_rate() == 10.0
    ingestion._note_embed_rate(200, 10.0)
    assert ingestion.measured_embed_rate() == pytest.approx(13.0)
    ingestion._embed_rate = None


def test_the_settings_check_reports_the_indexes_size() -> None:
    from api.doctor import diagnose
    from api.repository import IndexSize
    from tests.factories import make_test_settings

    settings = make_test_settings()

    def about_qdrant(index: IndexSize, memory: int) -> list[tuple[str, str]]:
        found = diagnose(
            settings, index=index, memory_limit=lambda: memory, ollama_models=lambda s: None
        )
        return [(f.level, f.message) for f in found if f.subject == "QDRANT"]

    small = about_qdrant(IndexSize(1000, 2, "green", 1000 * 1536), 8 * 1024**3)
    assert [level for level, _ in small] == ["note"] and "1000 passages" in small[0][1]
    large = about_qdrant(IndexSize(4_000_000, 9, "green", 4_000_000 * 1536), 8 * 1024**3)
    assert [level for level, _ in large] == ["warning"] and "QDRANT_QUANTIZATION" in large[0][1]
    red = about_qdrant(IndexSize(10, 1, "red", 15360), 8 * 1024**3)
    assert red[0][0] == "warning" and "red" in red[0][1]


def test_the_index_size_is_read_from_qdrant_and_exported(monkeypatch: pytest.MonkeyPatch) -> None:
    from api import metrics
    from api.repository import VectorRepository

    with _server(monkeypatch) as (client, qdrant):
        upload_and_wait(client, "a.txt", b"Intro\nalpha")
        from api.dependencies import get_app_settings

        size = VectorRepository(qdrant).index_size(get_app_settings())
        assert size is not None and size.points == 1 and size.vector_bytes > 0
        findings = client.get("/admin/doctor").json()["findings"]
        assert any(f["subject"] == "QDRANT" for f in findings)
    metrics.watch_index(lambda: (7, 700))
    assert metrics.index_points.collect()[0].samples[0].value == 7
    metrics.watch_index(lambda: None)


def test_the_listing_pages_by_name_and_filters_by_prefix_and_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _server(monkeypatch) as (client, _):
        for name in ("a1.txt", "a2.txt", "b1.txt", "b2.txt", "c1.txt"):
            upload_and_wait(client, name, f"Intro\n{name}".encode())
        assert client.patch("/documents/b1.txt/tags", json={"tags": ["HR"]}).status_code == 200
        assert client.patch("/documents/c1.txt/tags", json={"tags": ["hr"]}).status_code == 200

        whole = client.get("/documents").json()
        assert len(whole["filenames"]) == 5 and whole["total"] is None

        first = client.get("/documents", params={"limit": 2}).json()
        assert first["filenames"] == ["a1.txt", "a2.txt"] and first["total"] == 5
        assert set(first["chunk_counts"]) == {"a1.txt", "a2.txt"}
        second = client.get("/documents", params={"limit": 2, "cursor": first["next_cursor"]})
        assert second.json()["filenames"] == ["b1.txt", "b2.txt"]
        last = client.get("/documents", params={"limit": 2, "cursor": "b2.txt"}).json()
        assert last["filenames"] == ["c1.txt"] and last["next_cursor"] is None

        by_prefix = client.get("/documents", params={"prefix": "b"}).json()
        assert by_prefix["filenames"] == ["b1.txt", "b2.txt"] and by_prefix["total"] == 2
        by_tag = client.get("/documents", params={"tag": "hr"}).json()
        assert by_tag["filenames"] == ["b1.txt", "c1.txt"]
        assert client.get("/documents", params={"limit": 0}).status_code == 422


def test_missing_documents_are_the_ones_not_indexed_with_those_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    with _server(monkeypatch) as (client, _):
        upload_and_wait(client, "a.txt", b"Intro\nalpha")
        same = hashlib.sha256(b"Intro\nalpha").hexdigest()
        asked = {
            "documents": [
                {"name": "a.txt", "sha256": same},
                {"name": "b.txt", "sha256": same},
            ]
        }
        assert client.post("/documents/missing", json=asked).json() == {"missing": ["b.txt"]}
        changed = {"documents": [{"name": "a.txt", "sha256": "0" * 64}]}
        assert client.post("/documents/missing", json=changed).json() == {"missing": ["a.txt"]}
        bad = {"documents": [{"name": "a.txt", "sha256": "xyz"}]}
        assert client.post("/documents/missing", json=bad).status_code == 422
