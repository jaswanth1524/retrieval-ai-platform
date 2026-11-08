"""REQUEUE_INTERRUPTED_JOBS: a re-index a restart cut off is started again at startup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from api.chunking import HeuristicTokenCounter
from api.jobs import SqliteIngestJobStore
from api.schemas import DocumentJobAcceptedResponse
from tests.routes_http.support import FakeEmbeddingProvider, make_settings


def test_an_interrupted_reindex_is_started_again_and_an_upload_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import api.main as main

    path = str(tmp_path / "jobs.db")
    before = SqliteIngestJobStore(path, max_retained=10)
    reindex = before.create("guide.txt", kind="reindex")
    before.update(reindex.id, state="embedding")
    upload = before.create("notes.txt", kind="upload")
    before.update(upload.id, state="parsing")
    done = before.create("done.txt", kind="reindex")
    before.update(done.id, state="done")

    restarted = SqliteIngestJobStore(path, max_retained=10)  # the restart
    assert set(restarted.interrupted_at_open) == {reindex.id, upload.id}

    asked: list[str] = []

    def reindex_from_original(filename: str, *args: Any) -> DocumentJobAcceptedResponse:
        asked.append(filename)
        return DocumentJobAcceptedResponse(job_id="new-job", filename=filename, state="queued")

    monkeypatch.setattr(main, "get_ingest_job_store", lambda: restarted)
    monkeypatch.setattr(main, "reindex_from_original", reindex_from_original)
    monkeypatch.setattr(main, "get_qdrant_client", lambda: QdrantClient(":memory:"))
    monkeypatch.setattr(main, "get_token_counter", lambda: HeuristicTokenCounter())
    monkeypatch.setattr(main, "get_embedding_provider", lambda: FakeEmbeddingProvider())
    caplog.set_level("INFO", logger="api.audit")

    started = main.requeue_interrupted_jobs(make_settings(job_store_backend="sqlite"))

    assert started == ["new-job"]
    assert asked == ["guide.txt"]
    audited = [r.audit for r in caplog.records if r.name == "api.audit"]  # type: ignore[attr-defined]
    assert [(a["event"], a["job_id"], a["scheduled"]) for a in audited] == [
        ("job.retry", reindex.id, True)
    ]
