from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

import api.export as export_module
from api.export import MANIFEST_VERSION, build_export
from api.feedback import FeedbackStore
from api.pipeline import IngestService
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.settings import AppSettings
from tests.factories import make_test_settings
from tests.routes_http.support import FakeEmbeddingProvider


def _settings(**overrides: Any) -> AppSettings:
    return make_test_settings(
        qdrant_url=":memory:",
        qdrant_collection="export_documents",
        qdrant_dense_vector_size=3,
        embedding_model_tag="test-embedding:v1",
        **overrides,
    )


def _indexed(settings: AppSettings, *names: str) -> VectorRepository:
    repository = VectorRepository(QdrantClient(":memory:"))
    service = IngestService(repository, FakeEmbeddingProvider(), settings)
    for name in names:
        service.ingest(name, b"Intro\nalpha beta")
    return repository


class DeletedMidExportStore(RawDocumentStore):
    """Reports an original as stored, then loses it before it can be read — what a
    DELETE landing between the export's existence check and its read looks like."""

    def path(self, filename: str) -> Path | None:
        found = super().path(filename)
        if found is not None and filename == "gone.txt":
            found.unlink()
        return found


def test_an_original_deleted_mid_export_is_skipped_not_a_failure(tmp_path: Path) -> None:
    settings = _settings()
    repository = _indexed(settings, "kept.txt", "gone.txt")
    store = DeletedMidExportStore(str(tmp_path))
    store.save("kept.txt", b"kept bytes")
    store.save("gone.txt", b"gone bytes")

    archive = zipfile.ZipFile(io.BytesIO(b"".join(build_export(settings, repository, store, None))))
    manifest = json.loads(archive.read("manifest.json"))
    originals = {d["filename"]: d["original"] for d in manifest["documents"]}

    assert manifest["manifest_version"] == MANIFEST_VERSION == 1
    assert originals == {"gone.txt": None, "kept.txt": "originals/kept.txt"}
    # What the zip actually holds, not what the listing promised.
    assert manifest["originals_missing"] == ["gone.txt"]
    assert manifest["complete"] is False
    assert archive.read("originals/kept.txt") == b"kept bytes"
    assert "originals/gone.txt" not in archive.namelist()


def _export_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "docrag-export"]


def test_the_zip_streams_and_pages_feedback_oldest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(export_module, "_FEEDBACK_PAGE", 2)
    settings = _settings()
    repository = _indexed(settings, "a.txt")
    store = RawDocumentStore(str(tmp_path / "raw"))
    # Bigger than the queue holds, so the writer has to wait for the reader.
    store.save("a.txt", bytes(range(256)) * 8192)
    feedback = FeedbackStore(str(tmp_path / "feedback.db"))
    for index in range(5):
        feedback.add(
            trace_id=None,
            question=f"q{index}",
            answer_excerpt="a",
            cited_filenames=[],
            rating="up",
            citation_source_number=None,
        )

    blocks = list(build_export(settings, repository, store, feedback))

    assert len(blocks) > 1
    archive = zipfile.ZipFile(io.BytesIO(b"".join(blocks)))
    assert archive.testzip() is None
    assert archive.read("originals/a.txt") == bytes(range(256)) * 8192
    rows = [json.loads(line) for line in archive.read("feedback.jsonl").splitlines()]
    assert [row["question"] for row in rows] == ["q0", "q1", "q2", "q3", "q4"]


def test_a_failure_before_streaming_raises_from_the_call(tmp_path: Path) -> None:
    class BrokenRepository(VectorRepository):
        def filename_metadata(self, settings: AppSettings) -> Any:
            raise RuntimeError("qdrant down")

    with pytest.raises(RuntimeError, match="qdrant down"):
        build_export(_settings(), BrokenRepository(QdrantClient(":memory:")), None, None)


def test_a_failure_mid_stream_is_raised_not_ended_as_a_complete_zip(tmp_path: Path) -> None:
    class FailingStore(RawDocumentStore):
        def open(self, filename: str) -> Any:
            raise OSError("disk gone")

    settings = _settings()
    stream = build_export(settings, _indexed(settings, "a.txt"), FailingStore(str(tmp_path)), None)

    with pytest.raises(OSError, match="disk gone"):
        b"".join(stream)


def test_a_client_that_leaves_stops_the_writer_and_closes_the_files(tmp_path: Path) -> None:
    settings = _settings()
    repository = _indexed(settings, "a.txt")
    opened: list[Any] = []

    class TrackingStore(RawDocumentStore):
        def open(self, filename: str) -> Any:
            handle = super().open(filename)
            opened.append(handle)
            return handle

    store = TrackingStore(str(tmp_path))
    store.save("a.txt", b"x" * (4 * 1024 * 1024))
    stream = build_export(settings, repository, store, None)

    next(stream)
    stream.close()  # what Starlette does when the client disconnects

    deadline = time.monotonic() + 5
    while _export_threads() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _export_threads() == []
    assert opened and all(handle.closed for handle in opened)


def test_close_wakes_a_reader_waiting_on_the_next_block() -> None:
    """The response closes the stream when the client leaves; a worker thread blocked
    waiting for the writer's next block must return, not wait forever."""

    from api.export import ExportStream

    gate = threading.Event()
    stream = ExportStream(lambda writer: gate.wait(5))
    outcome: list[str] = []

    def read() -> None:
        try:
            next(stream)
            outcome.append("block")
        except StopIteration:
            outcome.append("stopped")

    reader = threading.Thread(target=read)
    reader.start()
    time.sleep(0.2)  # the reader is now waiting in next()
    stream.close()
    reader.join(timeout=2)
    gate.set()

    assert outcome == ["stopped"]
    with pytest.raises(StopIteration):
        next(stream)
