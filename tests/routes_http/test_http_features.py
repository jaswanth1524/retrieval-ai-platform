"""Access level, search, ingest-job cancellation and backup restore."""

from __future__ import annotations

import io
import json
import threading
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from api.admission import QuestionSlots
from api.dependencies import (
    get_embedding_provider,
    get_feedback_store,
    get_ingest_executor,
    get_question_slots,
    get_raw_document_store,
)
from api.embeddings import EmbeddedText
from api.feedback import FeedbackStore
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from tests.routes_http.support import (
    ApiTestContext,
    FakeEmbeddingProvider,
    keyed_client,
    upload_and_wait,
    wait_for_job,
)

FULL = {"X-API-Key": "full-key"}
READ = {"X-API-Key": "read-key"}


# --- GET /access -------------------------------------------------------------------


def test_access_reports_open_when_no_key_is_configured(api_context: ApiTestContext) -> None:
    assert api_context.client.get("/access").json() == {"access": "open"}


def test_access_reports_which_key_the_caller_holds() -> None:
    with keyed_client("full-key", api_read_key="read-key") as client:
        assert client.get("/access", headers=FULL).json() == {"access": "full"}
        assert client.get("/access", headers=READ).json() == {"access": "read"}
        assert client.get("/access").status_code == 401
        assert client.get("/access", headers={"X-API-Key": "wrong"}).status_code == 401


# --- POST /search -------------------------------------------------------------------


def test_search_returns_ranked_passages_without_calling_the_llm(
    api_context: ApiTestContext,
) -> None:
    client = api_context.client
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"

    response = client.post("/search", json={"query": "alpha"})

    assert response.status_code == 200
    body = response.json()
    first = body["results"][0]
    assert first["rank"] == 1
    assert first["filename"] == "guide.txt"
    assert "alpha" in first["text"]
    assert {"page", "section", "chunk_id", "chunk_ordinal", "retrieval_score", "rerank_score"} <= (
        first.keys()
    )
    assert set(body["timings"]) == {"embed_ms", "search_ms", "rerank_ms"}
    assert api_context.generator.messages == []


def test_search_skips_llm_query_expansion_even_when_it_is_on() -> None:
    from api.dependencies import get_generator
    from tests.routes_http.support import FakeGenerator

    generator = FakeGenerator("one\ntwo")
    with keyed_client(
        "",
        extra_overrides={get_generator: lambda: generator},
        query_expansion_enabled=True,
    ) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        assert client.post("/search", json={"query": "alpha"}).status_code == 200
    assert generator.messages == []


def test_search_validates_and_respects_the_question_cap(api_context: ApiTestContext) -> None:
    client = api_context.client
    assert client.post("/search", json={"query": ""}).status_code == 422
    too_many = api_context.settings.fused_top_n + 1
    assert client.post("/search", json={"query": "a", "rerank_top_k": too_many}).status_code in (
        400,
        422,
    )
    full = QuestionSlots(1)
    held = full.try_acquire()
    assert held is not None
    api_context.app.dependency_overrides[get_question_slots] = lambda: full
    from api.metrics import questions_total

    busy_questions = questions_total.labels(outcome="error", error_type="busy")
    before = busy_questions._value.get()
    busy = client.post("/search", json={"query": "alpha"})
    assert busy.status_code == 429
    # A search isn't a question: it stays out of docrag_questions_total.
    assert busy_questions._value.get() == before
    held()


def test_the_read_key_may_search() -> None:
    with keyed_client("full-key", api_read_key="read-key") as client:
        assert upload_and_wait(client, "g.txt", b"Intro\nalpha", headers=FULL)["state"] == "done"
        assert client.post("/search", json={"query": "alpha"}, headers=READ).status_code == 200
        assert client.post("/search", json={"query": "alpha"}).status_code == 401


# --- DELETE /documents/jobs/{job_id} ------------------------------------------------


def test_a_queued_job_can_be_cancelled_and_indexes_nothing(api_context: ApiTestContext) -> None:
    client = api_context.client
    single = ThreadPoolExecutor(max_workers=1)
    busy = threading.Event()
    single.submit(busy.wait, 5)
    api_context.app.dependency_overrides[get_ingest_executor] = lambda: single
    try:
        job_id = client.post(
            "/documents", files={"file": ("q.txt", b"Intro\nalpha", "text/plain")}
        ).json()["job_id"]
        cancelled = client.delete(f"/documents/jobs/{job_id}")
        busy.set()
    finally:
        busy.set()
        single.shutdown(wait=True)

    assert cancelled.status_code == 202
    job = client.get(f"/documents/jobs/{job_id}").json()
    assert job["state"] == "failed" and job["cancelled"] is True
    assert "q.txt" not in client.get("/documents").json()["filenames"]


class GatedEmbeddings(FakeEmbeddingProvider):
    """Blocks embedding batch number ``gate_on`` (1-based) until released."""

    def __init__(self, gate_on: int) -> None:
        super().__init__()
        self.gate_on = gate_on
        self.calls = 0
        self.reached = threading.Event()
        self.release = threading.Event()

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        self.calls += 1
        if self.calls == self.gate_on:
            self.reached.set()
            self.release.wait(5)
        return super().embed_texts(texts)


def _four_sections() -> bytes:
    sections = (f"Section {i}\n" + " ".join(f"w{i}x{j}" for j in range(30)) for i in range(4))
    return "\n\n".join(sections).encode()


def _start_gated_upload(api_context: ApiTestContext, gate_on: int) -> tuple[str, GatedEmbeddings]:
    embeddings = GatedEmbeddings(gate_on)
    api_context.app.dependency_overrides[get_embedding_provider] = lambda: embeddings
    job_id = api_context.client.post(
        "/documents", files={"file": ("big.txt", _four_sections(), "text/plain")}
    ).json()["job_id"]
    assert embeddings.reached.wait(5)
    return job_id, embeddings


def test_a_running_job_is_cancelled_at_its_last_point_before_writing(
    api_context: ApiTestContext, caplog: pytest.LogCaptureFixture
) -> None:
    job_id, embeddings = _start_gated_upload(api_context, gate_on=1)

    response = api_context.client.delete(f"/documents/jobs/{job_id}")
    embeddings.release.set()
    job = wait_for_job(api_context.client, job_id)

    assert response.status_code == 202
    assert job["state"] == "failed" and job["cancelled"] is True
    assert "big.txt" not in api_context.client.get("/documents").json()["filenames"]
    # A cancel the user asked for is not logged as a failure with a traceback.
    assert not [r for r in caplog.records if r.levelno >= 30 and r.name == "api.ingest_jobs"]


def test_a_job_still_embedding_after_its_first_batch_can_be_cancelled(
    api_context: ApiTestContext,
) -> None:
    """Nothing is written until every batch is embedded, so the cancel window covers
    the whole embedding run, not just its first batch."""

    api_context.settings.__dict__["ingest_batch_size"] = 1
    job_id, embeddings = _start_gated_upload(api_context, gate_on=2)

    response = api_context.client.delete(f"/documents/jobs/{job_id}")
    embeddings.release.set()
    job = wait_for_job(api_context.client, job_id)

    assert response.status_code == 202
    assert job["state"] == "failed" and job["cancelled"] is True
    assert embeddings.calls == 2  # stopped at the next checkpoint, not run to the end
    assert "big.txt" not in api_context.client.get("/documents").json()["filenames"]


def test_a_job_that_already_wrote_cannot_be_cancelled(
    api_context: ApiTestContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_context.settings.__dict__["ingest_batch_size"] = 1
    writing, release = threading.Event(), threading.Event()
    upsert = VectorRepository.upsert
    calls: list[int] = []

    def gated_upsert(self: VectorRepository, settings: Any, points: Any) -> None:
        calls.append(len(points))
        if len(calls) == 2:
            writing.set()
            release.wait(5)
        upsert(self, settings, points)

    monkeypatch.setattr(VectorRepository, "upsert", gated_upsert)
    job_id = api_context.client.post(
        "/documents", files={"file": ("big.txt", _four_sections(), "text/plain")}
    ).json()["job_id"]
    assert writing.wait(5)

    response = api_context.client.delete(f"/documents/jobs/{job_id}")
    release.set()
    job = wait_for_job(api_context.client, job_id)

    assert response.status_code == 409
    assert job["state"] == "done" and job["cancelled"] is False


def test_tags_and_delete_are_not_held_up_by_an_embedding_ingest(
    api_context: ApiTestContext,
) -> None:
    """Embedding used to run under the filename lock, so a tag change or a DELETE of
    the document waited on a request thread for the whole re-index."""

    client = api_context.client
    assert upload_and_wait(client, "big.txt", b"Intro\nalpha beta")["state"] == "done"
    job_id, embeddings = _start_gated_upload(api_context, gate_on=1)

    statuses: list[int] = []
    worker = threading.Thread(
        target=lambda: statuses.append(
            client.patch("/documents/big.txt/tags", json={"tags": ["finance"]}).status_code
        ),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=3)
    finished_while_embedding = not worker.is_alive()
    embeddings.release.set()
    worker.join(timeout=10)

    assert finished_while_embedding, "PATCH tags waited for the ingest's embedding"
    assert statuses == [200]
    assert wait_for_job(client, job_id)["state"] == "done"
    # Read under the lock at write time, the new tag reaches the re-indexed chunks.
    assert client.get("/documents").json()["tags"] == {"big.txt": ["finance"]}


def test_cancelling_an_unknown_or_finished_job(api_context: ApiTestContext) -> None:
    client = api_context.client
    assert client.delete("/documents/jobs/nope").status_code == 404
    job_id = client.post(
        "/documents", files={"file": ("d.txt", b"Intro\nalpha", "text/plain")}
    ).json()["job_id"]
    wait_for_job(client, job_id)
    assert client.delete(f"/documents/jobs/{job_id}").status_code == 409


def test_the_read_key_may_not_cancel_jobs() -> None:
    with keyed_client("full-key", api_read_key="read-key") as client:
        assert client.delete("/documents/jobs/x", headers=READ).status_code == 403


# --- POST /import -------------------------------------------------------------------


def _backup_client(tmp_path: Path, name: str) -> Any:
    raw = RawDocumentStore(str(tmp_path / f"{name}-raw"))
    feedback = FeedbackStore(str(tmp_path / f"{name}-feedback.db"))
    return keyed_client(
        "full-key",
        extra_overrides={get_raw_document_store: lambda: raw, get_feedback_store: lambda: feedback},
        feedback_enabled=True,
        api_read_key="read-key",
    )


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _import(client: Any, content: bytes, headers: dict[str, str] = FULL) -> Any:
    return client.post(
        "/import", files={"file": ("backup.zip", content, "application/zip")}, headers=headers
    )


def test_an_export_restores_into_a_fresh_instance(tmp_path: Path) -> None:
    with _backup_client(tmp_path, "old") as old:
        assert upload_and_wait(old, "guide.txt", b"Intro\nalpha beta", headers=FULL)["state"] == (
            "done"
        )
        old.patch("/documents/guide.txt/tags", json={"tags": ["hr"]}, headers=FULL)
        old.post(
            "/feedback",
            headers=FULL,
            json={
                "trace_id": None,
                "question": "q",
                "answer_excerpt": "a",
                "cited_filenames": ["guide.txt"],
                "rating": "up",
                "citation_source_number": None,
            },
        )
        before = old.get("/documents", headers=FULL).json()
        backup = old.get("/export", headers=FULL).content

    with _backup_client(tmp_path, "new") as new:
        response = _import(new, backup)
        assert response.status_code == 202
        body = response.json()
        assert [job["filename"] for job in body["jobs"]] == ["guide.txt"]
        assert body["feedback_imported"] == 1
        for job in body["jobs"]:
            assert wait_for_job_with(new, job["job_id"])["state"] == "done"
        after = new.get("/documents", headers=FULL).json()
        assert after["tags"] == {"guide.txt": ["hr"]}
        assert after["uploaded_ats"] == before["uploaded_ats"]
        assert new.get("/documents/guide.txt/original", headers=FULL).content == (
            b"Intro\nalpha beta"
        )

        again = _import(new, backup).json()
        assert again["jobs"] == [] and again["skipped"] == ["guide.txt"]
        assert again["feedback_imported"] == 0 and again["feedback_skipped"] == 1


def wait_for_job_with(client: Any, job_id: str) -> dict[str, Any]:
    return wait_for_job(client, job_id, headers=FULL)


@pytest.mark.parametrize(
    ("content", "detail"),
    [
        (b"not a zip", "isn't a zip"),
        (_zip({"readme.txt": b"x"}), "no manifest.json"),
        (_zip({"manifest.json": b"{"}), "not a DocRAG backup manifest"),
        (
            _zip({"manifest.json": json.dumps({"manifest_version": 99, "documents": []}).encode()}),
            "Unsupported backup format",
        ),
    ],
)
def test_import_refuses_what_is_not_a_backup(tmp_path: Path, content: bytes, detail: str) -> None:
    with _backup_client(tmp_path, "x") as client:
        response = _import(client, content)
    assert response.status_code == 400
    assert detail in response.json()["detail"]


def test_import_reads_only_expected_members_and_caps_each_one(tmp_path: Path) -> None:
    manifest = {
        "manifest_version": 1,
        "documents": [
            {"filename": "../evil.txt", "original": "originals/../evil.txt"},
            {"filename": "nooriginal.txt", "original": None},
            {"filename": "big.txt", "original": "originals/big.txt"},
            {"filename": "ok.txt", "original": "originals/ok.txt", "tags": ["a", 7]},
        ],
    }
    content = _zip(
        {
            "manifest.json": json.dumps(manifest).encode(),
            "originals/../evil.txt": b"Intro\nevil",
            "originals/big.txt": b"x" * 2000,
            "originals/ok.txt": b"Intro\nalpha",
        }
    )
    raw = RawDocumentStore(str(tmp_path / "raw"))
    with keyed_client(
        "full-key",
        extra_overrides={get_raw_document_store: lambda: raw},
        max_upload_bytes=1000,
    ) as client:
        body = _import(client, content).json()
        assert [job["filename"] for job in body["jobs"]] == ["ok.txt"]
        assert body["missing_originals"] == ["nooriginal.txt"]
        assert set(body["rejected"]) == {"../evil.txt", "big.txt"}
        wait_for_job_with(client, body["jobs"][0]["job_id"])
        assert client.get("/documents", headers=FULL).json()["tags"] == {"ok.txt": ["a"]}
    assert not (tmp_path / "evil.txt").exists()


def test_import_needs_the_full_key_and_is_size_capped(tmp_path: Path) -> None:
    with keyed_client("full-key", api_read_key="read-key", max_import_bytes=1000) as client:
        assert _import(client, b"x", headers=READ).status_code == 403
        assert _import(client, b"x" * 200_000).status_code == 413
