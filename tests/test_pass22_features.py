"""Pass 22: features."""

from __future__ import annotations

import threading
from typing import Any

from api.dependencies import clear_dependency_caches
from tests.routes_http.support import upload_and_wait, wait_until
from tests.test_pass21_features import _DOC, _client


def test_a_batch_runs_in_the_background_and_its_answers_can_be_read_as_they_come() -> None:
    answer = "Staff take annual leave of 28 days after asking their manager [1]."
    with _client(answer) as client:
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        started = client.post("/questions/batch/jobs", json={"questions": ["leave?", "days?"]})
        assert started.status_code == 202, started.text
        batch_id = started.json()["batch_id"]
        assert started.json()["total"] == 2
        wait_until(
            lambda: client.get(f"/questions/batch/jobs/{batch_id}").json()["state"] == "done"
        )
        job = client.get(f"/questions/batch/jobs/{batch_id}").json()
        assert job["done"] == 2 and [a["question"] for a in job["answers"]] == ["leave?", "days?"]
        assert all(a["answer"]["answer"] == answer for a in job["answers"])
        assert client.get("/questions/batch/jobs/nope").status_code == 404
        # Another workspace can't read it.
        other = client.get(f"/questions/batch/jobs/{batch_id}", headers={"X-Workspace": "x"})
        assert other.status_code in (400, 404)
    clear_dependency_caches()


def test_a_batch_waits_for_a_slot_and_can_be_cancelled() -> None:
    from api.batch_jobs import BatchJobStore, run_batch_job

    store = BatchJobStore()
    job = store.add("c", ["a", "b", "c"])
    assert job is not None
    free = threading.Event()
    answered: list[str] = []

    def take_slot() -> Any:
        return (lambda: None) if free.is_set() else None

    def answer(question: str, should_stop: Any) -> str:
        answered.append(question)
        if question == "a":
            job.cancel.set()
        return question

    runner = threading.Thread(target=run_batch_job, args=(job, take_slot, answer, None))
    runner.start()
    threading.Event().wait(0.2)
    assert job.state == "running" and answered == []  # waiting for a slot, not failing
    free.set()
    runner.join(5)
    assert answered == ["a"] and job.answers == ["a"] and job.state == "cancelled"


def test_at_most_two_batches_run_at_once() -> None:
    from api.batch_jobs import MAX_RUNNING_BATCH_JOBS, BatchJobStore

    store = BatchJobStore()
    jobs = [store.add("c", ["q"]) for _ in range(MAX_RUNNING_BATCH_JOBS)]
    assert all(jobs) and store.add("c", ["q"]) is None
    assert jobs[0] is not None
    jobs[0].state = "done"
    assert store.add("c", ["q"]) is not None


def test_the_cli_follows_a_background_batch(tmp_path: Any, monkeypatch: Any) -> None:
    import io

    import cli.question_commands as question_commands
    from cli.__main__ import main
    from cli.client import DocragClient

    monkeypatch.setattr(question_commands, "BATCH_POLL_SECONDS", 0.05)
    with _client("Staff take annual leave of 28 days after asking their manager [1].") as client:
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        questions = tmp_path / "q.txt"
        questions.write_text("leave?\nmanager?\n", encoding="utf-8")
        out = io.StringIO()
        code = main(
            ["ask", "--questions", str(questions), "--background", "--quiet"],
            client=DocragClient("http://testserver", None, http=client),
            out=out,
        )
        assert code == 0, out.getvalue()
        assert "1. leave?" in out.getvalue() and "2. manager?" in out.getvalue()
    clear_dependency_caches()


def test_a_batch_that_ends_early_fails_the_cli(tmp_path: Any, monkeypatch: Any) -> None:
    import argparse
    import io

    import cli.question_commands as question_commands

    class _Client:
        def start_batch_job(self, questions: list[str], **_: Any) -> dict[str, Any]:
            return {"batch_id": "b1", "total": len(questions)}

        def batch_job(self, batch_id: str) -> dict[str, Any]:
            answer = {"answer": "a [1].", "answer_status": "answered", "sources": []}
            return {
                "state": "cancelled",
                "done": 1,
                "total": 2,
                "answers": [{"question": "one?", "answer": answer}],
            }

    questions = tmp_path / "q.txt"
    questions.write_text("one?\ntwo?\n", encoding="utf-8")
    args = argparse.Namespace(
        questions=str(questions),
        background=True,
        batch=None,
        quiet=True,
        json=True,
        file=None,
        tag=None,
        provider=None,
        strict=False,
        format=None,
    )
    monkeypatch.setattr(question_commands, "question_options", lambda args: {})
    out = io.StringIO()
    assert question_commands.cmd_ask_many(_Client(), args, out) == 1  # type: ignore[arg-type]
    assert "not answered" in out.getvalue()


def test_a_batch_whose_thread_cant_start_doesnt_hold_a_running_place(monkeypatch: Any) -> None:
    import api.routes.questions as question_routes
    from api.batch_jobs import MAX_RUNNING_BATCH_JOBS, batch_job_store

    def no_thread(*args: Any) -> None:
        raise RuntimeError("can't start new thread")

    from fastapi.testclient import TestClient

    with _client("unused") as plain:
        client = TestClient(plain.app, raise_server_exceptions=False)
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        monkeypatch.setattr(question_routes, "_start_batch_thread", no_thread)
        for _ in range(MAX_RUNNING_BATCH_JOBS + 1):
            response = client.post("/questions/batch/jobs", json={"questions": ["q?"]})
            assert response.status_code == 500
        monkeypatch.undo()
        assert batch_job_store().add("elsewhere", ["q"]) is not None
    clear_dependency_caches()
