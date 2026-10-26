"""Readiness reflects whether the models loaded (api/model_health.py), and actions that
change or remove documents leave an audit line (api/audit.py)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping

import pytest

from api.dependencies import get_ollama_reachability_checker
from api.logging_config import JsonFormatter
from api.main import warm_models
from api.model_health import ModelHealth, ModelState
from tests.routes_http.support import (
    ApiTestContext,
    FakeReranker,
    keyed_client,
    make_settings,
    upload_and_wait,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_failed_model_degrades_and_a_tokenizer_fallback_does_not() -> None:
    health = ModelHealth()
    health.record({"embedding": "ok", "reranker": "ok", "tokenizer": "failed"})
    assert not health.degraded()

    health.record({"reranker": "failed"})
    assert health.degraded()
    assert health.states() == {"embedding": "ok", "reranker": "failed", "tokenizer": "failed"}


def test_retry_rewarms_only_the_failed_models_at_most_once_per_interval() -> None:
    clock = _Clock()
    asked: list[frozenset[str]] = []
    outcome: dict[str, ModelState] = {"reranker": "failed"}

    def rewarm(names: frozenset[str]) -> Mapping[str, ModelState]:
        asked.append(names)
        return dict(outcome)

    health = ModelHealth(rewarm, retry_seconds=60, clock=clock)
    health.record({"embedding": "ok", "reranker": "failed", "tokenizer": "failed"})

    first = health.retry_failed()
    assert first is not None
    first.join(5)
    # The tokenizer's own fallback is never re-warmed from here.
    assert asked == [frozenset({"reranker"})]
    assert health.degraded()

    clock.now = 30
    assert health.retry_failed() is None

    clock.now = 61
    outcome["reranker"] = "ok"
    second = health.retry_failed()
    assert second is not None
    second.join(5)
    assert not health.degraded()
    assert health.retry_failed() is None


def test_a_raising_retry_keeps_the_previous_state() -> None:
    def rewarm(names: frozenset[str]) -> Mapping[str, ModelState]:
        raise RuntimeError("still offline")

    health = ModelHealth(rewarm)
    health.record({"reranker": "failed"})
    thread = health.retry_failed()
    assert thread is not None
    thread.join(5)
    assert health.states() == {"reranker": "failed"}


def test_warm_models_counts_a_model_that_cannot_be_built_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> FakeReranker:
        raise ValueError("no such model")

    monkeypatch.setattr("api.main.get_reranker", broken)

    states = warm_models(make_settings(), frozenset({"reranker"}))

    assert states == {"reranker": "failed"}


def test_readiness_is_degraded_while_a_model_failed_and_recovers(
    api_context: ApiTestContext,
) -> None:
    api_context.app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: True
    outcome: dict[str, ModelState] = {"reranker": "failed"}
    health = ModelHealth(lambda names: dict(outcome), retry_seconds=0)
    health.record({"embedding": "ok", "reranker": "failed", "tokenizer": "ok"})
    api_context.app.state.model_health = health

    response = api_context.client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["models"] == {"embedding": "ok", "reranker": "failed", "tokenizer": "ok"}

    outcome["reranker"] = "ok"
    # The probe above started a retry that still failed; this one recovers.
    _wait_for_retry(health)
    api_context.client.get("/health/ready")
    _wait_for_retry(health)

    response = api_context.client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["models"]["reranker"] == "ok"


def _wait_for_retry(health: ModelHealth) -> None:
    deadline = time.monotonic() + 5
    while health._retrying and time.monotonic() < deadline:  # noqa: SLF001
        time.sleep(0.01)


def _audit_records(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [r.audit for r in caplog.records if r.name == "api.audit"]  # type: ignore[attr-defined]


def test_delete_and_tag_changes_are_audited(
    api_context: ApiTestContext, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="api.audit")
    upload_and_wait(api_context.client, "notes.txt", b"alpha beta gamma")

    assert (
        api_context.client.patch("/documents/notes.txt/tags", json={"tags": ["Finance"]})
    ).status_code == 200
    assert api_context.client.delete("/documents/notes.txt").status_code == 200

    records = _audit_records(caplog)
    assert [r["event"] for r in records] == ["document.upload", "document.tags", "document.delete"]
    upload, tags, delete = records
    assert upload["filename"] == "notes.txt"
    assert upload["size"] == len(b"alpha beta gamma")
    assert tags["tags"] == ["Finance"]
    assert delete["filename"] == "notes.txt"
    assert delete["workspace"] == "default"
    # No API_KEY configured.
    assert delete["key"] == "open"
    assert delete["client"] == "testclient"


def test_audit_records_which_key_was_used(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="api.audit")
    headers = {"X-API-Key": "secret"}
    with keyed_client("secret") as client:
        upload_and_wait(client, "notes.txt", b"alpha beta gamma", headers=headers)
        assert client.delete("/documents/notes.txt", headers=headers).status_code == 200

    upload, record = _audit_records(caplog)
    assert upload["event"] == "document.upload"
    assert record["key"] == "full"


def test_a_failed_delete_is_not_audited(
    api_context: ApiTestContext, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="api.audit")
    assert api_context.client.delete("/documents/missing.txt").status_code == 404
    assert _audit_records(caplog) == []


def test_audit_fields_are_an_object_in_json_logs() -> None:
    record = logging.LogRecord("api.audit", logging.INFO, __file__, 1, "x", None, None)
    record.audit = {"event": "document.delete", "filename": "a.txt"}
    record.request_id = "r1"

    entry = json.loads(JsonFormatter().format(record))

    assert entry["audit"] == {"event": "document.delete", "filename": "a.txt"}
