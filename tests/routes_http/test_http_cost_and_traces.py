"""Estimated LLM cost per answer (llm.usage_cost_usd) and traces kept on disk
(TRACE_STORE_PATH, api/trace_archive.py)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches, get_generator, get_trace_store
from api.llm import LiteLLMGenerator, model_prices, usage_cost_usd
from api.trace_archive import TraceArchive
from api.tracing import QueryTrace, TraceCandidate, TraceConfig, TraceStore, trace_from_json
from tests.routes_http.support import (
    hermetic_app,
    keyed_client,
    make_settings,
    parse_sse_events,
    upload_and_wait,
)


def test_cost_is_priced_for_listed_hosted_models_only() -> None:
    openai = make_settings(llm_provider="openai", openai_api_key="k", openai_model="gpt-4o-mini")
    prices = model_prices("gpt-4o-mini")
    assert prices is not None
    assert usage_cost_usd(openai, (1000, 100)) == pytest.approx(1000 * prices[0] + 100 * prices[1])
    assert usage_cost_usd(openai, None) is None
    unlisted = make_settings(llm_provider="openai", openai_api_key="k", openai_model="my-model")
    assert usage_cost_usd(unlisted, (1000, 100)) is None
    # Nothing is billed for a local model.
    assert usage_cost_usd(make_settings(llm_provider="ollama"), (1000, 100)) is None
    anthropic = make_settings(
        llm_provider="anthropic", anthropic_api_key="k", anthropic_model="claude-haiku-4-5"
    )
    assert usage_cost_usd(anthropic, (1000, 100)) is not None


def _openai_reply(text: str, prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def test_usage_and_cost_cover_every_call_of_a_question() -> None:
    """The condense call counts too (it used to start counting at generation)."""

    settings = make_settings(
        llm_provider="openai",
        openai_api_key="k",
        openai_model="gpt-4o-mini",
        citation_retry_enabled=False,
    )
    replies = [
        _openai_reply("What does alpha cover?", 100, 10),
        _openai_reply("Alpha is documented [1].", 1000, 20),
    ]

    def completion_client(**kwargs: Any) -> Any:
        return replies.pop(0)

    clear_dependency_caches()
    app = hermetic_app(settings, QdrantClient(":memory:"))
    app.dependency_overrides[get_generator] = lambda: LiteLLMGenerator(settings, completion_client)
    with TestClient(app) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        answer = client.post(
            "/questions",
            json={
                "question": "And what about it?",
                "history": [
                    {"role": "user", "content": "Tell me about alpha."},
                    {"role": "assistant", "content": "Alpha is a guide [1]."},
                ],
            },
        ).json()
        trace = client.get(f"/traces/{answer['trace_id']}").json()
    clear_dependency_caches()

    usage = answer["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (1100, 30)
    assert usage["cost_usd"] == pytest.approx(usage_cost_usd(settings, (1100, 30)))
    assert trace["usage"] == usage
    assert trace["config"]["llm_model"] == "gpt-4o-mini"


def _trace(trace_id: str, workspace: str = "", created_at: float | None = None) -> QueryTrace:
    return QueryTrace(
        trace_id=trace_id,
        created_at=created_at if created_at is not None else time.time(),
        question=f"question {trace_id}",
        mode="sync",
        status="ok",
        config=TraceConfig(
            llm_provider="ollama",
            rerank_top_k=3,
            max_context_chunks=2,
            llm_temperature=0.1,
            rerank_min_score=0.15,
            fused_top_n=5,
            filenames=None,
            narrowing={"page_min": 2},
        ),
        candidates=[
            TraceCandidate(
                point_id="p",
                filename="a.txt",
                page=1,
                section="Intro",
                chunk_id="c",
                retrieval_score=0.5,
                rerank_score=0.9,
                kept=True,
                drop_reason=None,
                selected_for_context=True,
                neighbor_expanded=False,
            )
        ],
        prompt_messages=[{"role": "user", "content": "q"}],
        answer="a [1]",
        cited_source_numbers=[1],
        timings={"total_ms": 1.0},
        error=None,
        usage=(10, 2),
        cost_usd=0.001,
        workspace=workspace,
    )


def test_an_archived_trace_survives_the_ring_and_a_restart(tmp_path: Path) -> None:
    path = str(tmp_path / "traces.db")
    store = TraceStore(max_retained=1, archive=TraceArchive(path, max_rows=100))
    for index in range(3):
        store.add(_trace(f"t{index}", created_at=1000.0 + index))

    # Evicted from the ring, read back from disk.
    assert store.get("t0") == _trace("t0", created_at=1000.0)

    restarted = TraceStore(max_retained=10, archive=TraceArchive(path, max_rows=100))
    assert [trace.trace_id for trace in restarted.list_traces("")] == ["t2", "t1", "t0"]
    lines = list(restarted.export_lines(""))
    assert [json.loads(line)["trace_id"] for line in lines] == ["t0", "t1", "t2"]
    assert all(line.endswith("\n") for line in lines)


def test_the_archive_is_trimmed_and_scoped_by_workspace(tmp_path: Path) -> None:
    archive = TraceArchive(str(tmp_path / "traces.db"), max_rows=2)
    for index in range(4):
        archive.add(_trace(f"t{index}", workspace="team" if index % 2 else ""))
    # Trimmed on the first insert of every hundred, so the 101st add trims to 2.
    for index in range(4, 101):
        archive.add(_trace(f"t{index}"))
    assert len(archive.recent(None, 1000)) == 2
    assert [t.trace_id for t in archive.recent("team", 10)] == []


def test_a_row_from_another_version_still_reads(tmp_path: Path) -> None:
    data = dataclasses.asdict(_trace("t"))
    data["from_the_future"] = 1
    data["config"]["also_new"] = True
    del data["cost_usd"]
    trace = trace_from_json(json.dumps(data))
    assert trace.trace_id == "t" and trace.cost_usd is None
    assert trace.usage == (10, 2)


def test_an_unwritable_archive_never_fails_the_answer() -> None:
    class BrokenArchive:
        def add(self, trace: QueryTrace) -> None:
            raise OSError("disk full")

        def get(self, trace_id: str) -> QueryTrace | None:
            raise OSError("disk full")

        def recent(self, workspace: str | None, limit: int) -> list[QueryTrace]:
            raise OSError("disk full")

        def export_lines(self, workspace: str):  # type: ignore[no-untyped-def]
            yield from ()

    store = TraceStore(max_retained=5, archive=BrokenArchive())
    store.add(_trace("t"))
    assert store.get("t") is not None
    assert store.get("missing") is None
    assert [trace.trace_id for trace in store.list_traces("")] == ["t"]


def test_traces_export_needs_the_full_key_and_streams_json_lines(tmp_path: Path) -> None:
    archive_path = str(tmp_path / "traces.db")
    store = TraceStore(max_retained=10, archive=TraceArchive(archive_path, max_rows=100))
    full = {"X-API-Key": "full-key"}
    with keyed_client(
        "full-key", {get_trace_store: lambda: store}, api_read_key="read-key"
    ) as client:
        assert (
            upload_and_wait(client, "a.txt", b"Intro\nalpha beta", headers=full)["state"] == "done"
        )
        asked = client.post("/questions", json={"question": "alpha"}, headers=full).json()

        assert client.get("/traces/export", headers={"X-API-Key": "read-key"}).status_code == 403
        response = client.get("/traces/export", headers=full)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert "attachment" in response.headers["content-disposition"]
    rows = [json.loads(line) for line in response.text.splitlines()]
    assert [row["trace_id"] for row in rows] == [asked["trace_id"]]


@pytest.mark.parametrize("path", ["/questions", "/questions/stream"])
def test_usage_covers_the_answer_not_the_follow_ups_on_both_endpoints(path: str) -> None:
    settings = make_settings(
        llm_provider="openai",
        openai_api_key="k",
        openai_model="gpt-4o-mini",
        citation_retry_enabled=False,
        followups_enabled=True,
    )
    replies = [
        _openai_reply("Alpha is documented [1].", 1000, 20),
        _openai_reply("What else is in alpha?\nWho wrote alpha?", 500, 40),
    ]

    def completion_client(**kwargs: Any) -> Any:
        reply = replies.pop(0)
        if kwargs.get("stream"):
            text = reply["choices"][0]["message"]["content"]
            return iter(
                [
                    {"choices": [{"delta": {"content": text}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": reply["usage"]},
                ]
            )
        return reply

    clear_dependency_caches()
    app = hermetic_app(settings, QdrantClient(":memory:"))
    app.dependency_overrides[get_generator] = lambda: LiteLLMGenerator(settings, completion_client)
    with TestClient(app) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        response = client.post(path, json={"question": "What is alpha?"})
        if path == "/questions":
            body = response.json()
        else:
            body = next(
                event for event in parse_sse_events(response.text) if event["type"] == "done"
            )
        trace = client.get(f"/traces/{body['trace_id']}").json()
    clear_dependency_caches()

    assert not replies  # the follow-ups were asked for
    usage = body["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (1000, 20)
    assert trace["usage"] == usage
    assert trace["answer_status"] == "answered"
