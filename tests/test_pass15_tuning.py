"""Pass 15 tuning tools: a trace re-run under other ranking settings, relevance settings
per workspace, the answer cache's view and clear, and each candidate's leg ranks."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.workspace_settings import WorkspaceSettingsStore
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait


def _client(**overrides: Any) -> Iterator[TestClient]:
    clear_dependency_caches()
    settings = make_settings(**overrides)
    with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as client:
        yield client
    clear_dependency_caches()


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield from _client(trace_leg_ranks=True, answer_cache_size=16, workspaces_enabled=True)


def _ask(client: TestClient, question: str, **body: Any) -> dict[str, Any]:
    response = client.post("/questions", json={"question": question, **body})
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_a_trace_is_run_again_under_another_bar_and_compared(client: TestClient) -> None:
    assert upload_and_wait(client, "guide.md", b"# Guide\nalpha beta gamma")["state"] == "done"
    answered = _ask(client, "What is alpha?")
    trace_id = answered["trace_id"]

    rerun = client.post(f"/traces/{trace_id}/rerun", json={"rerank_min_score": 1.0})
    assert rerun.status_code == 200, rerun.text
    body = rerun.json()
    assert body["query"] == "What is alpha?" and body["settings"] == {"rerank_min_score": 1.0}
    assert body["would_answer"] is False
    assert "rerank_min_score" in [s["name"] for s in body["diff"]["settings"]]
    assert body["candidates"]

    same = client.post(f"/traces/{trace_id}/rerun", json={}).json()
    assert same["would_answer"] is True and same["diff"]["settings"] == [
        s for s in same["diff"]["settings"] if s["name"] == "query_expansion_enabled"
    ]
    assert client.post("/traces/nope/rerun", json={}).status_code == 404
    bad = client.post(f"/traces/{trace_id}/rerun", json={"rerank_top_k": 0})
    assert bad.status_code == 422


def test_candidates_carry_their_leg_ranks_and_prompt_numbers(client: TestClient) -> None:
    assert upload_and_wait(client, "guide.md", b"# Guide\nalpha beta gamma")["state"] == "done"
    trace_id = _ask(client, "What is alpha?")["trace_id"]

    candidates = client.get(f"/traces/{trace_id}").json()["candidates"]

    assert candidates and all(c["dense_rank"] is not None for c in candidates)
    selected = [c for c in candidates if c["selected_for_context"]]
    assert [c["source_number"] for c in selected] == list(range(1, len(selected) + 1))


def test_a_workspace_bar_applies_to_its_questions_only_and_empties_its_cache(
    client: TestClient,
) -> None:
    assert client.post("/workspaces", json={"name": "legal"}).status_code == 201
    ws = {"X-Workspace": "legal"}
    for headers in ({}, ws):
        assert (
            upload_and_wait(client, "guide.md", b"# Guide\nalpha beta", headers=headers)["state"]
            == "done"
        )
    client.post("/questions", json={"question": "What is alpha?"}, headers=ws)
    assert client.get("/answer-cache", headers=ws).json()["entries"]

    put = client.put(
        "/workspaces/legal/settings", json={"rerank_min_score": 1.0, "answer_format": "bullets"}
    )
    assert put.status_code == 200, put.text
    assert put.json()["effective"]["rerank_min_score"] == 1.0
    # The change emptied the workspace's cached answers (given under the old bar).
    assert client.get("/answer-cache", headers=ws).json()["entries"] == []

    legal = client.post("/questions", json={"question": "What is alpha?"}, headers=ws).json()
    default = _ask(client, "What is alpha?")
    # Nothing clears a bar of 1.0 there: answered from the closest passages only.
    assert legal["weak_matches"] is True
    assert default["weak_matches"] is False and default["sources"]
    assert client.get("/workspaces/default/settings").json()["settings"] == {
        "rerank_min_score": None,
        "recency_boost_weight": None,
        "answer_language": None,
        "answer_format": None,
    }

    assert (
        client.put("/workspaces/legal/settings", json={}).json()["settings"]["rerank_min_score"]
        is None
    )
    assert client.get("/workspaces/nope/settings").status_code == 404
    bad = client.put("/workspaces/legal/settings", json={"answer_language": "<script>"})
    assert bad.status_code == 422


def test_workspace_settings_survive_a_restart(tmp_path: Path) -> None:
    path = str(tmp_path / "settings.db")
    WorkspaceSettingsStore(path).put("legal", {"rerank_min_score": 0.3, "unknown": 1})
    reopened = WorkspaceSettingsStore(path)
    assert reopened.get("legal") == {"rerank_min_score": 0.3}
    assert reopened.delete_workspace("legal") == 1
    assert WorkspaceSettingsStore(path).get("legal") == {}


def test_the_answer_cache_lists_and_clears_this_workspace(client: TestClient) -> None:
    assert upload_and_wait(client, "guide.md", b"# Guide\nalpha beta")["state"] == "done"
    _ask(client, "What is alpha?")
    _ask(client, "What is beta?")

    listing = client.get("/answer-cache").json()
    assert listing["enabled"] is True and listing["max_entries"] == 16
    assert sorted(entry["question"] for entry in listing["entries"]) == [
        "what is alpha?",
        "what is beta?",
    ]
    assert client.delete("/answer-cache").json() == {"removed": 2}
    assert client.get("/answer-cache").json()["entries"] == []
    assert _ask(client, "What is alpha?")["cached"] is False


def test_the_answer_cache_view_says_when_it_is_off() -> None:
    for client in _client(answer_cache_size=0):
        assert client.get("/answer-cache").json()["enabled"] is False
        assert client.delete("/answer-cache").json() == {"removed": 0}


def test_the_cli_runs_a_trace_again(client: TestClient) -> None:
    import io

    from cli.__main__ import main
    from cli.client import DocragClient

    assert upload_and_wait(client, "guide.md", b"# Guide\nalpha beta gamma")["state"] == "done"
    trace_id = _ask(client, "What is alpha?")["trace_id"]
    docrag = DocragClient("http://testserver", None, http=client, sleep=lambda _: None)

    out = io.StringIO()
    assert main(["traces", "rerun", trace_id, "--min-score", "1.0"], client=docrag, out=out) == 0
    assert "weak matches only" in out.getvalue()
    out = io.StringIO()
    assert main(["traces", "rerun", trace_id], client=docrag, out=out) == 0
    assert "would answer with the server settings" in out.getvalue()


def test_a_running_server_is_scored_through_its_search(client: TestClient, tmp_path: Path) -> None:
    from cli.client import DocragClient
    from eval.harness import RetrievalExample, main, run_retrieval_eval, server_ranker

    assert upload_and_wait(client, "guide.md", b"# Guide\nalpha beta gamma")["state"] == "done"
    docrag = DocragClient("http://testserver", None, http=client, sleep=lambda _: None)
    example = RetrievalExample(
        question="What is alpha?",
        relevant_filenames=frozenset({"guide.md"}),
        relevant_chunk_ids=frozenset(),
        reference=None,
    )

    result = run_retrieval_eval(
        [example],
        None,
        (1, 3),
        ranker=server_ranker(docrag, 3),
        fingerprint={"server": "http://testserver", "arm": "reranked"},
    )

    assert result["aggregate"]["filename"]["hit@1"] == 1.0
    assert result["fingerprint"]["server"] == "http://testserver"
    dataset = tmp_path / "set.jsonl"
    dataset.write_text('{"question": "q", "relevant_filenames": ["guide.md"]}\n')
    code = main(["retrieval", str(dataset), "--server", "http://x", "--baseline", str(dataset)])
    assert code == 2
