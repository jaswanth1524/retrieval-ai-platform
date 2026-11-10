"""Pass 16 answer-path fixes: summaries of a page range counted and read through the
index's filter, error traces that keep what was found, and the rest."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait


@pytest.fixture
def client() -> Iterator[TestClient]:
    clear_dependency_caches()
    with TestClient(hermetic_app(make_settings(), QdrantClient(":memory:"))) as client:
        yield client
    clear_dependency_caches()


def test_a_summary_of_pages_with_nothing_on_them_is_refused(client: TestClient) -> None:
    assert upload_and_wait(client, "a.txt", b"Intro\nalpha beta gamma")["state"] == "done"

    empty = client.post(
        "/questions/summary", json={"mode": "summarize", "filenames": ["a.txt"], "page_min": 5}
    )
    assert empty.status_code == 400
    assert "no passages from page 5 on" in empty.json()["detail"]
    missing = client.post(
        "/questions/summary", json={"mode": "summarize", "filenames": ["b.txt"], "page_min": 5}
    )
    assert missing.status_code == 404


def test_the_summary_cap_counts_the_page_range() -> None:
    from api.repository import PassageFilter
    from api.summarize import SUMMARY_MAX_DOCUMENT_CHUNKS, SummaryRequest, SummaryTooLargeError
    from tests.test_summarize import FakeRepository, ScriptedGenerator, _payload, _summarizer

    pages = [_payload("a.txt", n, f"Page {n} text.", page=n) for n in range(1, 7)]
    # Far too long to summarize whole; pages 2-3 are two passages.
    repository = FakeRepository({"a.txt": pages})
    repository.counts["a.txt"] = SUMMARY_MAX_DOCUMENT_CHUNKS + 1
    summarizer = _summarizer(repository, ScriptedGenerator(answer="Pages two and three [1]."))

    with pytest.raises(SummaryTooLargeError):
        summarizer.check_documents(["a.txt"])
    narrow = PassageFilter(page_min=2, page_max=3)
    summarizer.check_documents(["a.txt"], narrow)
    result = summarizer.summarize(SummaryRequest("summarize", ("a.txt",), narrow=narrow))
    assert result.chunks_total == 2


def test_an_answer_says_which_bar_it_was_given_under(client: TestClient) -> None:
    import json

    assert upload_and_wait(client, "a.txt", b"Intro\nalpha beta gamma")["state"] == "done"
    bar = client.get("/config").json()["rerank_min_score"]

    sync = client.post("/questions", json={"question": "alpha", "use_cache": False}).json()
    assert sync["rerank_min_score"] == bar
    with client.stream("POST", "/questions/stream", json={"question": "alpha"}) as response:
        frames = [
            json.loads(line[len("data:") :])
            for line in response.iter_lines()
            if line.startswith("data:")
        ]
    (done,) = [frame for frame in frames if frame["type"] == "done"]
    assert done["rerank_min_score"] == bar


def test_a_summary_part_is_retried_only_when_another_try_could_help() -> None:
    from collections.abc import Sequence

    from api.llm import ChatMessage, GenerationError
    from api.settings import AppSettings
    from api.summarize import SummaryRequest
    from tests.factories import make_test_settings
    from tests.test_summarize import FakeRepository, ScriptedGenerator, _payload, _summarizer

    class Failing(ScriptedGenerator):
        def __init__(self, error: GenerationError) -> None:
            super().__init__(answer="Summary [1].")
            self.error = error
            self.note_calls = 0

        def _reply(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
            if "Excerpts from" in messages[-1]["content"] and self.note_calls == 0:
                self.note_calls += 1
                raise self.error
            return super()._reply(messages, settings)

    text = "word " * 150
    repository = FakeRepository({"a.txt": [_payload("a.txt", n, text) for n in range(1, 9)]})
    settings = make_test_settings(
        llm_provider="ollama", summary_batch_chars=1000, summary_max_batches=4
    )
    request = SummaryRequest(mode="summarize", filenames=("a.txt",))

    hiccup = _summarizer(repository, Failing(GenerationError("hiccup")), settings)
    assert hiccup.summarize(request, use_cache=False).parts_skipped == 0
    timed_out = Failing(GenerationError("timed out", worth_retrying=False))
    result = _summarizer(repository, timed_out, settings).summarize(request, use_cache=False)
    assert result.parts_skipped == 1


def test_failed_follow_ups_keep_the_answer_out_of_the_cache() -> None:
    from api.llm import begin_usage, was_degraded
    from api.prompting import generate_followups
    from tests.factories import make_test_settings
    from tests.test_pass15_answers import _Failing, _scored

    begin_usage()
    settings = make_test_settings(followups_enabled=True)
    sources = [_scored("a.txt", 0.9)]
    assert generate_followups("q", "An answer.", sources, _Failing(), settings) == []
    assert was_degraded()
