"""Pass 21: features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches, get_generator
from api.settings import AppSettings
from tests.routes_http.support import hermetic_app, upload_and_wait
from tests.routes_http.support import make_settings as route_settings
from tests.test_pass17_answers import _sse_events

_DOC = b"Leave\nStaff take annual leave of 28 days after asking their manager in writing."


class _Says:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, messages: Sequence[Any], settings: AppSettings) -> str:
        return self.text

    def stream(self, messages: Sequence[Any], settings: AppSettings) -> Any:
        yield self.text


def _client(text: str, **overrides: Any) -> Any:
    clear_dependency_caches()
    app = hermetic_app(
        route_settings(citation_retry_enabled=False, followups_enabled=False, **overrides),
        QdrantClient(":memory:"),
    )
    app.dependency_overrides[get_generator] = lambda: _Says(text)
    return TestClient(app)


def test_each_answer_sentence_is_checked_against_the_passages_it_cites() -> None:
    answer = (
        "Staff take annual leave of 28 days after asking their manager [1]. "
        "Staff take annual leave of 35 days after asking their manager [1]. "
        "The moon orbits the earth every month in silence [1]."
    )
    with _client(answer) as client:
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        sync = client.post("/questions", json={"question": "leave?"}).json()
        statuses = [check["status"] for check in sync["verification"]]
        assert statuses == ["supported", "numbers_differ", "unsupported"]
        assert sync["verification"][0]["source_number"] == 1
        assert sync["verification"][1]["missing_numbers"] == ["35"]
        events = _sse_events(
            client.post("/questions/stream", json={"question": "leave?", "use_cache": False}).text
        )
        done = next(e for e in events if e["type"] == "done")
        assert [c["status"] for c in done["verification"]] == statuses
        # A cache hit is checked as well (worked out from the answer and its sources).
        hit = client.post("/questions", json={"question": "leave?"}).json()
        assert hit["cached"] is True and hit["verification"] == sync["verification"]
    clear_dependency_caches()


def test_the_self_check_can_be_turned_off() -> None:
    with _client("Staff take annual leave [1].", answer_self_check=False) as client:
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        assert client.post("/questions", json={"question": "leave?"}).json()["verification"] is None
    clear_dependency_caches()


def test_a_saved_answer_is_rechecked_against_the_documents_as_they_are_now() -> None:
    answer = "Staff take annual leave of 28 days after asking their manager [1]."
    with _client(answer, chunk_size_tokens=200) as client:
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        asked = client.post("/questions", json={"question": "leave?"}).json()
        sources = [
            {k: s[k] for k in ("source_number", "filename", "chunk_id")} for s in asked["sources"]
        ]
        body = {"answer": asked["answer"], "sources": sources}

        same = client.post("/questions/recheck", json=body).json()
        assert [s["state"] for s in same["sources"]] == ["unchanged"]
        assert same["still_holds"] is True and same["sentences"][0]["status"] == "supported"

        changed_doc = _DOC.replace(b"28", b"35")
        assert upload_and_wait(client, "leave.txt", changed_doc)["state"] == "done"
        changed = client.post("/questions/recheck", json=body).json()
        assert [s["state"] for s in changed["sources"]] == ["changed"]
        assert changed["sentences"][0]["status"] == "numbers_differ"
        assert changed["still_holds"] is False

        assert client.delete("/documents/leave.txt").status_code in (200, 204)
        gone = client.post("/questions/recheck", json=body).json()
        assert [s["state"] for s in gone["sources"]] == ["deleted"]
        assert gone["sentences"][0]["status"] == "unsupported" and gone["still_holds"] is False
    clear_dependency_caches()


def test_calibrate_reports_and_applies_the_workspaces_suggested_bar(monkeypatch: Any) -> None:
    import api.routes.traces as trace_routes
    from api.relevance import ScoreReport

    def report(*args: Any, **kwargs: Any) -> ScoreReport:
        return ScoreReport(
            current_min_score=0.2,
            bin_edges=[0.0, 1.0],
            answered=[25],
            unanswered=[25],
            answered_count=25,
            unanswered_count=25,
            suggested_min_score=0.35,
            note="from 50 traced questions",
            would_change=4,
        )

    monkeypatch.setattr(trace_routes, "score_report", report)
    with _client("unused") as client:
        dry = client.post("/admin/calibrate", json={})
        assert dry.status_code == 200 and dry.json()["applied"] is False
        assert dry.json()["suggested_min_score"] == 0.35
        applied = client.post("/admin/calibrate", json={"apply": True}).json()
        assert applied["applied"] is True
        kept = client.get("/workspaces/default/settings").json()
        assert kept["settings"]["rerank_min_score"] == 0.35, kept

        monkeypatch.setattr(
            trace_routes,
            "score_report",
            lambda *a, **k: report().__class__(
                **{**report().__dict__, "suggested_min_score": None}
            ),
        )
        assert client.post("/admin/calibrate", json={"apply": True}).status_code == 409
    clear_dependency_caches()


def test_docrag_traces_report_apply_calibrates_the_workspace(monkeypatch: Any) -> None:
    import io

    import api.routes.traces as trace_routes
    from api.relevance import ScoreReport
    from cli.__main__ import main
    from cli.client import DocragClient

    monkeypatch.setattr(
        trace_routes,
        "score_report",
        lambda *a, **k: ScoreReport(0.2, [0.0, 1.0], [25], [25], 25, 25, 0.4, "ok", 3),
    )
    with _client("unused") as client:
        out = io.StringIO()
        code = main(
            ["traces", "report", "--apply"],
            client=DocragClient("http://testserver", None, http=client),
            out=out,
        )
        assert code == 0, out.getvalue()
        assert "applied: this workspace now uses RERANK_MIN_SCORE=0.4" in out.getvalue()
    clear_dependency_caches()


def test_a_file_of_questions_is_answered_in_one_batch(tmp_path: Any) -> None:
    import io

    from cli.__main__ import main
    from cli.client import DocragClient

    with _client("Staff take annual leave of 28 days after asking their manager [1].") as client:
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        answered = client.post(
            "/questions/batch",
            json={"questions": ["leave?", "manager?"], "filenames": ["leave.txt"]},
        ).json()
        assert [a["question"] for a in answered["answers"]] == ["leave?", "manager?"]
        assert all(a["answer"]["answer_status"] == "answered" for a in answered["answers"])
        assert client.post("/questions/batch", json={"questions": []}).status_code == 422
        assert (
            client.post(
                "/questions/batch",
                json={"questions": ["a"], "history": [{"role": "user", "content": "x"}]},
            ).status_code
            == 422
        )

        questions = tmp_path / "q.txt"
        questions.write_text("# leave checklist\nleave?\n\nmanager?\n", encoding="utf-8")
        out = io.StringIO()
        code = main(
            ["ask", "--questions", str(questions)],
            client=DocragClient("http://testserver", None, http=client),
            out=out,
        )
        assert code == 0, out.getvalue()
        assert "1. leave?" in out.getvalue() and "2. manager?" in out.getvalue()
        assert "[1] leave.txt" in out.getvalue()
    clear_dependency_caches()


def test_a_sitemap_dry_run_lists_its_pages_and_fetches_none(monkeypatch: Any) -> None:
    import functools

    import httpx

    from api.url_fetch import fetch_document
    from tests.factories import wait_until
    from tests.test_pass15_sources import SITEMAP
    from tests.test_url_fetch import resolver, transport

    pages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, content=SITEMAP, headers={"content-type": "text/xml"})
        pages.append(request.url.path)
        return httpx.Response(
            200,
            content=f"Intro\nThe page {request.url.path}".encode(),
            headers={"content-type": "text/plain"},
        )

    fake = functools.partial(fetch_document, resolve=resolver({}), transport=transport(handler))
    monkeypatch.setattr("api.routes.documents_write.fetch_document", fake)
    with _client("unused", url_ingest_enabled=True) as client:
        body = {"url": "https://example.com/sitemap.xml", "dry_run": True}
        dry = client.post("/documents/from-sitemap", json=body).json()
        assert dry["import_id"] is None and pages == []
        assert [(p["url"], p["state"]) for p in dry["pages"]] == [
            ("https://example.com/a.txt", "new"),
            ("https://example.com/b.txt", "new"),
        ]
        client.post("/documents/from-sitemap", json={"url": body["url"]})
        wait_until(lambda: len(client.get("/documents").json()["filenames"]) == 2)
        again = client.post("/documents/from-sitemap", json=body).json()
        assert {p["state"] for p in again["pages"]} == {"check"}
    clear_dependency_caches()


def test_docrag_dedupe_lists_and_deletes_copies() -> None:
    import io
    import json

    from cli.__main__ import main
    from cli.client import DocragClient

    with _client("unused") as client:
        for name in ("a.txt", "b.txt", "c.txt"):
            assert upload_and_wait(client, name, _DOC)["state"] == "done"
        cli = DocragClient("http://testserver", None, http=client)
        out = io.StringIO()
        assert main(["dedupe"], client=cli, out=out) == 1
        assert "keep a.txt; copies: b.txt, c.txt" in out.getvalue()
        out = io.StringIO()
        assert main(["dedupe", "--delete", "--json"], client=cli, out=out) == 0
        assert json.loads(out.getvalue())["deleted"] == ["b.txt", "c.txt"]
        assert client.get("/documents").json()["filenames"] == ["a.txt"]
    clear_dependency_caches()


def test_the_duplicate_report_lists_identical_documents() -> None:
    with _client("unused") as client:
        for name in ("a.txt", "b.txt"):
            assert upload_and_wait(client, name, _DOC)["state"] == "done"
        assert upload_and_wait(client, "c.txt", b"Other\nsomething else")["state"] == "done"
        report = client.get("/documents/duplicates").json()
        assert report["copies"] == 1
        (group,) = report["groups"]
        assert group["filenames"] == ["a.txt", "b.txt"] and group["chunk_count"] >= 1
    clear_dependency_caches()


def test_a_sentence_saying_the_documents_lack_something_is_not_flagged() -> None:
    from api.citations import SourceCitation
    from api.verify import self_check

    source = SourceCitation(1, "it.txt", 1, "IT", "c1", "Laptops are replaced every four years.")
    checks = self_check(
        "Laptops are replaced every four years [1]. "
        "The provided documents do not contain enough information regarding the expense policy. "
        "The context provided does not include any information regarding an expense policy.",
        [source],
    )
    assert [c.status for c in checks] == ["supported", "no_claim", "no_claim"]
