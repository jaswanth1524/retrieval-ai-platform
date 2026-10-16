from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from cli.__main__ import ingestible, main
from cli.client import DocragClient, DocragError, sse_events
from tests.routes_http.support import hermetic_app, make_settings


@pytest.fixture
def client() -> Iterator[DocragClient]:
    clear_dependency_caches()
    settings = make_settings(api_key="full-key", followups_enabled=True)
    app = hermetic_app(settings, QdrantClient(":memory:"))
    with TestClient(app) as test_client:
        docrag = DocragClient(
            "http://testserver",
            "full-key",
            http=test_client,  # the app's own TestClient: the real routes, in process
            sleep=lambda _: None,
        )
        yield docrag
        docrag.close()
    clear_dependency_caches()


def run(client: DocragClient, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), client=client, out=out)
    return code, out.getvalue()


def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "nested").mkdir(parents=True)
    (root / "guide.txt").write_text("Intro\nAlpha is the first letter.")
    (root / "nested" / "notes.md").write_text("# Notes\nBeta comes second.")
    (root / "image.bin").write_bytes(b"\x00\x01")
    (root / ".hidden.txt").write_text("not this")
    return root


def test_ingest_walks_directories_and_reports_each_file(
    client: DocragClient, tmp_path: Path
) -> None:
    code, out = run(client, "ingest", str(corpus(tmp_path)))

    assert code == 0
    assert "skip  " in out and "image.bin" in out
    assert out.count("ok    ") == 2
    assert ".hidden" not in out
    assert "2 indexed, 0 failed, 1 skipped" in out
    code, out = run(client, "--json", "documents")
    assert sorted(json.loads(out)["filenames"]) == ["guide.txt", "notes.md"]


def test_ingestible_skips_dotfiles_and_unknown_types(tmp_path: Path) -> None:
    root = corpus(tmp_path)
    files, skipped = ingestible([root], {".txt", ".md"})
    assert [p.name for p in files] == ["guide.txt", "notes.md"]
    assert [p.name for p in skipped] == ["image.bin"]


def test_ask_streams_the_answer_with_sources_and_followups(
    client: DocragClient, tmp_path: Path
) -> None:
    run(client, "ingest", str(corpus(tmp_path)))

    code, out = run(client, "ask", "alpha", "--quiet")

    assert code == 0
    assert out.startswith("Alpha is documented [1].")
    assert "Sources:" in out and "[1] " in out
    code, out = run(client, "--json", "ask", "alpha")
    assert json.loads(out)["answer"] == "Alpha is documented [1]."


def test_summarize_and_compare(client: DocragClient, tmp_path: Path) -> None:
    run(client, "ingest", str(corpus(tmp_path)))

    code, out = run(client, "summarize", "guide.txt", "--quiet")
    assert code == 0 and "Alpha is documented [1]." in out
    code, out = run(client, "--json", "summarize", "guide.txt", "notes.md", "--focus", "letters")
    assert code == 0 and json.loads(out)["chunks_read"] == 2
    code, _ = run(client, "summarize", "a", "b", "c")
    assert code == 2


def test_search_lists_ranked_passages(client: DocragClient, tmp_path: Path) -> None:
    run(client, "ingest", str(corpus(tmp_path)))

    code, out = run(client, "search", "alpha", "--file", "guide.txt")

    assert code == 0
    assert out.startswith("1. guide.txt p.1")


def test_status_and_export(client: DocragClient, tmp_path: Path) -> None:
    code, out = run(client, "status")
    assert code == 0 and "documents: 0" in out

    target = tmp_path / "backup.zip"
    code, out = run(client, "export", str(target))
    assert code == 0
    assert "manifest.json" in zipfile.ZipFile(target).namelist()


def test_server_errors_exit_1_with_the_reason(
    client: DocragClient, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = run(client, "summarize", "missing.txt", "--quiet")
    assert code == 1
    assert "error (404): No indexed document named 'missing.txt'" in capsys.readouterr().err

    wrong = DocragClient("http://testserver", "nope", http=client._http)
    assert main(["status"], client=wrong, out=io.StringIO()) == 1
    assert "error (401)" in capsys.readouterr().err


def test_ingest_of_a_missing_path_is_a_usage_error(client: DocragClient) -> None:
    assert run(client, "ingest", "/no/such/path")[0] == 2


def test_upload_waits_out_a_full_indexing_queue(tmp_path: Path) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503, headers={"Retry-After": "7"}, json={"detail": "full"})
        return httpx.Response(202, json={"job_id": "j1", "filename": "a.txt", "state": "queued"})

    slept: list[float] = []
    docrag = DocragClient(
        "http://x", "", transport=httpx.MockTransport(handler), sleep=slept.append
    )
    path = tmp_path / "a.txt"
    path.write_text("x")

    assert docrag.upload(path)["job_id"] == "j1"
    assert slept == [7.0]


def test_stream_errors_and_unreachable_servers_raise() -> None:
    body = 'data: {"type": "error", "detail": "provider down"}\n\n'
    errors = DocragClient(
        "http://x",
        "",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body)),
    )
    with pytest.raises(DocragError, match="provider down"):
        errors.ask("q")

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    down = DocragClient("http://x", "", transport=httpx.MockTransport(refuse))
    with pytest.raises(DocragError, match="Cannot reach DocRAG"):
        down.health()
    with pytest.raises(DocragError, match="Cannot reach DocRAG"):
        down.ask("q")


def test_sse_events_follow_the_sse_rules() -> None:
    lines = [": ping", 'data:{"a":', "data: 1}", "", 'data: {"b": 2}\r']
    assert list(sse_events(iter(lines))) == [{"a": 1}, {"b": 2}]


def test_add_url_reports_what_the_server_says(client: DocragClient) -> None:
    # URL ingest is off on this server: the 404 comes back as the reason.
    code, _ = run(client, "add-url", "https://example.com/a.txt")
    assert code == 1


def test_add_url_waits_for_the_job() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/documents/from-url":
            assert json.loads(request.content) == {"url": "https://e.com/x", "filename": "x.txt"}
            return httpx.Response(202, json={"job_id": "j", "filename": "x.txt", "state": "queued"})
        return httpx.Response(
            200,
            json={
                "job_id": "j",
                "filename": "x.txt",
                "state": "done",
                "chunks_total": 1,
                "chunks_done": 1,
            },
        )

    docrag = DocragClient("http://x", "", transport=httpx.MockTransport(handler))
    code, out = run(docrag, "add-url", "https://e.com/x", "--name", "x.txt")
    assert code == 0 and "→ x.txt" in out


def test_the_workspace_header_is_sent_when_named(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("X-Workspace"))
        return httpx.Response(200, json={"filenames": []})

    DocragClient(
        "http://x", "", transport=httpx.MockTransport(handler), workspace="legal"
    ).documents()
    monkeypatch.setenv("DOCRAG_WORKSPACE", "research")
    DocragClient("http://x", "", transport=httpx.MockTransport(handler)).documents()
    monkeypatch.delenv("DOCRAG_WORKSPACE")
    DocragClient("http://x", "", transport=httpx.MockTransport(handler)).documents()
    assert seen == ["legal", "research", None]


def test_the_installed_command_names_itself_docrag(monkeypatch: pytest.MonkeyPatch) -> None:
    from cli.__main__ import build_parser

    monkeypatch.setattr("sys.argv", ["/home/me/.local/bin/docrag", "status"])
    assert build_parser().prog == "docrag"
    monkeypatch.setattr("sys.argv", ["/repo/cli/__main__.py", "status"])
    assert build_parser().prog == "python -m cli"
