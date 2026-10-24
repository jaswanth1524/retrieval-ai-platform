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
    # Three documents: one brief across them; unknown names are the server's 404.
    code, _ = run(client, "summarize", "a", "b", "c")
    assert code == 1


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


def test_status_names_the_model_of_the_provider_in_use() -> None:
    """It printed llm_model (Ollama's) for every provider: "openai (qwen2.5:latest)"."""

    from cli.__main__ import provider_model

    config = {
        "llm_model": "llama3.1:8b",
        "openai_model": "gpt-5-mini",
        "openai_compatible_model": "qwen2.5-7b",
        "anthropic_model": "claude-sonnet-5-5",
    }

    assert provider_model(config, "ollama") == "llama3.1:8b"
    assert provider_model(config, "openai") == "gpt-5-mini"
    assert provider_model(config, "openai_compatible") == "qwen2.5-7b"
    assert provider_model(config, "anthropic") == "claude-sonnet-5-5"
    assert provider_model({"openai_compatible_model": ""}, "openai_compatible") == "?"


def test_status_prints_the_default_providers_model(client: DocragClient) -> None:
    code, out = run(client, "status")

    assert code == 0
    assert "generation: ollama (llama3.1:8b)" in out


# -- review fixes ----------------------------------------------------------------------


DONE = {"type": "done", "answer": "The answer [1].", "sources": [], "answer_status": "answered"}


def frame(event: dict[str, object]) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode()


class BrokenStream(httpx.SyncByteStream):
    """A body that sends ``chunks`` and then loses the connection."""

    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks
        raise httpx.ReadError("connection reset")


def mock_client(handler: object, slept: list[float] | None = None) -> DocragClient:
    return DocragClient(
        "http://x",
        "",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        sleep=(slept.append if slept is not None else lambda _: None),
    )


def test_ingest_uploads_one_file_per_stored_name(client: DocragClient, tmp_path: Path) -> None:
    """The server keeps only the basename: a/notes.md and b/Notes.md overwrote each other."""

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "notes.md").write_text("# A\nAlpha from folder a.")
    (tmp_path / "b" / "Notes.md").write_text("# B\nBeta from folder b.")

    code, out = run(client, "ingest", str(tmp_path))

    assert code == 0
    assert out.count("ok    ") == 1
    assert (
        f"skip  {tmp_path / 'b' / 'Notes.md'} (another file is already indexed as notes.md" in out
    )
    assert "1 indexed, 0 failed, 1 skipped" in out
    _, listing = run(client, "--json", "documents")
    assert json.loads(listing)["filenames"] == ["notes.md"]


def test_an_explicitly_named_dotfile_is_not_dropped(client: DocragClient, tmp_path: Path) -> None:
    root = corpus(tmp_path)

    files, _ = ingestible([root / ".hidden.txt"], {".txt"})
    assert files == [root / ".hidden.txt"]
    code, out = run(client, "ingest", str(root / ".hidden.txt"))
    assert code == 0 and "ok    " in out and "1 indexed" in out


def test_ingest_skips_files_over_the_servers_upload_limit(tmp_path: Path) -> None:
    posted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/config":
            return httpx.Response(
                200, json={"supported_extensions": [".txt"], "max_upload_bytes": 5}
            )
        posted.append(request.url.path)
        return httpx.Response(500)

    (tmp_path / "big.txt").write_text("more than five bytes")

    code, out = run(mock_client(handler), "ingest", str(tmp_path / "big.txt"))

    assert code == 0 and posted == []
    assert "big.txt (too large: 20 bytes, the limit is 5)" in out
    assert "0 indexed, 0 failed, 1 skipped" in out


def test_an_unreadable_file_fails_alone_and_the_ingest_goes_on(
    client: DocragClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = corpus(tmp_path)
    real = Path.read_bytes

    def read_bytes(self: Path) -> bytes:
        if self.name == "guide.txt":
            raise PermissionError(13, "Permission denied")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    code, out = run(client, "ingest", str(root))

    assert code == 1
    assert "guide.txt: Cannot read" in out and "Permission denied" in out
    assert "1 indexed, 1 failed, 1 skipped" in out


def test_an_error_after_done_keeps_the_answer() -> None:
    """Only the follow-ups were still coming: the web UI's sawDone guard."""

    followups = {"type": "followups", "questions": ["Next?"]}
    error = {"type": "error", "detail": "follow-ups failed"}
    bodies = [
        frame(DONE) + frame(followups) + frame(error),
        frame(DONE) + b"data: {not json\n\n",
    ]
    for body in bodies:
        result = mock_client(lambda _, body=body: httpx.Response(200, content=body)).ask("q")
        assert result.done["answer"] == "The answer [1]."

    reset = mock_client(
        lambda _: httpx.Response(200, stream=BrokenStream(frame(DONE), frame(followups)))
    ).ask("q")
    assert reset.done["answer"] == "The answer [1]." and reset.followups == ["Next?"]

    # Before `done` the same reset is still a failure.
    with pytest.raises(DocragError, match="Cannot reach DocRAG"):
        mock_client(lambda _: httpx.Response(200, stream=BrokenStream(b""))).ask("q")


def test_export_never_truncates_an_existing_backup(tmp_path: Path) -> None:
    target = tmp_path / "backup.zip"
    target.write_bytes(b"the old backup")

    def interrupted(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BrokenStream(b"PK\x03\x04 half a zip"))

    with pytest.raises(DocragError, match="The download was interrupted") as caught:
        mock_client(interrupted).export(target)
    assert "Cannot reach" not in str(caught.value)
    assert target.read_bytes() == b"the old backup"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["backup.zip"]

    refused = mock_client(lambda _: httpx.Response(403, json={"detail": "full key needed"}))
    with pytest.raises(DocragError, match="full key needed"):
        refused.export(target)
    assert target.read_bytes() == b"the old backup"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["backup.zip"]

    assert mock_client(lambda _: httpx.Response(200, content=b"new zip")).export(target) == 7
    assert target.read_bytes() == b"new zip"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["backup.zip"]


def import_server(states: dict[str, object], **report: object) -> DocragClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/import":
            jobs = [{"job_id": job, "filename": f"{job}.txt", "state": "queued"} for job in states]
            body = {
                "jobs": jobs,
                "deferred": [],
                "skipped": [],
                "missing_originals": [],
                "rejected": [],
                "feedback_imported": 0,
                "feedback_skipped": 0,
                **report,
            }
            return httpx.Response(200, json=body)
        job = request.url.path.rsplit("/", 1)[-1]
        state = states[job]
        if isinstance(state, int):
            return httpx.Response(state, json={"detail": "Unknown job."})
        return httpx.Response(200, json={"job_id": job, "filename": f"{job}.txt", "state": state})

    return mock_client(handler)


def test_import_waits_for_every_job_and_fails_on_any_failure(tmp_path: Path) -> None:
    archive = tmp_path / "backup.zip"
    archive.write_bytes(b"zip")
    docrag = import_server({"evicted": 404, "broken": "failed", "fine": "done"})

    code, out = run(docrag, "import", str(archive))
    assert code == 1
    assert "unknown evicted.txt: Unknown job." in out
    assert "failed broken.txt" in out and "done   fine.txt" in out

    # --json waits too, and reports every job's final state.
    code, out = run(docrag, "import", str(archive), "--json")
    assert code == 1
    states = [status["state"] for status in json.loads(out)["results"]]
    assert states == ["unknown", "failed", "done"]

    assert run(import_server({"fine": "done"}), "import", str(archive))[0] == 0
    code, out = run(import_server({"fine": "done"}, deferred=["later.txt"]), "import", str(archive))
    assert code == 1 and "defer  later.txt" in out
    assert run(import_server({}, rejected=["bad.txt"]), "--json", "import", str(archive))[0] == 1


def test_local_failures_are_error_lines_not_tracebacks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    html = mock_client(lambda _: httpx.Response(200, text="<html>a proxy</html>"))
    assert run(html, "status")[0] == 1
    assert "error: The server sent a response that is not JSON" in capsys.readouterr().err

    garbled = mock_client(lambda _: httpx.Response(200, content=b"data: {oops\n\n"))
    assert run(garbled, "ask", "q", "--quiet")[0] == 1
    assert "error: The server sent a malformed stream frame." in capsys.readouterr().err
    not_an_object = mock_client(lambda _: httpx.Response(200, content=b"data: 1\n\n"))
    assert run(not_an_object, "ask", "q", "--quiet")[0] == 1
    assert "malformed stream frame" in capsys.readouterr().err

    zipped = mock_client(lambda _: httpx.Response(200, content=b"zip"))
    assert run(zipped, "export", str(tmp_path / "no-such-dir" / "backup.zip"))[0] == 1
    assert "error: Cannot write" in capsys.readouterr().err

    assert run(zipped, "import", str(tmp_path / "missing.zip"))[0] == 1
    assert "error: Cannot read" in capsys.readouterr().err


def test_global_flags_also_work_after_the_subcommand(client: DocragClient, tmp_path: Path) -> None:
    from cli.__main__ import build_parser

    parser = build_parser()
    args = parser.parse_args(["ask", "q", "--json", "--url", "http://srv", "--workspace", "w"])
    assert (args.json, args.url, args.workspace) == (True, "http://srv", "w")
    # A flag given before the subcommand survives it.
    args = parser.parse_args(["--json", "--api-key", "k", "status"])
    assert (args.json, args.api_key, args.url) == (True, "k", None)
    assert parser.parse_args(["status"]).json is False
    # add-url's own argument no longer replaces the server's address.
    args = parser.parse_args(["--url", "http://srv", "add-url", "https://e.com/x.html"])
    assert (args.url, args.source_url) == ("http://srv", "https://e.com/x.html")

    run(client, "ingest", str(corpus(tmp_path)))
    code, out = run(client, "ask", "alpha", "--json")
    assert code == 0 and json.loads(out)["answer"] == "Alpha is documented [1]."


def test_output_the_console_cannot_encode_is_replaced(
    client: DocragClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cp1252 (a Windows console) has no "→": the ok line raised UnicodeEncodeError."""

    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr("sys.stdout", console)

    code = main(["ingest", str(corpus(tmp_path) / "guide.txt")], client=client)

    console.flush()
    assert code == 0
    assert b"guide.txt ? guide.txt" in raw.getvalue()


def test_busy_answers_and_searches_are_retried_after_retry_after(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) % 2 == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"detail": "busy"})
        if request.url.path == "/search":
            return httpx.Response(200, json={"results": []})
        if request.url.path == "/documents":
            return httpx.Response(202, json={"job_id": "j", "filename": "a.txt"})
        return httpx.Response(200, content=frame(DONE))

    slept: list[float] = []
    docrag = mock_client(handler, slept)
    path = tmp_path / "a.txt"
    path.write_text("x")

    assert docrag.ask("q").done["answer"] == "The answer [1]."
    assert docrag.summarize(["a.txt"]).done["answer"] == "The answer [1]."
    assert docrag.search("q") == []
    assert docrag.upload(path)["job_id"] == "j"
    assert slept == [2.0, 2.0, 2.0, 2.0]

    # Bounded: a server that stays busy is reported, not waited on forever.
    always: list[int] = []

    def busy(_: httpx.Request) -> httpx.Response:
        always.append(1)
        return httpx.Response(429, json={"detail": "All question slots are busy."})

    with pytest.raises(DocragError, match="busy") as caught:
        mock_client(busy).ask("q")
    assert caught.value.status_code == 429 and len(always) == 4


def jobs_server(finish_after: dict[str, int]) -> tuple[DocragClient, list[str]]:
    """Uploads get a job each; job ``name`` is done after ``finish_after[name]`` polls
    (never, when absent). The log shows the order of uploads and polls."""

    log: list[str] = []
    polls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/config":
            return httpx.Response(200, json={"supported_extensions": [".txt"]})
        if request.url.path == "/documents/from-url":
            log.append("from-url")
            return httpx.Response(202, json={"job_id": "url.txt", "state": "queued"})
        if request.url.path == "/documents":
            name = request.content.split(b'filename="')[1].split(b'"')[0].decode()
            log.append(f"upload {name}")
            return httpx.Response(202, json={"job_id": name, "filename": name, "state": "queued"})
        name = request.url.path.rsplit("/", 1)[1]
        log.append(f"poll {name}")
        polls[name] = polls.get(name, 0) + 1
        done = name in finish_after and polls[name] >= finish_after[name]
        if done:
            log.append(f"done {name}")
        return httpx.Response(
            200,
            json={
                "job_id": name,
                "filename": name,
                "state": "done" if done else "embedding",
                "chunks_done": 1,
            },
        )

    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    docrag = DocragClient(
        "http://x", "", transport=httpx.MockTransport(handler), sleep=sleep, clock=lambda: now[0]
    )
    return docrag, log


def test_ingest_sends_several_files_before_waiting_and_reports_json(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("Intro\nText.")
    (tmp_path / "skip.bin").write_bytes(b"\x00")
    docrag, log = jobs_server({"a.txt": 3, "b.txt": 1, "c.txt": 2})

    code, out = run(docrag, "ingest", str(tmp_path), "--json")
    assert code == 0
    # All three on the server before the first poll: it indexes them side by side.
    assert log[:3] == ["upload a.txt", "upload b.txt", "upload c.txt"]
    report = json.loads(out)
    assert [item["filename"] for item in report["indexed"]] == ["b.txt", "c.txt", "a.txt"]
    assert report["failed"] == []
    assert report["skipped"] == [{"path": str(tmp_path / "skip.bin"), "reason": "unsupported type"}]


def test_ingest_stops_waiting_at_the_timeout(tmp_path: Path) -> None:
    (tmp_path / "quick.txt").write_text("Intro\nText.")
    (tmp_path / "stuck.txt").write_text("Intro\nText.")
    docrag, _ = jobs_server({"quick.txt": 1})

    code, out = run(docrag, "ingest", str(tmp_path), "--timeout", "5")
    assert code == 1
    assert "ok    " in out and "quick.txt" in out
    assert "still embedding when the wait ended" in out
    assert out.strip().endswith("1 indexed, 1 failed, 0 skipped")


def test_add_url_prints_json_and_honours_the_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/documents/from-url":
            return httpx.Response(202, json={"job_id": "j", "filename": "x.txt", "state": "queued"})
        return httpx.Response(200, json={"job_id": "j", "filename": "x.txt", "state": "done"})

    code, out = run(mock_client(handler), "add-url", "https://e.com/x.txt", "--json")
    assert code == 0 and json.loads(out)["state"] == "done"

    docrag, _ = jobs_server({})
    code, out = run(docrag, "add-url", "https://e.com/slow", "--json", "--timeout", "3")
    assert code == 1 and "still embedding" in json.loads(out)["error"]


def test_wait_for_job_gives_up_after_its_timeout() -> None:
    docrag, log = jobs_server({})
    with pytest.raises(DocragError, match=r"Still embedding after 4 s \(job j\)"):
        docrag.wait_for_job("j", timeout=4)
    assert 4 <= len(log) <= 6


def test_a_stream_stops_when_asked_even_between_events() -> None:
    """Only keepalive comments arrive while the server retrieves: the stop is still seen."""

    closed: list[bool] = []

    class Pings(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            while True:
                yield b": ping\n\n"

        def close(self) -> None:
            closed.append(True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Pings(), headers={"content-type": "text/event-stream"})

    checks = [0]

    def should_stop() -> bool:
        checks[0] += 1
        return checks[0] > 3

    from cli.client import Cancelled

    with pytest.raises(Cancelled):
        mock_client(handler).ask("q", should_stop=should_stop)
    assert closed == [True]  # the connection was closed: the server stops the answer


def test_index_files_keeps_at_most_a_window_of_jobs_indexing(tmp_path: Path) -> None:
    """Uploading a whole folder at once let finished jobs fall out of the server's job
    store (they then read as unknown) and later uploads wait on 503s."""

    paths = []
    for index in range(7):
        path = tmp_path / f"f{index}.txt"
        path.write_text("Intro\nText.")
        paths.append(path)
    docrag, log = jobs_server({path.name: 1 + index % 3 for index, path in enumerate(paths)})

    statuses = docrag.index_files(paths, window=2)
    assert [statuses[path]["state"] for path in paths] == ["done"] * 7
    indexing = most = 0
    for entry in log:
        if entry.startswith("upload "):
            indexing += 1
        elif entry.startswith("done "):
            indexing -= 1
        most = max(most, indexing)
    assert most == 2


def test_files_not_yet_sent_when_the_wait_ends_are_reported(tmp_path: Path) -> None:
    paths = []
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("Intro\nText.")
        paths.append(tmp_path / name)
    docrag, _ = jobs_server({})  # nothing ever finishes

    statuses = docrag.index_files(paths, window=1, timeout=3)
    assert [statuses[path]["state"] for path in paths] == ["embedding", "not_sent", "not_sent"]


def test_search_says_when_the_passages_are_only_the_closest() -> None:
    passage = {"rank": 1, "filename": "it.txt", "page": 1, "text": "Laptops.", "rerank_score": 0.1}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [passage], "timings": {}, "weak_matches": True})

    code, out = run(mock_client(handler), "search", "laptop")
    assert code == 0
    assert out.startswith("(no strong matches; the closest passages:)")
    assert "1. it.txt p.1 (0.10)" in out
    # --json keeps its shape: the list of passages.
    code, out = run(mock_client(handler), "search", "laptop", "--json")
    assert json.loads(out) == [passage]


def test_tags_read_delete_jobs_and_workspaces(client: DocragClient, tmp_path: Path) -> None:
    run(client, "ingest", str(corpus(tmp_path)))

    assert run(client, "tags", "set", "guide.txt", "hr", "policy") == (0, "guide.txt: hr, policy\n")
    assert run(client, "tags", "add", "guide.txt", "legal", "hr")[1] == (
        "guide.txt: hr, policy, legal\n"
    )
    assert run(client, "tags", "remove", "guide.txt", "HR")[1] == "guide.txt: policy, legal\n"
    code, out = run(client, "search", "alpha", "--tag", "legal")
    assert code == 0 and out.startswith("1. guide.txt")
    code, out = run(client, "search", "beta", "--tag", "legal")
    assert "notes.md" not in out

    code, out = run(client, "read", "guide.txt")
    assert code == 0 and "Alpha is the first letter." in out and "(1 passages in all)" in out
    chunk_id = json.loads(run(client, "--json", "read", "guide.txt")[1])["chunks"][0]["chunk_id"]
    code, out = run(client, "read", "guide.txt", "--around", chunk_id)
    assert code == 0 and chunk_id in out

    assert run(client, "jobs") == (0, "no indexing jobs running\n")
    code, out = run(client, "workspaces", "list")
    assert code == 0 and "not enabled" in out
    assert run(client, "workspaces", "create")[0] == 2

    assert run(client, "delete", "notes.md") == (0, "deleted notes.md\n")
    assert json.loads(run(client, "--json", "documents")[1])["filenames"] == ["guide.txt"]
    code, _ = run(client, "delete", "notes.md")
    assert code == 1  # already gone: the server's 404


def test_reindex_refresh_and_watch_report_the_servers_answer(client: DocragClient) -> None:
    # No stored originals, no URL ingest, no watched folder on this server: each says so.
    assert run(client, "reindex")[0] == 2
    assert run(client, "reindex", "guide.txt")[0] == 1
    assert run(client, "refresh", "guide.txt")[0] == 1
    assert run(client, "watch")[0] == 1


def test_versions_and_queries_say_what_the_server_says(client: DocragClient) -> None:
    # No stored originals and no query log on this server: the server's refusals.
    assert run(client, "versions", "guide.txt")[0] == 1
    assert run(client, "queries")[0] == 1


def test_an_answer_from_weak_matches_is_flagged_in_the_cli() -> None:
    import io

    from cli.__main__ import _print_sources
    from cli.client import StreamResult

    out = io.StringIO()
    _print_sources(StreamResult({"answer": "x [1]", "sources": [], "weak_matches": True}, []), out)
    assert "weak match: no passage matched the question closely" in out.getvalue()


def test_ask_strict_exits_3_when_the_documents_do_not_answer(
    client: DocragClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cli.client import StreamResult

    run(client, "ingest", str(corpus(tmp_path)))
    refused = StreamResult(
        {"answer": "The documents do not cover it.", "sources": [], "answer_status": "no_context"},
        [],
    )
    monkeypatch.setattr(client, "ask", lambda *a, **k: refused)
    code, out = run(client, "ask", "gym?", "--quiet", "--strict")
    assert code == 3
    assert "nothing in the searched documents matched this question" in out
    # Without --strict a refusal is still a successful run.
    assert run(client, "ask", "gym?", "--quiet")[0] == 0


def test_cli_notes_explain_uncited_and_auto_cited_answers() -> None:
    from cli.client import grounding_notes

    assert grounding_notes({"answer_status": "uncited"}) == [
        "answered without citing its sources — check it against the documents"
    ]
    assert "added by matching" in grounding_notes({"citation_backfilled": True})[0]
    assert grounding_notes({"answer_status": "answered"}) == []


def test_chat_sends_the_conversation_and_leaves_refusals_out(
    client: DocragClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cli.client import StreamResult

    seen: list[list[dict[str, str]] | None] = []
    answers = iter(
        [
            {"answer": "Fourteen days [1].", "sources": [], "answer_status": "answered"},
            {"answer": "Not covered.", "sources": [], "answer_status": "insufficient_context"},
            {"answer": "Sixty euros [1].", "sources": [], "answer_status": "answered"},
            {"answer": "Fresh [1].", "sources": [], "answer_status": "answered"},
        ]
    )

    def fake_ask(
        question: str, *args: object, history: object = None, **kwargs: object
    ) -> StreamResult:
        seen.append(history)  # type: ignore[arg-type]
        return StreamResult(next(answers), [])

    monkeypatch.setattr(client, "ask", fake_ask)
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("Flights?\nGym?\nAnd meals?\n/new\nHello?\n/quit\nnever\n")
    )
    code, out = run(client, "chat", "--quiet")
    assert code == 0
    assert seen[0] is None
    flights = [
        {"role": "user", "content": "Flights?"},
        {"role": "assistant", "content": "Fourteen days [1]."},
    ]
    # The refusal to "Gym?" isn't sent back with the next question.
    assert seen[1] == flights and seen[2] == flights
    assert seen[3] is None  # /new
    assert "(new conversation)" in out
    assert len(seen) == 4


def test_the_stage_reporter_ticks_while_nothing_arrives() -> None:
    from cli.__main__ import StageReporter

    err = io.StringIO()
    with StageReporter(err, tick=0.05) as reporter:
        reporter.stage({"stage": "reading", "eta_ms": 42000})
        import time

        time.sleep(0.2)
        reporter.delta()
    lines = err.getvalue().splitlines()
    assert lines[0] == "[reading the passages, about 42s]"
    assert any(line.startswith("[reading the passages… ") for line in lines[1:])


def _flaky_jobs(failures: list[int | None]) -> tuple[DocragClient, list[str]]:
    """A server whose job polls fail with ``failures`` (a status, or None for a dropped
    connection) in turn, then report the job done."""

    log: list[str] = []
    queue = list(failures)

    def handler(request: httpx.Request) -> httpx.Response:
        log.append(request.url.path)
        if request.url.path == "/documents":
            return httpx.Response(202, json={"job_id": "j", "filename": "a.txt", "state": "queued"})
        if queue:
            status = queue.pop(0)
            if status is None:
                raise httpx.ConnectError("connection reset")
            return httpx.Response(status, json={"detail": "nope"})
        return httpx.Response(200, json={"job_id": "j", "filename": "a.txt", "state": "done"})

    docrag = DocragClient(
        "http://x", "", transport=httpx.MockTransport(handler), sleep=lambda _: None
    )
    return docrag, log


def test_a_poll_that_fails_for_a_moment_does_not_end_the_wait(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Intro\nText.")
    docrag, _ = _flaky_jobs([502, None])
    assert docrag.wait_for_jobs(["j"])["j"]["state"] == "done"
    docrag, _ = _flaky_jobs([503, None])
    assert docrag.index_files([tmp_path / "a.txt"])[tmp_path / "a.txt"]["state"] == "done"
    docrag, _ = _flaky_jobs([502])
    assert docrag.wait_for_job("j")["state"] == "done"


def test_a_job_the_server_lost_or_a_server_down_for_good_is_unknown() -> None:
    docrag, log = _flaky_jobs([404])
    assert docrag.wait_for_jobs(["j"])["j"]["state"] == "unknown"
    assert len(log) == 1
    docrag, _ = _flaky_jobs([None, None, None, None])
    status = docrag.wait_for_jobs(["j"])["j"]
    assert status["state"] == "unknown" and "Cannot reach" in status["error"]


def test_history_is_trimmed_to_what_the_server_takes() -> None:
    from cli.client import HISTORY_MESSAGE_MAX_CHARS, trim_history

    long_answer = "word " * 2000  # 10 000 characters
    history = [
        {"role": "assistant", "content": "an answer to a question left out"},
        *[
            message
            for n in range(8)
            for message in (
                {"role": "user", "content": f"Question {n}?"},
                {"role": "assistant", "content": long_answer if n == 7 else f"Answer {n}."},
            )
        ],
        {"role": "system", "content": "ignored"},
    ]
    trimmed = trim_history(history)
    assert trimmed[0]["role"] == "user"
    assert len(trimmed) <= 12
    assert all(len(m["content"]) <= HISTORY_MESSAGE_MAX_CHARS for m in trimmed)
    assert sum(len(m["content"]) for m in trimmed) <= 8000
    assert trimmed[-1]["content"].endswith("word…")  # cut at a word, and marked
    assert trimmed[-2] == {"role": "user", "content": "Question 7?"}


def test_delete_goes_on_past_a_failure_and_reports_each() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/missing.txt"):
            return httpx.Response(404, json={"detail": "No indexed document named 'missing.txt'."})
        return httpx.Response(200, json={"filename": "x", "points_deleted": 1})

    code, out = run(mock_client(handler), "delete", "a.txt", "missing.txt", "b.txt", "--json")
    assert code == 1
    report = json.loads(out)
    assert report["deleted"] == ["a.txt", "b.txt"]
    assert [item["filename"] for item in report["failed"]] == ["missing.txt"]


def test_export_reports_json(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"PK\x05\x06" + b"\x00" * 18)

    target = tmp_path / "backup.zip"
    code, out = run(mock_client(handler), "export", str(target), "--json")
    assert code == 0
    assert json.loads(out) == {"output": str(target), "bytes": 22}


def test_ctrl_c_stops_an_answer_in_chat_but_not_the_conversation(
    client: DocragClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cli.client import StreamResult

    asked: list[str] = []

    def fake_ask(question: str, *args: object, **kwargs: object) -> StreamResult:
        asked.append(question)
        if question == "Slow?":
            raise KeyboardInterrupt
        return StreamResult({"answer": "Yes [1].", "sources": [], "answer_status": "answered"}, [])

    monkeypatch.setattr(client, "ask", fake_ask)
    monkeypatch.setattr("sys.stdin", io.StringIO("Slow?\nNext?\n"))
    code, _ = run(client, "chat", "--quiet")
    assert code == 0 and asked == ["Slow?", "Next?"]

    def interrupted(*args: object, **kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(client, "documents", interrupted)
    assert run(client, "documents")[0] == 130


def test_an_upload_waits_for_a_full_queue_only_until_the_deadline(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Intro\nText.")
    now = [0.0]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "queue full"}, headers={"Retry-After": "15"})

    def sleep(seconds: float) -> None:
        now[0] += seconds

    docrag = DocragClient(
        "http://x", "", transport=httpx.MockTransport(handler), sleep=sleep, clock=lambda: now[0]
    )
    statuses = docrag.index_files([tmp_path / "a.txt"], timeout=40)
    assert statuses[tmp_path / "a.txt"]["state"] == "failed"
    assert "when the wait ended" in statuses[tmp_path / "a.txt"]["error"]
    assert now[0] <= 40


class FakeServer:
    """Answers the routes the maintenance commands use; records each request."""

    def __init__(self, *, workspaces_enabled: bool = True) -> None:
        self.log: list[str] = []
        self.workspaces_enabled = workspaces_enabled

    def __call__(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.log.append(f"{method} {path}")
        job = {"job_id": "j1", "filename": "a.txt", "state": "queued"}
        done = {**job, "state": "done", "chunks_total": 3, "chunks_done": 3}
        routes: dict[tuple[str, str], object] = {
            ("POST", "/documents/reindex"): {
                "jobs": [job],
                "deferred": ["b.txt"],
                "in_progress": ["c.txt"],
                "failed": ["d.txt"],
            },
            ("POST", "/documents/a.txt/refresh"): job,
            ("POST", "/documents/same.txt/refresh"): {"filename": "same.txt", "unchanged": True},
            ("GET", "/documents/jobs/j1"): done,
            ("GET", "/documents/a.txt/versions"): {
                "versions": [{"version": "171", "size": 12}, {"version": "170", "size": 9}]
            },
            ("GET", "/documents/new.txt/versions"): {"versions": []},
            ("POST", "/documents/a.txt/versions/171/restore"): job,
            ("GET", "/query-log"): {
                "queries": [
                    {"status": "no_context", "top_score": 0.02, "question": "Gym?"},
                    {"status": "answered", "top_score": None, "question": "Leave?"},
                ]
            },
            ("GET", "/documents/jobs"): {
                "jobs": [{**job, "state": "embedding", "chunks_total": 10, "chunks_done": 4}]
            },
            ("DELETE", "/documents/jobs/j1"): {**job, "cancelled": True},
            ("GET", "/workspaces"): {
                "enabled": self.workspaces_enabled,
                "workspaces": ["legal"] if self.workspaces_enabled else [],
            },
            ("POST", "/workspaces"): {"name": "legal"},
            ("DELETE", "/workspaces/legal"): {"deleted": "legal"},
            ("GET", "/watch"): {
                "folder": "/inbox",
                "files": 3,
                "retrying": [{"filename": "x.pdf", "attempts": 2, "error": "Qdrant down"}],
                "gave_up": ["y.pdf"],
                "problems": ["z.bin: unsupported"],
            },
            ("POST", "/watch/scan"): {"folder": "/inbox"},
        }
        body = routes.get((method, path))
        if body is None:
            return httpx.Response(404, json={"detail": f"no route {method} {path}"})
        return httpx.Response(200 if method == "GET" else 202, json=body)


def test_reindex_stale_reports_every_outcome_and_follows_the_jobs() -> None:
    code, out = run(mock_client(FakeServer()), "reindex", "--stale")
    assert code == 1  # one deferred and one failed
    assert "1 queued, 1 deferred (queue full)" in out
    assert "already being indexed: c.txt" in out
    assert "could not be queued: d.txt" in out
    assert "ok    a.txt" in out
    code, _ = run(mock_client(FakeServer()), "reindex", "a.txt", "--stale")
    assert code == 2


def test_refresh_follows_a_job_or_says_unchanged() -> None:
    code, out = run(mock_client(FakeServer()), "refresh", "a.txt")
    assert code == 0 and "ok    a.txt" in out
    code, out = run(mock_client(FakeServer()), "refresh", "same.txt")
    assert code == 0 and "same.txt: unchanged at its source" in out
    code, out = run(mock_client(FakeServer()), "refresh", "same.txt", "--json")
    assert json.loads(out)["unchanged"] is True
    code, out = run(mock_client(FakeServer()), "refresh", "a.txt", "--json")
    assert json.loads(out)[0]["state"] == "done"


def test_versions_lists_and_restores() -> None:
    code, out = run(mock_client(FakeServer()), "versions", "a.txt")
    assert code == 0 and out.splitlines() == ["171  12 bytes", "170  9 bytes"]
    code, out = run(mock_client(FakeServer()), "versions", "new.txt")
    assert "no earlier versions of new.txt" in out
    code, out = run(mock_client(FakeServer()), "versions", "a.txt", "--json")
    assert [v["version"] for v in json.loads(out)] == ["171", "170"]
    server = FakeServer()
    code, out = run(mock_client(server), "versions", "a.txt", "--restore", "171")
    assert code == 0 and "POST /documents/a.txt/versions/171/restore" in server.log


def test_queries_and_jobs_print_rows_and_cancel() -> None:
    code, out = run(mock_client(FakeServer()), "queries")
    assert "no_context" in out and " 0.02  Gym?" in out and "    -  Leave?" in out
    code, out = run(mock_client(FakeServer()), "jobs")
    assert out.strip() == "j1  embedding 4/10  a.txt"
    code, out = run(mock_client(FakeServer()), "jobs", "--cancel", "j1")
    assert out.strip() == "cancel requested: a.txt (queued)"
    code, out = run(mock_client(FakeServer()), "jobs", "--cancel", "j1", "--json")
    assert json.loads(out)["cancelled"] is True


def test_workspaces_list_create_and_delete() -> None:
    code, out = run(mock_client(FakeServer()), "workspaces", "list")
    assert out.splitlines() == ["(default)", "legal"]
    code, out = run(mock_client(FakeServer(workspaces_enabled=False)), "workspaces", "list")
    assert "not enabled" in out
    server = FakeServer()
    assert run(mock_client(server), "workspaces", "create", "legal")[1].strip() == (
        "created workspace legal"
    )
    code, out = run(mock_client(server), "workspaces", "delete", "legal", "--json")
    assert json.loads(out) == {"delete": "legal"}
    assert "DELETE /workspaces/legal" in server.log
    assert run(mock_client(server), "workspaces", "create")[0] == 2


def test_watch_shows_retries_give_ups_and_starts_a_pass() -> None:
    server = FakeServer()
    code, out = run(mock_client(server), "watch", "--scan")
    assert code == 0
    assert "POST /watch/scan" in server.log
    assert "watching /inbox: 3 files" in out
    assert "(a pass was started" in out
    assert "retry   x.pdf (attempt 2): Qdrant down" in out
    assert "gave up y.pdf" in out
    assert "skip    z.bin: unsupported" in out


def test_chat_reports_a_server_error_and_goes_on(
    client: DocragClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from cli.client import StreamResult

    asked: list[str] = []

    def fake_ask(question: str, *args: object, **kwargs: object) -> StreamResult:
        asked.append(question)
        if question == "Busy?":
            raise DocragError("Every question slot is taken.", 429)
        return StreamResult({"answer": "Yes [1].", "sources": [], "answer_status": "answered"}, [])

    monkeypatch.setattr(client, "ask", fake_ask)
    monkeypatch.setattr("sys.stdin", io.StringIO("Busy?\nNext?\n"))
    code, _ = run(client, "chat", "--quiet")
    assert code == 0 and asked == ["Busy?", "Next?"]
    assert "error (429): Every question slot is taken." in capsys.readouterr().err


def test_ask_sends_the_narrowing_and_answer_style(client: DocragClient, tmp_path: Path) -> None:
    run(client, "ingest", str(corpus(tmp_path)))
    sent: list[dict[str, object]] = []
    original = client._stream  # noqa: SLF001

    def recording(path, body, *rest, **kwargs):  # type: ignore[no-untyped-def]
        sent.append(body)
        return original(path, body, *rest, **kwargs)

    client._stream = recording  # type: ignore[method-assign]  # noqa: SLF001

    code, out = run(
        client,
        "--json",
        "ask",
        "alpha",
        "--page-min",
        "1",
        "--section",
        "Intro",
        "--exclude",
        "notes.md",
        "--answer-format",
        "bullets",
        "--language",
        "match",
    )

    assert code == 0, out
    assert sent == [
        {
            "question": "alpha",
            "page_min": 1,
            "section": "Intro",
            "exclude_filenames": ["notes.md"],
            "answer_format": "bullets",
            "answer_language": "match",
        }
    ]
    assert json.loads(out)["sources"][0]["filename"] == "guide.txt"
