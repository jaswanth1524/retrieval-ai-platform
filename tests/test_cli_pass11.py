"""CLI profiles, shell completion and Markdown answers; eval run diffs and reports."""

from __future__ import annotations

import io
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from cli import __main__ as cli_main
from cli.client import DocragClient, StreamResult
from cli.config import Profile, ProfileError, load_profile, resolve
from eval.compare import diff_runs, format_diff, load_run, ragas_scores, render_report
from eval.harness import main as harness_main
from tests.routes_http.support import hermetic_app, make_settings


@pytest.fixture
def client() -> Iterator[DocragClient]:
    clear_dependency_caches()
    app = hermetic_app(make_settings(), QdrantClient(":memory:"))
    with TestClient(app) as test_client:
        docrag = DocragClient("http://testserver", "", http=test_client, sleep=lambda _: None)
        yield docrag
        docrag.close()
    clear_dependency_caches()


def run(client: DocragClient | None, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    return cli_main.main(list(argv), client=client, out=out), out.getvalue()


def write_config(tmp_path: Path, text: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    monkeypatch.setenv("DOCRAG_CONFIG", str(path))
    for name in ("DOCRAG_URL", "DOCRAG_API_KEY", "DOCRAG_WORKSPACE", "DOCRAG_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    return path


def test_a_profile_is_read_with_its_key_from_the_named_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(
        tmp_path,
        '[default]\nurl = "http://one"\n\n[profiles.prod]\nurl = "https://two"\n'
        'workspace = "legal"\nkey_env = "PROD_KEY"\n',
        monkeypatch,
    )
    monkeypatch.setenv("PROD_KEY", "s3cret")

    assert load_profile(None) == Profile(url="http://one")
    prod = load_profile("prod")
    assert prod == Profile(url="https://two", workspace="legal", key_env="PROD_KEY")
    assert prod is not None and prod.api_key() == "s3cret"
    monkeypatch.setenv("DOCRAG_PROFILE", "prod")
    assert load_profile(None) == prod


def test_flags_win_over_the_environment_which_wins_over_the_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCRAG_URL", raising=False)
    assert resolve("http://flag", "DOCRAG_URL", "http://profile") == "http://flag"
    assert resolve(None, "DOCRAG_URL", "http://profile") == "http://profile"
    monkeypatch.setenv("DOCRAG_URL", "http://env")
    # None: the client reads the environment itself.
    assert resolve(None, "DOCRAG_URL", "http://profile") is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[default]\napi_key = "x"\n', "holds a key"),
        ("[default]\nurl = 3\n", "must be a string"),
        ('[default]\nproxy = "x"\n', "unknown settings"),
        ("[default\n", "config.toml"),
    ],
)
def test_a_bad_config_file_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, message: str
) -> None:
    write_config(tmp_path, text, monkeypatch)
    with pytest.raises(ProfileError, match=message):
        load_profile(None)


def test_an_unknown_profile_stops_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_config(tmp_path, '[profiles.a]\nurl = "http://a"\n', monkeypatch)
    code, _ = run(None, "--profile", "b", "status")
    assert code == 2
    assert "no profile 'b' (it has: a)" in capsys.readouterr().err


def test_completion_scripts_need_no_server_and_parse(tmp_path: Path) -> None:
    code, script = run(None, "completion", "bash")
    assert code == 0
    assert "complete -F _docrag docrag" in script
    assert "--answer-format) COMPREPLY" in script
    path = tmp_path / "docrag.bash"
    path.write_text(script)
    probe = (
        f'source {path}; COMP_WORDS=(docrag work); COMP_CWORD=1; _docrag; echo "${{COMPREPLY[*]}}"'
    )
    shown = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, check=True)
    assert shown.stdout.split() == ["workspaces"]
    assert run(None, "completion", "zsh")[1].startswith("(( $+functions[compdef] ))")


def test_ask_writes_markdown_with_footnoted_sources(client: DocragClient, tmp_path: Path) -> None:
    (tmp_path / "guide.txt").write_text("Intro\nAlpha is the first letter.")
    assert run(client, "ingest", str(tmp_path / "guide.txt"))[0] == 0

    code, out = run(client, "ask", "alpha", "--format", "markdown", "--quiet")

    assert code == 0, out
    assert out.startswith("Alpha is documented [^1].")
    assert "[^1]: guide.txt, p.1 · Intro" in out


def test_answer_markdown_quotes_each_source() -> None:
    result = StreamResult(
        done={
            "answer": "Leave is 20 days [1][2].",
            "sources": [
                {
                    "source_number": 1,
                    "filename": "a.pdf",
                    "page": 2,
                    "page_end": 3,
                    "section": "Leave",
                    "spans": [{"quote": "twenty  days"}],
                },
                {"source_number": 2, "filename": "b.md", "page": 1, "section": ""},
            ],
        },
        followups=[],
    )
    text = cli_main.answer_markdown(result)
    assert text.splitlines()[0] == "Leave is 20 days [^1][^2]."
    assert "[^1]: a.pdf, p.2-3 · Leave — “twenty days”" in text
    assert "[^2]: b.md, p.1 · -" in text


def test_chat_history_is_kept_between_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("readline")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    save = cli_main._start_line_editing()  # noqa: SLF001
    assert save is not None
    import readline

    readline.add_history("leave-days?")
    save()
    # libedit (macOS) escapes some characters in the file; the question is still there.
    assert "leave-days?" in (tmp_path / "docrag" / "history").read_text()


def _retrieval(path: Path, scores: dict[str, float], aggregate: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "aggregate": {"filename": {"hit@1": aggregate, "mrr": aggregate}},
                "per_question": [
                    {"question": q, "filename": {"hit@1": s, "mrr": s}} for q, s in scores.items()
                ],
            }
        )
    )
    return path


def test_a_diff_names_each_question_that_flipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _retrieval(tmp_path / "a.json", {"q1": 1.0, "q2": 0.0, "q3": 1.0}, 0.67)
    after = _retrieval(tmp_path / "b.json", {"q1": 0.0, "q2": 1.0, "q4": 1.0}, 0.67)

    diff = diff_runs(load_run(before), load_run(after))
    assert [c.question for c in diff.regressed] == ["q1"]
    assert [c.question for c in diff.improved] == ["q2"]
    assert diff.only_before == ["q3"] and diff.only_after == ["q4"]
    assert "- q1  [filename hit@1 1.00→0.00; filename mrr 1.00→0.00]" in format_diff(diff)

    assert harness_main(["diff", str(before), str(after)]) == 0
    assert harness_main(["diff", str(before), str(after), "--fail-on-regression"]) == 1
    assert harness_main(["diff", str(before), str(tmp_path / "missing.json")]) == 2
    capsys.readouterr()


def test_answer_runs_diff_on_status_and_facts(tmp_path: Path) -> None:
    before = tmp_path / "a.json"
    after = tmp_path / "b.json"
    before.write_text(
        json.dumps(
            [{"question": "q", "answer": "x", "answer_status": "answered", "fact_hit": True}]
        )
    )
    after.write_text(
        json.dumps(
            [{"question": "q", "answer": "", "answer_status": "answered", "error": "timeout"}]
        )
    )
    diff = diff_runs(load_run(before), load_run(after))
    assert [c.question for c in diff.regressed] == ["q"]
    assert "status answered→error" in diff.regressed[0].details


def test_a_report_is_one_escaped_page(tmp_path: Path) -> None:
    run_path = tmp_path / "answers.json"
    run_path.write_text(
        json.dumps(
            [
                {
                    "question": "<script>alert(1)</script>",
                    "answer": "20 days [1]",
                    "answer_status": "answered",
                    "fact_hit": True,
                    "cited_filenames": ["a.pdf"],
                }
            ]
        )
    )
    ragas = tmp_path / "ragas.json"
    ragas.write_text(json.dumps({"rows": [{"faithfulness": 0.5}, {"faithfulness": 1.0}]}))
    output = tmp_path / "report.html"

    assert (
        harness_main(
            [
                "report",
                str(run_path),
                "--output",
                str(output),
                "--ragas",
                str(ragas),
                "--title",
                "T",
            ]
        )
        == 0
    )
    page = output.read_text()
    assert "<script>alert" not in page and "&lt;script&gt;" in page
    assert "faithfulness" in page and "0.750" in page
    assert "facts stated" in page
    assert ragas_scores({"faithfulness": 0.9, "label": "x"}) == {"faithfulness": 0.9}
    with_base = render_report(load_run(run_path), title="T", base=load_run(run_path))
    assert "0 regressed" in with_base


def test_latency_is_summarized_and_shown_but_never_judged(tmp_path: Path) -> None:
    from eval.compare import render_report
    from eval.harness import latency_summary

    assert latency_summary([]) == {}
    assert latency_summary([10.0, 30.0, 20.0, 400.0]) == {"p50": 20.0, "p95": 400.0, "mean": 115.0}

    before = _retrieval(tmp_path / "a.json", {"q1": 1.0}, 1.0)
    after = _retrieval(tmp_path / "b.json", {"q1": 1.0}, 1.0)
    for path, p50 in ((before, 100.0), (after, 160.0)):
        data = json.loads(path.read_text())
        data["latency_ms"] = {"p50": p50, "p95": p50 * 2, "mean": p50}
        data["per_question"][0]["latency_ms"] = {"total_ms": p50}
        path.write_text(json.dumps(data))

    diff = diff_runs(load_run(before), load_run(after))
    assert not diff.regressed and not diff.improved
    assert "  latency p50: 100 → 160 ms (+60)" in format_diff(diff)
    page = render_report(load_run(after), title="t")
    assert "<td>p50 (ms)</td><td>160</td>" in page
