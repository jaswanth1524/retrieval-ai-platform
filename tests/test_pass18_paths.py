"""Pass 18: document names with folders (reports/2024/q1.txt), opt-in per source."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from api.documents import UnsupportedDocumentError, normalize_document_name, validate_document_name
from api.raw_documents import RawDocumentStore
from api.watch import names_on_disk, scan_folder
from tests.routes_http.support import hermetic_app, make_settings, wait_until

NAME = "reports/2024/q1.txt"


@contextmanager
def _server(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Iterator[TestClient]:
    clear_dependency_caches()
    settings = make_settings(**overrides)
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    try:
        with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as client:
            yield client
    finally:
        monkeypatch.undo()
        clear_dependency_caches()


def _upload(client: TestClient, name: str, content: bytes) -> dict[str, object]:
    accepted = client.post("/documents", files={"file": ("q1.txt", content)}, data={"name": name})
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["job_id"]

    def finished() -> bool:
        return client.get(f"/documents/jobs/{job_id}").json()["state"] in ("done", "failed")

    wait_until(finished)
    result: dict[str, object] = client.get(f"/documents/jobs/{job_id}").json()
    return result


@pytest.mark.parametrize(
    "bad",
    [
        "../secret.txt",
        "a/../b.txt",
        "a//b.txt",
        "/abs.txt",
        "a/./b.txt",
        "a/b.txt/",
        ".hidden/b.txt",
        "jobs/x.txt",
        "a\\b/c.txt",
        "a/ b.txt",
        "a/b\x00.txt",
    ],
)
def test_a_name_that_could_mean_another_path_is_refused(bad: str) -> None:
    with pytest.raises(UnsupportedDocumentError):
        validate_document_name(bad)


def test_a_valid_name_with_folders_is_kept_as_it_is() -> None:
    assert normalize_document_name(NAME) == NAME
    assert validate_document_name("plain.txt") == "plain.txt"
    with pytest.raises(UnsupportedDocumentError, match="Unsupported"):
        validate_document_name("a/b.exe")


def test_a_name_with_folders_works_through_every_document_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    with _server(monkeypatch, raw_document_dir=str(raw), raw_document_versions=3) as client:
        assert _upload(client, NAME, b"Intro\nfirst quarter numbers")["state"] == "done"
        assert _upload(client, "reports/2025/q1.txt", b"Intro\nnext year")["state"] == "done"
        listing = client.get("/documents").json()
        assert listing["filenames"] == [NAME, "reports/2025/q1.txt"]
        assert listing["reindexable_filenames"] == listing["filenames"]
        assert (raw / "reports" / "2024" / "q1.txt").read_bytes() == b"Intro\nfirst quarter numbers"

        content = client.get(f"/documents/{NAME}/content").json()
        assert content["chunks"] and content["total_chunks"] >= 1
        assert client.get(f"/documents/{NAME}/outline").status_code == 200
        original = client.get(f"/documents/{NAME}/original")
        assert original.status_code == 200 and original.content.endswith(b"numbers")
        assert 'filename="q1.txt"' in original.headers["content-disposition"]
        assert client.patch(f"/documents/{NAME}/tags", json={"tags": ["fin"]}).status_code == 200
        assert client.get("/documents").json()["tags"] == {NAME: ["fin"]}
        reindex = client.post(f"/documents/{NAME}/reindex")
        assert reindex.status_code == 202, reindex.text
        wait_until(
            lambda: (
                client.get(f"/documents/jobs/{reindex.json()['job_id']}").json()["state"] == "done"
            )
        )
        assert _upload(client, NAME, b"Intro\nfirst quarter, corrected")["state"] == "done"
        versions = client.get(f"/documents/{NAME}/versions").json()["versions"]
        assert len(versions) == 1
        restored = client.post(f"/documents/{NAME}/versions/{versions[0]['version']}/restore")
        assert restored.status_code == 202, restored.text
        wait_until(
            lambda: (
                client.get(f"/documents/jobs/{restored.json()['job_id']}").json()["state"] == "done"
            )
        )
        assert client.get(f"/documents/{NAME}/original").content.endswith(b"numbers")
        # A URL-only route on an uploaded document: its own 404, not another route's.
        assert client.post(f"/documents/{NAME}/refresh").status_code in (404, 409)
        # The routes that share the prefix still answer as themselves.
        assert client.get("/documents/jobs").status_code == 200
        assert client.get("/documents/imports").status_code == 200

        search = client.post("/search", json={"query": "quarter numbers", "filenames": [NAME]})
        assert search.status_code == 200
        assert {r["filename"] for r in search.json()["results"]} <= {NAME}

        assert client.delete(f"/documents/{NAME}").status_code == 200
        assert client.get("/documents").json()["filenames"] == ["reports/2025/q1.txt"]
        # Its folders went with it; the sibling's stay.
        assert not (raw / "reports" / "2024").exists()
        assert (raw / "reports" / "2025" / "q1.txt").exists()
        assert not (raw / ".versions" / "reports" / "2024").exists()

        # Traversal through the path parameter finds nothing.
        # (Percent-encoded: the HTTP client would collapse a literal ".." itself.)
        for probe in ("reports/%2E%2E/reports/2025/q1.txt", "reports//2025/q1.txt"):
            assert client.get(f"/documents/{probe}/original").status_code == 404
        refused = client.post(
            "/documents", files={"file": ("x.txt", b"x")}, data={"name": "../escape.txt"}
        )
        assert refused.status_code == 400
        assert not (tmp_path / "escape.txt").exists()


def test_a_backup_carries_names_with_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _server(monkeypatch, raw_document_dir=str(tmp_path / "a")) as client:
        _upload(client, NAME, b"Intro\nfirst quarter numbers")
        backup = client.get("/export").content
    with zipfile.ZipFile(io.BytesIO(backup)) as archive:
        assert f"originals/{NAME}" in archive.namelist()
    with _server(monkeypatch, raw_document_dir=str(tmp_path / "b")) as client:
        restored = client.post("/import", files={"file": ("b.zip", backup)})
        assert restored.status_code == 202, restored.text
        assert [job["filename"] for job in restored.json()["jobs"]] == [NAME]
        wait_until(lambda: client.get("/documents").json()["filenames"] == [NAME])


def test_a_zip_keeps_its_paths_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("2024/readme.txt", b"Intro\nthe 2024 notes")
        archive.writestr("2025/readme.txt", b"Intro\nthe 2025 notes")
    with _server(monkeypatch) as client:
        plain = client.post("/documents/archive", files={"file": ("z.zip", buffer.getvalue())})
        assert [job["filename"] for job in plain.json()["jobs"]] == ["readme.txt"]
        kept = client.post(
            "/documents/archive",
            files={"file": ("z.zip", buffer.getvalue())},
            data={"keep_paths": "true"},
        )
        assert sorted(job["filename"] for job in kept.json()["jobs"]) == [
            "2024/readme.txt",
            "2025/readme.txt",
        ]


def test_the_watcher_names_files_by_their_path_when_asked(tmp_path: Path) -> None:
    (tmp_path / "2024").mkdir()
    (tmp_path / "2025").mkdir()
    (tmp_path / "2024" / "readme.txt").write_text("a")
    (tmp_path / "2025" / "readme.txt").write_text("b")
    plain = scan_folder(tmp_path, 1000)
    assert list(plain.files) == ["readme.txt"] and plain.problems
    kept = scan_folder(tmp_path, 1000, keep_paths=True)
    assert sorted(kept.files) == ["2024/readme.txt", "2025/readme.txt"] and not kept.problems
    assert names_on_disk(tmp_path, keep_paths=True) == {"2024/readme.txt", "2025/readme.txt"}


def test_the_store_lists_names_in_folders_and_never_its_own(tmp_path: Path) -> None:
    store = RawDocumentStore(str(tmp_path), keep_versions=2)
    store.save("top.txt", b"1")
    store.save(NAME, b"2")
    store.save(NAME, b"3")
    store.save_details(NAME, {"tags": []})
    (tmp_path / "workspaces" / "legal").mkdir(parents=True)
    (tmp_path / "workspaces" / "legal" / "theirs.txt").write_text("x")
    assert store.stored_names() == {"top.txt", NAME}
    assert store.versioned_names() == {NAME}
    assert store.names_with_details() == {NAME}


def test_a_name_and_a_folder_of_the_same_name_cannot_both_be_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api.document_names import FilenameConflictError, refuse_near_duplicate_name

    with pytest.raises(FilenameConflictError, match="file and a folder"):
        refuse_near_duplicate_name("a.txt/b.txt", ["a.txt"])
    with pytest.raises(FilenameConflictError, match="file and a folder"):
        refuse_near_duplicate_name("A.txt", ["a.txt/b.txt"])
    refuse_near_duplicate_name("a.txt.d/b.txt", ["a.txt"])  # only a whole part clashes

    raw = tmp_path / "raw"
    with _server(monkeypatch, raw_document_dir=str(raw), raw_document_versions=2) as client:
        assert _upload(client, "a.txt", b"Intro\nthe file")["state"] == "done"
        refused = client.post(
            "/documents", files={"file": ("b.txt", b"x")}, data={"name": "a.txt/b.txt"}
        )
        assert refused.status_code == 409
        assert _upload(client, "c/d.txt", b"Intro\nin a folder")["state"] == "done"
        assert _upload(client, "c/d.txt", b"Intro\nin a folder, again")["state"] == "done"
        refused = client.post("/documents", files={"file": ("c", b"x")}, data={"name": "c.txt"})
        assert refused.status_code == 202  # "c.txt" is not "c"
        assert (
            client.post(
                "/documents", files={"file": ("x.txt", b"x")}, data={"name": "c/d.txt/e.txt"}
            ).status_code
            == 409
        )
        # The folder's history is untouched by anything above.
        assert len(client.get("/documents/c/d.txt/versions").json()["versions"]) == 1
