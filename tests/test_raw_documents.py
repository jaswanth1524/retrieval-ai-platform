from __future__ import annotations

from pathlib import Path

from api.raw_documents import RawDocumentStore


def test_save_and_read_round_trips_bytes(tmp_path: Path) -> None:
    store = RawDocumentStore(str(tmp_path))

    store.save("guide.pdf", b"pdf-bytes")

    assert store.read("guide.pdf") == b"pdf-bytes"


def test_read_returns_none_for_unknown_filename(tmp_path: Path) -> None:
    store = RawDocumentStore(str(tmp_path))
    assert store.read("does-not-exist.pdf") is None


def test_delete_removes_a_stored_file(tmp_path: Path) -> None:
    store = RawDocumentStore(str(tmp_path))
    store.save("guide.pdf", b"pdf-bytes")

    store.delete("guide.pdf")

    assert store.read("guide.pdf") is None


def test_delete_is_a_noop_for_an_unknown_filename(tmp_path: Path) -> None:
    store = RawDocumentStore(str(tmp_path))
    store.delete("never-existed.pdf")  # must not raise


def test_normalizes_a_path_traversal_attempt_to_a_basename(tmp_path: Path) -> None:
    """save/read/delete all key on normalize_filename, not the raw client-supplied
    string — a filename containing path components must resolve inside the store's
    own directory, never escape it."""

    store = RawDocumentStore(str(tmp_path))

    store.save("../../etc/passwd", b"not actually passwd")

    escaped_path = tmp_path.parent.parent / "etc" / "passwd"
    assert not escaped_path.exists()
    stored_files = list(tmp_path.iterdir())
    assert len(stored_files) == 1
    assert stored_files[0].name == "passwd"
    assert store.read("../../etc/passwd") == b"not actually passwd"


def test_creates_the_directory_if_it_does_not_exist(tmp_path: Path) -> None:
    directory = tmp_path / "nested" / "raw"
    RawDocumentStore(str(directory))
    assert directory.is_dir()
