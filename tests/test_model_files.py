"""Model files are kept beside their model, where ONNX Runtime looks for external data."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from api.model_files import colocate, colocate_model_files, model_directory


def _snapshot(tmp_path: Path) -> Path:
    """The huggingface_hub 1.x layout: a snapshot of links into a sharded blob store."""

    for shard, blob, body in (("29", "graph", b"graph"), ("9e", "weights", b"weights")):
        (tmp_path / "blobs" / shard).mkdir(parents=True)
        (tmp_path / "blobs" / shard / blob).write_bytes(body)
    snapshot = tmp_path / "models--x" / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    (snapshot / "model.onnx").symlink_to(tmp_path / "blobs" / "29" / "graph")
    (snapshot / "model.onnx_data").symlink_to(tmp_path / "blobs" / "9e" / "weights")
    (snapshot / "config.json").write_text("{}")
    (tmp_path / "blobs" / "aa").mkdir()
    (tmp_path / "blobs" / "aa" / "quantized").write_bytes(b"quantized")
    (snapshot / "onnx").mkdir()
    (snapshot / "onnx" / "model_quantized.onnx").symlink_to(tmp_path / "blobs" / "aa" / "quantized")
    return snapshot


def test_links_become_hard_links_to_the_same_blobs(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)

    assert colocate_model_files(snapshot) == 3
    graph, weights = snapshot / "model.onnx", snapshot / "model.onnx_data"
    assert not graph.is_symlink() and not weights.is_symlink()
    # What ONNX Runtime compares: both now resolve into the snapshot itself.
    assert graph.resolve().parent == weights.resolve().parent == snapshot.resolve()
    # The same files, not copies.
    assert os.path.samefile(weights, tmp_path / "blobs" / "9e" / "weights")
    assert weights.read_bytes() == b"weights"
    assert (snapshot / "config.json").read_text() == "{}"
    assert sorted(p.name for p in snapshot.iterdir()) == [
        "config.json",
        "model.onnx",
        "model.onnx_data",
        "onnx",
    ]
    nested = snapshot / "onnx" / "model_quantized.onnx"
    assert not nested.is_symlink() and nested.read_bytes() == b"quantized"
    assert colocate_model_files(snapshot) == 0  # nothing left to do


def test_a_fastembed_model_is_found_through_its_wrapper(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    wrapper = SimpleNamespace(model=SimpleNamespace(_model_dir=str(snapshot)))

    assert model_directory(wrapper) == snapshot
    colocate(wrapper)
    assert not (snapshot / "model.onnx_data").is_symlink()

    # A test double (no download directory) is left alone.
    assert model_directory(object()) is None
    colocate(object())
    assert colocate_model_files(tmp_path / "missing") == 0
