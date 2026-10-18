"""Keep a downloaded model's files in its own directory, for ONNX Runtime's sake.

A model larger than protobuf's 2 GB limit (``intfloat/multilingual-e5-large``, the
stronger multilingual profile) keeps its weights beside the graph, in ``model.onnx_data``.
ONNX Runtime (1.27 and later) refuses external data that resolves outside the directory
the model file resolves into: "External data path escapes model directory".

FastEmbed downloads through the Hugging Face cache, where a snapshot's files are
symbolic links to blobs. huggingface_hub 1.x stores blobs in a store shared by every
repository and sharded by hash (``<cache>/blobs/29/…``, ``<cache>/blobs/9e/…``), so the
graph and its data resolve into two different directories and a freshly downloaded
large model never loads. (A cache filled by an older huggingface_hub, with one blobs
directory per repository, happens to work.)

``colocate_model_files`` replaces each link in the model's snapshot directory with a
hard link to the same blob: the same file, no extra disk, but now inside the directory
ONNX Runtime checks against. The Hugging Face cache accepts regular files in a snapshot
(it writes copies itself where symbolic links aren't available).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def model_directory(model: object) -> Path | None:
    """The directory a FastEmbed model was downloaded to, if it has one.

    ``TextEmbedding``/``SparseTextEmbedding``/``TextCrossEncoder`` wrap the concrete
    ONNX model in ``.model``, which records its ``_model_dir`` when constructed (before
    any lazy load)."""

    inner = getattr(model, "model", model)
    directory = getattr(inner, "_model_dir", None)
    return Path(directory) if directory else None


def colocate_model_files(directory: Path) -> int:
    """Replace every symbolic link under ``directory`` (subdirectories too: many
    exports keep the graph and its data in ``onnx/``) with a hard link to what it
    points at (a copy where a hard link can't be made); the number replaced.

    Never fails the caller: a file that can't be replaced (a read-only cache) is left as
    it was and logged, and loading then reports what ONNX Runtime thinks of it.
    """

    replaced = 0
    try:
        entries = sorted(directory.rglob("*"))
    except OSError:
        return 0
    for entry in entries:
        if not entry.is_symlink():
            continue
        target = entry.resolve()
        if not target.is_file():
            continue
        temporary = entry.with_name(f".{entry.name}.docrag-link")
        try:
            try:
                os.link(target, temporary)
            except OSError:
                shutil.copyfile(target, temporary)
            os.replace(temporary, entry)
            replaced += 1
        except OSError as exc:
            logger.warning("Could not move %s beside its model (%s).", entry, exc)
            temporary.unlink(missing_ok=True)
    return replaced


def colocate(model: object) -> None:
    """``colocate_model_files`` for a FastEmbed model's directory, when it has one."""

    directory = model_directory(model)
    if directory is not None:
        colocate_model_files(directory)
