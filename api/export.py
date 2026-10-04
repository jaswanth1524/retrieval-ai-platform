"""``GET /export``: one zip holding what DocRAG knows beyond its vectors.

- ``manifest.json``: every indexed document's metadata (chunks, pages, size, upload time,
  chunker version, tags, whether an original is kept) plus the settings that decide
  whether an index built elsewhere is compatible (embedding tag, sparse model).
- ``originals/<filename>``: the stored uploads (``RAW_DOCUMENT_DIR``), which are what a
  rebuild needs — re-indexing from them after an embedding model change, or moving to a
  new instance, without asking anyone to upload again.
- ``feedback.jsonl``: every answer rating, when feedback is on.

Qdrant's own data is backed up with a Qdrant snapshot (README "Backup and upgrade");
this complements it rather than duplicating vectors a new model would replace anyway.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from collections.abc import Iterator
from dataclasses import asdict
from tempfile import SpooledTemporaryFile
from typing import IO, BinaryIO

from api.documents import CHUNKER_VERSION
from api.feedback import FeedbackStore
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.settings import AppSettings
from api.version import app_version

# In memory up to this, then a temporary file: originals can add up to gigabytes.
_SPOOL_BYTES = 32 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024
_FEEDBACK_PAGE = 1_000_000
MANIFEST_VERSION = 1


def _write_member(archive: zipfile.ZipFile, name: str, source: BinaryIO) -> None:
    """Copy ``source`` into ``archive`` in blocks, never whole into memory."""

    info = zipfile.ZipInfo(name, date_time=time.localtime(os.fstat(source.fileno()).st_mtime)[:6])
    # ZIP_STORED: uploads are mostly already compressed (PDF, DOCX).
    info.compress_type = zipfile.ZIP_STORED
    with archive.open(info, "w", force_zip64=True) as member:
        shutil.copyfileobj(source, member, _CHUNK_BYTES)


def build_export(
    settings: AppSettings,
    repository: VectorRepository,
    raw_store: RawDocumentStore | None,
    feedback_store: FeedbackStore | None,
) -> IO[bytes]:
    """Write the export zip to a spooled temporary file, rewound, and return it."""

    metadata = repository.filename_metadata(settings)
    spool: IO[bytes] = SpooledTemporaryFile(max_size=_SPOOL_BYTES)
    with zipfile.ZipFile(spool, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        documents = []
        for filename, meta in sorted(metadata.items()):
            # A document deleted while the export runs simply has no original here,
            # instead of failing the whole export with FileNotFoundError.
            handle = raw_store.open(filename) if raw_store is not None else None
            original = handle is not None
            if handle is not None:
                with handle:
                    _write_member(archive, f"originals/{filename}", handle)
            documents.append(
                {
                    "filename": filename,
                    "chunk_count": meta.chunk_count,
                    "page_count": meta.page_count,
                    "byte_size": meta.byte_size,
                    "uploaded_at": meta.uploaded_at,
                    "chunker_version": meta.chunker_version,
                    "tags": list(meta.tags),
                    "original": f"originals/{filename}" if original else None,
                }
            )
        manifest = {
            # Bumped on any change a restore (POST /import) must handle differently.
            "manifest_version": MANIFEST_VERSION,
            "docrag_version": app_version(),
            "exported_at": time.time(),
            "qdrant_collection": settings.qdrant_collection,
            "embedding_model_tag": settings.embedding_model_tag,
            "sparse_embedding_model": settings.sparse_embedding_model,
            "chunker_version": CHUNKER_VERSION,
            "documents": documents,
        }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        if feedback_store is not None:
            rows = feedback_store.list_recent(limit=_FEEDBACK_PAGE)
            archive.writestr(
                "feedback.jsonl",
                "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in reversed(rows)),
            )
    spool.seek(0)
    return spool


def iter_file(handle: IO[bytes]) -> Iterator[bytes]:
    """Stream ``handle`` in chunks, closing (and so deleting) it when done."""

    with handle:
        while block := handle.read(_CHUNK_BYTES):
            yield block
