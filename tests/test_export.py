from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from api.export import MANIFEST_VERSION, build_export
from api.pipeline import IngestService
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.settings import AppSettings
from tests.factories import make_test_settings
from tests.routes_http.support import FakeEmbeddingProvider


def _settings(**overrides: Any) -> AppSettings:
    return make_test_settings(
        qdrant_url=":memory:",
        qdrant_collection="export_documents",
        qdrant_dense_vector_size=3,
        embedding_model_tag="test-embedding:v1",
        **overrides,
    )


def _indexed(settings: AppSettings, *names: str) -> VectorRepository:
    repository = VectorRepository(QdrantClient(":memory:"))
    service = IngestService(repository, FakeEmbeddingProvider(), settings)
    for name in names:
        service.ingest(name, b"Intro\nalpha beta")
    return repository


class DeletedMidExportStore(RawDocumentStore):
    """Reports an original as stored, then loses it before it can be read — what a
    DELETE landing between the export's existence check and its read looks like."""

    def path(self, filename: str) -> Path | None:
        found = super().path(filename)
        if found is not None and filename == "gone.txt":
            found.unlink()
        return found


def test_an_original_deleted_mid_export_is_skipped_not_a_failure(tmp_path: Path) -> None:
    settings = _settings()
    repository = _indexed(settings, "kept.txt", "gone.txt")
    store = DeletedMidExportStore(str(tmp_path))
    store.save("kept.txt", b"kept bytes")
    store.save("gone.txt", b"gone bytes")

    with build_export(settings, repository, store, None) as spool:
        archive = zipfile.ZipFile(spool)
        manifest = json.loads(archive.read("manifest.json"))
        originals = {d["filename"]: d["original"] for d in manifest["documents"]}

        assert manifest["manifest_version"] == MANIFEST_VERSION == 1
        assert originals == {"gone.txt": None, "kept.txt": "originals/kept.txt"}
        assert archive.read("originals/kept.txt") == b"kept bytes"
        assert "originals/gone.txt" not in archive.namelist()
