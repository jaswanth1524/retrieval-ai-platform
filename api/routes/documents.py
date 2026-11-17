"""Document routes: list, upload, re-index, delete, tags, jobs, content, original, export.

Split by what they do — ``documents_read`` (the listing, passages, originals,
versions), ``documents_write`` (everything that changes a document), ``documents_jobs``
(ingest jobs) and ``backup`` (export and import) — and registered here as one set.
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import backup, documents_jobs, documents_read, documents_write, migrate


def register(app: FastAPI) -> None:
    """Register the document and export routes."""

    # Jobs first: a document's name may have folders (``/documents/{filename:path}``),
    # which would otherwise also match ``/documents/jobs/{id}`` and the imports.
    for module in (documents_jobs, documents_read, documents_write, backup, migrate):
        app.include_router(module.router)
