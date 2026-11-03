"""Document read routes: the listing, a document's passages, its original and its versions."""

from __future__ import annotations

import logging
import mimetypes
import os
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from api.dependencies import (
    get_url_refresh_state_store,
    require_api_key,
)
from api.document_names import stored_filename
from api.document_ops import (
    require_raw_store,
    staleness,
)
from api.documents import (
    DocumentNotFoundError,
)
from api.metrics import corpus_documents
from api.repository import DocumentMetadata
from api.routes.deps import (
    RawDocumentStoreDep,
    SettingsDep,
    TokenCounterDep,
    VectorRepositoryDep,
)
from api.routes.responses import content_chunk
from api.schemas import (
    DocumentContentResponse,
    DocumentListResponse,
    DocumentVersionResponse,
    DocumentVersionsResponse,
    UrlRefreshStatusResponse,
)
from api.settings import AppSettings
from api.streaming import ClosingStreamingResponse
from api.url_fetch import (
    loggable_url,
)
from api.url_refresh import refresh_due
from api.workspaces import workspace_of

logger = logging.getLogger("api.routes.documents")

router = APIRouter()


# A page of /content is at most 2 * this + 1 chunks.
CONTENT_MAX_RADIUS = 100


# Far past any real document; a 400-digit `start` used to reach Qdrant's filter and 500.
CONTENT_MAX_ORDINAL = 1_000_000


# /content with no paging parameters returns at most this many chunks.
CONTENT_UNPAGED_MAX_CHUNKS = 2000


_ORIGINAL_CHUNK_BYTES = 64 * 1024


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    dependencies=[Depends(require_api_key)],
)
def list_documents(
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    raw_store: RawDocumentStoreDep,
    token_counter: TokenCounterDep,
) -> DocumentListResponse:
    metadata = repository.filename_metadata(settings)
    corpus_documents.labels(workspace=workspace_of(settings) or "default").set(len(metadata))
    is_stale = staleness(settings, token_counter)
    stored_originals = raw_store.stored_names() if raw_store is not None else set()
    return DocumentListResponse(
        url_refresh=_url_refresh_statuses(metadata, settings),
        filenames=sorted(metadata),
        chunk_counts={name: meta.chunk_count for name, meta in metadata.items()},
        page_counts={name: meta.page_count for name, meta in metadata.items()},
        byte_sizes={
            name: meta.byte_size for name, meta in metadata.items() if meta.byte_size is not None
        },
        uploaded_ats={
            name: meta.uploaded_at
            for name, meta in metadata.items()
            if meta.uploaded_at is not None
        },
        stale_filenames=sorted(name for name, meta in metadata.items() if is_stale(meta)),
        incomplete_filenames=sorted(name for name, meta in metadata.items() if meta.incomplete),
        duplicate_groups=duplicate_groups(metadata),
        reindexable_filenames=sorted(set(metadata) & stored_originals),
        tags={name: list(meta.tags) for name, meta in metadata.items() if meta.tags},
        # Without the query string or fragment, which can carry a signed token: the
        # listing is readable with the read key. Refresh uses the stored full URL.
        source_urls={
            name: loggable_url(meta.source_url)
            for name, meta in metadata.items()
            if meta.source_url
        },
    )


def duplicate_groups(metadata: dict[str, DocumentMetadata]) -> list[list[str]]:
    """Names sharing one content hash, two or more to a group, groups in order of their
    first name. A document with mixed or no hashes (incomplete, or indexed before
    hashes were stored) is in none."""

    by_hash: dict[str, list[str]] = {}
    for name, meta in sorted(metadata.items()):
        if meta.content_sha256:
            by_hash.setdefault(meta.content_sha256, []).append(name)
    return sorted(names for names in by_hash.values() if len(names) > 1)


def _url_refresh_statuses(
    metadata: dict[str, DocumentMetadata], settings: AppSettings
) -> dict[str, UrlRefreshStatusResponse]:
    """Each fetched document's scheduled refresh: its interval, last check and when
    it is next due (never, with the refresh off)."""

    fetched = {name: meta for name, meta in metadata.items() if meta.source_url}
    if not fetched:
        return {}
    running = bool(settings.url_refresh_interval_minutes and settings.url_ingest_enabled)
    default_seconds = float(settings.url_refresh_interval_minutes) * 60.0
    states = get_url_refresh_state_store().get_all(settings.qdrant_collection)
    statuses = {}
    for name, meta in fetched.items():
        state = states.get(name)
        statuses[name] = UrlRefreshStatusResponse(
            refresh_minutes=meta.source_refresh_minutes,
            last_checked=state.last_checked if state is not None else None,
            last_outcome=state.last_outcome if state is not None else None,
            failures=state.failures if state is not None else 0,
            next_due=refresh_due(meta, state, default_seconds) if running else None,
        )
    return statuses


# Earlier versions of a document's original, kept by re-uploads when
# RAW_DOCUMENT_VERSIONS is set; 409 without stored originals, like /reindex.
@router.get(
    "/documents/{filename}/versions",
    response_model=DocumentVersionsResponse,
    dependencies=[Depends(require_api_key)],
)
def list_document_versions(
    filename: str, raw_store: RawDocumentStoreDep
) -> DocumentVersionsResponse:
    store = require_raw_store(raw_store)
    name = stored_filename(filename)
    return DocumentVersionsResponse(
        filename=name,
        versions=[
            DocumentVersionResponse(version=v.version, saved_at=v.saved_at, size=v.size)
            for v in store.versions(name)
        ],
    )


# Guarded alongside GET /documents: this returns every chunk's raw text, so leaving
# it open would let an unauthenticated caller list the corpus and then dump it in
# full — the key would protect Q&A while the documents themselves stayed readable.
@router.get(
    "/documents/{filename}/content",
    response_model=DocumentContentResponse,
    dependencies=[Depends(require_api_key)],
)
def document_content(
    filename: str,
    repository: VectorRepositoryDep,
    settings: SettingsDep,
    around: Annotated[str | None, Query(max_length=200)] = None,
    radius: Annotated[int, Query(ge=0, le=CONTENT_MAX_RADIUS)] = 30,
    start: Annotated[int | None, Query(ge=1, le=CONTENT_MAX_ORDINAL)] = None,
    end: Annotated[int | None, Query(ge=1, le=CONTENT_MAX_ORDINAL)] = None,
) -> DocumentContentResponse:
    # No parameters: the whole document, as before. `around` (a chunk id) returns
    # `radius` chunks either side of it; `start`/`end` a range of ordinals. A 50 MB
    # CSV used to ship every chunk so the viewer could show the few around a citation.
    filename = stored_filename(filename)
    total = repository.chunk_count_for_filename(settings, filename)
    if total == 0:
        raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
    if around is None and start is None and end is None:
        # The whole document, up to CONTENT_UNPAGED_MAX_CHUNKS: unbounded, one
        # request could pull a 100k-chunk document into memory. total_chunks still
        # reports the real size, so a client can tell the list was cut short.
        if total <= CONTENT_UNPAGED_MAX_CHUNKS:
            payloads = repository.chunks_for_filename(settings, filename)
        else:
            payloads = (
                repository.chunks_in_ordinal_range(
                    settings, filename, 1, CONTENT_UNPAGED_MAX_CHUNKS
                )
                or repository.chunks_for_filename(settings, filename)[:CONTENT_UNPAGED_MAX_CHUNKS]
            )  # a legacy document with no ordinals to range over
        return DocumentContentResponse(
            filename=filename,
            chunks=[content_chunk(payload) for payload in payloads],
            total_chunks=total,
        )
    target_found: bool | None = None
    if around is not None:
        ordinal = repository.chunk_ordinal_of(settings, filename, around)
        target_found = ordinal is not None
        # A chunk a re-index replaced: show the start of the document instead.
        center = ordinal if ordinal is not None else 1
        first, last = max(1, center - radius), center + radius
    else:
        first = start if start is not None else 1
        last = end if end is not None else first + 2 * CONTENT_MAX_RADIUS
        if last < first:
            raise HTTPException(status_code=422, detail="`end` must not be before `start`.")
        last = min(last, first + 2 * CONTENT_MAX_RADIUS)
    payloads = repository.chunks_in_ordinal_range(settings, filename, first, last)
    return DocumentContentResponse(
        filename=filename,
        chunks=[content_chunk(payload) for payload in payloads],
        total_chunks=total,
        target_found=target_found,
    )


# Guarded at least as strictly as GET /documents/{filename}/content above: raw
# original bytes are strictly more sensitive than derived chunk text. Returns 404
# both when raw_document_dir is unset (feature off) and when it's set but this
# particular filename predates the feature or was never stored — the client sees
# the same "not found" either way, not a distinct "feature unavailable" state.
@router.get("/documents/{filename}/original", dependencies=[Depends(require_api_key)])
def document_original(filename: str, raw_store: RawDocumentStoreDep) -> Response:
    filename = stored_filename(filename)
    path = raw_store.path(filename) if raw_store is not None else None
    if path is None:
        raise DocumentNotFoundError(f"No stored original for '{filename}'.")
    # Open before building the response. FileResponse re-stats and opens the path
    # only when the response is sent, so a DELETE landing in between turned this
    # 404 into a 500; an already-open handle keeps streaming the bytes it opened
    # even if the file is unlinked meanwhile.
    try:
        handle = path.open("rb")
    except FileNotFoundError:
        raise DocumentNotFoundError(f"No stored original for '{filename}'.") from None
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    # Headers from FileResponse (given the stat, it does no I/O of its own): it
    # RFC 5987-encodes the name — a hand-built latin-1 header 500'd on any
    # non-Latin-1 filename and broke on one containing a quote.
    template = FileResponse(
        path,
        media_type=media_type,
        filename=filename,
        stat_result=os.fstat(handle.fileno()),
        headers={"X-Content-Type-Options": "nosniff"},
    )
    headers = {
        name: value
        for name, value in template.headers.items()
        # Ranges need seeking the streamed body can't do; its own type is set below.
        if name not in ("accept-ranges", "content-type")
    }

    def chunks() -> Iterator[bytes]:
        with handle:
            while block := handle.read(_ORIGINAL_CHUNK_BYTES):
                yield block

    # Closing: a client that leaves mid-download lets go of the file at once.
    return ClosingStreamingResponse(chunks(), media_type=media_type, headers=headers)
