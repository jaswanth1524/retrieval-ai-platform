"""Ingestion orchestration: parse -> chunk -> embed -> index, behind one call.

Split out of ``api.pipeline``, which re-exports these names.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from api.chunking import HeuristicTokenCounter, TokenCounter
from api.corpus import bump_corpus_generation
from api.documents import (
    CHUNKER_VERSION,
    DocumentSource,
    chunk_sections,
    chunking_fingerprint,
    normalize_filename,
)
from api.ingestion import EmbeddingProvider as IngestEmbeddingProvider
from api.ingestion import IngestResult, filename_write_lock, ingest_chunks
from api.metrics import ingest_stage_seconds
from api.parse_isolation import parse_with_limit
from api.repository import VectorRepository
from api.settings import AppSettings
from api.workspaces import document_key

# The name ingestion has always logged under ("Chunked x into n chunks ..."), kept so
# log filters and dashboards built on it still match.
logger = logging.getLogger("api.pipeline")


@dataclass(frozen=True)
class DocumentIngestOutcome:
    """Result of parsing, chunking, and indexing one uploaded document."""

    filename: str
    sections_parsed: int
    chunks_ingested: int
    collection_name: str
    # Already indexed from these exact bytes: nothing was parsed, embedded or written.
    unchanged: bool = False


class IngestService:
    """Composes parsing, chunking, embedding, and indexing behind a single call."""

    def __init__(
        self,
        repository: VectorRepository,
        embedding_provider: IngestEmbeddingProvider,
        settings: AppSettings,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._settings = settings
        # Default to the heuristic (no model fetch) so direct constructions in tests
        # stay hermetic; production DI passes the dense model's real tokenizer.
        self._token_counter = token_counter or HeuristicTokenCounter()

    @property
    def settings(self) -> AppSettings:
        """The settings it indexes with (the request's workspace, see api/workspaces.py)."""

        return self._settings

    def ingest(
        self,
        filename: str,
        content: bytes,
        on_progress: Callable[[int, int], None] | None = None,
        on_indexed: Callable[[], None] | None = None,
        uploaded_at: float | None = None,
        precondition: Callable[[], None] | None = None,
        *,
        before_first_write: Callable[[], None] | None = None,
        tags: tuple[str, ...] | None = None,
        source: DocumentSource | None = None,
        dedupe: bool = False,
    ) -> DocumentIngestOutcome:
        """Parse, chunk, embed, and index an uploaded document. Blocking — run off-loop.

        ``on_indexed`` runs under the filename write lock once indexing succeeded (see
        ``ingest_chunks``). ``uploaded_at`` overrides the upload stamp — a re-index of a
        stored original keeps the document's original upload time rather than "now".
        ``precondition`` is checked under that lock before anything is written, and
        ``before_first_write`` right before the first batch lands (see ``ingest_chunks``).
        ``tags`` sets the document's tags (a restore) instead of carrying its current ones.
        ``source`` records where a URL-ingested document came from (a plain upload
        has none, and drops a previous one). With ``dedupe``, bytes identical to the
        indexed version leave it as it is (see ``_leave_unchanged``).
        """

        # Before parsing: an index built for another embedding setup refuses the
        # upload here unless it is empty, in which case it is rebuilt for this one.
        self._repository.ensure_writable(self._settings)
        digest = hashlib.sha256(content).hexdigest()
        if dedupe:
            unchanged = self._leave_unchanged(
                filename, digest, uploaded_at, precondition, on_indexed, tags, source
            )
            if unchanged is not None:
                return unchanged
        elif precondition is not None:
            # Once now, before the parse (up to PARSE_TIMEOUT_SECONDS, OCR included): a
            # job already refused (cancelled, superseded, its original deleted while it
            # waited) fails here instead of after it. The check under the filename lock
            # still decides; this only saves the work.
            precondition()
        parse_start = time.monotonic()
        sections = parse_with_limit(filename, content, self._settings)
        chunk_start = time.monotonic()
        ingest_stage_seconds.labels(stage="parse").observe(chunk_start - parse_start)
        chunks = chunk_sections(sections, self._settings, self._token_counter)
        ingest_stage_seconds.labels(stage="chunk").observe(time.monotonic() - chunk_start)
        # Stamped here (not in chunk_sections) because this is the first point in the
        # pipeline that has both the raw upload's byte count and the upload instant —
        # chunking itself works purely from parsed text.
        upload_stamp = uploaded_at if uploaded_at is not None else time.time()
        upload_size = len(content)
        fingerprint = chunking_fingerprint(self._settings)
        # Read after chunking: the tokenizer loads (or fails to) on first use.
        counter_mode = getattr(self._token_counter, "mode", None)
        chunk_total = len(chunks)
        chunks = [
            replace(
                chunk,
                byte_size=upload_size,
                uploaded_at=upload_stamp,
                chunker_version=CHUNKER_VERSION,
                chunking_fingerprint=fingerprint,
                token_counter=counter_mode if counter_mode in ("hf", "heuristic") else None,
                content_sha256=digest,
                document_chunks=chunk_total,
                source_url=source.url if source else None,
                source_etag=source.etag if source else None,
                source_last_modified=source.last_modified if source else None,
                source_refresh_minutes=source.refresh_minutes if source else None,
            )
            for chunk in chunks
        ]
        logger.info(
            "Chunked %s into %d chunks (chunker v%d).",
            sections[0].filename,
            len(chunks),
            CHUNKER_VERSION,
        )
        writing = False

        def start_writing() -> None:
            nonlocal writing
            if before_first_write is not None:
                before_first_write()  # may refuse: a cancel that arrived in time
            writing = True

        try:
            result: IngestResult = ingest_chunks(
                self._repository,
                self._settings,
                chunks,
                self._embedding_provider,
                on_progress,
                on_indexed,
                precondition,
                # A re-upload or re-index replaces the document's points; its tags are
                # a property of the document, not of one upload, so they carry over.
                carry_tags=True,
                before_first_write=start_writing,
                tags=tags,
            )
        finally:
            # Even a failed ingest may have written points; cached answers could
            # describe a corpus that no longer exists either way. One that stopped
            # before its first write (cancelled, refused, an embedding error) changed
            # nothing, and used to empty the answer cache all the same.
            if writing:
                bump_corpus_generation(
                    {chunk.filename for chunk in chunks},
                    collection=self._settings.qdrant_collection,
                )
        return DocumentIngestOutcome(
            filename=sections[0].filename,
            sections_parsed=len(sections),
            chunks_ingested=result.points_count,
            collection_name=result.collection_name,
        )

    def _leave_unchanged(
        self,
        filename: str,
        digest: str,
        uploaded_at: float | None,
        precondition: Callable[[], None] | None,
        on_indexed: Callable[[], None] | None,
        tags: tuple[str, ...] | None,
        source: DocumentSource | None,
    ) -> DocumentIngestOutcome | None:
        """The outcome of leaving an identical document alone, or None to index it.

        Skipped only when re-indexing could change nothing: the indexed points are one
        complete ingest of these exact bytes (as many as it wrote, all agreeing), under
        today's chunking (not stale), with no tags to set and the same source URL (new
        validators are stored in place). Decided under the filename lock, after the same
        sequencing check a write makes, so an older upload still queued can't be left
        "unchanged" over a newer one.
        ``on_indexed`` still runs (it records the upload as applied and stores the
        original, which may have been missing). A watched file that was only touched
        gets its new modification time, which is how the watcher knows the document is
        current.
        """

        name = normalize_filename(filename)
        settings = self._settings
        with filename_write_lock(document_key(settings, name)):
            if precondition is not None:
                precondition()
            # One point first: a changed file (the usual case) shows it at once, where
            # the full scan read every point under the lock (~0.6 s at 50k points).
            if self._repository.content_sha256_of(settings, name) != digest:
                return None
            meta = self._repository.document_metadata(settings, name)
            if meta is None or meta.content_sha256 != digest or tags is not None:
                return None
            # Every point of one complete ingest: a failed one leaves fewer than it meant
            # to write (all with this hash), and a retry of the same bytes must finish it.
            if meta.document_chunks != meta.chunk_count:
                return None
            # A tokenizer not loaded yet may well load: judged as if it had.
            mode = getattr(self._token_counter, "mode", None)
            counter_mode = mode if mode == "heuristic" else "hf"
            if meta.is_stale(chunking_fingerprint(settings), counter_mode):
                return None
            if meta.source_url != (source.url if source else None):
                return None
            changed = False
            # The same bytes under new validators (a dynamic page, a CDN that re-stamps)
            # or another refresh interval: only those change, or every refresh would
            # re-embed the document.
            if source is not None and (
                meta.source_etag,
                meta.source_last_modified,
                meta.source_refresh_minutes,
            ) != (source.etag, source.last_modified, source.refresh_minutes):
                self._repository.set_source_details(
                    settings, name, source.etag, source.last_modified, source.refresh_minutes
                )
                changed = True
            if uploaded_at is not None and meta.uploaded_at != uploaded_at:
                self._repository.set_uploaded_at(settings, name, uploaded_at)
                changed = True
            if changed:
                bump_corpus_generation([name], collection=settings.qdrant_collection)
            if on_indexed is not None:
                on_indexed()
        logger.info("%s is unchanged (identical bytes); not re-indexed.", name)
        return DocumentIngestOutcome(
            filename=name,
            sections_parsed=0,
            chunks_ingested=meta.chunk_count,
            collection_name=settings.qdrant_collection,
            unchanged=True,
        )
