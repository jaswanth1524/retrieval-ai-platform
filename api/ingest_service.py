"""Ingestion orchestration: parse -> chunk -> embed -> index, behind one call.

Split out of ``api.pipeline``, which re-exports these names.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from api.chunking import HeuristicTokenCounter, TokenCounter
from api.corpus import bump_corpus_generation
from api.documents import CHUNKER_VERSION, chunk_sections, chunking_fingerprint
from api.ingestion import EmbeddingProvider as IngestEmbeddingProvider
from api.ingestion import IngestResult, ingest_chunks
from api.metrics import ingest_stage_seconds
from api.parse_isolation import parse_with_limit
from api.repository import VectorRepository
from api.settings import AppSettings

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
    ) -> DocumentIngestOutcome:
        """Parse, chunk, embed, and index an uploaded document. Blocking — run off-loop.

        ``on_indexed`` runs under the filename write lock once indexing succeeded (see
        ``ingest_chunks``). ``uploaded_at`` overrides the upload stamp — a re-index of a
        stored original keeps the document's original upload time rather than "now".
        ``precondition`` is checked under that lock before anything is written, and
        ``before_first_write`` right before the first batch lands (see ``ingest_chunks``).
        ``tags`` sets the document's tags (a restore) instead of carrying its current ones.
        """

        # Before parsing: an index built for another embedding setup refuses the
        # upload here unless it is empty, in which case it is rebuilt for this one.
        self._repository.ensure_writable(self._settings)
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
        chunks = [
            replace(
                chunk,
                byte_size=upload_size,
                uploaded_at=upload_stamp,
                chunker_version=CHUNKER_VERSION,
                chunking_fingerprint=fingerprint,
                token_counter=counter_mode if counter_mode in ("hf", "heuristic") else None,
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
                bump_corpus_generation({chunk.filename for chunk in chunks})
        return DocumentIngestOutcome(
            filename=sections[0].filename,
            sections_parsed=len(sections),
            chunks_ingested=result.points_count,
            collection_name=result.collection_name,
        )
