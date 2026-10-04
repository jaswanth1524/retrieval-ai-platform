"""Document parsing and chunking utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
from typing import ClassVar

from api.chunking import (
    TokenCounter,
    _group_sections_for_windowing,
    _sentence_units,
    _window_units,
    merge_tiny_semantic_sections,
)
from api.parsers.common import (
    DEFAULT_SECTION,
    DocumentError,
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    UnsupportedDocumentError,
    infer_section,
    normalize_text,
)
from api.parsers.csv import parse_csv_document
from api.parsers.docx import parse_docx_document
from api.parsers.html import parse_html_document
from api.parsers.pdf import parse_pdf_document
from api.parsers.text import (
    decode_text_document,
    looks_like_markdown,
    parse_markdown_document,
    parse_plain_text_document,
)
from api.settings import AppSettings

# Re-exported: the parser types and helpers callers have always imported from here.
# Deliberately not the parsers' internals (PdfReader, _extract_pdf_tables, ...): a test
# patching those must patch the parser module that uses them.
__all__ = [
    "CHUNKER_VERSION",
    "DEFAULT_SECTION",
    "MAX_FILENAME_BYTES",
    "MAX_TAGS_PER_DOCUMENT",
    "PARSERS",
    "SUPPORTED_EXTENSIONS",
    "ChunkConfigError",
    "DocumentChunk",
    "DocumentError",
    "DocumentNotFoundError",
    "DocumentParseError",
    "DocumentSection",
    "EmptyDocumentError",
    "UnsupportedDocumentError",
    "build_chunk_id",
    "chunk_sections",
    "contextual_text",
    "infer_section",
    "looks_like_markdown",
    "merge_tiny_semantic_sections",
    "normalize_filename",
    "normalize_tags",
    "normalize_text",
    "parse_document_bytes",
    "parse_markdown_document",
    "validate_upload_filename",
]


Parser = Callable[[str, bytes, AppSettings | None], list[DocumentSection]]


def _parse_txt(
    filename: str, content: bytes, settings: AppSettings | None
) -> list[DocumentSection]:
    text = decode_text_document(filename, content)
    if looks_like_markdown(text):
        return parse_markdown_document(filename, text)
    return parse_plain_text_document(filename, text)


def _parse_markdown(
    filename: str, content: bytes, settings: AppSettings | None
) -> list[DocumentSection]:
    return parse_markdown_document(filename, decode_text_document(filename, content))


def _parse_html(
    filename: str, content: bytes, settings: AppSettings | None
) -> list[DocumentSection]:
    return parse_html_document(filename, decode_text_document(filename, content))


def _parse_csv(
    filename: str, content: bytes, settings: AppSettings | None
) -> list[DocumentSection]:
    return parse_csv_document(filename, decode_text_document(filename, content), settings)


def _parse_docx(
    filename: str, content: bytes, settings: AppSettings | None
) -> list[DocumentSection]:
    return parse_docx_document(filename, content)


# One parser per supported suffix (api/parsers/*); adding a format is one entry here.
PARSERS: dict[str, Parser] = {
    ".pdf": parse_pdf_document,
    ".docx": _parse_docx,
    ".md": _parse_markdown,
    ".markdown": _parse_markdown,
    ".html": _parse_html,
    ".htm": _parse_html,
    ".csv": _parse_csv,
    ".txt": _parse_txt,
}
SUPPORTED_EXTENSIONS = frozenset(PARSERS)


# Chunker output format version — bumped when the chunking algorithm changes in a way
# that alters chunk contents (v2: token-aware, sentence-boundary windowing). Logged per
# ingest for trace/debug clarity. Not stored on the collection: the vector space is
# unchanged (same embedding model), so old and new chunks stay mutually queryable — a
# hard refusal would punish existing corpora for a quality improvement. Re-uploading a
# document replaces its chunks (deterministic point IDs + stale-chunk cleanup).
# v3: token counts exclude [CLS]/[SEP] per unit (reserved once per chunk instead), so
# windows fill their budget rather than stopping ~2 tokens short per sentence.
CHUNKER_VERSION = 3


class ChunkConfigError(DocumentError):
    """Raised when chunk-sizing configuration is invalid."""


class DocumentNotFoundError(RuntimeError):
    """Raised when a filename has no indexed chunks in the collection.

    Deliberately not a ``DocumentError`` subclass — that hierarchy maps to 400
    (user-correctable upload errors); "no such indexed document" is a 404.
    """


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk ready for embedding and storage with citation metadata.

    ``ordinal`` is the chunk's 1-based position within its source document (stable
    across re-ingests of the same content) — stored in the payload so retrieval can
    fetch a chunk's immediate neighbors for small-to-big context expansion without
    needing a separate ordering scheme.
    """

    filename: str
    page: int
    section: str
    chunk_id: str
    text: str
    ordinal: int = 1
    # Document-level metadata, the same value stamped onto every chunk of one ingest
    # (redundant per-point storage, same tradeoff as `filename` itself) — set by
    # IngestService.ingest, which is the first place both the upload time and the raw
    # byte count are known. None on chunks built directly in tests/other call sites
    # that skip that stamping, and permanently None on points ingested before this
    # field existed (their stored payload simply lacks the key).
    byte_size: int | None = None
    uploaded_at: float | None = None
    # CHUNKER_VERSION that produced this chunk, so a document chunked by an older
    # algorithm can be found and re-indexed from its stored original. None (key absent)
    # on points indexed before it was stamped — those count as stale.
    chunker_version: int | None = None
    # Document-level labels (PATCH /documents/{filename}/tags), stamped on every chunk
    # so a question can be scoped by tag with the same payload filter as by filename.
    tags: tuple[str, ...] = ()

    PAYLOAD_TEXT_KEY: ClassVar[str] = "text"
    PAYLOAD_ORDINAL_KEY: ClassVar[str] = "chunk_ordinal"
    PAYLOAD_BYTE_SIZE_KEY: ClassVar[str] = "byte_size"
    PAYLOAD_UPLOADED_AT_KEY: ClassVar[str] = "uploaded_at"
    PAYLOAD_CHUNKER_VERSION_KEY: ClassVar[str] = "chunker_version"
    PAYLOAD_TAGS_KEY: ClassVar[str] = "tags"

    def to_payload(self) -> dict[str, str | int | float | list[str]]:
        """Return the Qdrant payload shape needed for grounded citations."""

        payload: dict[str, str | int | float | list[str]] = {
            "filename": self.filename,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
            self.PAYLOAD_TEXT_KEY: self.text,
            self.PAYLOAD_ORDINAL_KEY: self.ordinal,
        }
        if self.byte_size is not None:
            payload[self.PAYLOAD_BYTE_SIZE_KEY] = self.byte_size
        if self.uploaded_at is not None:
            payload[self.PAYLOAD_UPLOADED_AT_KEY] = self.uploaded_at
        if self.chunker_version is not None:
            payload[self.PAYLOAD_CHUNKER_VERSION_KEY] = self.chunker_version
        if self.tags:
            payload[self.PAYLOAD_TAGS_KEY] = list(self.tags)
        return payload


def parse_document_bytes(
    filename: str, content: bytes, settings: AppSettings | None = None
) -> list[DocumentSection]:
    """Parse uploaded document bytes into text sections with source metadata.

    ``settings`` is only consulted for CSV row grouping and the PDF page limits; None
    uses the defaults, keeping the two-argument call working.
    """

    safe_filename = validate_upload_filename(filename)
    suffix = PurePosixPath(safe_filename).suffix.lower()
    return PARSERS[suffix](safe_filename, content, settings)


def chunk_sections(
    sections: list[DocumentSection],
    settings: AppSettings,
    token_counter: TokenCounter,
) -> list[DocumentChunk]:
    """Split extracted sections into overlapping, token-bounded chunks.

    ``chunk_size_tokens``/``chunk_overlap_tokens`` count the dense embedding model's
    real subword tokens (via ``token_counter``). Windows are built from whole
    sentences and never exceed ``chunk_size_tokens`` minus the contextual-embedding
    prefix's token cost, so the string actually embedded
    (``filename › section\\n{chunk}``) stays within the model's limit instead of
    silently dropping its tail. An oversized single sentence is split into
    token-bounded pieces rather than emitted overweight.

    Consecutive ``boundary="physical"`` sections (PDF pages, plain-text paragraphs)
    are windowed as one continuous stream, so overlap carries across page/paragraph
    breaks. ``boundary="semantic"`` sections (Markdown headings) are always windowed
    independently — crossing a heading would mix two topics into one chunk and make
    its section citation label lie about which topic it covers.
    """

    if settings.chunk_overlap_tokens >= settings.chunk_size_tokens:
        raise ChunkConfigError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS.")

    chunk_size = int(settings.chunk_size_tokens)
    overlap = int(settings.chunk_overlap_tokens)
    merged_sections = merge_tiny_semantic_sections(sections, int(settings.min_section_words))

    chunks: list[DocumentChunk] = []
    for group in _group_sections_for_windowing(merged_sections):
        filename = group[0].filename
        # Reserve headroom for the contextual-embedding prefix (ingestion prepends
        # "filename › section\n" before embedding), so prefix + chunk fits the model.
        # Budget against the group's LONGEST section label, not the first one: a
        # physical group spans many sections and each chunk is prefixed with its own,
        # so sizing on group[0] under-reserves for every later, longer heading. Today's
        # 448/512 slack absorbs the difference, but it would become live truncation the
        # moment chunk_size_tokens is raised toward the model's limit.
        prefix_tokens = max(
            token_counter.count(f"{filename} › {section.section}\n") for section in group
        )
        # Counts exclude special tokens; the ones the model wraps around each embedded
        # string ([CLS]/[SEP]) are reserved once per chunk here.
        special = int(getattr(token_counter, "special_tokens_per_sequence", 0))
        budget = max(1, chunk_size - prefix_tokens - special)

        units = _sentence_units(group, budget, token_counter)
        for window in _window_units(units, budget, overlap):
            chunk_text = " ".join(unit.text for unit in window).strip()
            if not chunk_text:
                continue
            page = window[0].page
            section = window[0].section
            ordinal = len(chunks) + 1
            chunks.append(
                DocumentChunk(
                    filename=filename,
                    page=page,
                    section=section,
                    chunk_id=build_chunk_id(filename, page, section, ordinal, chunk_text),
                    text=chunk_text,
                    ordinal=ordinal,
                )
            )

    if not chunks:
        raise EmptyDocumentError("Document has no extractable text to chunk.")
    return chunks


MAX_TAGS_PER_DOCUMENT = 20


MAX_TAG_LENGTH = 40


def normalize_tags(tags: Sequence[str]) -> tuple[str, ...]:
    """Trimmed, whitespace-collapsed, de-duplicated (case-insensitively), in given order.

    Raises ``UnsupportedDocumentError`` (a 400) for an empty or over-long tag, or too
    many of them.
    """

    seen: set[str] = set()
    result: list[str] = []
    for raw in tags:
        tag = " ".join(raw.split())
        if not tag or len(tag) > MAX_TAG_LENGTH or any(ord(char) < 32 for char in tag):
            raise UnsupportedDocumentError(f"Tags must be 1-{MAX_TAG_LENGTH} printable characters.")
        if tag.casefold() in seen:
            continue
        seen.add(tag.casefold())
        result.append(tag)
    if len(result) > MAX_TAGS_PER_DOCUMENT:
        raise UnsupportedDocumentError(f"A document can have at most {MAX_TAGS_PER_DOCUMENT} tags.")
    return tuple(result)


def contextual_text(filename: str, section: str, text: str) -> str:
    """``text`` prefixed with where it came from, as the embedders and reranker see it.

    One definition for both stages: the dense and sparse models embed chunks with this
    prefix, and the cross-encoder scores the same string, so a question naming a
    document or heading keeps that signal at the stage that applies rerank_min_score.
    """

    return f"{filename} › {section}\n{text}"


MAX_FILENAME_BYTES = 255


def normalize_filename(filename: str) -> str:
    """Return a basename-only upload filename.

    "." and ".." survive basename extraction (``PurePosixPath("..").name == ".."``) and
    name a directory, and control characters (NUL above all) make the OS call itself
    fail — both used to surface as a 500 from the raw-document store.
    """

    normalized = filename.replace("\\", "/")
    safe_filename = PurePosixPath(normalized).name.strip()
    if not safe_filename:
        raise UnsupportedDocumentError("Document filename is required.")
    if safe_filename in {".", ".."} or any(ord(char) < 32 for char in safe_filename):
        raise UnsupportedDocumentError("Document filename is not valid.")
    return safe_filename


def validate_upload_filename(filename: str) -> str:
    """Normalize ``filename`` and check its type is ingestible, before any work starts.

    Same checks ``parse_document_bytes`` applies, run at upload time so an unusable
    file is rejected with a 4xx instead of being accepted (202) and failing later in
    the background job.
    """

    safe_filename = normalize_filename(filename)
    # Most filesystems cap a name at 255 bytes. Past that, storing the original fails
    # with ENAMETOOLONG, and on Python 3.12 so does every later existence check on it,
    # which turned one long name into a 500 for the whole document list. Checked here,
    # at upload, not in normalize_filename, so a document already indexed under a
    # longer name can still be found and deleted.
    if len(safe_filename.encode("utf-8")) > MAX_FILENAME_BYTES:
        raise UnsupportedDocumentError(
            f"Document filename is longer than {MAX_FILENAME_BYTES} bytes; rename it."
        )
    suffix = PurePosixPath(safe_filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported document type '{suffix or '<none>'}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    return safe_filename


def build_chunk_id(filename: str, page: int, section: str, ordinal: int, text: str) -> str:
    """Build a stable chunk id from source metadata and chunk text."""

    raw = "\n".join([filename, str(page), section, str(ordinal), text])
    return sha256(raw.encode("utf-8")).hexdigest()[:20]
