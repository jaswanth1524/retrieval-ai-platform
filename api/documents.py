"""Document parsing and chunking utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from typing import ClassVar

from pypdf import PdfReader

from api.settings import AppSettings

DEFAULT_SECTION = "Document"
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".markdown"})
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")


class DocumentError(ValueError):
    """Base error for document parsing and chunking failures."""


class UnsupportedDocumentError(DocumentError):
    """Raised when an uploaded document type is not supported."""


class EmptyDocumentError(DocumentError):
    """Raised when a document has no extractable text."""


class DocumentParseError(DocumentError):
    """Raised when a supported document cannot be parsed."""


@dataclass(frozen=True)
class DocumentSection:
    """Extracted text with source metadata before chunking."""

    filename: str
    page: int
    section: str
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk ready for embedding and storage with citation metadata."""

    filename: str
    page: int
    section: str
    chunk_id: str
    text: str

    PAYLOAD_TEXT_KEY: ClassVar[str] = "text"

    def to_payload(self) -> dict[str, str | int]:
        """Return the Qdrant payload shape needed for grounded citations."""

        return {
            "filename": self.filename,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
            self.PAYLOAD_TEXT_KEY: self.text,
        }


def parse_document_bytes(filename: str, content: bytes) -> list[DocumentSection]:
    """Parse uploaded document bytes into text sections with source metadata."""

    safe_filename = normalize_filename(filename)
    suffix = PurePosixPath(safe_filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported document type '{suffix or '<none>'}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if suffix == ".pdf":
        return parse_pdf_document(safe_filename, content)

    text = decode_text_document(safe_filename, content)
    if suffix in {".md", ".markdown"}:
        return parse_markdown_document(safe_filename, text)
    return [DocumentSection(safe_filename, page=1, section=infer_section(text), text=text)]


def chunk_sections(
    sections: list[DocumentSection],
    settings: AppSettings,
) -> list[DocumentChunk]:
    """Split extracted sections into overlapping chunks."""

    if settings.chunk_overlap_tokens >= settings.chunk_size_tokens:
        raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS.")

    chunks: list[DocumentChunk] = []
    chunk_size = int(settings.chunk_size_tokens)
    overlap = int(settings.chunk_overlap_tokens)
    step = chunk_size - overlap

    for section in sections:
        tokens = section.text.split()
        if not tokens:
            continue

        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = " ".join(tokens[start:end]).strip()
            if chunk_text:
                ordinal = len(chunks) + 1
                chunks.append(
                    DocumentChunk(
                        filename=section.filename,
                        page=section.page,
                        section=section.section,
                        chunk_id=build_chunk_id(section, ordinal, chunk_text),
                        text=chunk_text,
                    )
                )
            if end == len(tokens):
                break
            start += step

    if not chunks:
        raise EmptyDocumentError("Document has no extractable text to chunk.")
    return chunks


def parse_pdf_document(filename: str, content: bytes) -> list[DocumentSection]:
    """Extract text from a PDF, preserving one-based page numbers."""

    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:  # pypdf raises several parser-specific exceptions.
        raise DocumentParseError(f"Could not parse PDF '{filename}'.") from exc

    sections: list[DocumentSection] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue
        sections.append(
            DocumentSection(
                filename=filename,
                page=page_number,
                section=infer_section(text),
                text=text,
            )
        )

    if not sections:
        raise EmptyDocumentError(f"PDF '{filename}' has no extractable text.")
    return sections


def decode_text_document(filename: str, content: bytes) -> str:
    """Decode a UTF-8 text-like document and normalize whitespace."""

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParseError(f"Could not decode text document '{filename}' as UTF-8.") from exc

    text = normalize_text(text)
    if not text:
        raise EmptyDocumentError(f"Document '{filename}' has no extractable text.")
    return text


def parse_markdown_document(filename: str, text: str) -> list[DocumentSection]:
    """Split Markdown into sections using heading text as citation metadata."""

    sections: list[DocumentSection] = []
    current_section = DEFAULT_SECTION
    buffer: list[str] = []

    def flush() -> None:
        section_text = normalize_text("\n".join(buffer))
        if section_text:
            sections.append(
                DocumentSection(
                    filename=filename,
                    page=1,
                    section=current_section,
                    text=section_text,
                )
            )

    for line in text.splitlines():
        match = _MARKDOWN_HEADING_RE.match(line)
        if match:
            flush()
            current_section = match.group(1).strip() or DEFAULT_SECTION
            buffer = [line]
        else:
            buffer.append(line)

    flush()
    if not sections:
        raise EmptyDocumentError(f"Markdown document '{filename}' has no extractable text.")
    return sections


def normalize_filename(filename: str) -> str:
    """Return a basename-only upload filename."""

    normalized = filename.replace("\\", "/")
    safe_filename = PurePosixPath(normalized).name.strip()
    if not safe_filename:
        raise UnsupportedDocumentError("Document filename is required.")
    return safe_filename


def normalize_text(text: str) -> str:
    """Normalize line endings and trim surrounding whitespace."""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def infer_section(text: str) -> str:
    """Infer a stable section label when richer structure is unavailable."""

    for line in text.splitlines():
        candidate = line.strip().strip("#").strip()
        if candidate:
            return candidate[:120]
    return DEFAULT_SECTION


def build_chunk_id(section: DocumentSection, ordinal: int, text: str) -> str:
    """Build a stable chunk id from source metadata and chunk text."""

    raw = "\n".join(
        [
            section.filename,
            str(section.page),
            section.section,
            str(ordinal),
            text,
        ]
    )
    return sha256(raw.encode("utf-8")).hexdigest()[:20]
