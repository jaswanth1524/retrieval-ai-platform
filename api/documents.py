"""Document parsing and chunking utilities."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from typing import ClassVar, Literal

from pypdf import PdfReader

from api.settings import AppSettings

logger = logging.getLogger(__name__)

DEFAULT_SECTION = "Document"
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".markdown"})
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
# Setext H1 only ("Title\n==="). A "---" underline is intentionally not treated as a
# setext H2 heading — it's ambiguous with a CommonMark thematic break (<hr>).
_SETEXT_H1_UNDERLINE_RE = re.compile(r"^\s{0,3}=+\s*$")
# .txt content sniffing: >=2 heading lines is treated as evidence the file is
# Markdown saved with a .txt extension. One stray "# TODO"-style line in plain prose
# is too common to trust alone.
_MIN_MARKDOWN_HEADINGS_FOR_TXT_SNIFF = 2

# infer_section: candidate first-lines that look like page furniture rather than an
# actual heading, so a PDF page's literal first line (often a page number or footer)
# doesn't become a meaningless citation section label.
_DIGITS_ONLY_RE = re.compile(r"^\d+$")
_PAGE_LABEL_RE = re.compile(r"^page\s+\d+$", re.IGNORECASE)
_ROMAN_NUMERAL_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)

# Rule-of-thumb ~4 characters per BPE-style subword token; used only for a best-effort
# ingest-time warning, not exact counting (loading the real tokenizer would require a
# model download, conflicting with hermetic tests). This estimate under-counts tokens
# for code/URL-heavy text specifically (the content this guard targets), so it errs
# toward under-warning, not over-warning.
_CHARS_PER_TOKEN_ESTIMATE = 4
_TRUNCATION_WARNING_CHAR_THRESHOLD = 512 * _CHARS_PER_TOKEN_ESTIMATE


class DocumentError(ValueError):
    """Base error for document parsing and chunking failures."""


class UnsupportedDocumentError(DocumentError):
    """Raised when an uploaded document type is not supported."""


class EmptyDocumentError(DocumentError):
    """Raised when a document has no extractable text."""


class DocumentParseError(DocumentError):
    """Raised when a supported document cannot be parsed."""


class ChunkConfigError(DocumentError):
    """Raised when chunk-sizing configuration is invalid."""


@dataclass(frozen=True)
class DocumentSection:
    """Extracted text with source metadata before chunking.

    ``boundary`` marks whether the edge before the *next* section is a physical
    artifact of the source format (a PDF page break, a plain-text paragraph break —
    safe for chunk windows to cross with overlap) or a semantic one (a Markdown
    heading — a real topic break that chunk windows must not cross).
    """

    filename: str
    page: int
    section: str
    text: str
    boundary: Literal["physical", "semantic"] = "physical"


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
    if suffix == ".txt" and looks_like_markdown(text):
        return parse_markdown_document(safe_filename, text)
    return parse_plain_text_document(safe_filename, text)


def chunk_sections(
    sections: list[DocumentSection],
    settings: AppSettings,
) -> list[DocumentChunk]:
    """Split extracted sections into overlapping chunks.

    ``chunk_size_tokens``/``chunk_overlap_tokens`` count whitespace-delimited words
    (this function splits on ``str.split()``), not the embedding model's subword
    tokens — see the settings field's docstring for why the default is 300, not 500.

    Consecutive ``boundary="physical"`` sections (PDF pages, plain-text paragraphs)
    are windowed as one continuous word stream, so overlap carries across page/
    paragraph breaks instead of resetting to zero at every one. ``boundary="semantic"``
    sections (Markdown headings) are always windowed independently — crossing a
    heading would mix two different topics into one chunk and make its section
    citation label lie about which topic it actually covers.
    """

    if settings.chunk_overlap_tokens >= settings.chunk_size_tokens:
        raise ChunkConfigError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS.")

    chunk_size = int(settings.chunk_size_tokens)
    overlap = int(settings.chunk_overlap_tokens)
    step = chunk_size - overlap

    merged_sections = merge_tiny_semantic_sections(sections, int(settings.min_section_words))

    chunks: list[DocumentChunk] = []
    for group in _group_sections_for_windowing(merged_sections):
        tokens: list[tuple[str, int, str]] = [
            (word, section.page, section.section)
            for section in group
            for word in section.text.split()
        ]
        if not tokens:
            continue

        filename = group[0].filename
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            window = tokens[start:end]
            chunk_text = " ".join(word for word, _, _ in window).strip()
            if chunk_text:
                window_page = window[0][1]
                window_section = window[0][2]
                ordinal = len(chunks) + 1
                chunk = DocumentChunk(
                    filename=filename,
                    page=window_page,
                    section=window_section,
                    chunk_id=build_chunk_id(
                        filename, window_page, window_section, ordinal, chunk_text
                    ),
                    text=chunk_text,
                )
                chunks.append(chunk)
                if len(chunk_text) > _TRUNCATION_WARNING_CHAR_THRESHOLD:
                    logger.warning(
                        "Chunk %s (%s p.%s, %r) is ~%d chars and may exceed the dense "
                        "embedding model's 512-subword-token limit — the embedded "
                        "vector may silently drop the chunk's tail. Consider lowering "
                        "CHUNK_SIZE_TOKENS or pre-splitting dense code/table/URL content.",
                        chunk.chunk_id,
                        chunk.filename,
                        chunk.page,
                        chunk.section,
                        len(chunk_text),
                    )
            if end == len(tokens):
                break
            start += step

    if not chunks:
        raise EmptyDocumentError("Document has no extractable text to chunk.")
    return chunks


def _group_sections_for_windowing(
    sections: list[DocumentSection],
) -> list[list[DocumentSection]]:
    """Group sections into runs that should be windowed as one continuous stream.

    Consecutive same-filename ``physical`` sections merge into one group (chunk
    overlap crosses the boundary between them); each ``semantic`` section is always
    its own singleton group (windowed alone, exactly as before this change).
    """

    groups: list[list[DocumentSection]] = []
    for section in sections:
        if (
            section.boundary == "physical"
            and groups
            and groups[-1][-1].boundary == "physical"
            and groups[-1][-1].filename == section.filename
        ):
            groups[-1].append(section)
        else:
            groups.append([section])
    return groups


def merge_tiny_semantic_sections(
    sections: list[DocumentSection],
    min_words: int,
) -> list[DocumentSection]:
    """Fold undersized Markdown heading sections into a neighbor before windowing.

    Only operates on ``boundary == "semantic"`` runs (Markdown headings) — physical
    sections don't need this, cross-boundary windowing already absorbs short
    paragraphs/pages into their neighbors.
    """

    result: list[DocumentSection] = []
    index = 0
    while index < len(sections):
        if sections[index].boundary != "semantic":
            result.append(sections[index])
            index += 1
            continue
        run_start = index
        while index < len(sections) and sections[index].boundary == "semantic":
            index += 1
        result.extend(_merge_semantic_run(sections[run_start:index], min_words))
    return result


def _merge_semantic_run(
    run: list[DocumentSection],
    min_words: int,
) -> list[DocumentSection]:
    """Fold sections under ``min_words`` forward into the next section in the run.

    A trailing run of tiny sections with no bigger section after them folds
    backward into the last merged section instead (or, if the whole run is tiny,
    collapses into one section under the last section's label).
    """

    merged: list[DocumentSection] = []
    pending: list[str] = []
    for section in run:
        if len(section.text.split()) < min_words:
            pending.append(section.text)
            continue
        text = "\n\n".join([*pending, section.text]) if pending else section.text
        pending = []
        merged.append(replace(section, text=text))
    if pending:
        if merged:
            merged[-1] = replace(
                merged[-1], text=merged[-1].text + "\n\n" + "\n\n".join(pending)
            )
        else:
            merged.append(replace(run[-1], text="\n\n".join(pending)))
    return merged


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


_TEXT_DECODE_FALLBACKS: tuple[str, ...] = ("utf-8-sig", "cp1252", "latin-1")


def decode_text_document(filename: str, content: bytes) -> str:
    """Decode a text-like document, falling back across common encodings.

    ``utf-8-sig`` only tolerates a BOM, not a different encoding — a `.txt`/`.md`
    file saved as cp1252 or latin-1 (common from Windows tools) would otherwise be
    rejected outright even though it decodes cleanly under one of these fallbacks.
    latin-1 never raises (every byte maps to a codepoint), so it's the final,
    always-succeeding fallback rather than a real "unsupported encoding" signal.
    """

    text: str | None = None
    for encoding in _TEXT_DECODE_FALLBACKS:
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            continue
        else:
            break

    if text is None:
        raise DocumentParseError(f"Could not decode text document '{filename}'.")

    text = normalize_text(text)
    if not text:
        raise EmptyDocumentError(f"Document '{filename}' has no extractable text.")
    return text


def parse_markdown_document(filename: str, text: str) -> list[DocumentSection]:
    """Split Markdown into sections using heading text as citation metadata.

    Recognizes ATX headings (``# Heading``) and setext H1 headings (``Title`` on its
    own line followed by a line of ``=``). Setext H2 (``---`` underline) is not
    treated as a heading — it's ambiguous with a CommonMark thematic break (``<hr>``).
    """

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
                    boundary="semantic",
                )
            )

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        atx_match = _MARKDOWN_HEADING_RE.match(line)
        if atx_match:
            flush()
            current_section = atx_match.group(1).strip() or DEFAULT_SECTION
            buffer = [line]
            index += 1
            continue

        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if line.strip() and _SETEXT_H1_UNDERLINE_RE.match(next_line):
            flush()
            current_section = line.strip() or DEFAULT_SECTION
            buffer = [line, next_line]
            index += 2
            continue

        buffer.append(line)
        index += 1

    flush()
    if not sections:
        raise EmptyDocumentError(f"Markdown document '{filename}' has no extractable text.")
    return sections


def looks_like_markdown(text: str) -> bool:
    """Heuristic used only for ``.txt`` uploads: content with two or more heading
    lines (ATX ``#`` or setext ``===``) is treated as Markdown saved with the wrong
    extension, so it gets heading-based section splitting instead of one giant chunk.
    """

    lines = text.splitlines()
    heading_count = 0
    for index, line in enumerate(lines):
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        is_heading = bool(_MARKDOWN_HEADING_RE.match(line)) or (
            bool(line.strip()) and bool(_SETEXT_H1_UNDERLINE_RE.match(next_line))
        )
        if is_heading:
            heading_count += 1
        if heading_count >= _MIN_MARKDOWN_HEADINGS_FOR_TXT_SNIFF:
            return True
    return False


def parse_plain_text_document(filename: str, text: str) -> list[DocumentSection]:
    """Split plain text into blank-line-delimited paragraph sections.

    Each paragraph becomes its own ``DocumentSection`` labeled via ``infer_section``,
    so a multi-paragraph .txt upload no longer collapses into a single section (and
    therefore, for short files, a single chunk) covering the whole document.
    """

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    return [
        DocumentSection(filename=filename, page=1, section=infer_section(paragraph), text=paragraph)
        for paragraph in paragraphs
    ]


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
    """Infer a stable section label when richer structure is unavailable.

    Skips candidate first-lines that look like page furniture — a bare page number,
    a "Page N" label, or a lone roman numeral — since PDF page text often has exactly
    one of these as its literal first line, which would otherwise become a
    meaningless citation section label. Falls back to the first non-blank line if
    every candidate line looks like furniture.
    """

    first_candidate: str | None = None
    for line in text.splitlines():
        candidate = line.strip().strip("#").strip()
        if not candidate:
            continue
        if first_candidate is None:
            first_candidate = candidate
        if _looks_like_page_furniture(candidate):
            continue
        return candidate[:120]
    if first_candidate is not None:
        return first_candidate[:120]
    return DEFAULT_SECTION


def _looks_like_page_furniture(candidate: str) -> bool:
    """True for page numbers/footers unlikely to be a meaningful heading."""

    if len(candidate) < 3:
        return True
    if _DIGITS_ONLY_RE.match(candidate):
        return True
    if _PAGE_LABEL_RE.match(candidate):
        return True
    if _ROMAN_NUMERAL_RE.match(candidate):
        return True
    return False


def build_chunk_id(filename: str, page: int, section: str, ordinal: int, text: str) -> str:
    """Build a stable chunk id from source metadata and chunk text."""

    raw = "\n".join([filename, str(page), section, str(ordinal), text])
    return sha256(raw.encode("utf-8")).hexdigest()[:20]
