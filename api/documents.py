"""Document parsing and chunking utilities."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, ClassVar, Literal, cast

from pypdf import PdfReader

from api.chunking import TokenCounter
from api.settings import AppSettings

logger = logging.getLogger(__name__)

DEFAULT_SECTION = "Document"
SUPPORTED_EXTENSIONS = frozenset(
    {".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm", ".csv"}
)
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

# Chunker output format version — bumped when the chunking algorithm changes in a way
# that alters chunk contents (v2: token-aware, sentence-boundary windowing). Logged per
# ingest for trace/debug clarity. Not stored on the collection: the vector space is
# unchanged (same embedding model), so old and new chunks stay mutually queryable — a
# hard refusal would punish existing corpora for a quality improvement. Re-uploading a
# document replaces its chunks (deterministic point IDs + stale-chunk cleanup).
CHUNKER_VERSION = 2

# Sentence boundaries: whitespace after .!? that precedes a capital/digit (optionally
# behind an opening quote/bracket), OR a blank line. Windows prefer to break here rather
# than mid-sentence. Not linguistically perfect (abbreviations, decimals) — a pragmatic
# heuristic that keeps most chunk edges on real sentence ends.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n{2,}")


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


class DocumentNotFoundError(RuntimeError):
    """Raised when a filename has no indexed chunks in the collection.

    Deliberately not a ``DocumentError`` subclass — that hierarchy maps to 400
    (user-correctable upload errors); "no such indexed document" is a 404.
    """


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

    PAYLOAD_TEXT_KEY: ClassVar[str] = "text"
    PAYLOAD_ORDINAL_KEY: ClassVar[str] = "chunk_ordinal"

    def to_payload(self) -> dict[str, str | int]:
        """Return the Qdrant payload shape needed for grounded citations."""

        return {
            "filename": self.filename,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
            self.PAYLOAD_TEXT_KEY: self.text,
            self.PAYLOAD_ORDINAL_KEY: self.ordinal,
        }


def parse_document_bytes(
    filename: str, content: bytes, settings: AppSettings | None = None
) -> list[DocumentSection]:
    """Parse uploaded document bytes into text sections with source metadata.

    ``settings`` is only consulted for CSV row grouping (``csv_rows_per_section``);
    None uses the default, keeping the two-argument call working for other formats.
    """

    safe_filename = normalize_filename(filename)
    suffix = PurePosixPath(safe_filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported document type '{suffix or '<none>'}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if suffix == ".pdf":
        return parse_pdf_document(safe_filename, content)
    if suffix == ".docx":
        return parse_docx_document(safe_filename, content)

    text = decode_text_document(safe_filename, content)
    if suffix in {".md", ".markdown"}:
        return parse_markdown_document(safe_filename, text)
    if suffix in {".html", ".htm"}:
        return parse_html_document(safe_filename, text)
    if suffix == ".csv":
        return parse_csv_document(safe_filename, text, settings)
    if suffix == ".txt" and looks_like_markdown(text):
        return parse_markdown_document(safe_filename, text)
    return parse_plain_text_document(safe_filename, text)


@dataclass(frozen=True)
class _Unit:
    """One sentence (or forced fragment of an oversized sentence) with its metadata."""

    text: str
    tokens: int
    page: int
    section: str


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
        prefix = f"{filename} › {group[0].section}\n"
        budget = max(1, chunk_size - token_counter.count(prefix))

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


def _sentence_units(
    group: list[DocumentSection], budget: int, token_counter: TokenCounter
) -> list[_Unit]:
    """Flatten a group's sections into sentence units, splitting oversized sentences."""

    units: list[_Unit] = []
    for section in group:
        for raw_sentence in _SENTENCE_SPLIT_RE.split(section.text):
            # Collapse intra-sentence whitespace (newlines, runs of spaces) to single
            # spaces so chunk text is clean and stable regardless of source formatting.
            sentence = " ".join(raw_sentence.split())
            if not sentence:
                continue
            for piece in _split_oversized(sentence, budget, token_counter):
                units.append(
                    _Unit(
                        text=piece,
                        tokens=token_counter.count(piece),
                        page=section.page,
                        section=section.section,
                    )
                )
    return units


def _split_oversized(sentence: str, budget: int, token_counter: TokenCounter) -> list[str]:
    """Greedily pack a single sentence's words into <=budget-token pieces.

    A sentence within budget is returned unchanged. A single word longer than budget
    is still emitted alone (a word cannot be split) — the only case a unit exceeds
    budget, and rare enough to accept.
    """

    if token_counter.count(sentence) <= budget:
        return [sentence]

    pieces: list[str] = []
    current: list[str] = []
    for word in sentence.split():
        if current and token_counter.count(" ".join([*current, word])) > budget:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _window_units(units: list[_Unit], budget: int, overlap: int) -> list[list[_Unit]]:
    """Slide a token-bounded window over units with token-bounded overlap.

    Each window holds as many whole units as fit in ``budget`` (always at least one,
    guaranteeing forward progress). The next window starts by stepping back over the
    trailing units whose token sum is within ``overlap`` — but never so far that the
    window fails to advance by at least one unit.
    """

    windows: list[list[_Unit]] = []
    start = 0
    total = len(units)
    while start < total:
        end = start
        window_tokens = 0
        while end < total and (end == start or window_tokens + units[end].tokens <= budget):
            window_tokens += units[end].tokens
            end += 1
        windows.append(units[start:end])
        if end >= total:
            break

        overlap_tokens = 0
        next_start = end
        while next_start > start + 1 and overlap_tokens + units[next_start - 1].tokens <= overlap:
            next_start -= 1
            overlap_tokens += units[next_start].tokens
        start = next_start

    return windows


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
    """Extract text from a PDF, preserving one-based page numbers.

    pypdf provides the text stream; pdfplumber (best-effort) appends any extracted
    tables as Markdown blocks so table Q&A is answerable. Pages with no extractable
    text fall back to OCR when the optional ``ocr`` extra is installed.
    """

    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:  # pypdf raises several parser-specific exceptions.
        raise DocumentParseError(f"Could not parse PDF '{filename}'.") from exc

    tables_by_page = _extract_pdf_tables(content)

    sections: list[DocumentSection] = []
    empty_pages: list[int] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        table_blocks = tables_by_page.get(page_number, [])
        if table_blocks:
            table_text = "\n\n".join(f"[Table]\n{block}" for block in table_blocks)
            text = f"{text}\n\n{table_text}".strip() if text else table_text
        if not text:
            empty_pages.append(page_number)
            continue
        sections.append(
            DocumentSection(
                filename=filename,
                page=page_number,
                section=infer_section(text),
                text=text,
            )
        )

    if not sections and empty_pages:
        sections = _ocr_pdf_pages(filename, content, empty_pages)

    if not sections:
        raise EmptyDocumentError(f"PDF '{filename}' has no extractable text.")
    return sections


def _extract_pdf_tables(content: bytes) -> dict[int, list[str]]:
    """Return page-number -> rendered Markdown tables. Best-effort: failure -> {}.

    Table extraction is enrichment, never a parse gate — any pdfplumber error (or a
    page with no tables) simply yields no table text for that document/page.
    """

    try:
        import pdfplumber

        result: dict[int, list[str]] = {}
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                blocks = [_table_to_markdown(table) for table in page.extract_tables() if table]
                blocks = [block for block in blocks if block]
                if blocks:
                    result[page_number] = blocks
        return result
    except Exception:
        logger.warning("PDF table extraction failed; continuing with text only.", exc_info=True)
        return {}


def _table_to_markdown(rows: Sequence[Sequence[object]]) -> str:
    """Render an extracted table (list of rows of cells) as a Markdown table."""

    cleaned = [
        [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
        for row in rows
        if any(cell is not None and str(cell).strip() for cell in row)
    ]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    lines = ["| " + " | ".join(row + [""] * (width - len(row))) + " |" for row in cleaned]
    header_sep = "| " + " | ".join(["---"] * width) + " |"
    return "\n".join([lines[0], header_sep, *lines[1:]])


def _ocr_pdf_pages(filename: str, content: bytes, page_numbers: list[int]) -> list[DocumentSection]:
    """OCR the given PDF pages when the ``ocr`` extra is installed.

    Raises ``EmptyDocumentError`` with an install hint when the extra is absent, so a
    scanned PDF gives an actionable message rather than a generic "no text" error.
    """

    try:
        import pdfplumber
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise EmptyDocumentError(
            f"PDF '{filename}' has no extractable text — it may be scanned images. "
            "Install OCR support with: uv sync --extra ocr "
            "(or pip install 'docrag[ocr]')."
        ) from exc

    engine = _get_ocr_engine(RapidOCR)
    sections: list[DocumentSection] = []
    with pdfplumber.open(BytesIO(content)) as pdf:
        for page_number in page_numbers:
            if page_number > len(pdf.pages):
                continue
            image = pdf.pages[page_number - 1].to_image(resolution=200)
            import numpy as np

            ocr_result, _ = engine(np.array(image.original))
            text = normalize_text("\n".join(line[1] for line in (ocr_result or [])))
            if text:
                sections.append(
                    DocumentSection(
                        filename=filename,
                        page=page_number,
                        section=infer_section(text),
                        text=text,
                    )
                )
    return sections


_OCR_ENGINE: object | None = None


def _get_ocr_engine(engine_cls: type) -> Any:
    """Lazily construct and cache the OCR engine (model load is expensive)."""

    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = engine_cls()
    return _OCR_ENGINE


_HTML_STRIP_TAGS = ("script", "style", "noscript", "template")
_HTML_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_DEFAULT_CSV_ROWS_PER_SECTION = 50


def parse_docx_document(filename: str, content: bytes) -> list[DocumentSection]:
    """Extract DOCX text, using heading styles as semantic section boundaries.

    DOCX has no fixed pagination, so ``page`` is always 1 and the section label
    carries the location meaning. Tables are rendered as one text block of
    ``" | "``-joined cell rows in document order.
    """

    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = Document(BytesIO(content))
    except Exception as exc:
        raise DocumentParseError(f"Could not parse DOCX '{filename}'.") from exc

    sections: list[DocumentSection] = []
    current_section = DEFAULT_SECTION
    buffer: list[str] = []

    def flush() -> None:
        text = normalize_text("\n".join(buffer))
        if text:
            sections.append(
                DocumentSection(
                    filename=filename,
                    page=1,
                    section=current_section,
                    text=text,
                    boundary="semantic",
                )
            )

    for block in _iter_docx_blocks(document, Paragraph, Table):
        if isinstance(block, Paragraph):
            style_name = block.style.name if block.style is not None else ""
            text = block.text.strip()
            if not text:
                continue
            if style_name.startswith("Heading"):
                flush()
                current_section = text[:120] or DEFAULT_SECTION
                buffer = [text]
            else:
                buffer.append(text)
        elif isinstance(block, Table):
            rows = [
                " | ".join(cell.text.strip() for cell in row.cells)
                for row in block.rows
            ]
            table_text = "\n".join(row for row in rows if row.strip())
            if table_text:
                buffer.append(table_text)

    flush()
    if not sections:
        raise EmptyDocumentError(f"DOCX '{filename}' has no extractable text.")
    return sections


def _iter_docx_blocks(document: object, paragraph_cls: type, table_cls: type) -> Iterator[object]:
    """Yield a DOCX body's paragraphs and tables in document order.

    Prefers python-docx >= 1.1's ``iter_inner_content`` and falls back to manual XML
    body iteration on older versions.
    """

    iter_inner = getattr(document, "iter_inner_content", None)
    if callable(iter_inner):
        yield from iter_inner()
        return
    body = document.element.body  # type: ignore[attr-defined]
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield paragraph_cls(child, document)
        elif child.tag.endswith("}tbl"):
            yield table_cls(child, document)


def parse_html_document(filename: str, text: str) -> list[DocumentSection]:
    """Extract readable HTML text, splitting on h1-h6 into semantic sections."""

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "html.parser")
    for element in soup(list(_HTML_STRIP_TAGS)):
        element.decompose()

    body = soup.body or soup
    title_tag = soup.title.get_text(strip=True) if soup.title else ""

    sections: list[DocumentSection] = []
    current_section = title_tag[:120] or DEFAULT_SECTION
    buffer: list[str] = []

    def flush() -> None:
        section_text = normalize_text(" ".join(buffer))
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

    for element in cast(Iterator[Any], body.descendants):
        name = getattr(element, "name", None)
        if name in _HTML_HEADING_TAGS:
            flush()
            current_section = element.get_text(" ", strip=True)[:120] or DEFAULT_SECTION
            buffer = []
        elif name is None:  # NavigableString
            chunk = str(element).strip()
            if chunk:
                buffer.append(chunk)

    flush()
    if not sections:
        # No headings and/or text only outside body: fall back to whole-document text.
        whole = normalize_text(soup.get_text(" ", strip=True))
        if not whole:
            raise EmptyDocumentError(f"HTML '{filename}' has no extractable text.")
        return [
            DocumentSection(filename=filename, page=1, section=current_section, text=whole)
        ]
    return sections


def parse_csv_document(
    filename: str, text: str, settings: AppSettings | None = None
) -> list[DocumentSection]:
    """Render CSV rows as self-describing text, grouped into row-range sections."""

    import csv

    rows_per_section = (
        int(settings.csv_rows_per_section) if settings is not None
        else _DEFAULT_CSV_ROWS_PER_SECTION
    )
    lines = text.splitlines()
    try:
        dialect = csv.Sniffer().sniff(text[:2048])
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.reader(lines, delimiter=delimiter)
    records = [row for row in reader if any(cell.strip() for cell in row)]
    if not records:
        raise EmptyDocumentError(f"CSV '{filename}' has no rows.")

    header = [cell.strip() for cell in records[0]]
    data_rows = records[1:]
    if not data_rows:
        # Header-only file: treat the header row itself as the single section.
        return [
            DocumentSection(
                filename=filename,
                page=1,
                section="Header",
                text="; ".join(header),
            )
        ]

    sections: list[DocumentSection] = []
    for start in range(0, len(data_rows), rows_per_section):
        group = data_rows[start : start + rows_per_section]
        rendered_rows = [_render_csv_row(header, row) for row in group]
        text_block = "\n".join(line for line in rendered_rows if line)
        if not text_block:
            continue
        first_row = start + 2  # 1-based, and row 1 is the header
        last_row = start + len(group) + 1
        label = f"Rows {first_row}-{last_row}"
        sections.append(
            DocumentSection(filename=filename, page=1, section=label, text=text_block)
        )
    if not sections:
        raise EmptyDocumentError(f"CSV '{filename}' has no data rows.")
    return sections


def _render_csv_row(header: list[str], row: list[str]) -> str:
    """Render one CSV data row as "col1: v1; col2: v2" so any chunk slice self-describes."""

    parts: list[str] = []
    for index, value in enumerate(row):
        cell = value.strip()
        if not cell:
            continue
        column = header[index] if index < len(header) and header[index] else f"col{index + 1}"
        parts.append(f"{column}: {cell}")
    return "; ".join(parts)


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
