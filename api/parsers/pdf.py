"""PDF parsing: pypdf text, pdfplumber tables on pages that draw lines, OCR fallback."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from io import BytesIO
from threading import Lock
from typing import Any

from pypdf import PdfReader

from api.parsers.common import (
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    infer_section,
    normalize_text,
)
from api.settings import AppSettings

logger = logging.getLogger(__name__)


def parse_pdf_document(
    filename: str, content: bytes, settings: AppSettings | None = None
) -> list[DocumentSection]:
    """Extract text from a PDF, preserving one-based page numbers.

    pypdf provides the text stream; pdfplumber (best-effort) appends any extracted
    tables as Markdown blocks so table Q&A is answerable. Pages with no extractable
    text fall back to OCR when the optional ``ocr`` extra is installed, up to
    ``max_ocr_pages`` of them; a PDF over ``max_pdf_pages`` is refused up front.
    """

    limits = settings if settings is not None else AppSettings.model_construct()
    max_pdf_pages = int(limits.max_pdf_pages)
    max_ocr_pages = int(limits.max_ocr_pages)

    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:  # pypdf raises several parser-specific exceptions.
        raise DocumentParseError(f"Could not parse PDF '{filename}'.") from exc
    if reader.is_encrypted:
        # An owner-password-only PDF opens with the empty user password; anything else
        # can't be read, and used to fail as an "unexpected server error".
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:
            raise DocumentParseError(f"PDF '{filename}' is password-protected.") from exc
        if not decrypted:
            raise DocumentParseError(f"PDF '{filename}' is password-protected.")
    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise DocumentParseError(f"Could not parse PDF '{filename}'.") from exc
    if page_count > max_pdf_pages:
        raise DocumentParseError(
            f"PDF '{filename}' has {page_count} pages; the limit is {max_pdf_pages} "
            "(MAX_PDF_PAGES). Split it into smaller files."
        )

    tables_by_page = _extract_pdf_tables(content, _pages_that_draw_lines(reader))

    page_texts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_texts.append(normalize_text(page.extract_text() or ""))
        except Exception:
            # One malformed content stream shouldn't fail the whole document; the page
            # goes to OCR like a page with no text layer.
            logger.warning(
                "PDF %s: could not extract text from page %d.",
                filename,
                page_number,
                exc_info=True,
            )
            page_texts.append("")
    cleaned, kept_once = strip_running_lines_keeping_first(page_texts)

    sections: list[DocumentSection] = []
    empty_pages: list[int] = []
    for page_number, (raw, text) in enumerate(zip(page_texts, cleaned, strict=True), start=1):
        if raw and not text and not tables_by_page.get(page_number):
            continue  # nothing but the running header/footer: a blank page, not a scan
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
                section=infer_section(text, skip=kept_once),
                text=text,
            )
        )

    if len(empty_pages) > max_ocr_pages:
        logger.warning(
            "PDF %s: %d pages have no text layer; OCR'ing the first %d (MAX_OCR_PAGES) "
            "and skipping pages %s.",
            filename,
            len(empty_pages),
            max_ocr_pages,
            empty_pages[max_ocr_pages:],
        )
        empty_pages = empty_pages[:max_ocr_pages]
    if empty_pages:
        # Every empty page, not only all-empty PDFs: a scanned page bound into an
        # otherwise digital PDF used to be dropped without a word.
        try:
            ocr_sections = _ocr_pdf_pages(filename, content, empty_pages)
        except EmptyDocumentError:
            if not sections:
                raise
            logger.warning(
                "PDF %s: pages %s have no text layer and OCR is not installed "
                "(uv sync --extra ocr); indexing the remaining pages only.",
                filename,
                empty_pages,
            )
            ocr_sections = []
        except Exception:
            # OCR itself broke (engine load, unreadable PDF for the rasterizer). A fully
            # scanned PDF has nothing else to offer, so that still fails; a mixed one
            # keeps its text pages, as it did before its empty pages were OCR'd.
            if not sections:
                raise
            logger.warning(
                "PDF %s: OCR failed for pages %s; indexing the remaining pages only.",
                filename,
                empty_pages,
                exc_info=True,
            )
            ocr_sections = []
        sections = sorted([*sections, *ocr_sections], key=lambda section: section.page)

    if not sections:
        raise EmptyDocumentError(f"PDF '{filename}' has no extractable text.")
    return sections


# A line repeated at the top or bottom of most pages is a running header or footer
# ("ACME Handbook 2024", "Page 3 of 40", "Confidential"): it went into every chunk and
# became most pages' section label. Looked for in the first and last
# _RUNNING_EDGE_LINES lines of each page, on documents of at least
# _RUNNING_MIN_PAGES pages, and counted as running when it is on at least
# _RUNNING_MIN_SHARE of them (digits ignored, so page numbers match).
_RUNNING_EDGE_LINES = 2
_RUNNING_MIN_PAGES = 3
_RUNNING_MIN_SHARE = 0.6


# A page number, once digits are "#": the whole line ("3", "- 3 -", "Page 3 of 40",
# "3/40"), or one at either end set off by a separator or a page word ("ACME Handbook |
# 3", "Handbook 2024 — Page 3", "3 · ACME"). "Article 3" or "Notice period is 3 days"
# is not one: matched with the digits ignored, such a line was dropped from every page.
_PAGE_WORD = r"(?:page|pg\.?|p\.|seite|página|pagina)"
_PAGE_NUMBER = rf"(?:{_PAGE_WORD}\s*)?#(?:\s*(?:of|/|von|de|di|sur)\s*#)?"
_SEPARATOR = r"[-–—|•·:]"
_PAGE_NUMBER_LINE_RE = re.compile(
    rf"^(?:{_SEPARATOR}\s*)*{_PAGE_NUMBER}(?:\s*{_SEPARATOR})*$"
    rf"|(?:{_SEPARATOR}|\s{_PAGE_WORD})\s*{_PAGE_NUMBER}(?:\s*{_SEPARATOR})*$"
    rf"|^(?:{_SEPARATOR}\s*)*{_PAGE_NUMBER}\s*{_SEPARATOR}"
    rf"|^{_PAGE_WORD}\s*#"
)


def _line_keys(line: str, outermost: bool) -> set[str]:
    """What a line is matched across pages by: its text (whitespace and case aside),
    and — on a page's very first or last line, where page numbers sit, and only when it
    reads as a page number — also its text with the digits ignored ("Page 3 of 40")."""

    exact = " ".join(line.split()).casefold()
    if not exact:
        return set()
    keys = {exact}
    if outermost and re.search(r"\d", exact):
        digits_ignored = re.sub(r"\d+", "#", exact)
        if _PAGE_NUMBER_LINE_RE.search(digits_ignored):
            keys.add("#digits:" + digits_ignored)
    return keys


def strip_running_lines(page_texts: Sequence[str]) -> list[str]:
    """``page_texts`` with running headers and footers removed from each page's edges.

    Only the edge lines are touched: a sentence that happens to repeat inside the text
    stays. A document shorter than _RUNNING_MIN_PAGES pages is returned unchanged.
    """

    return strip_running_lines_keeping_first(page_texts)[0]


def strip_running_lines_keeping_first(
    page_texts: Sequence[str],
) -> tuple[list[str], set[str]]:
    """``strip_running_lines``, plus the lines it kept once (see below), casefolded.

    A page's outermost line that repeats is page furniture and goes from every page. A
    repeat one line in (a table's header row under the running header, a heading that
    starts each page) can be content: its first copy stays, and only the repeats go.
    The kept copies are returned so they aren't taken for their page's section label.
    """

    pages = [text.split("\n") for text in page_texts]
    with_text = [lines for lines in pages if any(line.strip() for line in lines)]
    if len(with_text) < _RUNNING_MIN_PAGES:
        return list(page_texts), set()

    def edges(lines: list[str]) -> dict[int, set[str]]:
        indexed = [index for index, line in enumerate(lines) if line.strip()]
        if not indexed:
            return {}
        picked = indexed[:_RUNNING_EDGE_LINES] + indexed[-_RUNNING_EDGE_LINES:]
        outer = {indexed[0], indexed[-1]}
        return {index: _line_keys(lines[index], index in outer) for index in picked}

    def outermost(lines: list[str]) -> set[int]:
        indexed = [index for index, line in enumerate(lines) if line.strip()]
        return {indexed[0], indexed[-1]} if indexed else set()

    counts: dict[str, int] = {}
    for lines in with_text:
        for key in set().union(*edges(lines).values()):
            counts[key] = counts.get(key, 0) + 1
    needed = max(_RUNNING_MIN_PAGES, int(len(with_text) * _RUNNING_MIN_SHARE + 0.999))
    running = {key for key, count in counts.items() if count >= needed}
    if not running:
        return list(page_texts), set()
    cleaned: list[str] = []
    kept_once: set[str] = set()
    for lines in pages:
        outer = outermost(lines)
        drop: set[int] = set()
        for index, keys in edges(lines).items():
            matched = keys & running
            if not matched:
                continue
            exact = " ".join(lines[index].split()).casefold()
            if index not in outer and matched == {exact} and exact not in kept_once:
                kept_once.add(exact)
                continue
            drop.add(index)
        kept = [line for index, line in enumerate(lines) if index not in drop]
        cleaned.append(normalize_text("\n".join(kept)))
    return cleaned, kept_once


# Content-stream operators that draw a rectangle or a line, plus XObject paints (a
# form can draw either). A table's ruling edges need one of them; text doesn't.
_RULING_OPERATOR_RE = re.compile(rb"(?<![A-Za-z])(?:re|l|Do)(?![A-Za-z])")


def _pages_that_draw_lines(reader: PdfReader) -> set[int] | None:
    """1-based pages whose content stream draws rectangles, lines or XObjects.

    A cheap byte scan of the stream pypdf already decoded — laying a page out with
    pdfplumber just to learn it has no ruling edges parsed most prose PDFs twice.
    Over-inclusive by design (an "l" inside a text string counts): a false positive
    only costs the old full check. None when a stream can't be read: check every page.
    """

    pages: set[int] = set()
    try:
        for page_number, page in enumerate(reader.pages, start=1):
            contents = page.get_contents()
            if contents is None:
                continue
            if _RULING_OPERATOR_RE.search(contents.get_data()):
                pages.add(page_number)
    except Exception:
        return None
    return pages


def _extract_pdf_tables(content: bytes, pages: set[int] | None = None) -> dict[int, list[str]]:
    """Return page-number -> rendered Markdown tables. Best-effort: failure -> {}.

    Table extraction is enrichment, never a parse gate — any pdfplumber error (or a
    page with no tables) simply yields no table text for that document/page. ``pages``
    limits the work to those 1-based pages (None: all of them).
    """

    if pages is not None and not pages:
        return {}
    try:
        import pdfplumber

        result: dict[int, list[str]] = {}
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                if pages is not None and page_number not in pages:
                    continue
                try:
                    # The default "lines" strategy builds tables only from ruling edges
                    # (rects, lines, curves); a page with none can't yield one, so skip
                    # the table finder — most pages of a prose PDF.
                    if not (page.rects or page.lines or page.curves):
                        continue
                    blocks = [_table_to_markdown(table) for table in page.extract_tables() if table]
                    blocks = [block for block in blocks if block]
                    if blocks:
                        result[page_number] = blocks
                finally:
                    # Parsed layout objects are cached per page until closed; on a long
                    # PDF they otherwise accumulate for the whole document.
                    page.close()
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


# OCR renders a page at OCR_DPI, as an RGB bitmap in memory. MAX_OCR_PAGES bounds how
# many pages, not how big one is: a MediaBox of a few hundred inches square (a valid PDF
# of a few hundred bytes) asked for gigabytes. A page past OCR_MAX_PAGE_PIXELS is
# rendered at a lower resolution, down to OCR_MIN_DPI, and skipped below that.
OCR_DPI = 200.0
OCR_MIN_DPI = 72.0
OCR_MAX_PAGE_PIXELS = 40_000_000


def ocr_resolution(width_pt: float, height_pt: float) -> float | None:
    """The resolution to render a page at for OCR, or None when it is too large."""

    area_in = (width_pt / 72.0) * (height_pt / 72.0)
    if area_in <= 0:
        return None
    if area_in * OCR_DPI * OCR_DPI <= OCR_MAX_PAGE_PIXELS:
        return OCR_DPI
    dpi = (OCR_MAX_PAGE_PIXELS / area_in) ** 0.5
    return dpi if dpi >= OCR_MIN_DPI else None


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
            page = pdf.pages[page_number - 1]
            resolution = ocr_resolution(float(page.width), float(page.height))
            if resolution is None:
                logger.warning(
                    "PDF %s: page %d is too large to OCR (%.0f x %.0f pt); skipping it.",
                    filename,
                    page_number,
                    page.width,
                    page.height,
                )
                page.close()
                continue
            try:
                image = page.to_image(resolution=resolution)
                import numpy as np

                ocr_result, _ = engine(np.array(image.original))
            except Exception:
                # One unreadable image page must not fail the whole document — before
                # mixed PDFs were OCR'd, such a page was simply skipped.
                logger.warning(
                    "PDF %s: OCR failed on page %d; skipping it.",
                    filename,
                    page_number,
                    exc_info=True,
                )
                continue
            finally:
                # A page keeps its parsed objects cached until closed; a long scan
                # otherwise held every OCR'd page in memory until the PDF closed.
                page.close()
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


# The ingest executor runs two workers, so two scanned PDFs arriving together could both
# see _OCR_ENGINE as None and each construct one — doubling a heavyweight model load.
_OCR_ENGINE_LOCK = Lock()


def _get_ocr_engine(engine_cls: type) -> Any:
    """Lazily construct and cache the OCR engine (model load is expensive)."""

    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        with _OCR_ENGINE_LOCK:
            # Re-check inside the lock: another worker may have built it while we waited.
            if _OCR_ENGINE is None:
                _OCR_ENGINE = engine_cls()
    return _OCR_ENGINE
