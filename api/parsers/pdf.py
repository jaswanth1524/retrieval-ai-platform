"""PDF parsing: pypdf text, pdfplumber tables on pages that draw lines, OCR fallback."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from io import BytesIO
from threading import Lock
from typing import Any

from pypdf import PdfReader
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, StreamObject

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
    repaired = repair_ligature_maps(reader)
    if repaired:
        logger.info("PDF %s: repaired the ligature mapping of %d font(s).", filename, repaired)

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
    cleaned = rejoin_hyphenated(cleaned)

    sections: list[DocumentSection] = []
    empty_pages: list[int] = []
    # Scanned pages carrying a few characters of real text (a scanner's stamp, a Bates
    # number): OCR'd like empty ones, keeping the stamp's text if OCR gives nothing.
    stamped: dict[int, DocumentSection] = {}
    for page_number, (raw, text) in enumerate(zip(page_texts, cleaned, strict=True), start=1):
        if raw and not text and not tables_by_page.get(page_number):
            # Nothing but the running header/footer: a blank page — or a scan whose
            # stamp repeats on every page.
            if _draws_a_large_image(reader.pages[page_number - 1]):
                empty_pages.append(page_number)
            continue
        table_blocks = tables_by_page.get(page_number, [])
        if table_blocks:
            table_text = "\n\n".join(f"[Table]\n{block}" for block in table_blocks)
            text = f"{text}\n\n{table_text}".strip() if text else table_text
        if not text:
            empty_pages.append(page_number)
            continue
        section = DocumentSection(
            filename=filename,
            page=page_number,
            section=infer_section(text, skip=kept_once),
            text=text,
        )
        if len(text) < _STAMP_MAX_CHARS and _draws_a_large_image(reader.pages[page_number - 1]):
            stamped[page_number] = section
            empty_pages.append(page_number)
            continue
        sections.append(section)

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
            if not sections and not stamped:
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
            if not sections and not stamped:
                raise
            logger.warning(
                "PDF %s: OCR failed for pages %s; indexing the remaining pages only.",
                filename,
                empty_pages,
                exc_info=True,
            )
            ocr_sections = []
        ocr_texts = rejoin_hyphenated([section.text for section in ocr_sections])
        ocr_sections = [
            DocumentSection(
                filename=section.filename,
                page=section.page,
                section=section.section,
                text=text,
                boundary=section.boundary,
            )
            for section, text in zip(ocr_sections, ocr_texts, strict=True)
        ]
        read = {section.page for section in ocr_sections}
        sections = sorted(
            [
                *sections,
                *ocr_sections,
                *(section for page, section in stamped.items() if page not in read),
            ],
            key=lambda section: section.page,
        )

    if not sections:
        raise EmptyDocumentError(f"PDF '{filename}' has no extractable text.")
    return sections


# A page with less text than this that draws a large image is a scan with a stamp on it.
_STAMP_MAX_CHARS = 20
# Width x height, in image pixels: about a third of a page scanned at 150 dpi.
_LARGE_IMAGE_PIXELS = 500_000


def _draws_a_large_image(page: Any) -> bool:
    """Whether the page's resources hold an image big enough to be a page scan."""

    try:
        resources = page.get("/Resources")
        resources = resources.get_object() if resources is not None else None
        xobjects = resources.get("/XObject") if isinstance(resources, DictionaryObject) else None
        xobjects = xobjects.get_object() if xobjects is not None else None
        if not isinstance(xobjects, DictionaryObject):
            return False
        for reference in xobjects.values():
            image = reference.get_object()
            if not isinstance(image, DictionaryObject) or image.get("/Subtype") != "/Image":
                continue
            pixels = int(image.get("/Width", 0)) * int(image.get("/Height", 0))
            if pixels >= _LARGE_IMAGE_PIXELS:
                return True
    except Exception:
        return False
    return False


# Ligature glyphs by their font-encoding names (the Adobe names, and the f_i form), as
# text. A font's /Differences can name the "fi" glyph at a code its /ToUnicode map says
# is "!" (seen in a published book: "speci!c", "!rst", "\"oor") — every extractor
# trusts the map, so the words came out broken.
_LIGATURE_GLYPHS = {
    "/fi": "fi",
    "/fl": "fl",
    "/ff": "ff",
    "/ffi": "ffi",
    "/ffl": "ffl",
    "/f_i": "fi",
    "/f_l": "fl",
    "/f_f": "ff",
    "/f_f_i": "ffi",
    "/f_f_l": "ffl",
    "/uniFB00": "ff",
    "/uniFB01": "fi",
    "/uniFB02": "fl",
    "/uniFB03": "ffi",
    "/uniFB04": "ffl",
}
# Form XObjects nest resources; deeper than this is not worth following.
_MAX_RESOURCE_DEPTH = 4


def _ligature_codes(font: DictionaryObject) -> dict[int, str]:
    """The single-byte codes a simple font's /Differences assigns ligature glyphs to."""

    encoding = font.get("/Encoding")
    encoding = encoding.get_object() if encoding is not None else None
    if not isinstance(encoding, DictionaryObject):
        return {}
    differences = encoding.get("/Differences")
    differences = differences.get_object() if differences is not None else None
    if not isinstance(differences, list):
        return {}
    codes: dict[int, str] = {}
    code: int | None = None
    for item in differences:
        if isinstance(item, int):
            code = item
            continue
        if code is None:
            continue
        text = _LIGATURE_GLYPHS.get(str(item))
        if text is not None and 0 <= code < 256:
            codes[code] = text
        code += 1
    return codes


def repair_ligature_maps(reader: PdfReader) -> int:
    """Correct, in memory, fonts whose /ToUnicode map contradicts the ligature glyphs
    their /Differences name; returns how many fonts were changed.

    Each ligature code gets a /ToUnicode entry for its glyph appended (pypdf reads the
    map in order, so the last entry for a code wins): a map that already agreed is
    unchanged in effect, a wrong one is corrected. Simple fonts only (a /Differences
    array means one-byte codes); best-effort — a font it can't read is left alone.
    """

    repaired: set[int] = set()

    def visit(resources: Any, depth: int) -> None:
        resources = resources.get_object() if resources is not None else None
        if not isinstance(resources, DictionaryObject) or depth > _MAX_RESOURCE_DEPTH:
            return
        fonts = resources.get("/Font")
        fonts = fonts.get_object() if fonts is not None else None
        if isinstance(fonts, DictionaryObject):
            for reference in fonts.values():
                font = reference.get_object()
                if not isinstance(font, DictionaryObject) or id(font) in repaired:
                    continue
                to_unicode = font.get("/ToUnicode")
                to_unicode = to_unicode.get_object() if to_unicode is not None else None
                codes = _ligature_codes(font)
                if not codes or not isinstance(to_unicode, StreamObject):
                    continue
                entries = "\n".join(
                    f"<{code:02X}> <{text.encode('utf-16-be').hex().upper()}>"
                    for code, text in sorted(codes.items())
                )
                fixed = DecodedStreamObject()
                fixed.set_data(
                    to_unicode.get_data()
                    + f"\n{len(codes)} beginbfchar\n{entries}\nendbfchar\n".encode()
                )
                font[NameObject("/ToUnicode")] = fixed
                repaired.add(id(font))
        xobjects = resources.get("/XObject")
        xobjects = xobjects.get_object() if xobjects is not None else None
        if isinstance(xobjects, DictionaryObject):
            for reference in xobjects.values():
                xobject = reference.get_object()
                if isinstance(xobject, DictionaryObject) and xobject.get("/Subtype") == "/Form":
                    visit(xobject.get("/Resources"), depth + 1)

    try:
        for page in reader.pages:
            visit(page.get("/Resources"), 0)
    except Exception:
        logger.debug("Could not check the PDF's fonts for ligature mappings.", exc_info=True)
    return len(repaired)


# A word broken over a line end: "pur-\nchase". Letters only on both sides.
_HYPHEN_BREAK_RE = re.compile(r"([^\W\d_]+)-\n([^\W\d_]+)")
_WORDS_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")


def rejoin_hyphenated(page_texts: Sequence[str]) -> list[str]:
    """``page_texts`` with words broken across a line end ("pur-\nchase") made whole.

    The document's own words decide each one: joined when the joined word appears
    elsewhere, or when either half isn't a word of its own ("com-pliant"); kept
    hyphenated when that form appears elsewhere, when the second half is capitalized,
    or when both halves are words the document uses ("cost-\neffective").
    """

    unbroken = [_HYPHEN_BREAK_RE.sub(" ", text) for text in page_texts]
    words: set[str] = set()
    compounds: set[str] = set()
    for text in unbroken:
        for match in _WORDS_RE.finditer(text):
            word = match.group(0).casefold()
            (compounds if "-" in word else words).add(word)
            words.update(word.split("-"))

    def rejoin(match: re.Match[str]) -> str:
        first, second = match.group(1), match.group(2)
        joined, hyphenated = first + second, f"{first}-{second}"
        if hyphenated.casefold() in compounds:
            return hyphenated
        if joined.casefold() in words:
            return joined
        if second[0].isupper():
            return hyphenated
        if first.casefold() in words and second.casefold() in words:
            return hyphenated
        return joined

    return [_HYPHEN_BREAK_RE.sub(rejoin, text) for text in page_texts]


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

    A book's running head carries its page number at one end ("26 Chapter 1:
    Prerequisites", "1.2 About Your App 27") and alternates between left and right
    pages, so each version is on only half of them: an outermost line of two or more
    words whose leading or trailing number moves in step with the page is matched page
    by page, with even and odd pages counted apart. "Article 3" isn't one.
    """

    pages = [text.split("\n") for text in page_texts]
    numbered = [
        (number, lines)
        for number, lines in enumerate(pages, start=1)
        if any(line.strip() for line in lines)
    ]
    if len(numbered) < _RUNNING_MIN_PAGES:
        return list(page_texts), set()

    def outermost(lines: list[str]) -> set[int]:
        indexed = [index for index, line in enumerate(lines) if line.strip()]
        return {indexed[0], indexed[-1]} if indexed else set()

    def edges(number: int, lines: list[str]) -> dict[int, set[str]]:
        indexed = [index for index, line in enumerate(lines) if line.strip()]
        if not indexed:
            return {}
        picked = indexed[:_RUNNING_EDGE_LINES] + indexed[-_RUNNING_EDGE_LINES:]
        outer = {indexed[0], indexed[-1]}
        result: dict[int, set[str]] = {}
        for index in picked:
            keys = _line_keys(lines[index], index in outer)
            if index in outer:
                tracking = _page_tracking_key(lines[index], number)
                if tracking is not None:
                    keys.add(tracking)
            result[index] = keys
        return result

    counts: dict[str, int] = {}
    for number, lines in numbered:
        for key in set().union(*edges(number, lines).values()):
            counts[key] = counts.get(key, 0) + 1
    needed = max(_RUNNING_MIN_PAGES, int(len(numbered) * _RUNNING_MIN_SHARE + 0.999))
    by_parity = [sum(1 for number, _ in numbered if number % 2 == parity) for parity in (0, 1)]

    def is_running(key: str) -> bool:
        if key.startswith("#page:"):
            parity = int(key.split(":", 2)[1])
            return counts[key] >= max(2, math.ceil(by_parity[parity] * _RUNNING_MIN_SHARE))
        return counts[key] >= needed

    running = {key for key in counts if is_running(key)}
    if not running:
        return list(page_texts), set()
    # Page-number offsets a running head proved ("26 Chapter 1" on page 1: 25). The
    # other parity's head can change with every subsection ("1.2 About Your App 27",
    # "1.2.2 Online versus Offline 29") and so never repeats; its number is still the
    # page number, and goes — the title stays, as the page's section label.
    offsets = {int(key.split(":", 3)[2]) for key in running if key.startswith("#page:")}
    cleaned: list[str] = []
    kept_once: set[str] = set()
    for number, lines in enumerate(pages, start=1):
        outer = outermost(lines)
        drop: set[int] = set()
        for index, keys in edges(number, lines).items():
            matched = keys & running
            if not matched:
                continue
            exact = " ".join(lines[index].split()).casefold()
            if index not in outer and matched == {exact} and exact not in kept_once:
                kept_once.add(exact)
                continue
            drop.add(index)
        if offsets:
            for index in outer - drop:
                lines[index] = _without_page_number(lines[index], number, offsets)
        kept = [line for index, line in enumerate(lines) if index not in drop]
        cleaned.append(normalize_text("\n".join(kept)))
    return cleaned, kept_once


# A number at either end of a line, set off by a space: "26 Chapter 1", "Your App 27".
_LEADING_NUMBER_RE = re.compile(r"^(\d{1,4})\s+(?=\S)")
_TRAILING_NUMBER_RE = re.compile(r"(?<=\S)\s+(\d{1,4})$")


def _without_page_number(line: str, page_number: int, offsets: set[int]) -> str:
    """``line`` minus a leading or trailing number that is ``page_number`` plus one of
    ``offsets`` — when what is left is a head of two or more words."""

    text = line.strip()
    for pattern in (_LEADING_NUMBER_RE, _TRAILING_NUMBER_RE):
        match = pattern.search(text)
        if match is None or int(match.group(1)) - page_number not in offsets:
            continue
        rest = (text[: match.start()] + text[match.end() :]).strip()
        if len([word for word in rest.split() if re.search(r"[^\W\d_]", word)]) >= 2:
            return rest
    return line


def _page_tracking_key(line: str, page_number: int) -> str | None:
    """A key matching the same line on pages of the same parity when its number at
    either end is the page number plus a fixed offset (a book's running head)."""

    exact = " ".join(line.split()).casefold()
    for pattern in (_LEADING_NUMBER_RE, _TRAILING_NUMBER_RE):
        match = pattern.search(exact)
        if match is None:
            continue
        rest = exact[: match.start()] + "#" + exact[match.end() :]
        # A running head says more than a numbered heading: "Article 3" or "Step 5"
        # starting each page in turn is content, and keeps its number.
        if len([word for word in rest.split() if re.search(r"[^\W\d_]", word)]) < 2:
            return None
        offset = int(match.group(1)) - page_number
        return f"#page:{page_number % 2}:{offset}:{rest}"
    return None


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
        no_text_pages = 0
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
                    if not getattr(page, "chars", True):
                        no_text_pages += 1
                    blocks = [_table_to_markdown(table) for table in page.extract_tables() if table]
                    blocks = [block for block in blocks if block]
                    if blocks:
                        result[page_number] = blocks
                finally:
                    # Parsed layout objects are cached per page until closed; on a long
                    # PDF they otherwise accumulate for the whole document.
                    page.close()
        if no_text_pages and not result:
            # pdfplumber found no characters where it looked for tables (a font with no
            # usable text map): any table on those pages is indexed as plain text lines.
            logger.info(
                "PDF table extraction read no text on %d page(s); tables there are "
                "indexed as plain text.",
                no_text_pages,
            )
        return result
    except Exception:
        logger.warning("PDF table extraction failed; continuing with text only.", exc_info=True)
        return {}


def _table_to_markdown(rows: Sequence[Sequence[object]]) -> str:
    """Render an extracted table (list of rows of cells) as a Markdown table."""

    cleaned = [
        # A "|" inside a cell would split it into two columns.
        [
            str(cell).strip().replace("\n", " ").replace("|", "\\|") if cell is not None else ""
            for cell in row
        ]
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
