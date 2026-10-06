"""Plain text and Markdown: encoding detection, heading sections, paragraph sections."""

from __future__ import annotations

import codecs
import re

from api.parsers.common import (
    DEFAULT_SECTION,
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    cap_section,
    infer_section,
    normalize_text,
)

# Setext H1 only ("Title\n==="). A "---" underline is intentionally not treated as a
# setext H2 heading — it's ambiguous with a CommonMark thematic break (<hr>).
_SETEXT_H1_UNDERLINE_RE = re.compile(r"^\s{0,3}=+\s*$")


_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")


# A fenced code block's opening/closing line: three or more backticks or tildes. Lines
# inside one are code — a "# comment" in a bash block is not a heading.
_CODE_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


class _FenceTracker:
    """Tracks whether a line sits inside a fenced code block (CommonMark rules).

    A fence closes on a line of the same character at least as long as the opener; an
    unclosed fence runs to the end of the document.
    """

    def __init__(self) -> None:
        self._open: str | None = None

    def is_code(self, line: str) -> bool:
        """Feed one line; True when it is a fence line or inside a fenced block."""

        match = _CODE_FENCE_RE.match(line)
        if self._open is None:
            if match:
                self._open = match.group(1)
                return True
            return False
        if match and match.group(1)[0] == self._open[0] and len(match.group(1)) >= len(self._open):
            if not line.strip().strip(match.group(1)[0]):
                self._open = None
        return True


# .txt content sniffing: >=2 heading lines is treated as evidence the file is
# Markdown saved with a .txt extension. One stray "# TODO"-style line in plain prose
# is too common to trust alone.
_MIN_MARKDOWN_HEADINGS_FOR_TXT_SNIFF = 2


_TEXT_DECODE_FALLBACKS: tuple[str, ...] = ("utf-8-sig", "cp1252", "latin-1")


_TEXT_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def decode_text_document(filename: str, content: bytes) -> str:
    """Decode a text-like document, falling back across common encodings.

    ``utf-8-sig`` only tolerates a BOM, not a different encoding — a `.txt`/`.md`
    file saved as cp1252 or latin-1 (common from Windows tools) would otherwise be
    rejected outright even though it decodes cleanly under one of these fallbacks.
    latin-1 never raises (every byte maps to a codepoint), so it's the final,
    always-succeeding fallback rather than a real "unsupported encoding" signal.
    """

    text: str | None = None
    # A byte-order mark names the encoding outright. UTF-16 ("Unicode" in Notepad's
    # save dialog) otherwise fell through to cp1252/latin-1 and indexed as mojibake
    # interleaved with NULs. UTF-32 first: its little-endian BOM starts with UTF-16's.
    for bom, encoding in _TEXT_BOMS:
        if content.startswith(bom):
            try:
                text = content.decode(encoding)
            except UnicodeDecodeError as exc:
                raise DocumentParseError(
                    f"Could not decode text document '{filename}' as {encoding}."
                ) from exc
            break
    for encoding in _TEXT_DECODE_FALLBACKS if text is None else ():
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
    fences = _FenceTracker()
    index = 0
    while index < len(lines):
        line = lines[index]
        if fences.is_code(line):
            buffer.append(line)
            index += 1
            continue
        atx_match = _MARKDOWN_HEADING_RE.match(line)
        if atx_match:
            flush()
            current_section = cap_section(atx_match.group(1).strip())
            buffer = [line]
            index += 1
            continue

        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if line.strip() and _SETEXT_H1_UNDERLINE_RE.match(next_line):
            flush()
            current_section = cap_section(line.strip())
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
    fences = _FenceTracker()
    heading_count = 0
    for index, line in enumerate(lines):
        if fences.is_code(line):
            continue
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
