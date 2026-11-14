"""RTF: the text of a Rich Text Format document, paragraph by paragraph.

A small reader, no dependency: control words are dropped except the ones that are text
(``\\par``, ``\\line``, ``\\tab``, ``\\'hh`` bytes, ``\\uN`` characters, the escaped
braces and backslash), and the groups that hold no body text (font and colour tables,
the style sheet, document info, pictures, ``{\\*...}`` destinations) are skipped whole.
Paragraphs then go through the plain-text parser, which splits and labels them.
"""

from __future__ import annotations

import re

from api.parsers.common import DocumentParseError, DocumentSection, EmptyDocumentError
from api.parsers.text import parse_plain_text_document

# Groups whose content is never body text.
_SKIPPED_DESTINATIONS = frozenset(
    {
        "fonttbl",
        "colortbl",
        "stylesheet",
        "info",
        "pict",
        "object",
        "header",
        "footer",
        "headerl",
        "headerr",
        "footerl",
        "footerr",
        "listtable",
        "listoverridetable",
        "rsidtbl",
        "generator",
        "xmlnstbl",
        "themedata",
        "datastore",
        "latentstyles",
        "colorschememapping",
    }
)
_TOKEN_RE = re.compile(
    r"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\([^a-zA-Z])|([{}])|([^\\{}\r\n]+)|[\r\n]+"
)
# Nesting deeper than this is no real document (a crafted file blowing the stack).
_MAX_DEPTH = 256


def rtf_to_text(source: str) -> str:
    """The body text of the RTF document ``source``."""

    if not source.lstrip().startswith("{\\rtf"):
        raise DocumentParseError("It isn't an RTF document (no {\\rtf header).")
    out: list[str] = []
    # Per open group: whether its content is skipped, and how many characters follow a
    # \uN (\ucN, 1 by default) as its fallback, to be dropped.
    stack: list[tuple[bool, int]] = []
    skipping = False
    uc = 1
    pending_skip = 0
    encoding = "cp1252"
    hex_bytes = bytearray()

    def flush_hex() -> None:
        if hex_bytes:
            out.append(hex_bytes.decode(encoding, errors="replace"))
            hex_bytes.clear()

    for match in _TOKEN_RE.finditer(source):
        word, number, hex_pair, symbol, brace, text = match.groups()
        if hex_pair is None:
            flush_hex()
        if brace == "{":
            if len(stack) >= _MAX_DEPTH:
                raise DocumentParseError("The RTF document nests too deeply.")
            stack.append((skipping, uc))
            continue
        if brace == "}":
            if stack:
                skipping, uc = stack.pop()
            continue
        if pending_skip and (text or hex_pair or symbol):
            # The fallback characters of a \uN, already given as the character itself.
            if text:
                dropped = min(pending_skip, len(text))
                text = text[dropped:]
                pending_skip -= dropped
            else:
                pending_skip -= 1
                continue
        if symbol is not None:
            if symbol == "*":
                skipping = True  # an ignorable destination: {\*\name ...}
            elif not skipping and symbol in "\\{}":
                out.append(symbol)
            elif not skipping and symbol == "~":
                out.append("\u00a0")
            elif not skipping and symbol in "-_":
                out.append("-" if symbol == "_" else "")
            continue
        if hex_pair is not None:
            if not skipping:
                hex_bytes.append(int(hex_pair, 16))
            continue
        if word is not None:
            if word in _SKIPPED_DESTINATIONS:
                skipping = True
            elif word == "ansicpg" and number:
                encoding = f"cp{number}"
            elif word == "uc" and number:
                uc = max(0, int(number))
            elif skipping:
                pass
            elif word == "u" and number:
                code = int(number)
                out.append(chr(code + 65536 if code < 0 else code))
                pending_skip = uc
            elif word in ("par", "sect", "page"):
                out.append("\n\n")
            elif word in ("line", "row"):
                out.append("\n")
            elif word in ("tab", "cell"):
                out.append("\t")
            continue
        if text and not skipping:
            out.append(text)
    flush_hex()
    return "".join(out)


def parse_rtf_document(filename: str, content: bytes) -> list[DocumentSection]:
    """``filename``'s paragraphs, split and labelled as plain text is."""

    try:
        # RTF is 7-bit; anything else comes through \'hh or \uN.
        source = content.decode("latin-1")
        text = rtf_to_text(source)
    except LookupError as exc:  # an \ansicpg Python has no codec for
        raise DocumentParseError(f"'{filename}' names an unknown code page: {exc}") from exc
    text = re.sub(r"[ \t]+\n", "\n", text)
    if not text.strip():
        raise EmptyDocumentError(f"'{filename}' has no text.")
    return parse_plain_text_document(filename, text)
