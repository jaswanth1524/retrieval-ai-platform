"""Jupyter notebooks (.ipynb): markdown cells as text, code cells as fenced code."""

from __future__ import annotations

import json
import re

from api.parsers.common import (
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    note_parse_warning,
)
from api.parsers.text import parse_markdown_document

# A cell longer than this is cut (a pasted data dump in a markdown cell).
MAX_CELL_CHARS = 100_000


def parse_ipynb_document(filename: str, text: str) -> list[DocumentSection]:
    """The notebook as the Markdown it reads as: its markdown cells, and its code cells
    as fenced blocks in the kernel's language, sectioned by the markdown headings
    (``parse_markdown_document``). Outputs are skipped: they are results, often large,
    and rerun at will."""

    try:
        notebook = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise DocumentParseError(f"Could not parse notebook '{filename}'.") from exc
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        raise DocumentParseError(f"'{filename}' is not a Jupyter notebook (no cells).")
    metadata = notebook.get("metadata") if isinstance(notebook.get("metadata"), dict) else {}
    language = _language(metadata)
    blocks: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        source = cell.get("source")
        body = "".join(source) if isinstance(source, list) else source
        if not isinstance(body, str) or not body.strip():
            continue
        if len(body) > MAX_CELL_CHARS:
            note_parse_warning(
                f"A cell longer than {MAX_CELL_CHARS} characters was cut at that length."
            )
            body = body[:MAX_CELL_CHARS]
        if cell.get("cell_type") == "markdown":
            blocks.append(body.strip())
        elif cell.get("cell_type") == "code":
            # Longer than any backtick run in the cell (CommonMark): a line of ``` in
            # the code mustn't close the block and turn its comments into headings.
            longest = max((len(run) for run in re.findall(r"`+", body)), default=0)
            fence = "`" * max(3, longest + 1)
            blocks.append(f"{fence}{language}\n{body.rstrip()}\n{fence}")
    if not blocks:
        raise EmptyDocumentError(f"Notebook '{filename}' has no cells with text.")
    return parse_markdown_document(filename, "\n\n".join(blocks))


def _language(metadata: dict[str, object]) -> str:
    for key in ("kernelspec", "language_info"):
        info = metadata.get(key)
        if isinstance(info, dict):
            name = info.get("language") or info.get("name")
            if isinstance(name, str) and name.isidentifier():
                return name
    return ""
