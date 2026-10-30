"""JSON documents and source code: text formats with a structure of their own."""

from __future__ import annotations

import json
from typing import Any

from api.parsers.common import (
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    cap_section,
)
from api.parsers.text import parse_plain_text_document

# Past this depth a value is written out as compact JSON rather than walked further.
_MAX_JSON_DEPTH = 32
# A list of plain values is one line ("tags: a, b, c") up to this many of them.
_MAX_INLINE_ITEMS = 50

# Source code and configuration read as text, split at blank lines.
CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".go",
        ".java",
        ".rs",
        ".rb",
        ".c",
        ".cpp",
        ".h",
        ".sh",
        ".sql",
        ".yaml",
        ".yml",
        ".toml",
    }
)


def parse_json_document(filename: str, text: str) -> list[DocumentSection]:
    """One section per top-level key (or per item of a top-level list), each value
    written out as ``path › to › key: value`` lines, one fact a line.

    Each line is a unit of its own (``line_units``): a chunk never ends in the middle
    of a value, and a citation quotes the one fact it rests on.
    """

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DocumentParseError(
            f"'{filename}' is not valid JSON: {exc.msg} (line {exc.lineno})."
        ) from exc
    except RecursionError as exc:
        raise DocumentParseError(f"'{filename}' is nested too deeply to read.") from exc
    except ValueError as exc:
        # Python's limit on integer digits (a number thousands of digits long).
        raise DocumentParseError(f"'{filename}' holds a number too long to read.") from exc

    if isinstance(data, dict):
        items: list[tuple[str, Any]] = [(str(key), value) for key, value in data.items()]
    elif isinstance(data, list):
        items = [(f"Item {number}", value) for number, value in enumerate(data, start=1)]
    else:
        items = [("Value", data)]

    sections: list[DocumentSection] = []
    for label, value in items:
        lines = list(_lines(value, label, 0))
        body = "\n".join(" ".join(line.split()) for line in lines if line.strip())
        if body:
            sections.append(
                DocumentSection(
                    filename=filename,
                    page=1,
                    section=cap_section(label),
                    text=body,
                    boundary="semantic",
                    line_units=True,
                )
            )
    if not sections:
        raise EmptyDocumentError(f"JSON document '{filename}' has no values to index.")
    return sections


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _lines(value: Any, path: str, depth: int) -> list[str]:
    if depth >= _MAX_JSON_DEPTH:
        return [f"{path}: {json.dumps(value, ensure_ascii=False)}"]
    if isinstance(value, dict):
        if not value:
            return []
        lines: list[str] = []
        for key, item in value.items():
            lines.extend(_lines(item, f"{path} › {key}", depth + 1))
        return lines
    if isinstance(value, list):
        if not value:
            return []
        if len(value) <= _MAX_INLINE_ITEMS and not any(
            isinstance(item, dict | list) for item in value
        ):
            return [f"{path}: {', '.join(_scalar(item) for item in value)}"]
        lines = []
        for number, item in enumerate(value, start=1):
            lines.extend(_lines(item, f"{path} [{number}]", depth + 1))
        return lines
    return [f"{path}: {_scalar(value)}"]


def parse_code_document(filename: str, text: str) -> list[DocumentSection]:
    """Source code and config files, split at blank lines like plain text: a block's
    first line (a function's signature, a config table's name) labels it."""

    if not text.strip():
        raise EmptyDocumentError(f"'{filename}' is empty.")
    return parse_plain_text_document(filename, text)
