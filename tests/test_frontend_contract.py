"""The frontend's hand-written API types against the API's own schema.

``frontend/src/api/types.ts`` is written by hand, and the Playwright suite mocks every
route, so a field renamed or dropped on the server passed both test suites and broke
only the real app. Every interface there named like a response or request schema must
use only fields that schema has (a client may read fewer, never others).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPES = ROOT / "frontend" / "src" / "api" / "types.ts"
OPENAPI = ROOT / "tests" / "golden" / "openapi.json"

_INTERFACE_RE = re.compile(r"^export interface (\w+) \{\n(.*?)^\}", re.M | re.S)
_FIELD_RE = re.compile(r"^  (\w+)\??:", re.M)


def _interfaces() -> dict[str, set[str]]:
    source = TYPES.read_text(encoding="utf-8")
    return {name: set(_FIELD_RE.findall(body)) for name, body in _INTERFACE_RE.findall(source)}


def test_frontend_types_use_only_fields_the_api_schema_has() -> None:
    schemas = json.loads(OPENAPI.read_text(encoding="utf-8"))["components"]["schemas"]
    checked = 0
    problems: list[str] = []
    for name, fields in _interfaces().items():
        schema = schemas.get(name)
        if schema is None or "properties" not in schema:
            continue  # client-only types (QuestionOptions, ...) have no server schema
        checked += 1
        unknown = sorted(fields - set(schema["properties"]))
        if unknown:
            problems.append(f"{name}: {', '.join(unknown)} not in the API schema")
    assert not problems, "\n".join(problems)
    # The check must be doing something: most response types share their names.
    assert checked >= 20
