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


USE_CHAT = ROOT / "frontend" / "src" / "hooks" / "useChat.ts"


def _server_stages() -> set[str]:
    stages: set[str] = set()
    for path in (ROOT / "api").rglob("*.py"):
        stages.update(re.findall(r'"stage": "([a-z_]+)"', path.read_text(encoding="utf-8")))
    return stages


def test_every_stage_the_server_sends_is_named_by_the_ui_and_the_cli() -> None:
    """An unknown stage falls back to a generic label, so a new one never breaks a
    client; but it should be named in both, which are written by hand."""

    from cli.__main__ import STAGE_NAMES

    block = re.search(
        r"const STAGE_LABELS: Record<string, string> = \{(.*?)\};", USE_CHAT.read_text(), re.S
    )
    assert block is not None
    ui_stages = set(re.findall(r"^\s*(\w+):", block.group(1), re.M))
    server = _server_stages()
    assert server, "no stage frames found"
    assert server <= ui_stages, server - ui_stages
    assert server <= set(STAGE_NAMES), server - set(STAGE_NAMES)


def test_the_cli_keeps_history_to_the_ui_s_limits() -> None:
    from cli import client

    source = USE_CHAT.read_text(encoding="utf-8")
    for ts_name, value in (
        ("HISTORY_MAX_MESSAGES", client.HISTORY_MAX_MESSAGES),
        ("HISTORY_CHAR_BUDGET", client.HISTORY_CHAR_BUDGET),
        ("HISTORY_MESSAGE_MAX_CHARS", client.HISTORY_MESSAGE_MAX_CHARS),
    ):
        match = re.search(rf"const {ts_name} = ([\d_]+);", source)
        assert match is not None, ts_name
        assert int(match.group(1).replace("_", "")) == value, ts_name
