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


# The field check above can't see a field the server may send as null that the client
# declares non-null, a number declared as a string, or a request field the server
# requires left optional — each type-checks in the app and fails only against a real
# server. These read each field's TypeScript type and compare its shape with the schema.

_SCHEMA_REF_RE = re.compile(r"#/components/schemas/(\w+)$")


def _field_types(body: str) -> dict[str, tuple[bool, str]]:
    """Each field of an interface body: (optional, its type with whitespace collapsed).
    A type may span lines; it ends at the first `;` outside brackets."""

    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    fields: dict[str, tuple[bool, str]] = {}
    for match in re.finditer(r"^  (\w+)(\??):", body, re.M):
        depth = 0
        end = match.end()
        while end < len(body):
            char = body[end]
            if char in "{(<[":
                depth += 1
            elif char in "})>]":
                depth -= 1
            elif char == ";" and depth == 0:
                break
            end += 1
        fields[match.group(1)] = (match.group(2) == "?", " ".join(body[match.end() : end].split()))
    return fields


def _union_members(ts_type: str) -> list[str]:
    members: list[str] = []
    depth = 0
    current = ""
    for char in ts_type:
        if char in "{(<[":
            depth += 1
        elif char in "})>]":
            depth -= 1
        if char == "|" and depth == 0:
            members.append(current.strip())
            current = ""
        else:
            current += char
    members.append(current.strip())
    return [member for member in members if member]


def _ts_kind(member: str) -> str:
    """A union member's JSON kind, or ``named:<Type>`` for a named type (an interface or
    an alias this check doesn't resolve)."""

    if member in ("string", "number", "boolean", "null"):
        return member
    if re.fullmatch(r"'[^']*'", member):
        return "string"
    if re.fullmatch(r"-?\d+(\.\d+)?", member):
        return "number"
    if member in ("true", "false"):
        return "boolean"
    if member.endswith("[]") or member.startswith(("Array<", "readonly ")):
        return "array"
    if member.startswith(("Record<", "{")):
        return "object"
    if member in ("unknown", "any"):
        return "any"
    if re.fullmatch(r"\w+", member):
        return f"named:{member}"
    return "other"


def _schema_options(prop: dict[str, object]) -> list[dict[str, object]]:
    options = prop.get("anyOf") or prop.get("oneOf") or [prop]
    assert isinstance(options, list)
    return options


def _schema_kind(option: dict[str, object]) -> str:
    if "$ref" in option:
        return "object"
    kind = option.get("type")
    if isinstance(kind, str):
        return "number" if kind == "integer" else kind
    if "enum" in option or "const" in option:
        values = option.get("enum") or [option.get("const")]
        assert isinstance(values, list)
        return "string" if isinstance(values[0], str) else "any"
    return "any"


def _schema_refs(prop: dict[str, object]) -> set[str]:
    refs: set[str] = set()
    for option in _schema_options(prop):
        ref = option.get("$ref")
        if isinstance(ref, str) and (match := _SCHEMA_REF_RE.search(ref)):
            refs.add(match.group(1))
    return refs


def type_shape_problems(types_source: str, schemas: dict[str, dict[str, object]]) -> list[str]:
    """Where an interface's field types disagree with the schema of the same name."""

    problems: list[str] = []
    for name, body in _INTERFACE_RE.findall(types_source):
        schema = schemas.get(name)
        if schema is None or "properties" not in schema:
            continue
        properties = schema["properties"]
        assert isinstance(properties, dict)
        required = set(schema.get("required", []))  # type: ignore[arg-type]
        is_request = name.endswith("Request")
        for field, (optional, ts_type) in _field_types(body).items():
            prop = properties.get(field)
            if prop is None:
                continue  # reported by the field check above
            schema_kinds = {_schema_kind(option) for option in _schema_options(prop)}
            ts_kinds = {_ts_kind(member) for member in _union_members(ts_type)}
            where = f"{name}.{field}"
            if not is_request and "null" in schema_kinds and "null" not in ts_kinds:
                problems.append(f"{where}: the server may send null; the type is {ts_type!r}")
            if is_request and field in required and optional:
                problems.append(f"{where}: the server requires it; the type makes it optional")
            plain = {kind for kind in ts_kinds if ":" not in kind and kind != "other"} - {
                "null",
                "any",
            }
            expected = schema_kinds - {"null", "any"}
            if plain and expected and "any" not in schema_kinds and not plain <= expected:
                problems.append(
                    f"{where}: {sorted(plain)} in TypeScript, {sorted(expected)} in the schema"
                )
            # A field typed as another response interface must name the schema it refers to.
            named = {kind[6:] for kind in ts_kinds if kind.startswith("named:")} & set(schemas)
            refs = _schema_refs(prop)
            if named and refs and not named <= refs:
                problems.append(
                    f"{where}: {sorted(named)} in TypeScript, {sorted(refs)} in the schema"
                )
    return problems


def test_frontend_field_types_match_the_api_schema() -> None:
    schemas = json.loads(OPENAPI.read_text(encoding="utf-8"))["components"]["schemas"]

    problems = type_shape_problems(TYPES.read_text(encoding="utf-8"), schemas)

    assert not problems, "\n".join(problems)


def test_the_type_shape_check_catches_each_kind_of_drift() -> None:
    schemas: dict[str, dict[str, object]] = {
        "ThingResponse": {
            "properties": {
                "score": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "count": {"type": "integer"},
                "child": {"$ref": "#/components/schemas/ChildResponse"},
            }
        },
        "ChildResponse": {"properties": {"id": {"type": "string"}}},
        "OtherResponse": {"properties": {"id": {"type": "string"}}},
        "ThingRequest": {"properties": {"question": {"type": "string"}}, "required": ["question"]},
    }
    good = (
        "export interface ThingResponse {\n  score: number | null;\n  count: number;\n"
        "  /** A comment; with a semicolon. */\n  child: ChildResponse;\n}\n"
        "export interface ThingRequest {\n  question: string;\n}\n"
    )
    bad = (
        "export interface ThingResponse {\n  score: number;\n  count: string;\n"
        "  child: OtherResponse;\n}\n"
        "export interface ThingRequest {\n  question?: string;\n}\n"
    )

    assert type_shape_problems(good, schemas) == []
    found = type_shape_problems(bad, schemas)
    assert [problem.split(":")[0] for problem in found] == [
        "ThingResponse.score",
        "ThingResponse.count",
        "ThingResponse.child",
        "ThingRequest.question",
    ], found


USE_CHAT = ROOT / "frontend" / "src" / "hooks" / "useChat.ts"
HISTORY = ROOT / "frontend" / "src" / "utils" / "history.ts"


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

    source = HISTORY.read_text(encoding="utf-8")
    for ts_name, value in (
        ("HISTORY_MAX_MESSAGES", client.HISTORY_MAX_MESSAGES),
        ("HISTORY_CHAR_BUDGET", client.HISTORY_CHAR_BUDGET),
        ("HISTORY_MESSAGE_MAX_CHARS", client.HISTORY_MESSAGE_MAX_CHARS),
    ):
        match = re.search(rf"const {ts_name} = ([\d_]+);", source)
        assert match is not None, ts_name
        assert int(match.group(1).replace("_", "")) == value, ts_name
