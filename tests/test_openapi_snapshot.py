"""Pins the OpenAPI document, so every API change shows up as a reviewed diff.

The frontend's types are written by hand, so nothing else notices a field that went
missing. When the schema changes on purpose, regenerate the snapshot with

    DOCRAG_UPDATE_GOLDEN=1 uv run pytest tests/test_openapi_snapshot.py

and read the diff of ``tests/golden/openapi.json``. A failing run lists the changes that
would break an existing client (removed paths or fields, new required fields, removed
enum values, new or tighter length limits) before the plain mismatch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from api.dependencies import clear_dependency_caches
from tests.routes_http.support import hermetic_app, make_settings

GOLDEN_PATH = Path(__file__).parent / "golden" / "openapi.json"
_LIMITS = ("maxLength", "maxItems", "maximum")


def current_openapi() -> dict[str, Any]:
    clear_dependency_caches()
    try:
        schema = hermetic_app(make_settings(), QdrantClient(":memory:")).openapi()
    finally:
        clear_dependency_caches()
    # The version moves every release and is pinned by test_release_metadata already.
    schema = json.loads(json.dumps(schema))
    schema["info"].pop("version", None)
    return schema


def _schema_problems(where: str, old: Any, new: Any) -> list[str]:
    if not isinstance(old, dict) or not isinstance(new, dict):
        return []
    problems: list[str] = []
    for key, value in old.get("properties", {}).items():
        if key not in new.get("properties", {}):
            problems.append(f"{where}: property '{key}' removed")
        else:
            problems += _schema_problems(f"{where}.{key}", value, new["properties"][key])
    for key in sorted(set(new.get("required", [])) - set(old.get("required", []))):
        problems.append(f"{where}: '{key}' is newly required")
    if "enum" in old and "enum" in new:
        for value in old["enum"]:
            if value not in new["enum"]:
                problems.append(f"{where}: enum value {value!r} removed")
    for limit in _LIMITS:
        if limit in new and (limit not in old or new[limit] < old[limit]):
            problems.append(f"{where}: {limit} {old.get(limit)} -> {new[limit]}")
    if "items" in old and "items" in new:
        problems += _schema_problems(f"{where}[]", old["items"], new["items"])
    for combinator in ("anyOf", "oneOf", "allOf"):
        for index, (a, b) in enumerate(
            zip(old.get(combinator, []), new.get(combinator, []), strict=False)
        ):
            problems += _schema_problems(f"{where}<{combinator}{index}>", a, b)
    return problems


def breaking_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Changes from ``old`` to ``new`` that an existing client could trip over.

    Conservative: a schema shared by requests and responses is held to both rules.
    """

    problems: list[str] = []
    for path, methods in old.get("paths", {}).items():
        if path not in new.get("paths", {}):
            problems.append(f"path {path} removed")
            continue
        for method, operation in methods.items():
            new_operation = new["paths"][path].get(method)
            if new_operation is None:
                problems.append(f"{method.upper()} {path} removed")
                continue
            old_params = {(p["name"], p["in"]) for p in operation.get("parameters", [])}
            for param in new_operation.get("parameters", []):
                if param.get("required") and (param["name"], param["in"]) not in old_params:
                    problems.append(f"{method.upper()} {path}: new required parameter")
            for param in operation.get("parameters", []):
                match = next(
                    (
                        p
                        for p in new_operation.get("parameters", [])
                        if (p["name"], p["in"]) == (param["name"], param["in"])
                    ),
                    None,
                )
                if match is None:
                    problems.append(f"{method.upper()} {path}: parameter {param['name']} removed")
                else:
                    problems += _schema_problems(
                        f"{method.upper()} {path} ?{param['name']}",
                        param.get("schema"),
                        match.get("schema"),
                    )
    old_schemas = old.get("components", {}).get("schemas", {})
    new_schemas = new.get("components", {}).get("schemas", {})
    for name, schema in old_schemas.items():
        if name not in new_schemas:
            problems.append(f"schema {name} removed")
        else:
            problems += _schema_problems(name, schema, new_schemas[name])
    return problems


def test_the_openapi_document_matches_its_snapshot() -> None:
    actual = current_openapi()
    if os.environ.get("DOCRAG_UPDATE_GOLDEN") == "1":
        GOLDEN_PATH.parent.mkdir(exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
        pytest.skip("OpenAPI snapshot regenerated")
    expected = json.loads(GOLDEN_PATH.read_text())
    breaking = breaking_changes(expected, actual)
    assert not breaking, "Breaking API changes (regenerate only if intended):\n" + "\n".join(
        breaking
    )
    assert actual == expected, (
        "The API changed in an additive way; regenerate the snapshot with "
        "DOCRAG_UPDATE_GOLDEN=1 and review the diff."
    )


def test_the_checker_names_each_kind_of_breaking_change() -> None:
    old = {
        "paths": {"/a": {"get": {}, "post": {}}, "/b": {"get": {}}},
        "components": {
            "schemas": {
                "Gone": {},
                "Thing": {
                    "properties": {
                        "kept": {"type": "string", "maxLength": 10},
                        "dropped": {"type": "string"},
                        "kind": {"enum": ["x", "y"]},
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["kept"],
                },
            }
        },
    }
    new = {
        "paths": {"/a": {"get": {}}},
        "components": {
            "schemas": {
                "Thing": {
                    "properties": {
                        "kept": {"type": "string", "maxLength": 5},
                        "kind": {"enum": ["x"]},
                        "items": {"type": "array", "items": {"type": "string", "maxLength": 3}},
                        "extra": {"type": "string"},
                    },
                    "required": ["kept", "extra"],
                },
            }
        },
    }

    problems = breaking_changes(old, new)

    assert "path /b removed" in problems
    assert "POST /a removed" in problems
    assert "schema Gone removed" in problems
    assert "Thing: property 'dropped' removed" in problems
    assert "Thing: 'extra' is newly required" in problems
    assert "Thing.kind: enum value 'y' removed" in problems
    assert "Thing.kept: maxLength 10 -> 5" in problems
    assert "Thing.items[]: maxLength None -> 3" in problems
    # Additions alone are fine.
    assert breaking_changes(new, new) == []
