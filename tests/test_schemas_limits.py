"""Limits the frontend mirrors by hand must stay equal to the API's."""

from __future__ import annotations

import re
from pathlib import Path

from api.documents import CORE_EXTENSIONS
from api.parsers.structured import CODE_EXTENSIONS
from api.schemas import MAX_QUESTION_CHARS, REQUEST_FILENAMES_MAX

UTILS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "utils"
LIMITS_TS = UTILS / "limits.ts"


def test_the_frontend_question_limit_matches_the_api() -> None:
    match = re.search(r"export const MAX_QUESTION_CHARS = (\d+);", LIMITS_TS.read_text())
    assert match is not None
    assert int(match.group(1)) == MAX_QUESTION_CHARS


def test_the_frontends_fallback_upload_types_match_the_api() -> None:
    """Used until /config arrives: a missing type refused a valid file before that."""

    source = (UTILS / "uploadValidation.ts").read_text()
    block = re.search(r"ACCEPTED_EXTENSIONS = \[(.*?)\];", source, flags=re.DOTALL)
    code = re.search(r"CODE_EXTENSIONS = \[(.*?)\];", source, flags=re.DOTALL)
    assert block is not None and code is not None
    listed = set(re.findall(r"'(\.[a-z]+)'", block.group(1)))
    assert "...CODE_EXTENSIONS" in block.group(1)
    listed |= set(re.findall(r"'(\.[a-z]+)'", code.group(1)))
    # Images need the server's OCR extra: only a server that has it offers them.
    assert listed == set(CORE_EXTENSIONS)
    assert set(CODE_EXTENSIONS) == set(re.findall(r"'(\.[a-z]+)'", code.group(1)))


def test_the_frontend_workspace_name_rule_matches_the_api() -> None:
    from api.workspaces import _NAME_RE

    match = re.search(r"export const WORKSPACE_NAME_PATTERN = /(.+)/;", LIMITS_TS.read_text())
    assert match is not None
    assert match.group(1) == _NAME_RE.pattern


def test_the_mcp_servers_workspace_name_rule_matches_the_api() -> None:
    from api.workspaces import _NAME_RE
    from cli.mcp_server import WORKSPACE_NAME_RE

    assert WORKSPACE_NAME_RE.pattern == _NAME_RE.pattern


def test_the_frontend_scope_limit_matches_the_api() -> None:
    match = re.search(r"export const MAX_SCOPE_FILENAMES = (\d+);", LIMITS_TS.read_text())
    assert match is not None
    assert int(match.group(1)) == REQUEST_FILENAMES_MAX
