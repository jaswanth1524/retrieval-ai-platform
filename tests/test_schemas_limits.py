"""Limits the frontend mirrors by hand must stay equal to the API's."""

from __future__ import annotations

import re
from pathlib import Path

from api.documents import SUPPORTED_EXTENSIONS
from api.schemas import MAX_QUESTION_CHARS

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
    assert block is not None
    assert set(re.findall(r"'(\.[a-z]+)'", block.group(1))) == set(SUPPORTED_EXTENSIONS)
