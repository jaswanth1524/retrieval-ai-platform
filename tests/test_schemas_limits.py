"""Limits the frontend mirrors by hand must stay equal to the API's."""

from __future__ import annotations

import re
from pathlib import Path

from api.schemas import MAX_QUESTION_CHARS

LIMITS_TS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "utils" / "limits.ts"


def test_the_frontend_question_limit_matches_the_api() -> None:
    match = re.search(r"export const MAX_QUESTION_CHARS = (\d+);", LIMITS_TS.read_text())
    assert match is not None
    assert int(match.group(1)) == MAX_QUESTION_CHARS
