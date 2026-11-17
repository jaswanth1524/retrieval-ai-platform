"""``docs/settings.md`` is generated from ``AppSettings`` and kept current."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from api.settings import AppSettings
from api.settings_docs import DOCS_PATH, SECTIONS, field_comments, main, render


def test_every_setting_is_in_exactly_one_section() -> None:
    placed = Counter(name for _, _, names in SECTIONS for name in names)

    assert [name for name, count in placed.items() if count > 1] == []
    settings = set(AppSettings.model_fields)
    assert set(placed) == settings, (
        f"add to api/settings_docs.SECTIONS: {sorted(settings - set(placed))}; "
        f"no longer settings: {sorted(set(placed) - settings)}"
    )


def test_every_setting_has_a_comment_of_its_own() -> None:
    comments = field_comments()

    missing = sorted(name for name in AppSettings.model_fields if not comments.get(name))

    assert missing == [], f"add a comment directly above these fields in api/settings.py: {missing}"


def test_the_settings_reference_is_up_to_date() -> None:
    assert DOCS_PATH.read_text(encoding="utf-8") == render(), (
        "docs/settings.md is out of date: run `make settings-docs`"
    )


def test_descriptions_are_safe_markdown() -> None:
    text = render()

    # A comment's "<name>" would vanish as an HTML tag; "*.py" could start emphasis.
    assert "__ws_&lt;name&gt;" in text
    assert '"\\*.py"' in text
    assert "`QDRANT_URL`" in text and "`LOG_FORMAT`" in text


def test_main_writes_the_reference_to_a_given_path(tmp_path: Path) -> None:
    target = tmp_path / "settings.md"

    assert main([str(target)]) == 0

    assert target.read_text(encoding="utf-8") == render()
