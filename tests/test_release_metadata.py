"""The version, its changelog section and the pinned Qdrant stay in step.

A tag push publishes whatever these say. Each of these once disagreed unnoticed: the
release notes were empty because the changelog had no section for the version, and the
Qdrant version lives in three files no tool updates together.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    assert isinstance(version, str)
    return version


def changelog_section(changelog: str, version: str) -> str:
    """The section ``release.yml`` publishes: the lines after the heading ``## <version>``
    (alone or followed by a space), up to the next ``## `` heading — the same rule as
    its awk script."""

    found = False
    body: list[str] = []
    for line in changelog.splitlines():
        if line.startswith("## "):
            if found:
                break
            # The whole version: "## 0.2.0" must not match "## 0.2.0-rc.1".
            if line == f"## {version}" or line.startswith(f"## {version} "):
                found = True
            continue
        if found:
            body.append(line)
    return "\n".join(body)


def test_the_changelog_has_a_section_for_the_current_version() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text()

    section = changelog_section(changelog, _project_version())

    assert section.strip(), f"CHANGELOG.md needs a '## {_project_version()}' section"


def test_the_changelog_rule_matches_what_the_release_workflow_runs() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text()

    # The same match, in the workflow's own words: if one is edited, so is the other.
    assert 'if ($0 == "## " v || index($0, "## " v " ") == 1)' in workflow
    assert changelog_section("## 1.2.3 — soon\nnotes\n## 1.2.30\nother", "1.2.3") == "notes"


def test_the_frontend_package_carries_the_same_version() -> None:
    version = _project_version()
    package = json.loads((ROOT / "frontend/package.json").read_text())
    lock = json.loads((ROOT / "frontend/package-lock.json").read_text())

    assert package["version"] == version
    assert lock["version"] == version
    assert lock["packages"][""]["version"] == version


def test_the_cli_package_carries_the_same_version_and_matching_pins() -> None:
    """``cli/pyproject.toml`` is a separate distribution (``uv tool install ./cli``); its
    version and its httpx/mcp bounds must not drift from the app's."""

    with (ROOT / "cli/pyproject.toml").open("rb") as handle:
        cli = tomllib.load(handle)["project"]
    with (ROOT / "pyproject.toml").open("rb") as handle:
        app = tomllib.load(handle)["project"]

    assert cli["version"] == _project_version()
    assert cli["requires-python"] == app["requires-python"]
    httpx_pin = next(dep for dep in app["dependencies"] if dep.startswith("httpx"))
    assert httpx_pin in cli["dependencies"]
    assert cli["optional-dependencies"]["mcp"] == app["optional-dependencies"]["mcp"]
    assert set(cli["scripts"]) == {"docrag", "docrag-mcp"}


def test_the_qdrant_image_is_pinned_to_one_version_everywhere() -> None:
    pattern = re.compile(r"qdrant/qdrant:(v\d+\.\d+\.\d+)")
    files = [".env.example", "docker-compose.yml", ".github/workflows/ci.yml"]

    pins = {name: set(pattern.findall((ROOT / name).read_text())) for name in files}

    assert all(len(found) == 1 for found in pins.values()), pins
    assert len({next(iter(found)) for found in pins.values()}) == 1, pins


def test_a_version_heading_does_not_match_a_longer_version() -> None:
    changelog = "## 0.2.0-rc.1\n- rc notes\n## 0.2.0 — 2026-11-01\n- final notes\n"

    assert changelog_section(changelog, "0.2.0").strip() == "- final notes"
    assert changelog_section(changelog, "0.2.0-rc.1").strip() == "- rc notes"
