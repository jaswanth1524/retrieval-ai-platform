"""Skipped dependency advisories stay justified.

``.github/ci/pip-audit-ignore.txt`` lists advisories the audit skips, each with the
package versions it applies to. Once the lockfile moves past those, the skip has no
reason left and could hide a new advisory with the same ID, so it must go.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORE_FILE = ROOT / ".github" / "ci" / "pip-audit-ignore.txt"


def _release(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version.split("+")[0])[:3])


def _entries() -> list[tuple[str, str, str]]:
    """(advisory, package, version it is fixed in) — entries read ``ID package<fixed``."""

    entries = []
    for line in IGNORE_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        advisory, requirement = line.split()
        package, below = requirement.split("<")
        entries.append((advisory, package, below))
    return entries


def _locked_versions() -> dict[str, str]:
    with (ROOT / "uv.lock").open("rb") as handle:
        lock = tomllib.load(handle)
    return {package["name"]: package["version"] for package in lock["package"]}


def test_each_skipped_advisory_still_applies_to_the_locked_version() -> None:
    entries = _entries()
    assert entries, "an empty ignore list: drop the file and its readers instead"
    locked = _locked_versions()
    for advisory, package, below in entries:
        assert re.fullmatch(r"(CVE-\d{4}-\d+|GHSA(-[0-9a-z]{4}){3}|PYSEC-\d{4}-\d+)", advisory)
        assert package in locked, f"{advisory}: {package} is not locked"
        version = locked[package]
        assert _release(version) < _release(below), (
            f"{advisory} is skipped for {package}<{below}, but the lockfile has "
            f"{package} {version}: remove it from {IGNORE_FILE.name}"
        )


def test_ci_and_make_check_read_the_same_list() -> None:
    reader = ".github/ci/pip-audit-ignore.txt"
    assert reader in (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert reader in (ROOT / "Makefile").read_text(encoding="utf-8")
