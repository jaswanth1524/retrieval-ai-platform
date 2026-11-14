"""The shipped dashboard and alert rules name only metrics the API exports."""

from __future__ import annotations

import json
import re
from pathlib import Path

from prometheus_client import REGISTRY

import api.metrics  # noqa: F401 - registers every metric

ROOT = Path(__file__).resolve().parent.parent / "docs" / "observability"
_METRIC_RE = re.compile(r"\bdocrag_[a-z_]+")
# Series a histogram or counter exports beside its own name.
_SUFFIXES = ("_bucket", "_count", "_sum", "_total", "_created")


def _exported() -> set[str]:
    names: set[str] = set()
    for family in REGISTRY.collect():
        names.add(family.name)
        names.update(sample.name for sample in family.samples)
    return names


def _base(name: str) -> str:
    for suffix in _SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _expressions() -> list[str]:
    # The alert file is read as text (no YAML dependency): metric names only appear in
    # its expressions and descriptions.
    dashboard = json.loads((ROOT / "grafana-dashboard.json").read_text(encoding="utf-8"))
    exprs = [t["expr"] for panel in dashboard["panels"] for t in panel["targets"]]
    return [*exprs, (ROOT / "alerts.yml").read_text(encoding="utf-8")]


def test_every_metric_the_dashboard_and_alerts_name_is_exported() -> None:
    exported = _exported()
    known = exported | {_base(name) for name in exported}
    named = {name for expr in _expressions() for name in _METRIC_RE.findall(expr)}
    assert named, "no metric found in the dashboard or alerts"
    missing = sorted(name for name in named if name not in known and _base(name) not in known)
    assert missing == []
