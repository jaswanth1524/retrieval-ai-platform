"""Risky configurations, found (``GET /admin/doctor``, ``docrag doctor``). A report only.

Each check looks at settings, the machine, or the index, and says what is at risk and
what to change: backups that hold nothing to restore, or that sit on the disk they
back up; a disk nearly full; a memory budget the models and the ingest backlog can
exceed; an open server; settings that can't work together. A finding is advice — the
server runs either way.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from api.settings import AppSettings
from api.storage import enabled_store_directories

# What the models hold resident, measured (CLAUDE.md: the int8 reranker peaks around
# 1.4 GB, fp32 2.7 GB; the embedders add ~0.5 GB).
_MODELS_BYTES_INT8 = 2 * 1024**3
_MODELS_BYTES_FP32 = 3_300 * 1024**2
# A parsed document and its vectors take several times its upload size.
_INGEST_EXPANSION = 4
_LOW_DISK_BYTES = 1024**3
_LOW_DISK_SHARE = 0.05
# A document past this many passages ties up an ingest worker for a long time and makes
# a summary sample it.
LARGE_DOCUMENT_CHUNKS = 20_000


@dataclass(frozen=True)
class Finding:
    # "warning" (something will go wrong, or is at risk) or "note" (worth knowing).
    level: str
    # The setting or resource it is about.
    subject: str
    message: str


def _memory_limit() -> int | None:
    """The cgroup's memory limit (a container), else the machine's memory; None when
    neither can be read."""

    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 1 << 60:
            return int(raw)
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _device(path: Path) -> int | None:
    for candidate in (path, *path.parents):
        try:
            return candidate.stat().st_dev
        except OSError:
            continue
    return None


def diagnose(
    settings: AppSettings,
    *,
    chunk_counts: dict[str, int] | None = None,
    memory_limit: Callable[[], int | None] = _memory_limit,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
    environ: dict[str, str] | None = None,
    cpu_count: int | None = None,
) -> list[Finding]:
    """Every finding for ``settings`` on this machine; ``chunk_counts`` (document ->
    passages) adds the index's."""

    env = os.environ if environ is None else environ
    found: list[Finding] = []

    if settings.backup_dir and not settings.raw_document_dir:
        found.append(
            Finding(
                "warning",
                "BACKUP_DIR",
                "Backups are written but hold no documents: originals are kept only with "
                "RAW_DOCUMENT_DIR, so a restore would have nothing to index.",
            )
        )
    if settings.backup_dir and settings.raw_document_dir:
        backup_device = _device(Path(settings.backup_dir))
        data_device = _device(Path(settings.raw_document_dir))
        if backup_device is not None and backup_device == data_device:
            found.append(
                Finding(
                    "note",
                    "BACKUP_DIR",
                    "Backups are on the same disk (or volume) as the originals they back "
                    "up: a lost disk loses both. Copy them elsewhere too.",
                )
            )

    for setting, directory in enabled_store_directories(settings):
        try:
            usage = disk_usage(str(directory))
        except OSError:
            continue
        if usage.free < _LOW_DISK_BYTES or usage.free < usage.total * _LOW_DISK_SHARE:
            found.append(
                Finding(
                    "warning",
                    setting,
                    f"Only {usage.free // 1024**2} MB free where {setting} writes "
                    f"({directory}): uploads, backups and the job store fail when it fills.",
                )
            )

    limit = memory_limit()
    if limit:
        models = (
            _MODELS_BYTES_FP32 if not settings.reranker_onnx_file.strip() else _MODELS_BYTES_INT8
        )
        ingest = int(settings.ingest_max_pending_bytes) * _INGEST_EXPANSION
        needed = models + ingest
        if needed > limit * 0.9:
            found.append(
                Finding(
                    "warning",
                    "INGEST_MAX_PENDING_BYTES",
                    f"The models (~{models // 1024**2} MB) and a full ingest backlog "
                    f"(~{ingest // 1024**2} MB once parsed) can need ~{needed // 1024**2} MB, "
                    f"against {limit // 1024**2} MB available: the process may be killed "
                    "for memory. Lower INGEST_MAX_PENDING_BYTES or raise the limit.",
                )
            )
    if not int(settings.parse_memory_limit_mb):
        found.append(
            Finding(
                "note",
                "PARSE_MEMORY_LIMIT_MB",
                "Parsing has no memory cap (0): a pathological file can take the server's "
                "memory with it. On Linux, set PARSE_MEMORY_LIMIT_MB (e.g. 1500).",
            )
        )

    if not settings.api_key:
        found.append(
            Finding(
                "note",
                "API_KEY",
                "No API key: anyone who can reach the port can read, upload and delete. "
                "Fine on 127.0.0.1; set API_KEY before exposing it.",
            )
        )
    elif not int(settings.rate_limit_per_minute):
        found.append(
            Finding(
                "note",
                "RATE_LIMIT_PER_MINUTE",
                "With a key and no rate limit (0), a wrong key can be guessed at full speed.",
            )
        )

    workers = env.get("WEB_CONCURRENCY", "").strip()
    if workers.isdigit() and int(workers) > 1:
        found.append(
            Finding(
                "warning",
                "WEB_CONCURRENCY",
                f"{workers} server processes: the caches, locks, question slots and job "
                "queue are per process, so they wouldn't see each other's work. Run one.",
            )
        )
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    if int(settings.ingest_workers) > cpus:
        found.append(
            Finding(
                "note",
                "INGEST_WORKERS",
                f"{settings.ingest_workers} ingest workers on {cpus} CPUs: they wait on "
                "each other, and each holds a document in memory.",
            )
        )

    for name, count in sorted((chunk_counts or {}).items()):
        if count > LARGE_DOCUMENT_CHUNKS:
            found.append(
                Finding(
                    "note",
                    name,
                    f"{count} passages: a re-index ties up an ingest worker for a long time, "
                    "and a summary reads a sample of it. Splitting it may serve better.",
                )
            )
    return found
