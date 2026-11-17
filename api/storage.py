"""Startup check that every enabled on-disk store can actually be written."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import HTTPException

from api.settings import AppSettings


class StorageNotWritableError(RuntimeError):
    """An enabled store's directory can't be written by this process."""


def enabled_store_directories(settings: AppSettings) -> list[tuple[str, Path]]:
    """The directory each enabled store writes to, named by the setting that sets it."""

    directories: list[tuple[str, Path]] = []
    if settings.raw_document_dir:
        directories.append(("RAW_DOCUMENT_DIR", Path(settings.raw_document_dir)))
    if settings.job_store_backend == "sqlite":
        directories.append(("JOB_STORE_PATH", Path(settings.job_store_path).parent))
    if settings.feedback_enabled:
        directories.append(("FEEDBACK_STORE_PATH", Path(settings.feedback_store_path).parent))
    if settings.query_log_enabled:
        directories.append(("QUERY_LOG_PATH", Path(settings.query_log_path).parent))
    if settings.url_refresh_interval_minutes and settings.url_refresh_state_path:
        directories.append(("URL_REFRESH_STATE_PATH", Path(settings.url_refresh_state_path).parent))
    if settings.backup_dir:
        directories.append(("BACKUP_DIR", Path(settings.backup_dir)))
    if settings.import_spool_dir:
        directories.append(("IMPORT_SPOOL_DIR", Path(settings.import_spool_dir)))
    if settings.audit_log_path:
        directories.append(("AUDIT_LOG_PATH", Path(settings.audit_log_path).parent))
    if settings.workspace_settings_path:
        directories.append(
            ("WORKSPACE_SETTINGS_PATH", Path(settings.workspace_settings_path).parent)
        )
    if settings.trace_enabled and settings.trace_store_path:
        directories.append(("TRACE_STORE_PATH", Path(settings.trace_store_path).parent))
    return directories


def check_writable_storage(settings: AppSettings) -> None:
    """Fail startup, naming the setting, when an enabled store can't write its directory.

    The stores are created on first use, so on the read-only container filesystem an
    operator who turned one on without mounting a volume found out from a 500 on the
    first upload or rating. Stores that are off are not checked.
    """

    for name, directory in enabled_store_directories(settings):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".docrag-write-check-"):
                pass
        except OSError as exc:
            raise StorageNotWritableError(
                f"{name} needs a writable directory, and {directory} is not "
                f"({exc.strerror or exc}). In Docker, keep it under /app/data (the "
                "docrag_data volume in docker-compose.yml)."
            ) from exc


def free_bytes(directory: str | Path) -> int | None:
    """Free space on the filesystem holding ``directory`` (or its nearest existing
    parent); None when it can't be read."""

    path = Path(directory)
    for candidate in (path, *path.parents):
        try:
            return shutil.disk_usage(candidate).free
        except FileNotFoundError:
            continue
        except OSError:
            return None
    return None


def low_on_space(settings: AppSettings, directory: str | Path | None) -> str | None:
    """Why nothing more should be written to ``directory`` (MIN_FREE_BYTES), or None:
    the guard is off, the directory isn't used, or there is room."""

    floor = int(settings.min_free_bytes)
    if floor <= 0 or not directory:
        return None
    free = free_bytes(directory)
    if free is None or free >= floor:
        return None
    return (
        f"Only {free // (1024 * 1024)} MB are free where {directory} is kept, and "
        f"MIN_FREE_BYTES keeps {floor // (1024 * 1024)} MB clear. Free some space, then "
        "try again."
    )


def require_free_space(settings: AppSettings, directory: str | Path | None) -> None:
    """A 507 before anything is written to a nearly full disk (MIN_FREE_BYTES): a disk
    that fills mid-write costs a half-written backup or an original that can't be kept."""

    reason = low_on_space(settings, directory)
    if reason is not None:
        raise HTTPException(status_code=507, detail=reason)
