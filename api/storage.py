"""Startup check that every enabled on-disk store can actually be written."""

from __future__ import annotations

import tempfile
from pathlib import Path

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
