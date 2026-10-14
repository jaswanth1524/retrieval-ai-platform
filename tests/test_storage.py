from __future__ import annotations

import os
from pathlib import Path

import pytest

from api.storage import StorageNotWritableError, check_writable_storage, enabled_store_directories
from tests.factories import make_test_settings


def test_only_enabled_stores_are_checked(tmp_path: Path) -> None:
    assert enabled_store_directories(make_test_settings()) == []

    settings = make_test_settings(
        raw_document_dir=str(tmp_path / "raw"),
        job_store_backend="sqlite",
        job_store_path=str(tmp_path / "jobs" / "jobs.db"),
        feedback_enabled=True,
        feedback_store_path=str(tmp_path / "feedback" / "f.db"),
    )
    check_writable_storage(settings)

    assert [name for name, _ in enabled_store_directories(settings)] == [
        "RAW_DOCUMENT_DIR",
        "JOB_STORE_PATH",
        "FEEDBACK_STORE_PATH",
    ]
    for directory in ("raw", "jobs", "feedback"):
        assert (tmp_path / directory).is_dir()
        assert list((tmp_path / directory).iterdir()) == []  # the probe file is gone


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only directory")
def test_an_unwritable_store_directory_fails_startup_naming_the_setting(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(StorageNotWritableError, match="RAW_DOCUMENT_DIR"):
            check_writable_storage(make_test_settings(raw_document_dir=str(locked)))
    finally:
        locked.chmod(0o700)
