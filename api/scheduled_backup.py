"""Back the corpus up on a timer (BACKUP_DIR, off by default).

The same zip ``GET /export`` streams — manifest, stored originals, feedback — written
to ``BACKUP_DIR/<workspace>/docrag-<UTC time>.zip`` every BACKUP_INTERVAL_HOURS, for the
default workspace and every other one, keeping the newest BACKUP_KEEP of each. A file is
written under a temporary name and renamed when complete, so a backup cut off by a
restart never looks like one. ``/import`` restores any of them.

A pass checks once an hour whether a workspace's newest backup is older than the
interval, so a restart doesn't add one. It stands aside while a ``/export`` streams
(they share one lock): the next pass catches up.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from api.audit import audit
from api.background import BackgroundLoop
from api.export import build_export
from api.feedback import FeedbackStore
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.settings import AppSettings
from api.workspaces import workspace_of

logger = logging.getLogger(__name__)

BACKUP_CHECK_SECONDS = 3600.0
_NAME_RE = re.compile(r"^docrag-\d{8}-\d{6}\.zip$")


@dataclass(frozen=True)
class BackupTarget:
    """One workspace to back up: its settings and its originals store."""

    settings: AppSettings
    raw_store: RawDocumentStore | None


@dataclass(frozen=True)
class BackupStatus:
    """The last backup written or attempted (``GET /health/ready``'s ``last_backup``)."""

    at: float
    workspace: str
    ok: bool
    file: str | None = None
    error: str | None = None


def backups_in(directory: Path) -> list[Path]:
    """The finished backups in ``directory``, oldest first."""

    try:
        return sorted(
            path for path in directory.iterdir() if _NAME_RE.fullmatch(path.name) is not None
        )
    except OSError:
        return []


class ScheduledBackup(BackgroundLoop):
    """The background loop; ``run_once`` is one pass (tests call it directly)."""

    def __init__(
        self,
        settings: AppSettings,
        repository: VectorRepository,
        targets: Callable[[], Sequence[BackupTarget]],
        feedback_store: FeedbackStore | None,
        export_lock: threading.Lock,
        *,
        clock: Callable[[], float] = time.time,
        interval: float = BACKUP_CHECK_SECONDS,
    ) -> None:
        super().__init__("docrag-backup", interval, "Scheduled backup pass failed.", logger)
        self._root = Path(settings.backup_dir)
        self._every = float(settings.backup_interval_hours) * 3600.0
        self._keep = int(settings.backup_keep)
        self._repository = repository
        self._targets = targets
        self._feedback_store = feedback_store
        self._export_lock = export_lock
        self._clock = clock
        self.last: BackupStatus | None = None

    def run_pass(self) -> list[Path]:
        return self.run_once()

    def run_once(self) -> list[Path]:
        """Back up each workspace whose newest backup is older than the interval; the
        files written."""

        written: list[Path] = []
        for target in self._targets():
            name = workspace_of(target.settings) or "default"
            directory = self._root / name
            existing = backups_in(directory)
            if existing and self._clock() - existing[-1].stat().st_mtime < self._every:
                continue
            if not self._export_lock.acquire(blocking=False):
                logger.info("A /export is running; backing up %r next pass.", name)
                return written
            try:
                written.append(self._write(target, directory, name))
            except Exception as exc:  # noqa: BLE001 - one workspace's failure is reported
                logger.exception("Could not back up workspace %r.", name)
                self.last = BackupStatus(self._clock(), name, False, error=str(exc)[:200])
            finally:
                self._export_lock.release()
            self._prune(directory)
        return written

    def _write(self, target: BackupTarget, directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(self._clock()))
        final = directory / f"docrag-{stamp}.zip"
        partial = directory / f".docrag-{stamp}.zip.part"
        # With every document's kept earlier versions: a backup restores history too.
        stream = build_export(
            target.settings,
            self._repository,
            target.raw_store,
            self._feedback_store,
            include_versions=True,
        )
        try:
            with partial.open("wb") as handle:
                for block in stream:
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            partial.replace(final)
        except BaseException:
            stream.close()
            partial.unlink(missing_ok=True)
            raise
        audit(
            "corpus.export",
            workspace=workspace_of(target.settings),
            scheduled=True,
            file=final.name,
            originals_missing=stream.originals_missing,
        )
        logger.info("Backed up workspace %r to %s.", name, final)
        self.last = BackupStatus(self._clock(), name, True, file=final.name)
        return final

    def _prune(self, directory: Path) -> None:
        for old in backups_in(directory)[: -self._keep]:
            try:
                old.unlink()
            except OSError:
                logger.warning("Could not remove the old backup %s.", old, exc_info=True)
