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

import dataclasses
import json
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
from api.repository import VectorRepository
from api.restore import BackupCheck, verify_backup
from api.settings import AppSettings
from api.workspace_target import WorkspaceTarget
from api.workspaces import workspace_of

logger = logging.getLogger(__name__)

BACKUP_CHECK_SECONDS = 3600.0
BACKUP_NAME_RE = re.compile(r"^docrag-\d{8}-\d{6}\.zip$")
# A backup being written; one left behind by a killed process is removed later.
_PART_RE = re.compile(r"^\.docrag-\d{8}-\d{6}\.zip\.part$")


def check_path(backup: Path) -> Path:
    """Where ``backup``'s last check is kept: a hidden file beside it, which neither
    BACKUP_NAME_RE (listing, pruning) nor _PART_RE (abandoned parts) matches."""

    return backup.with_name(f".{backup.name}.check.json")


def save_check(backup: Path, check: BackupCheck) -> None:
    """Keep ``check`` beside ``backup`` so a restart still lists it as checked. Keyed
    on the file's size and mtime: a backup replaced since never reads as checked. A
    failure is logged; it never fails the backup."""

    try:
        stat = backup.stat()
        record = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "check": dataclasses.asdict(check),
        }
        target = check_path(backup)
        partial = target.with_name(target.name + ".tmp")
        partial.write_text(json.dumps(record), encoding="utf-8")
        partial.replace(target)
    except OSError:
        logger.warning("Could not keep the check of %s.", backup, exc_info=True)


def load_check(backup: Path) -> BackupCheck | None:
    """The kept check of ``backup``, or None (none kept, unreadable, or the file
    changed since it was checked)."""

    try:
        record = json.loads(check_path(backup).read_text(encoding="utf-8"))
        stat = backup.stat()
        if record["size"] != stat.st_size or record["mtime_ns"] != stat.st_mtime_ns:
            return None
        fields = record["check"]
        return BackupCheck(
            documents=int(fields["documents"]),
            originals=int(fields["originals"]),
            originals_missing=int(fields["originals_missing"]),
            versions=int(fields["versions"]),
            feedback=bool(fields["feedback"]),
            problems=[str(problem) for problem in fields["problems"]],
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


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
            path for path in directory.iterdir() if BACKUP_NAME_RE.fullmatch(path.name) is not None
        )
    except OSError:
        return []


def _modified_at(backups: list[Path]) -> float | None:
    """The newest backup's modification time; None when there is none (or it went
    between the listing and this read)."""

    for path in reversed(backups):
        try:
            return path.stat().st_mtime
        except OSError:
            continue
    return None


class ScheduledBackup(BackgroundLoop):
    """The background loop; ``run_once`` is one pass (tests call it directly)."""

    def __init__(
        self,
        settings: AppSettings,
        repository: VectorRepository,
        targets: Callable[[], Sequence[WorkspaceTarget]],
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
        self._part_max_age = float(interval)
        # The newest attempt, and each workspace's own last attempt.
        self.last: BackupStatus | None = None
        self.by_workspace: dict[str, BackupStatus] = {}
        # What checking each backup found, by file path (verify_backup); also kept on
        # disk beside it (save_check), which ``check_for`` falls back to.
        self.checks: dict[str, BackupCheck] = {}
        # Workspaces asked to be backed up on the next pass, whatever their newest.
        self._forced: set[str] = set()
        self._forced_lock = threading.Lock()

    def directory_of(self, settings: AppSettings) -> Path:
        """Where ``settings``' workspace's backups are written."""

        return self._root / (workspace_of(settings) or "default")

    def request_now(self, settings: AppSettings) -> None:
        """Back ``settings``' workspace up on a pass started now (POST /backups/run)."""

        with self._forced_lock:
            self._forced.add(workspace_of(settings) or "default")
        self.wake()

    def record_check(self, backup: Path, check: BackupCheck) -> None:
        """Remember what checking ``backup`` found, here and beside the file."""

        self.checks[str(backup)] = check
        save_check(backup, check)

    def check_for(self, backup: Path) -> BackupCheck | None:
        """The last check of ``backup``: this process's, else the one kept on disk
        (a restart forgot the first)."""

        check = self.checks.get(str(backup))
        if check is None:
            check = load_check(backup)
            if check is not None:
                self.checks[str(backup)] = check
        return check

    def failing(self) -> list[str]:
        """The workspaces whose last backup attempt failed: one failing every pass
        would hide behind another's success in ``last``."""

        return sorted(name for name, status in self.by_workspace.items() if not status.ok)

    def _note(self, status: BackupStatus) -> None:
        self.last = status
        self.by_workspace[status.workspace] = status

    def run_pass(self) -> list[Path]:
        return self.run_once()

    def run_once(self) -> list[Path]:
        """Back up each workspace whose newest backup is older than the interval; the
        files written."""

        written: list[Path] = []
        for target in self._targets():
            name = workspace_of(target.settings) or "default"
            directory = self._root / name
            self._remove_abandoned_parts(directory)
            with self._forced_lock:
                forced = name in self._forced
            newest = _modified_at(backups_in(directory))
            if not forced and newest is not None and self._clock() - newest < self._every:
                continue
            if not self._export_lock.acquire(blocking=False):
                logger.info("A /export is running; backing up %r next pass.", name)
                return written
            with self._forced_lock:
                self._forced.discard(name)
            try:
                written.append(self._write(target, directory, name))
            except Exception as exc:  # noqa: BLE001 - one workspace's failure is reported
                logger.exception("Could not back up workspace %r.", name)
                self._note(BackupStatus(self._clock(), name, False, error=str(exc)[:200]))
            finally:
                self._export_lock.release()
            self._prune(directory)
        return written

    def _write(self, target: WorkspaceTarget, directory: Path, name: str) -> Path:
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
        # Read back as a restore would: a backup that can't be is a failure now, not
        # when it is needed.
        with final.open("rb") as written:
            check = verify_backup(written)
        self.record_check(final, check)
        if not check.ok:
            logger.warning("The backup %s did not check out: %s", final, "; ".join(check.problems))
        self._note(
            BackupStatus(
                self._clock(),
                name,
                check.ok,
                file=final.name,
                error=None if check.ok else check.problems[0][:200],
            )
        )
        return final

    def _remove_abandoned_parts(self, directory: Path) -> None:
        """Remove partial backups a killed process left (its ``except`` never ran):
        they never match a backup's name, so pruning never removed them. Only ones
        older than a check interval — a younger one may still be being written."""

        try:
            parts = [path for path in directory.iterdir() if _PART_RE.fullmatch(path.name)]
        except OSError:
            return
        for part in parts:
            try:
                if self._clock() - part.stat().st_mtime >= self._part_max_age:
                    part.unlink()
                    logger.info("Removed the unfinished backup %s.", part)
            except OSError:
                continue

    def _prune(self, directory: Path) -> None:
        for old in backups_in(directory)[: -self._keep]:
            try:
                old.unlink()
            except OSError:
                logger.warning("Could not remove the old backup %s.", old, exc_info=True)
                continue
            self.checks.pop(str(old), None)
            check_path(old).unlink(missing_ok=True)
