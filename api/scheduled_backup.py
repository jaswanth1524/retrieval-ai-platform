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
import shlex
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from api.audit import audit
from api.background import BackgroundLoop
from api.export import build_export
from api.feedback import FeedbackStore
from api.metrics import backup_last_ok, backup_last_success_timestamp
from api.notify import notify
from api.repository import VectorRepository
from api.restore import BackupCheck, verify_backup
from api.settings import AppSettings
from api.storage import low_on_space
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
            would_restore=int(fields.get("would_restore", 0)),
            rejected=[str(item) for item in fields.get("rejected", [])],
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
    # Indexed documents the backup holds no original of: a restore can't bring them back.
    originals_missing: int = 0


# A backup holding under half the documents of the one before it is reported as failed
# once the one before held at least this many (fewer: ordinary deletes swing the share).
DROP_ALARM_MIN_DOCUMENTS = 12


def copy_backup(settings: AppSettings, backup: Path) -> str | None:
    """Run BACKUP_COPY_COMMAND on ``backup`` (its path as the last argument, no shell):
    why the copy failed, or None (done, or no command set)."""

    command = settings.backup_copy_command.strip()
    if not command:
        return None
    try:
        argv = [*shlex.split(command), str(backup)]
    except ValueError as exc:
        return f"BACKUP_COPY_COMMAND can't be read ({exc})"
    try:
        result = subprocess.run(  # noqa: S603 - the operator's own command, no shell
            argv,
            capture_output=True,
            timeout=int(settings.backup_copy_timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            "the copy command ran past BACKUP_COPY_TIMEOUT_SECONDS "
            f"({settings.backup_copy_timeout_seconds} s)"
        )
    except OSError as exc:
        return f"the copy command could not run ({exc.strerror or exc})"
    if result.returncode != 0:
        said = (result.stderr or result.stdout).decode("utf-8", "replace").strip()[-200:]
        return f"the copy command failed (exit {result.returncode})" + (f": {said}" if said else "")
    return None


# Where a deleted workspace's backups are moved (never a workspace name: those can't
# start with a dot).
DELETED_DIRECTORY = ".deleted"


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
        if not status.ok:
            notify(
                "backup_failed",
                f"The scheduled backup of workspace '{status.workspace}' failed: "
                f"{status.error or 'see the server log'}",
            )
        backup_last_ok.labels(workspace=status.workspace).set(1 if status.ok else 0)
        if status.ok:
            backup_last_success_timestamp.labels(workspace=status.workspace).set(status.at)

    def run_pass(self) -> list[Path]:
        return self.run_once()

    def run_once(self) -> list[Path]:
        """Back up each workspace whose newest backup is older than the interval; the
        files written."""

        written: list[Path] = []
        try:
            targets = list(self._targets())
        except Exception as exc:  # noqa: BLE001 - Qdrant down: no workspace could be listed
            # Recorded, so readiness and the backup alert see a pass that backed up
            # nothing (it used to fail before any status was noted).
            logger.warning("Could not list the workspaces to back up.", exc_info=True)
            self._note(BackupStatus(self._clock(), "default", False, error=str(exc)[:200]))
            return written
        for target in targets:
            name = workspace_of(target.settings) or "default"
            directory = self._root / name
            self._remove_abandoned_parts(directory)
            with self._forced_lock:
                forced = name in self._forced
            newest = _modified_at(backups_in(directory))
            if not forced and newest is not None and self._clock() - newest < self._every:
                # Not due: the newest backup on disk is the last success (after a
                # restart the gauge had none until the next write, a day away, and
                # the staleness alert had no series to judge).
                if name not in self.by_workspace:
                    backup_last_success_timestamp.labels(workspace=name).set(newest)
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
            # Off the export lock: a copy may take minutes (an upload elsewhere), and
            # /export and the other workspaces' backups would wait on it.
            if written and written[-1].parent == directory:
                self._copy(target, name, written[-1])
            self._prune(directory)
        return written

    def _copy(self, target: WorkspaceTarget, name: str, backup: Path) -> None:
        """BACKUP_COPY_COMMAND on a backup just written that checked out; a failed copy
        turns this workspace's status into a failure."""

        status = self.by_workspace.get(name)
        if status is None or not status.ok or status.file != backup.name:
            return
        failed = copy_backup(target.settings, backup)
        if failed is not None:
            logger.warning("The backup %s: %s", backup, failed)
            self._note(dataclasses.replace(status, at=self._clock(), ok=False, error=failed))

    def _write(self, target: WorkspaceTarget, directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        low = low_on_space(target.settings, directory)
        if low is not None:
            raise OSError(low)
        # What is indexed now: a listing cached from before a wiped collection showed
        # its documents still there, and the backup (and its drop check) believed it.
        self._repository.forget_listing(target.settings)
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
            check = verify_backup(written, int(target.settings.max_upload_bytes), stored=True)
        self.record_check(final, check)
        if not check.ok:
            logger.warning("The backup %s did not check out: %s", final, "; ".join(check.problems))
        drop = self._document_drop(directory, final, check)
        if drop is not None:
            logger.warning("The backup %s: %s", final, drop)
        self._note(
            BackupStatus(
                self._clock(),
                name,
                check.ok and drop is None,
                file=final.name,
                error=(check.problems[0][:200] if not check.ok else drop),
                originals_missing=stream.originals_missing,
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

    def forget_workspace(self, name: str) -> None:
        """A deleted workspace: its last attempt no longer counts as failing (readiness),
        and its backups are set aside — kept (BACKUP_DIR/.deleted/<name>-<time>), but no
        longer where a workspace created later under the same name would list them,
        restore the old one's documents from them, or count them as its own latest."""

        self.by_workspace.pop(name, None)
        directory = self._root / (name or "default")
        if not directory.is_dir():
            return
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(self._clock()))
        aside = self._root / DELETED_DIRECTORY
        try:
            aside.mkdir(parents=True, exist_ok=True)
            directory.rename(aside / f"{name or 'default'}-{stamp}")
        except OSError:
            logger.warning(
                "Could not set aside the backups of deleted workspace %r (%s).",
                name,
                directory,
                exc_info=True,
            )

    def _document_drop(self, directory: Path, final: Path, check: BackupCheck) -> str | None:
        """Why this backup is suspect next to the one before it: it holds no documents
        where that one held some, or under half of a dozen or more (a lost index, a
        mass delete) — reported once, as a failed backup, so it is seen before pruning
        would leave only backups like it."""

        earlier = [b for b in backups_in(directory) if b != final]
        previous = self.check_for(earlier[-1]) if earlier else None
        # What a restore could bring back: documents with their original.
        before = previous.originals if previous is not None else 0
        now = check.originals
        if before == 0:
            return None
        if now == 0 or (before >= DROP_ALARM_MIN_DOCUMENTS and now * 2 < before):
            return (
                f"it holds {now} documents with their originals where the backup before "
                f"it held {before}: check that the index and originals weren't lost "
                "(docrag doctor) before older backups are pruned"
            )
        return None

    def _prune(self, directory: Path) -> None:
        backups = backups_in(directory)
        doomed = backups[: -self._keep]
        # The newest backup that passed its check stays, even past BACKUP_KEEP: with a
        # small keep, failed ones would otherwise push out the last one worth restoring.
        good = next(
            (b for b in reversed(backups) if (check := self.check_for(b)) and check.ok), None
        )
        # So does the newest that held any document: backups taken after the index was
        # lost hold none, and pruning by count alone would remove the last full one.
        holding = next(
            (
                b
                for b in reversed(backups)
                if (check := self.check_for(b)) and check.ok and check.originals > 0
            ),
            None,
        )
        for old in doomed:
            if old in (good, holding):
                continue
            try:
                old.unlink()
            except OSError:
                logger.warning("Could not remove the old backup %s.", old, exc_info=True)
                continue
            self.checks.pop(str(old), None)
            check_path(old).unlink(missing_ok=True)
