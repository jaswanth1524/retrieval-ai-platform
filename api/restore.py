"""``POST /import``: restore a ``GET /export`` zip into this instance.

Each stored original is re-ingested (same chunker, same embedding model as everything
else on this server) with its tags and upload time, and the feedback rows are added
back. Vectors are never imported: a backup is meant to survive a model change.

The zip is untrusted input. Nothing is ever extracted to disk; only members named
exactly ``manifest.json``, ``feedback.jsonl`` and ``originals/<a valid upload name>``
are read, each through a capped read, so a zip bomb or a ``../`` name gets nowhere.
"""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import IO, Any

from fastapi import HTTPException

from api.documents import DocumentError, normalize_tags, validate_upload_filename
from api.export import MANIFEST_VERSION
from api.feedback import Feedback, FeedbackStore
from api.schemas import DocumentJobAcceptedResponse

logger = logging.getLogger(__name__)

MAX_MEMBERS = 10_000
_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
_FEEDBACK_MAX_BYTES = 256 * 1024 * 1024


class BackupImportError(ValueError):
    """The upload isn't a usable DocRAG backup (a 400)."""


@dataclass
class ImportOutcome:
    jobs: list[DocumentJobAcceptedResponse] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing_originals: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    feedback_imported: int = 0
    feedback_skipped: int = 0


Enqueue = Callable[[str, bytes, float | None, tuple[str, ...]], DocumentJobAcceptedResponse]


def _read_capped(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes | None:
    """The member's bytes, or None when it decompresses past ``limit`` (the header's
    sizes aren't trusted)."""

    with archive.open(info) as handle:
        data = handle.read(limit + 1)
    return None if len(data) > limit else data


def restore_backup(
    upload: IO[bytes],
    *,
    existing_filenames: set[str],
    max_original_bytes: int,
    has_room: Callable[[int], bool],
    enqueue: Enqueue,
    feedback_store: FeedbackStore | None,
) -> ImportOutcome:
    try:
        archive = zipfile.ZipFile(upload)
    except (zipfile.BadZipFile, OSError) as exc:
        raise BackupImportError("That file isn't a zip archive.") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise BackupImportError(f"The archive has more than {MAX_MEMBERS} entries.")
        members = {info.filename: info for info in infos}
        manifest = _manifest(archive, members)
        outcome = ImportOutcome()
        for document in manifest["documents"]:
            _restore_document(
                archive,
                members,
                document,
                outcome,
                existing_filenames=existing_filenames,
                max_original_bytes=max_original_bytes,
                has_room=has_room,
                enqueue=enqueue,
            )
        if feedback_store is not None and "feedback.jsonl" in members:
            _restore_feedback(archive, members["feedback.jsonl"], feedback_store, outcome)
    return outcome


def _manifest(archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo]) -> dict[str, Any]:
    info = members.get("manifest.json")
    if info is None:
        raise BackupImportError("The archive has no manifest.json; is it a DocRAG backup?")
    raw = _read_capped(archive, info, _MANIFEST_MAX_BYTES)
    try:
        manifest = json.loads(raw) if raw is not None else None
    except ValueError:
        manifest = None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("documents"), list):
        raise BackupImportError("manifest.json is not a DocRAG backup manifest.")
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise BackupImportError(
            f"Unsupported backup format version {manifest.get('manifest_version')!r}; "
            f"this server reads version {MANIFEST_VERSION}."
        )
    return manifest


def _restore_document(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    document: object,
    outcome: ImportOutcome,
    *,
    existing_filenames: set[str],
    max_original_bytes: int,
    has_room: Callable[[int], bool],
    enqueue: Enqueue,
) -> None:
    if not isinstance(document, dict) or not isinstance(document.get("filename"), str):
        return
    raw_name = document["filename"]
    try:
        filename = validate_upload_filename(raw_name)
    except DocumentError:
        outcome.rejected.append(raw_name)
        return
    if filename != raw_name:
        outcome.rejected.append(raw_name)
        return
    if filename in existing_filenames:
        # Never overwrite: restoring into a live instance keeps what's there.
        outcome.skipped.append(filename)
        return
    info = members.get(f"originals/{filename}")
    if document.get("original") != f"originals/{filename}" or info is None:
        outcome.missing_originals.append(filename)
        return
    if not has_room(min(info.file_size, max_original_bytes)):
        outcome.deferred.append(filename)
        return
    content = _read_capped(archive, info, max_original_bytes)
    if content is None:
        outcome.rejected.append(filename)
        return
    uploaded_at = document.get("uploaded_at")
    raw_tags = document.get("tags")
    tags = (
        normalize_tags([tag for tag in raw_tags if isinstance(tag, str)])
        if isinstance(raw_tags, list)
        else ()
    )
    try:
        outcome.jobs.append(
            enqueue(
                filename,
                content,
                float(uploaded_at) if isinstance(uploaded_at, int | float) else None,
                tags,
            )
        )
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        outcome.deferred.append(filename)
    except DocumentError:
        outcome.rejected.append(filename)


def _restore_feedback(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    store: FeedbackStore,
    outcome: ImportOutcome,
) -> None:
    raw = _read_capped(archive, info, _FEEDBACK_MAX_BYTES)
    if raw is None:
        logger.warning("Backup feedback.jsonl is over %d bytes; not restored.", _FEEDBACK_MAX_BYTES)
        return
    rows: list[Feedback] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        row = _feedback_row(line)
        if row is None:
            outcome.feedback_skipped += 1
        else:
            rows.append(row)
    added = store.import_rows(rows)
    outcome.feedback_imported = added
    outcome.feedback_skipped += len(rows) - added


def _feedback_row(line: str) -> Feedback | None:
    if not line.strip():
        return None
    try:
        data = json.loads(line)
        if data["rating"] not in ("up", "down") or not isinstance(data["question"], str):
            return None
        cited = data.get("cited_filenames") or []
        return Feedback(
            id=str(data["id"]),
            trace_id=data.get("trace_id") if isinstance(data.get("trace_id"), str) else None,
            question=data["question"],
            answer_excerpt=str(data.get("answer_excerpt", "")),
            cited_filenames=[name for name in cited if isinstance(name, str)],
            rating=data["rating"],
            citation_source_number=data.get("citation_source_number")
            if isinstance(data.get("citation_source_number"), int)
            else None,
            created_at=float(data["created_at"]),
        )
    except (ValueError, KeyError, TypeError):
        return None
