"""``POST /import``: restore a ``GET /export`` zip into this instance.

Each stored original is re-ingested (same chunker, same embedding model as everything
else on this server) with its tags and upload time, and the feedback rows are added
back. Vectors are never imported: a backup is meant to survive a model change.

The zip is untrusted input. Nothing is ever extracted to disk; only members named
exactly ``manifest.json``, ``feedback.jsonl`` and ``originals/<a valid upload name>``
are read, each through a capped read, so a zip bomb or a ``../`` name gets nowhere.
"""

from __future__ import annotations

import io
import json
import logging
import math
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import IO, Any

from fastapi import HTTPException

from api.documents import DocumentError, normalize_tags, validate_upload_filename
from api.export import MANIFEST_VERSION
from api.feedback import Feedback, FeedbackStore
from api.schemas import (
    FEEDBACK_CITATION_NUMBER_MAX,
    FEEDBACK_CITED_FILENAMES_MAX,
    FEEDBACK_EXCERPT_MAX_CHARS,
    MAX_QUESTION_CHARS,
    REQUEST_FILENAME_MAX_CHARS,
    REQUEST_TRACE_ID_MAX_CHARS,
    DocumentJobAcceptedResponse,
)

logger = logging.getLogger(__name__)

MAX_MEMBERS = 10_000
_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
_FEEDBACK_MAX_BYTES = 256 * 1024 * 1024
# A feedback row is a few KB at most (the request caps below); a longer line is junk.
_FEEDBACK_LINE_MAX_CHARS = 64 * 1024
_FEEDBACK_BATCH_ROWS = 1000
# The same limits POST /feedback enforces, so a restored row is one the API could store.
_FEEDBACK_QUESTION_MAX = MAX_QUESTION_CHARS
_FEEDBACK_EXCERPT_MAX = FEEDBACK_EXCERPT_MAX_CHARS
_FEEDBACK_CITED_MAX = FEEDBACK_CITED_FILENAMES_MAX
_FEEDBACK_FILENAME_MAX = REQUEST_FILENAME_MAX_CHARS
_FEEDBACK_ID_MAX = REQUEST_TRACE_ID_MAX_CHARS


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


# What reading one damaged member raises: a CRC mismatch or truncation (BadZipFile,
# EOFError, zlib.error), an encrypted member (RuntimeError), an unsupported compression
# method (NotImplementedError). The member is skipped; the rest of the import goes on —
# documents before it are already queued, and a 500 hid their job ids.
_UNREADABLE_MEMBER = (
    zipfile.BadZipFile,
    zlib.error,
    EOFError,
    RuntimeError,
    NotImplementedError,
    OSError,
)


class _UnreadableMemberError(Exception):
    pass


def _read_capped(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes | None:
    """The member's bytes, or None when it decompresses past ``limit`` (the header's
    sizes aren't trusted). ``_UnreadableMemberError`` when it can't be read at all."""

    try:
        with archive.open(info) as handle:
            data = handle.read(limit + 1)
    except _UNREADABLE_MEMBER as exc:
        logger.warning("Backup member %r could not be read: %s", info.filename, exc)
        raise _UnreadableMemberError(info.filename) from exc
    return None if len(data) > limit else data


def restore_backup(
    upload: IO[bytes],
    *,
    existing_filenames: set[str],
    max_original_bytes: int,
    has_room: Callable[[int], bool],
    enqueue: Enqueue,
    feedback_store: FeedbackStore | None,
    workspace: str = "",
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
            try:
                _restore_feedback(
                    archive, members["feedback.jsonl"], feedback_store, outcome, workspace
                )
            except _UNREADABLE_MEMBER as exc:
                # The rows before the damage are stored; the member is reported.
                logger.warning("Backup feedback.jsonl could not be read to the end: %s", exc)
                outcome.rejected.append("feedback.jsonl")
    return outcome


def _manifest(archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo]) -> dict[str, Any]:
    info = members.get("manifest.json")
    if info is None:
        raise BackupImportError("The archive has no manifest.json; is it a DocRAG backup?")
    try:
        raw = _read_capped(archive, info, _MANIFEST_MAX_BYTES)
    except _UnreadableMemberError:
        raise BackupImportError("manifest.json in the archive is damaged or encrypted.") from None
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
    raw_tags = document.get("tags")
    try:
        tags = (
            normalize_tags([tag for tag in raw_tags if isinstance(tag, str)])
            if isinstance(raw_tags, list)
            else ()
        )
    except DocumentError:
        # Too many or too long: this document is rejected, not the whole import —
        # the documents before it are already queued.
        outcome.rejected.append(filename)
        return
    if not has_room(min(info.file_size, max_original_bytes)):
        outcome.deferred.append(filename)
        return
    try:
        content = _read_capped(archive, info, max_original_bytes)
    except _UnreadableMemberError:
        content = None
    if content is None:
        outcome.rejected.append(filename)
        return
    try:
        outcome.jobs.append(
            enqueue(filename, content, _uploaded_at(document.get("uploaded_at")), tags)
        )
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        outcome.deferred.append(filename)
    except DocumentError:
        outcome.rejected.append(filename)


def _uploaded_at(value: object) -> float | None:
    """A manifest's ``uploaded_at``, if it is a real timestamp (JSON allows NaN, and
    ``true`` is an int to Python; either would end up in every chunk's payload)."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _restore_feedback(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    store: FeedbackStore,
    outcome: ImportOutcome,
    workspace: str,
) -> None:
    """Add the backup's feedback rows to ``workspace``, read line by line and stored in
    batches.

    Reading the whole member and then decoding it held the file twice (up to 512 MiB);
    this holds one line and one batch.
    """

    batch: list[Feedback] = []

    def flush() -> None:
        added = store.import_rows(batch, workspace)
        outcome.feedback_imported += added
        outcome.feedback_skipped += len(batch) - added
        batch.clear()

    try:
        _read_feedback_lines(archive, info, batch, flush, outcome)
    except _UNREADABLE_MEMBER:
        if batch:
            flush()  # keep the rows read before the damage
        raise
    if batch:
        flush()


def _read_feedback_lines(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    batch: list[Feedback],
    flush: Callable[[], None],
    outcome: ImportOutcome,
) -> None:
    consumed = 0
    with (
        archive.open(info) as handle,
        # Only "\n" ends a row: a JSON string may hold a raw U+2028, which splitlines()
        # treated as a line break and so cut the row in two.
        io.TextIOWrapper(handle, encoding="utf-8", errors="replace", newline="\n") as text,
    ):
        while True:
            line = text.readline(_FEEDBACK_LINE_MAX_CHARS)
            if not line:
                break
            consumed += len(line)
            too_long = not line.endswith("\n") and len(line) >= _FEEDBACK_LINE_MAX_CHARS
            while too_long:
                rest = text.readline(_FEEDBACK_LINE_MAX_CHARS)
                consumed += len(rest)
                if not rest or rest.endswith("\n") or consumed > _FEEDBACK_MAX_BYTES:
                    break
            if consumed > _FEEDBACK_MAX_BYTES:
                logger.warning(
                    "Backup feedback.jsonl is over %d bytes; restored only the rows before it.",
                    _FEEDBACK_MAX_BYTES,
                )
                break
            row = None if too_long else _feedback_row(line)
            if row is None:
                if line.strip():
                    outcome.feedback_skipped += 1
                continue
            batch.append(row)
            if len(batch) >= _FEEDBACK_BATCH_ROWS:
                flush()


def _feedback_row(line: str) -> Feedback | None:
    """One backup line as a feedback row, or None when it isn't one ``POST /feedback``
    could have stored (a NaN timestamp used to fail the whole import in sqlite)."""

    if not line.strip():
        return None
    try:
        data = json.loads(line)
        if not isinstance(data, dict) or data.get("rating") not in ("up", "down"):
            return None
        question = data["question"]
        excerpt = str(data.get("answer_excerpt", ""))
        row_id = str(data["id"])
        trace_id = data.get("trace_id")
        cited = data.get("cited_filenames") or []
        number = data.get("citation_source_number")
        created_at = _uploaded_at(data.get("created_at"))
        if (
            not isinstance(question, str)
            or not 0 < len(question) <= _FEEDBACK_QUESTION_MAX
            or len(excerpt) > _FEEDBACK_EXCERPT_MAX
            or not 0 < len(row_id) <= _FEEDBACK_ID_MAX
            or (trace_id is not None and not isinstance(trace_id, str))
            or (isinstance(trace_id, str) and len(trace_id) > _FEEDBACK_ID_MAX)
            or not isinstance(cited, list)
            or len(cited) > _FEEDBACK_CITED_MAX
            or (number is not None and (isinstance(number, bool) or not isinstance(number, int)))
            or (isinstance(number, int) and not 1 <= number <= FEEDBACK_CITATION_NUMBER_MAX)
            or created_at is None
        ):
            return None
        return Feedback(
            id=row_id,
            trace_id=trace_id,
            question=question,
            answer_excerpt=excerpt,
            cited_filenames=[
                name
                for name in cited
                if isinstance(name, str) and len(name) <= _FEEDBACK_FILENAME_MAX
            ],
            rating=data["rating"],
            citation_source_number=number,
            created_at=created_at,
        )
    except (ValueError, KeyError, TypeError):
        return None
