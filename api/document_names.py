"""Document names: the stored form of a path parameter, near-duplicate refusal, and the
names indexed or being indexed in a workspace."""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Iterable

from api.documents import (
    DocumentError,
    DocumentNotFoundError,
    UnsupportedDocumentError,
    normalize_filename,
)
from api.ingestion import (
    ingest_sequencer,
)
from api.repository import VectorRepository
from api.settings import AppSettings
from api.workspaces import filename_of

logger = logging.getLogger(__name__)


def stored_filename(filename: str) -> str:
    """``filename`` as a path parameter naming a stored document, or a 404.

    Every stored name went through ``normalize_filename`` at upload, so a name it would
    change (padding spaces, a backslash path, control characters) was never stored. The
    raw-document store normalizes on its own, so passing such a name through let
    ``DELETE /documents/%20a.pdf`` remove ``a.pdf``'s original while holding a lock on,
    and deleting the points of, a different name.
    """

    try:
        name = normalize_filename(filename)
    except UnsupportedDocumentError:
        name = ""
    if name != filename:
        raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
    return name


class FilenameConflictError(DocumentError):
    """Another document's name differs from this one only by case or Unicode form."""


def _folded(filename: str) -> str:
    return unicodedata.normalize("NFC", filename).casefold()


def refuse_near_duplicate_name(filename: str, existing: Iterable[str]) -> None:
    """Refuse a name that only case or Unicode normalization tells apart from another.

    Only matters when originals are stored: on a case-insensitive or normalizing disk
    (a macOS host, a Windows bind mount) ``Report.pdf`` and ``report.pdf`` are one file,
    so the second upload replaced the first document's original — and deleting either
    removed both. The index itself tells them apart, so this is checked only with a raw
    store, and re-uploading the exact same name is always allowed.
    """

    folded = _folded(filename)
    for other in existing:
        if other != filename and _folded(other) == folded:
            raise FilenameConflictError(
                f"'{filename}' differs from the existing document '{other}' only by "
                "letter case or accents, and their stored originals would overwrite each "
                f"other. Rename the file, or delete '{other}' first."
            )


def indexing_now(settings: AppSettings) -> set[str]:
    """Filenames in ``settings``' workspace with an upload, re-index or delete admitted
    and not yet finished."""

    in_flight = (filename_of(settings, key) for key in ingest_sequencer.in_flight())
    return {name for name in in_flight if name}


def known_filenames(repository: VectorRepository, settings: AppSettings) -> list[str]:
    """Every indexed filename in ``settings``' workspace plus those with an upload or
    delete in flight: what a new name is checked against by
    ``refuse_near_duplicate_name`` (uploads, URL ingest, import and the folder watcher)."""

    return [*repository.filename_metadata(settings), *indexing_now(settings)]
