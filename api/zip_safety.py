"""A zip's central directory sized before ``zipfile`` reads it.

``zipfile.ZipFile()`` builds an object per entry before any caller can count them:
about half a kilobyte of heap and 25 µs each, so a 256 MiB upload of three million
empty entries cost ~1.5 GB and over a minute on a request thread before a member cap
was ever checked. The end-of-central-directory record says how many entries there are
and how many bytes the directory takes; ``zipfile`` reads the directory by its byte
size (not the count), so both are bounded here.
"""

from __future__ import annotations

import struct
from typing import IO

_EOCD = b"PK\x05\x06"
_EOCD64_LOCATOR = b"PK\x06\x07"
_EOCD64 = b"PK\x06\x06"
# The record is 22 bytes plus a comment of at most 65535.
_TAIL_BYTES = 22 + 65535 + 20
# The smallest a directory entry can be (a one-letter name).
_MIN_ENTRY_BYTES = 47


def central_directory(upload: IO[bytes]) -> tuple[int, int] | None:
    """(entries declared, directory bytes) from ``upload``'s end record, or None when it
    has none (not a zip: ``zipfile`` will say so). The position is left at the start."""

    try:
        upload.seek(0, 2)
        size = upload.tell()
        start = max(0, size - _TAIL_BYTES)
        upload.seek(start)
        tail = upload.read(size - start)
        at = tail.rfind(_EOCD)
        if at < 0 or len(tail) - at < 22:
            return None
        entries, directory_bytes = struct.unpack_from("<HL", tail, at + 10)
        # zipfile reads the zip64 record whenever its locator sits right before the end
        # record, whatever the end record's own fields say: so must this, or a zip
        # whose end record claims one entry and whose zip64 record claims millions got
        # past the cap and then built every entry.
        locator = at - 20
        has_locator = locator >= 0 and tail.startswith(_EOCD64_LOCATOR, locator)
        if not has_locator and (entries == 0xFFFF or directory_bytes == 0xFFFFFFFF):
            found = tail.rfind(_EOCD64_LOCATOR, 0, at)
            if found < 0 or len(tail) - found < 20:
                return None
            locator, has_locator = found, True
        if has_locator:
            (record_at,) = struct.unpack_from("<Q", tail, locator + 8)
            # Python 3.12's zipfile reads the record right before the locator and never
            # looks at the offset the locator gives; newer ones read at that offset and
            # fall back to the record before the locator. Either may be the one believed,
            # so both are read and the larger claim counts: reading only the offset let a
            # locator pointing at junk keep the end record's small count.
            found = False
            for position in {start + locator - 56, record_at}:
                if position < 0:
                    continue
                upload.seek(position)
                record = upload.read(56)
                if len(record) < 56 or not record.startswith(_EOCD64):
                    continue
                found = True
                wide_entries, wide_bytes = struct.unpack_from("<QQ", record, 32)
                entries = max(entries, wide_entries)
                directory_bytes = max(directory_bytes, wide_bytes)
            if not found and (entries == 0xFFFF or directory_bytes == 0xFFFFFFFF):
                return None
        return int(entries), int(directory_bytes)
    except (OSError, struct.error, ValueError):
        return None
    finally:
        try:
            upload.seek(0)
        except (OSError, ValueError):
            pass


def too_many_entries(upload: IO[bytes], limit: int, name_bytes: int = 128) -> bool:
    """Whether ``upload``'s directory declares more than ``limit`` entries, or is larger
    than that many entries with ``name_bytes`` of name each could need (a count that
    lies is caught by the bytes). Unreadable records are left to ``zipfile``."""

    found = central_directory(upload)
    if found is None:
        return False
    entries, directory_bytes = found
    return entries > limit or directory_bytes > limit * (_MIN_ENTRY_BYTES + name_bytes)
