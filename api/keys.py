"""Named API keys (API_KEYS_FILE): who sent a request, and which workspaces it may use.

The file is TOML, one table per key, holding the key's SHA-256 (never the key itself)::

    [[key]]
    name = "alice"
    sha256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    level = "full"            # or "read": what API_READ_KEY may do
    workspaces = ["*"]        # every workspace, or names: ["default", "legal"]

``docrag new-key NAME`` prints a fresh key and its table. The file is read again
when it changes, so a key is added or revoked without a restart. API_KEY and
API_READ_KEY keep working next to it (named "full" and "read", every workspace).

Not user accounts: no passwords, sessions or sign-in — a key is still a shared secret,
now with a name and a list of workspaces.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import threading
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

KeyLevel = Literal["full", "read"]
ALL_WORKSPACES = "*"
DEFAULT_WORKSPACE = "default"
KEY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Keys one file may name: each request compares against every one of them.
MAX_KEYS = 500


class KeyFileError(ValueError):
    """API_KEYS_FILE can't be read or says something it may not."""


@dataclass(frozen=True, slots=True)
class KeyGrant:
    """A key that matched: its name, what it may do, and where (None: everywhere)."""

    name: str
    level: KeyLevel
    workspaces: frozenset[str] | None = None

    def can_use(self, workspace: str) -> bool:
        """Whether this key may use ``workspace`` ("" is the default workspace)."""

        return self.workspaces is None or (workspace or DEFAULT_WORKSPACE) in self.workspaces

    @property
    def everywhere(self) -> bool:
        return self.workspaces is None


@dataclass(frozen=True, slots=True)
class KeyRing:
    entries: tuple[tuple[bytes, KeyGrant], ...] = ()

    def match(self, supplied: bytes) -> KeyGrant | None:
        """The grant whose key ``supplied`` is. Every entry is compared, in constant
        time, whichever matches: timing doesn't say which key (or whether any) did."""

        digest = hashlib.sha256(supplied).digest()
        found: KeyGrant | None = None
        for expected, grant in self.entries:
            if secrets.compare_digest(digest, expected) and found is None:
                found = grant
        return found

    @property
    def names(self) -> list[str]:
        return [grant.name for _, grant in self.entries]


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def new_key() -> str:
    return "drk_" + secrets.token_urlsafe(32)


def key_table(name: str, key: str, level: KeyLevel, workspaces: Iterable[str]) -> str:
    """The TOML table for API_KEYS_FILE that grants ``key``."""

    names = ", ".join(f'"{workspace}"' for workspace in workspaces)
    return (
        "[[key]]\n"
        f'name = "{name}"\n'
        f'sha256 = "{hash_key(key)}"\n'
        f'level = "{level}"\n'
        f"workspaces = [{names}]\n"
    )


def parse_key_file(
    text: str, *, reserved: Iterable[str] = (), shared_hashes: Iterable[str] = ()
) -> KeyRing:
    """API_KEYS_FILE's text as a ``KeyRing``; ``KeyFileError`` saying what is wrong.
    ``shared_hashes``: API_KEY's and API_READ_KEY's SHA-256 — a file key that is one of
    them would be matched as that key, its name and workspaces silently ignored."""

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise KeyFileError(f"API_KEYS_FILE is not valid TOML: {exc}") from exc
    unknown = set(data) - {"key"}
    if unknown:
        raise KeyFileError(f"API_KEYS_FILE: unknown table(s) {sorted(unknown)}; use [[key]].")
    tables = data.get("key", [])
    if not isinstance(tables, list):
        raise KeyFileError("API_KEYS_FILE: write each key as a [[key]] table.")
    if len(tables) > MAX_KEYS:
        raise KeyFileError(f"API_KEYS_FILE names more than {MAX_KEYS} keys.")
    taken = set(reserved)
    hashes: set[str] = set()
    entries: list[tuple[bytes, KeyGrant]] = []
    for index, table in enumerate(tables, start=1):
        where = f"API_KEYS_FILE key {index}"
        if not isinstance(table, dict):
            raise KeyFileError(f"{where}: not a table.")
        extra = set(table) - {"name", "sha256", "level", "workspaces"}
        if extra:
            raise KeyFileError(f"{where}: unknown field(s) {sorted(extra)}.")
        name = table.get("name")
        if not isinstance(name, str) or not KEY_NAME_RE.match(name):
            raise KeyFileError(f"{where}: name must be 1-64 letters, digits, '.', '_', '@' or '-'.")
        if name in taken:
            raise KeyFileError(f"{where}: the name '{name}' is used twice (or is reserved).")
        taken.add(name)
        digest = table.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.match(digest.lower()):
            raise KeyFileError(f"{where} ({name}): sha256 must be 64 hex characters.")
        digest = digest.lower()
        if digest in set(shared_hashes):
            raise KeyFileError(
                f"{where} ({name}): the same key as API_KEY or API_READ_KEY; give it its own."
            )
        if digest in hashes:
            raise KeyFileError(f"{where} ({name}): the same key is listed twice.")
        hashes.add(digest)
        level = table.get("level", "full")
        if level not in ("full", "read"):
            raise KeyFileError(f'{where} ({name}): level must be "full" or "read".')
        spaces = table.get("workspaces", [ALL_WORKSPACES])
        if (
            not isinstance(spaces, list)
            or not spaces
            or not all(isinstance(space, str) and space for space in spaces)
        ):
            raise KeyFileError(f'{where} ({name}): workspaces must be a list of names, or ["*"].')
        workspaces = None if ALL_WORKSPACES in spaces else frozenset(spaces)
        entries.append((bytes.fromhex(digest), KeyGrant(name, level, workspaces)))
    return KeyRing(tuple(entries))


class _KeyFile:
    """API_KEYS_FILE, read again when its modification time or size changes. A file
    that becomes unreadable keeps no key working (it fails closed), and says why."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path = ""
        self._stamp: tuple[int, int, tuple[str, ...]] | None = None
        self._ring = KeyRing()
        self._error: str | None = None

    def ring(
        self, path: str, reserved: Iterable[str] = (), shared_hashes: tuple[str, ...] = ()
    ) -> KeyRing:
        try:
            stat = os.stat(path)
            stamp = (stat.st_mtime_ns, stat.st_size, shared_hashes)
        except OSError as exc:
            return self._failed(path, None, f"API_KEYS_FILE can't be read: {exc.strerror}")
        with self._lock:
            if path == self._path and stamp == self._stamp:
                return self._ring
        try:
            with open(path, encoding="utf-8") as file:
                ring = parse_key_file(file.read(), reserved=reserved, shared_hashes=shared_hashes)
        except (OSError, UnicodeDecodeError) as exc:
            return self._failed(path, stamp, f"API_KEYS_FILE can't be read: {exc}")
        except KeyFileError as exc:
            return self._failed(path, stamp, str(exc))
        with self._lock:
            if self._path == path and self._stamp is not None:
                logger.info("API_KEYS_FILE changed: %d key(s).", len(ring.entries))
            self._path, self._stamp, self._ring, self._error = path, stamp, ring, None
        return ring

    def _failed(
        self, path: str, stamp: tuple[int, int, tuple[str, ...]] | None, error: str
    ) -> KeyRing:
        with self._lock:
            if (path, stamp, error) != (self._path, self._stamp, self._error):
                logger.error("%s No key from it is accepted until it is fixed.", error)
            self._path, self._stamp, self._ring, self._error = path, stamp, KeyRing(), error
        return KeyRing()

    def error(self, path: str, shared_hashes: tuple[str, ...] = ()) -> str | None:
        self.ring(path, RESERVED_NAMES, shared_hashes)
        with self._lock:
            return self._error

    def clear(self) -> None:
        with self._lock:
            self._path, self._stamp, self._ring, self._error = "", None, KeyRing(), None


_key_file = _KeyFile()

# The names API_KEY and API_READ_KEY are known by: a file key can't take them.
RESERVED_NAMES = ("full", "read", "open", "system")


def shared_key_hashes(*keys: str) -> tuple[str, ...]:
    """The SHA-256 of each set shared key (API_KEY, API_READ_KEY)."""

    return tuple(hash_key(key) for key in keys if key)


def key_ring(path: str, shared_hashes: tuple[str, ...] = ()) -> KeyRing:
    """API_KEYS_FILE's keys (empty with no file, or one that can't be read)."""

    if not path:
        return KeyRing()
    return _key_file.ring(path, RESERVED_NAMES, shared_hashes)


def key_file_error(path: str, shared_hashes: tuple[str, ...] = ()) -> str | None:
    """Why API_KEYS_FILE gives no keys, or None when it reads."""

    return _key_file.error(path, shared_hashes) if path else None


def clear_key_file_cache() -> None:
    _key_file.clear()
