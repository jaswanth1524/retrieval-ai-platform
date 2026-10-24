"""Named server profiles for the CLI, from ``~/.config/docrag/config.toml``.

    [default]                      # used when no --profile is given
    url = "http://127.0.0.1:8000"

    [profiles.prod]
    url = "https://docs.example.com"
    workspace = "legal"
    key_env = "DOCRAG_PROD_KEY"    # the NAME of the variable holding the key

A key is never written in the file itself — ``key_env`` names an environment variable
— so the file can be shared or committed to a dotfiles repository. Flags win over
environment variables (``DOCRAG_URL``, ``DOCRAG_API_KEY``, ``DOCRAG_WORKSPACE``), which
win over the profile. ``DOCRAG_CONFIG`` points at another file, ``DOCRAG_PROFILE``
picks a profile when ``--profile`` isn't given.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_KNOWN_KEYS = frozenset({"url", "workspace", "key_env"})


class ProfileError(ValueError):
    """The config file can't be read, or names something it doesn't define."""


@dataclass(frozen=True)
class Profile:
    url: str | None = None
    workspace: str | None = None
    # The environment variable the API key is read from.
    key_env: str | None = None

    def api_key(self) -> str | None:
        return os.environ.get(self.key_env) if self.key_env else None


def config_path() -> Path:
    explicit = os.environ.get("DOCRAG_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "docrag" / "config.toml"


def _profile(name: str, table: Any, path: Path) -> Profile:
    if not isinstance(table, dict):
        raise ProfileError(f"{path}: [{name}] is not a table.")
    if "api_key" in table or "key" in table:
        raise ProfileError(
            f"{path}: [{name}] holds a key. Put it in an environment variable and name "
            'that variable instead: key_env = "DOCRAG_API_KEY".'
        )
    unknown = sorted(set(table) - _KNOWN_KEYS)
    if unknown:
        raise ProfileError(f"{path}: [{name}] has unknown settings: {', '.join(unknown)}.")
    values = {}
    for key in _KNOWN_KEYS:
        value = table.get(key)
        if value is not None and not isinstance(value, str):
            raise ProfileError(f"{path}: [{name}] {key} must be a string.")
        values[key] = value or None
    return Profile(**values)


def load_profile(name: str | None, path: Path | None = None) -> Profile | None:
    """The profile ``name`` (``[profiles.NAME]``), or ``[default]`` when no name is
    given; None when there is no config file or no default. Naming a profile the file
    doesn't define, or a file that doesn't parse, is a ``ProfileError``."""

    path = path or config_path()
    name = name or os.environ.get("DOCRAG_PROFILE") or None
    if not path.is_file():
        if name:
            raise ProfileError(f"No config file at {path} to read profile {name!r} from.")
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProfileError(f"{path}: {exc}") from exc
    if name:
        profiles = data.get("profiles", {})
        if not isinstance(profiles, dict) or name not in profiles:
            known = ", ".join(sorted(profiles)) if isinstance(profiles, dict) else ""
            raise ProfileError(
                f"{path} has no profile {name!r}" + (f" (it has: {known})." if known else ".")
            )
        return _profile(f"profiles.{name}", profiles[name], path)
    if "default" in data:
        return _profile("default", data["default"], path)
    return None


def resolve(flag: str | None, env_name: str, profile_value: str | None) -> str | None:
    """A setting by precedence: the flag, then the environment, then the profile.
    None leaves it to the client's own default."""

    if flag is not None:
        return flag
    if os.environ.get(env_name):
        return None  # the client reads the environment itself
    return profile_value
