"""``docrag`` commands that add, sync, read and change documents, and their jobs."""

from __future__ import annotations

import argparse
import fnmatch
import functools
import hashlib
import json
import os
import sys
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from cli.client import (
    FINISHED_JOB_STATES,
    MISSING_BATCH,
    DocragClient,
    DocragError,
)
from cli.common import _NOT_GIVEN, _follow_import, _job_error, _wait_limit


def _matches(relative: str, patterns: Sequence[str]) -> bool:
    """A pattern without "/" matches the file name, one with "/" the path under the
    directory given (as the server's WATCH_INCLUDE / WATCH_EXCLUDE do)."""

    name = relative.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatchcase(relative if "/" in pattern else name, pattern) for pattern in patterns
    )


def ingestible(
    paths: Sequence[Path],
    extensions: set[str],
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    follow_symlinks: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Files to upload under ``paths`` (directories walked recursively, sorted) and the
    ones skipped for an unsupported type. Dotfiles and dot-directories inside a walked
    directory are ignored; a file named on the command line is always considered.
    ``include`` / ``exclude`` glob patterns choose files (left out silently), and an
    excluded directory name leaves its whole tree out. A symbolic link met while
    walking is left out unless ``follow_symlinks``: a folder holding
    ``notes.txt -> ~/.ssh/id_rsa`` would otherwise upload the key (the server's folder
    watcher refuses them too)."""

    found: list[Path] = []
    skipped: list[Path] = []
    for path in paths:
        if path.is_dir():
            candidates = sorted(
                p
                for p in path.rglob("*")
                if p.is_file()
                and (follow_symlinks or not p.is_symlink())
                and not any(part.startswith(".") for part in p.relative_to(path).parts)
                and not any(
                    _matches("/".join(p.relative_to(path).parts[: depth + 1]), exclude)
                    for depth in range(len(p.relative_to(path).parts) - 1)
                )
            )
            relative = {p: p.relative_to(path).as_posix() for p in candidates}
        else:
            candidates = [path]
            relative = {path: path.name}
        for candidate in candidates:
            where = relative[candidate]
            if (include and not _matches(where, include)) or _matches(where, exclude):
                continue
            if candidate.suffix.lower() in extensions:
                found.append(candidate)
            else:
                skipped.append(candidate)
    return found, skipped


def document_names(
    files: Sequence[Path], roots: Sequence[Path], keep_paths: bool
) -> dict[Path, str]:
    """The name each file is stored under: its base name, or with ``keep_paths`` its
    path under the folder named on the command line that holds it (``reports/q1.pdf``;
    a file named on its own keeps its base name)."""

    names: dict[Path, str] = {}
    folders = [root for root in roots if root.is_dir()]
    for path in files:
        name = path.name
        if keep_paths:
            for root in folders:
                try:
                    name = path.relative_to(root).as_posix()
                    break
                except ValueError:
                    continue
        names[path] = name
    return names


def stored_name_key(name: str) -> str:
    """The server keeps only a file's basename, and refuses (or, without stored
    originals, conflates) names that differ only by letter case or Unicode form."""

    return unicodedata.normalize("NFC", name.strip()).casefold()


def cmd_documents(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    listing = client.documents()
    if args.json:
        print(json.dumps(listing), file=out)
        return 0
    counts = listing.get("chunk_counts", {})
    for name in listing.get("filenames", []):
        print(f"{name}\t{counts.get(name, '?')} chunks", file=out)
    return 0


def cmd_ingest(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Upload and follow several files at a time (``DocragClient.index_files``): the
    server indexes them side by side instead of one file after another."""

    paths = [Path(p) for p in args.paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"error: not found: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2
    archives = [p for p in paths if p.is_file() and p.suffix.lower() == ".zip"]
    paths = [p for p in paths if p not in archives]
    config = client.config()
    extensions = {ext.lower() for ext in config.get("supported_extensions", [])}
    max_bytes = config.get("max_upload_bytes")
    files, unsupported = ingestible(
        paths,
        extensions,
        include=args.include or (),
        exclude=args.exclude or (),
        follow_symlinks=bool(getattr(args, "follow_symlinks", False)),
    )
    # A zip found while walking a directory is sent as an archive too, as one named on
    # the command line is (it used to be skipped as an unsupported type).
    archives += [p for p in unsupported if p.suffix.lower() == ".zip"]
    unsupported = [p for p in unsupported if p.suffix.lower() != ".zip"]
    text = None if args.json else out
    skipped: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    indexed: list[dict[str, Any]] = []
    queued_in_background: list[dict[str, Any]] = []

    def skip(path: Path, reason: str) -> None:
        skipped.append({"path": str(path), "reason": reason})
        if text:
            print(f"skip  {path} ({reason})", file=text)

    def fail(path: Path, error: str) -> None:
        failed.append({"path": str(path), "error": error})
        if text:
            print(f"fail  {path}: {error}", file=text)

    for path in unsupported:
        skip(path, "unsupported type")
    uploads: list[Path] = []
    first_of: dict[str, Path] = {}
    given = getattr(args, "names", None)
    names: dict[Path, str] = (
        {path: given.get(str(path), path.name) for path in files}
        if given is not None
        else document_names(files, paths, bool(getattr(args, "keep_paths", False)))
    )
    for path in files:
        try:
            size = path.stat().st_size
        except OSError as exc:
            fail(path, f"cannot read it ({exc.strerror or exc})")
            continue
        if isinstance(max_bytes, int) and size > max_bytes:
            skip(path, f"too large: {size} bytes, the limit is {max_bytes}")
            continue
        # Same basename from two folders: the second upload would replace the first.
        first = first_of.setdefault(stored_name_key(names[path]), path)
        if first is not path:
            skip(path, f"another file is already indexed as {names[first]}: {first}")
            continue
        uploads.append(path)
    if getattr(args, "skip_unchanged", False) and uploads:
        cache = HashCache()
        unchanged = _unchanged_on_server(client, uploads, cache, names)
        cache.save()
        for path in uploads:
            if path in unchanged:
                skip(path, "unchanged: already indexed with these bytes")
        uploads = [path for path in uploads if path not in unchanged]

    def finished(path: Path, status: dict[str, Any]) -> None:
        if status.get("state") != "done":
            fail(path, _job_error(status))
            return
        result = status.get("result") or {}
        chunks = result.get("chunks_ingested", status.get("chunks_done"))
        warnings = [str(warning) for warning in result.get("warnings") or []]
        indexed.append(
            {
                "path": str(path),
                "filename": status.get("filename"),
                "chunks": chunks,
                **({"warnings": warnings} if warnings else {}),
            }
        )
        if text:
            print(f"ok    {path} → {status.get('filename')} ({chunks} chunks)", file=text)
            # What the server left out or couldn't read.
            for warning in warnings:
                print(f"      note: {warning}", file=text)

    statuses = client.index_files(
        uploads,
        timeout=_wait_limit(args),
        on_finished=finished,
        tags=args.tag or (),
        names={path: names[path] for path in uploads if names[path] != path.name},
    )
    for path, status in statuses.items():
        if status.get("state") not in FINISHED_JOB_STATES:
            fail(path, _job_error(status))
    # A zip named on the command line: the server indexes each document in it.
    for archive in archives:
        try:
            report = client.upload_archive(archive, tags=args.tag or (), background=args.background)
        except DocragError as exc:
            # One zip refused (too large, not a zip): the others still go, and the
            # summary still counts what was indexed.
            fail(archive, str(exc))
            continue
        if args.background:
            _follow_archive_import(client, args, archive, report, fail, queued_in_background, text)
            continue
        for item in report.get("skipped", []):
            skip(Path(f"{archive}:{item.get('name')}"), str(item.get("reason")))
        for name in report.get("deferred", []):
            fail(Path(f"{archive}:{name}"), "the indexing queue was full; ingest the zip again")
        member = {
            job["job_id"]: Path(f"{archive}:{job.get('filename', '')}")
            for job in report.get("jobs", [])
        }
        waited = client.wait_for_jobs(
            list(member),
            timeout=_wait_limit(args),
            on_finished=functools.partial(_finished_member, member, finished),
        )
        for job_id, status in waited.items():
            if status.get("state") not in FINISHED_JOB_STATES:
                fail(member[job_id], _job_error(status))
    queued = sum(item["queued"] for item in queued_in_background)
    if args.json:
        report_json: dict[str, Any] = {"indexed": indexed, "failed": failed, "skipped": skipped}
        if queued_in_background:
            report_json["queued_in_background"] = queued_in_background
        print(json.dumps(report_json), file=out)
    else:
        background = f", {queued} queued from zips (`docrag jobs` follows them)" if queued else ""
        print(
            f"{len(indexed)} indexed, {len(failed)} failed, {len(skipped)} skipped{background}",
            file=out,
        )
    return 1 if failed else 0


def _follow_archive_import(
    client: DocragClient,
    args: argparse.Namespace,
    archive: Path,
    started: dict[str, Any],
    fail: Callable[[Path, str], None],
    queued_in_background: list[dict[str, Any]],
    text: TextIO | None,
) -> None:
    """``ingest --background``: wait until the server has queued every document of a
    zip (as room frees in its indexing queue; nothing is deferred)."""

    import_id = started.get("import_id")
    if not import_id:
        fail(archive, "the server didn't start a background import (too old?)")
        return
    try:
        status = client.wait_for_import(import_id, timeout=_wait_limit(args))
    except DocragError as exc:
        fail(archive, str(exc))
        return
    for problem in status.get("problems", []):
        fail(archive, str(problem))
    if status.get("state") != "done":
        fail(archive, f"the import ended {status.get('state')}")
    queued_in_background.append(
        {
            "path": str(archive),
            "import_id": import_id,
            "queued": int(status.get("queued", 0)),
            "skipped": int(status.get("skipped", 0)),
        }
    )
    if text:
        print(
            f"ok    {archive}: {status.get('queued', 0)} queued, "
            f"{status.get('skipped', 0)} skipped",
            file=text,
        )


# Tags are at most 40 characters: "sync:", the folder's name (cut to fit) and a short
# digest of its full path.
_SYNC_TAG_PREFIX = "sync:"


_MAX_TAG_CHARS = 40


_SYNC_DIGEST_CHARS = 8


def legacy_sync_tag(directory: Path) -> str:
    """The tag earlier versions used: the folder's name alone, which two folders of the
    same name shared (so one's --delete removed the other's documents)."""

    name = directory.resolve().name or "root"
    return (_SYNC_TAG_PREFIX + name)[:_MAX_TAG_CHARS]


def sync_tag(directory: Path) -> str:
    """The tag ``docrag sync`` marks a folder's documents with (what --delete may
    remove): the folder's name and a digest of its resolved path, so ``~/a/docs`` and
    ``~/b/docs`` each have their own."""

    resolved = directory.resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:_SYNC_DIGEST_CHARS]
    room = _MAX_TAG_CHARS - len(_SYNC_TAG_PREFIX) - len(digest) - 1
    name = (resolved.name or "root").lower()[:room]
    return f"{_SYNC_TAG_PREFIX}{name}-{digest}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class HashCache:
    """SHA-256 of local files remembered between runs, by path, size and modification
    time: a sync of a large folder hashes only what changed since the last one. Kept in
    the user's cache directory; a file it can't read or write is simply not used."""

    _MAX_ENTRIES = 200_000

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else self._default_path()
        self._entries: dict[str, list[Any]] = {}
        self._dirty = False
        try:
            loaded = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._entries = {k: v for k, v in loaded.items() if isinstance(v, list)}
        except (OSError, ValueError):
            pass

    @staticmethod
    def _default_path() -> Path:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        return Path(base) / "docrag" / "hashes.json"

    def sha256(self, path: Path) -> str:
        stat = path.stat()
        key = str(path.resolve())
        entry = self._entries.get(key)
        if entry and entry[:2] == [stat.st_size, stat.st_mtime_ns] and len(entry) == 3:
            return str(entry[2])
        digest = _sha256(path)
        self._entries[key] = [stat.st_size, stat.st_mtime_ns, digest]
        self._dirty = True
        return digest

    def save(self) -> None:
        if not self._dirty:
            return
        if len(self._entries) > self._MAX_ENTRIES:
            self._entries = dict(list(self._entries.items())[-self._MAX_ENTRIES :])
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            partial = self._path.with_suffix(".json.part")
            partial.write_text(json.dumps(self._entries), encoding="utf-8")
            os.replace(partial, self._path)
        except OSError:
            pass


def _unchanged_on_server(
    client: DocragClient,
    files: list[Path],
    cache: HashCache,
    names: dict[Path, str] | None = None,
) -> set[Path]:
    """The files the server already holds under their name with exactly these bytes
    (``POST /documents/missing``, a batch at a time). A server without that route (or
    a file that can't be read) leaves everything to be sent."""

    def name_of(path: Path) -> str:
        return (names or {}).get(path, path.name)

    hashed: list[tuple[Path, str]] = []
    for path in files:
        try:
            hashed.append((path, cache.sha256(path)))
        except OSError:
            continue
    unchanged: set[Path] = set()
    for start in range(0, len(hashed), MISSING_BATCH):
        batch = hashed[start : start + MISSING_BATCH]
        try:
            missing = set(client.missing([(name_of(path), digest) for path, digest in batch]))
        except DocragError:
            return set()
        unchanged.update(path for path, _ in batch if name_of(path) not in missing)
    return unchanged


def _names_on_disk(directory: Path, keep_paths: bool = False) -> set[str] | None:
    """The name of every file under ``directory``, whatever its type or the filters say
    (what --delete must not remove); None when the folder can't be listed."""

    names: set[str] = set()
    try:
        for path in directory.rglob("*"):
            if path.is_file() or path.is_symlink():
                name = path.relative_to(directory).as_posix() if keep_paths else path.name
                names.add(stored_name_key(name))
    except OSError:
        return None
    return names


def cmd_sync(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Make the index match a folder: upload files that are new or changed (by their
    SHA-256 against the server's), and with --delete remove documents this folder's
    sync put there whose file is gone. Unchanged files aren't sent at all.

    --delete only ever removes a document that carries this folder's own tag and whose
    file is no longer anywhere under the folder: a file left out by --include/--exclude
    or of a type no longer accepted is still there, and stays indexed. A folder with no
    files at all (an unmounted share) deletes nothing without --force."""

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2
    tag, legacy = sync_tag(directory), legacy_sync_tag(directory)
    config = client.config()
    extensions = {ext.lower() for ext in config.get("supported_extensions", [])}
    files, _ = ingestible(
        [directory],
        extensions,
        include=args.include or (),
        exclude=args.exclude or (),
        follow_symlinks=args.follow_symlinks,
    )
    listing = client.documents()
    hashes: dict[str, str] = listing.get("content_sha256s") or {}
    indexed = set(listing.get("filenames", []))
    tags: dict[str, list[str]] = listing.get("tags") or {}
    code = 0
    names = document_names(files, [directory], args.keep_paths)
    local: dict[str, Path] = {}
    for path in files:
        first = local.setdefault(stored_name_key(names[path]), path)
        if first is not path:
            print(f"skip  {path} (another file is indexed as {names[first]}: {first})", file=out)
    changed: list[Path] = []
    adopted: list[str] = []
    unchanged = 0
    sizes: dict[str, int] = listing.get("byte_sizes") or {}
    cache = HashCache()
    for path in local.values():
        try:
            # The size first: a file of another size has changed, without reading it.
            same = (
                names[path] in indexed
                and sizes.get(names[path], path.stat().st_size) == path.stat().st_size
                and hashes.get(names[path]) == cache.sha256(path)
            )
        except OSError as exc:
            print(f"skip  {path} (cannot read it: {exc.strerror or exc})", file=out)
            continue
        held = tags.get(names[path], [])
        ours = tag in held or legacy in held
        if legacy in held and tag not in held:
            adopted.append(names[path])  # an older sync's (the name-only tag): marked ours
        if same:
            unchanged += 1
        elif names[path] in indexed and not ours and not args.adopt:
            # Indexed from somewhere else (an upload, a URL, another folder) with other
            # content: replacing it — and later deleting it — isn't this sync's call.
            print(
                f"skip  {path} (a different document named {names[path]} is already "
                "indexed; --adopt replaces it)",
                file=out,
            )
            code = 1
        else:
            changed.append(path)
    cache.save()
    on_disk = _names_on_disk(directory, args.keep_paths)
    gone = sorted(
        name
        for name, document_tags in tags.items()
        if tag in document_tags and (on_disk is None or stored_name_key(name) not in on_disk)
    )
    # Nothing readable under the folder while documents came from it: an unmounted
    # share or a wrong path looks exactly like "everything was deleted".
    stand_down = bool(gone) and not on_disk and not args.force
    if args.dry_run:
        for path in changed:
            print(f"send    {path}", file=out)
        for name in gone if args.delete and not stand_down else []:
            print(f"delete  {name}", file=out)
        print(
            f"{len(changed)} to send, {unchanged} unchanged"
            + (
                f", {len(gone)} to delete"
                if args.delete and not stand_down
                else f", {len(gone)} gone (--delete)"
            ),
            file=out,
        )
        if stand_down and args.delete:
            print(_SYNC_STAND_DOWN.format(directory=directory), file=out)
        return code
    if changed:
        ingest_args = argparse.Namespace(
            **{
                **vars(args),
                "paths": [str(path) for path in changed],
                "tag": [*(args.tag or []), tag],
                "include": None,
                "exclude": None,
                "background": False,
                "names": {str(path): names[path] for path in changed},
            }
        )
        code = cmd_ingest(client, ingest_args, out) or code
    # An upload with tags replaces a document's tags: the ones it had (set by hand,
    # "hr") are put back beside this folder's, and a document taken over from an older
    # sync is marked as this folder's.
    wanted = [*(args.tag or []), tag]
    sent = {names[path] for path in changed}
    for name in sorted(sent | set(adopted)):
        before = tags.get(name, [])
        if name in sent:
            desired = [*before, *(t for t in wanted if t not in before)]
            if set(desired) == set(wanted):
                continue  # it had no other tags to put back
        else:
            desired = [*before, tag]
        try:
            client.set_tags(name, desired[:20])
        except DocragError as exc:
            print(f"fail  tags of {name}: {exc}", file=out)
            code = 1
    deleted = 0
    if args.delete and stand_down:
        print(_SYNC_STAND_DOWN.format(directory=directory), file=out)
        code = 1
    for name in gone if args.delete and not stand_down else []:
        try:
            client.delete(name)
            deleted += 1
            print(f"deleted {name}", file=out)
        except DocragError as exc:
            print(f"fail  delete {name}: {exc}", file=out)
            code = 1
    print(
        f"sync: {len(changed)} sent, {unchanged} unchanged, {deleted} deleted"
        + ("" if args.delete or not gone else f", {len(gone)} gone (--delete removes them)"),
        file=out,
    )
    return code


_SYNC_STAND_DOWN = (
    "nothing deleted: {directory} has no files at all, which is also what an unmounted "
    "or mistyped folder looks like (--force deletes anyway)"
)


def _finished_member(
    member: dict[str, Path],
    finished: Callable[[Path, dict[str, Any]], None],
    job_id: str,
    status: dict[str, Any],
) -> None:
    finished(member[job_id], status)


def cmd_add_url(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.archive:
        started = client.add_archive_url(
            args.source_url, tags=args.tag or (), keep_paths=args.keep_paths
        )
        import_id = started.get("import_id")
        if not import_id:
            print("error: the server didn't start an import (too old?)", file=sys.stderr)
            return 1
        return _follow_import(client, import_id, args, out)
    if args.sitemap:
        added = client.add_sitemap(
            args.source_url,
            tags=args.tag or (),
            refresh_minutes=args.refresh_minutes,
            max_pages=args.max_pages,
            same_host=not args.any_host,
        )
        import_id = added.get("import_id")
        if args.wait and import_id:
            status = client.wait_for_sitemap_import(import_id, timeout=_wait_limit(args))
            if args.json:
                print(json.dumps(status), file=out)
                return 0 if status.get("failed", 0) == 0 else 1
            for problem in status.get("problems", []):
                print(f"  {problem}", file=out)
            print(
                f"{status.get('queued', 0)} queued, {status.get('unchanged', 0)} unchanged, "
                f"{status.get('failed', 0)} failed, {status.get('skipped', 0)} skipped "
                f"of {status.get('total', 0)} page(s)"
                f"{'' if status.get('finished_at') is not None else ' so far (still going)'}",
                file=out,
            )
            return 0 if status.get("failed", 0) == 0 else 1
        if args.json:
            print(json.dumps(added), file=out)
            return 0
        for url in added.get("urls", []):
            print(f"queue {url}", file=out)
        print(
            f"{len(added.get('urls', []))} page(s) being added in the background "
            f"({added.get('left_out', 0)} left out); `docrag jobs` shows them",
            file=out,
        )
        return 0
    job = client.upload_url(
        args.source_url,
        args.name,
        tags=args.tag or (),
        refresh_minutes=args.refresh_minutes,
        replace=args.replace,
    )
    status = client.wait_for_jobs([job["job_id"]], timeout=_wait_limit(args))[job["job_id"]]
    succeeded = status.get("state") == "done"
    if args.json:
        print(
            json.dumps(status if succeeded else {**status, "error": _job_error(status)}), file=out
        )
    elif succeeded:
        print(f"ok    {args.source_url} → {status.get('filename')}", file=out)
    else:
        print(f"fail  {args.source_url}: {_job_error(status)}", file=out)
    return 0 if succeeded else 1


def cmd_read(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """A document's indexed passages: around a cited chunk, or from passage --start."""

    if args.around:
        page = client.content(args.file, around=args.around, radius=args.radius)
    else:
        start = max(1, args.start)
        page = client.content(args.file, start=start, end=start + max(1, args.count) - 1)
    if args.json:
        print(json.dumps(page), file=out)
        return 0
    if args.around and not page.get("target_found", True):
        print(f"(no passage {args.around} in {args.file}; showing its start)", file=sys.stderr)
    for chunk in page.get("chunks", []):
        page_no, end = chunk.get("page"), chunk.get("page_end")
        pages = f"p.{page_no}-{end}" if end else f"p.{page_no}"
        print(
            f"[{chunk.get('chunk_ordinal')}] {pages} · {chunk.get('section') or '-'} "
            f"({chunk.get('chunk_id')})\n{chunk.get('text', '')}\n",
            file=out,
        )
    print(f"({page.get('total_chunks')} passages in all)", file=out)
    return 0


def cmd_delete(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """Delete each file named; one that fails doesn't stop the rest (exit 1 if any did)."""

    deleted: list[str] = []
    failed: list[dict[str, Any]] = []
    for name in args.files:
        try:
            client.delete(name)
        except DocragError as exc:
            failed.append({"filename": name, "error": str(exc), "status": exc.status_code})
            if not args.json:
                code = f" ({exc.status_code})" if exc.status_code else ""
                print(f"error{code}: {name}: {exc}", file=sys.stderr)
            continue
        deleted.append(name)
        if not args.json:
            print(f"deleted {name}", file=out)
    if args.json:
        print(json.dumps({"deleted": deleted, "failed": failed}), file=out)
    return 1 if failed else 0


def cmd_tags(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    """``set`` replaces a document's tags; ``add``/``remove`` change the current ones."""

    if args.action == "set":
        wanted = list(args.tags)
    else:
        current = list(client.documents().get("tags", {}).get(args.file, []))
        if args.action == "add":
            wanted = current + [tag for tag in args.tags if tag not in current]
        else:
            drop = {tag.casefold() for tag in args.tags}
            wanted = [tag for tag in current if tag.casefold() not in drop]
    stored = client.set_tags(args.file, wanted)
    if args.json:
        print(json.dumps({"filename": args.file, "tags": stored}), file=out)
    else:
        print(f"{args.file}: {', '.join(stored) or '(no tags)'}", file=out)
    return 0


def _follow(
    client: DocragClient, jobs: list[dict[str, Any]], args: argparse.Namespace, out: TextIO
) -> int:
    """Wait for ``jobs`` and print each outcome; 1 when any failed or wasn't followed."""

    if not jobs:
        return 0
    names = {job["job_id"]: job.get("filename") for job in jobs}
    statuses = client.wait_for_jobs(list(names), timeout=_wait_limit(args))
    failed = 0
    for job_id, status in statuses.items():
        ok = status.get("state") == "done"
        failed += not ok
        if not args.json:
            note = " (unchanged)" if status.get("unchanged") else ""
            detail = "" if ok else f": {_job_error(status)}"
            label = "ok   " if ok else "fail "
            print(f"{label} {status.get('filename') or names[job_id]}{note}{detail}", file=out)
    if args.json:
        print(json.dumps(list(statuses.values())), file=out)
    return 1 if failed else 0


def cmd_reindex(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if bool(args.stale) == bool(args.file):
        print("error: name a document, or pass --stale (not both)", file=sys.stderr)
        return 2
    if args.stale:
        report = client.reindex_stale()
        jobs, deferred = report.get("jobs", []), report.get("deferred", [])
        busy, failed = report.get("in_progress", []), report.get("failed", [])
        if not args.json:
            print(f"{len(jobs)} queued, {len(deferred)} deferred (queue full)", file=out)
            if busy:
                print(f"already being indexed: {', '.join(busy)}", file=out)
            if failed:
                print(f"could not be queued: {', '.join(failed)}", file=out)
        return _follow(client, jobs, args, out) or (1 if deferred or failed else 0)
    return _follow(client, [client.reindex(args.file)], args, out)


def cmd_refresh(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.every is not _NOT_GIVEN:
        result = client.set_refresh_interval(args.file, args.every)
        if args.json:
            print(json.dumps(result), file=out)
        else:
            minutes = result.get("refresh_minutes")
            said = (
                "the server's interval"
                if minutes is None
                else "never"
                if minutes == 0
                else f"every {minutes} min"
            )
            print(f"{args.file}: checked {said}", file=out)
        return 0
    result = client.refresh(args.file)
    if result.get("unchanged") and "job_id" not in result:
        if args.json:
            print(json.dumps(result), file=out)
        else:
            print(f"{args.file}: unchanged at its source", file=out)
        return 0
    return _follow(client, [result], args, out)


def cmd_versions(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.restore:
        return _follow(client, [client.restore_version(args.file, args.restore)], args, out)
    versions = client.versions(args.file)
    if args.json:
        print(json.dumps(versions), file=out)
    elif not versions:
        print(f"no earlier versions of {args.file} (RAW_DOCUMENT_VERSIONS)", file=out)
    for version in versions if not args.json else []:
        print(f"{version['version']}  {version['size']} bytes", file=out)
    return 0


def cmd_preview(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    path = Path(args.file)
    preview = client.preview(path)
    if args.json:
        print(json.dumps(preview), file=out)
        return 0
    print(
        f"{preview['filename']}: {preview['chunk_count']} passages in "
        f"{preview['section_count']} sections, {preview['page_count']} page(s)",
        file=out,
    )
    for chunk in preview.get("first_chunks", [])[: args.count]:
        text = " ".join(str(chunk.get("text", "")).split())
        print(
            f"\n#{chunk.get('ordinal')} p.{chunk.get('page')} · {chunk.get('section')}\n"
            f"   {text[:240]}",
            file=out,
        )
    return 0


def cmd_jobs(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.retry:
        return _follow(client, [client.retry_job(args.retry)], args, out)
    if args.cancel:
        status = client.cancel_job(args.cancel)
        if args.json:
            print(json.dumps(status), file=out)
        else:
            print(f"cancel requested: {status.get('filename')} ({status.get('state')})", file=out)
        return 0
    if args.recent or args.failed:
        jobs = client.finished_jobs(failed_only=args.failed, limit=args.limit)
    else:
        jobs = client.active_jobs()
    if args.json:
        print(json.dumps(jobs), file=out)
        return 0
    if not jobs:
        print(
            "no failed jobs kept"
            if args.failed
            else "no finished jobs kept"
            if args.recent
            else "no indexing jobs running",
            file=out,
        )
    for job in jobs:
        progress = (
            f" {job.get('chunks_done')}/{job.get('chunks_total')}"
            if job.get("chunks_total") and job.get("state") not in ("done", "failed")
            else ""
        )
        line = f"{job.get('job_id')}  {job.get('state')}{progress}  {job.get('filename')}"
        if job.get("error"):
            line += f"\n    {job['error']}"
            if job.get("retryable"):
                line += f"\n    retry: docrag jobs --retry {job.get('job_id')}"
        print(line, file=out)
    return 0


def cmd_original(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    destination = Path(args.output or Path(args.file).name)
    written = client.original(args.file, destination)
    print(f"wrote {written} bytes to {destination}", file=sys.stderr)
    return 0


def cmd_workspaces(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.action == "list":
        listing = client.workspaces()
        if args.json:
            print(json.dumps(listing), file=out)
        elif not listing.get("enabled"):
            print("workspaces are not enabled on this server (WORKSPACES_ENABLED)", file=out)
        else:
            for name in ["(default)", *listing.get("workspaces", [])]:
                print(name, file=out)
        return 0
    if not args.name:
        print(f"error: workspaces {args.action} needs a name", file=sys.stderr)
        return 2
    if args.action == "create":
        client.create_workspace(args.name)
    else:
        client.delete_workspace(args.name)
    if args.json:
        print(json.dumps({args.action: args.name}), file=out)
    else:
        print(f"{args.action}d workspace {args.name}", file=out)
    return 0


def cmd_watch(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    if args.scan:
        client.watch_scan()
    status = client.watch_status()
    if args.json:
        print(json.dumps(status), file=out)
        return 0
    print(f"watching {status.get('folder')}: {status.get('files', 0)} files", file=out)
    if args.scan:
        print("(a pass was started; run again to see its result)", file=out)
    for retry in status.get("retrying", []):
        print(
            f"retry   {retry.get('filename')} (attempt {retry.get('attempts')}): "
            f"{retry.get('error')}",
            file=out,
        )
    for name in status.get("gave_up", []):
        print(f"gave up {name} (tried again when it changes)", file=out)
    for problem in status.get("problems", []):
        print(f"skip    {problem}", file=out)
    return 0


def cmd_outline(client: DocragClient, args: argparse.Namespace, out: TextIO) -> int:
    outline = client.outline(args.filename)
    if args.json:
        print(json.dumps(outline), file=out)
        return 0
    for entry in outline.get("sections", []):
        print(f"p. {entry.get('page'):>4}  {entry.get('section')}", file=out)
    if outline.get("truncated"):
        print("(more sections than the outline lists)", file=out)
    return 0
