"""A small synchronous client for the DocRAG REST API (shared by the CLI and MCP server)."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

DEFAULT_URL = "http://127.0.0.1:8000"
# A summary or a whole-corpus question on a local model takes minutes; the stream sends
# a keepalive every 15 s, so this only bounds a server that went silent.
READ_TIMEOUT_SECONDS = 120.0
# Uploads that meet a full indexing queue (503 + Retry-After) wait and try again, as
# the web UI does.
UPLOAD_MAX_ATTEMPTS = 40
# A question, summary or search turned away with a 429 (every question slot taken, or
# RATE_LIMIT_PER_MINUTE) is tried again this many times in all, after Retry-After.
BUSY_MAX_ATTEMPTS = 4
JOB_POLL_SECONDS = 1.0
# Job states after which a job never changes; "unknown" is this client's own, for a job
# the server no longer knows (evicted from its job store, or the server restarted).
FINISHED_JOB_STATES = ("done", "failed", "unknown")
# `index_files` keeps at most this many uploads indexing at once: enough to keep the
# server's ingest workers busy, well under its job retention (INGEST_JOBS_MAX_RETAINED,
# 50 by default — a finished job is evicted past it, and would read as unknown) and its
# in-flight cap (an upload past it waits on a 503).
INGEST_WINDOW = 8


class Cancelled(Exception):
    """``should_stop`` asked a stream to stop (an MCP tool call was cancelled). The
    connection is closed, which stops the answer on the server and frees its slot."""


class DocragError(RuntimeError):
    """The server refused a request, or could not be reached; ``str()`` says why."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class StreamResult:
    """A finished streamed answer: the ``done`` frame, plus any ``followups``."""

    done: dict[str, Any]
    followups: list[str]


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or response.reason_phrase
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(str(item.get("msg", item)) for item in detail if isinstance(item, dict))
    return str(detail) if detail else response.reason_phrase


def _retry_after(response: httpx.Response, default: float = 5.0) -> float:
    try:
        return max(float(response.headers.get("Retry-After", default)), 0.0)
    except ValueError:
        return default


def _response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:  # a proxy's HTML page, a truncated body
        raise DocragError(
            f"The server sent a response that is not JSON (HTTP {response.status_code}); "
            "is the URL a DocRAG server?"
        ) from exc


def _event(data: list[str]) -> dict[str, Any]:
    try:
        event = json.loads("\n".join(data))
    except ValueError as exc:
        raise DocragError("The server sent a malformed stream frame.") from exc
    if not isinstance(event, dict):
        raise DocragError("The server sent a malformed stream frame.")
    return event


def sse_events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """JSON events from an SSE body, by the SSE rules: ``data:`` lines (optional space,
    several per event) joined until a blank line; comments (``: ping``) ignored. A frame
    that is not a JSON object raises :class:`DocragError`."""

    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if data:
                yield _event(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield _event(data)


class DocragClient:
    """Thin wrapper over the endpoints the CLI and the MCP server use."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        workspace: str | None = None,
    ) -> None:
        url = (base_url or os.environ.get("DOCRAG_URL") or DEFAULT_URL).rstrip("/")
        key = api_key if api_key is not None else os.environ.get("DOCRAG_API_KEY", "")
        # Sent per request, not set on the client, so an injected client (tests pass the
        # app's own TestClient) gets it too without being changed.
        self._headers = {"X-API-Key": key} if key else {}
        # The workspace to work in (WORKSPACES_ENABLED); none is the default workspace.
        space = workspace if workspace is not None else os.environ.get("DOCRAG_WORKSPACE", "")
        if space:
            self._headers["X-Workspace"] = space
        self.workspace = space
        self._owns_http = http is None
        self._http = http or httpx.Client(
            base_url=url,
            timeout=httpx.Timeout(30.0, read=READ_TIMEOUT_SECONDS),
            transport=transport,
        )
        self._sleep = sleep
        self._clock = clock

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> DocragClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _unreachable(self, exc: httpx.HTTPError) -> DocragError:
        return DocragError(f"Cannot reach DocRAG at {self._http.base_url}: {exc}")

    def _request(
        self, method: str, path: str, *, attempts: int = 1, **kwargs: Any
    ) -> httpx.Response:
        """One request; a 429 is tried again after Retry-After, ``attempts`` times in all
        (only for bodies that can be sent twice)."""

        for attempt in range(attempts):
            try:
                response = self._http.request(method, path, headers=self._headers, **kwargs)
            except httpx.HTTPError as exc:
                raise self._unreachable(exc) from exc
            if response.status_code == 429 and attempt + 1 < attempts:
                self._sleep(_retry_after(response))
                continue
            break
        if response.status_code >= 400:
            raise DocragError(_detail(response), response.status_code)
        return response

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        return _response_json(self._request(method, path, **kwargs))

    # -- reads -------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        result: dict[str, Any] = self._json("GET", "/health")
        return result

    def config(self) -> dict[str, Any]:
        result: dict[str, Any] = self._json("GET", "/config")
        return result

    def documents(self) -> dict[str, Any]:
        result: dict[str, Any] = self._json("GET", "/documents")
        return result

    def content(self, filename: str, **params: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._json(
            "GET", f"/documents/{_quote(filename)}/content", params=params
        )
        return result

    def search(
        self,
        query: str,
        filenames: list[str] | None = None,
        top_k: int | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = self.search_response(query, filenames, top_k, tags)[
            "results"
        ]
        return results

    def search_response(
        self,
        query: str,
        filenames: list[str] | None = None,
        top_k: int | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """The whole ``POST /search`` response: ``results`` and ``weak_matches`` (none
        cleared the relevance bar; these are the closest)."""

        body: dict[str, Any] = {"query": query}
        if filenames:
            body["filenames"] = filenames
        if tags:
            body["tags"] = tags
        if top_k:
            body["rerank_top_k"] = top_k
        response: dict[str, Any] = self._json(
            "POST", "/search", json=body, attempts=BUSY_MAX_ATTEMPTS
        )
        return response

    # -- answers -----------------------------------------------------------------------

    def ask(
        self,
        question: str,
        filenames: list[str] | None = None,
        provider: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_stage: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        tags: list[str] | None = None,
    ) -> StreamResult:
        body: dict[str, Any] = {"question": question}
        if filenames:
            body["filenames"] = filenames
        if tags:
            body["tags"] = tags
        if provider:
            body["llm_provider"] = provider
        return self._stream("/questions/stream", body, on_delta, on_stage, should_stop)

    def summarize(
        self,
        filenames: list[str],
        instruction: str | None = None,
        provider: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_stage: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> StreamResult:
        body: dict[str, Any] = {
            # One document: a summary; two: a comparison; more: one brief across them.
            "mode": {1: "summarize", 2: "compare"}.get(len(filenames), "brief"),
            "filenames": filenames,
        }
        if instruction:
            body["instruction"] = instruction
        if provider:
            body["llm_provider"] = provider
        return self._stream("/questions/summary/stream", body, on_delta, on_stage, should_stop)

    def _stream(
        self,
        path: str,
        body: dict[str, Any],
        on_delta: Callable[[str], None] | None,
        on_stage: Callable[[dict[str, Any]], None] | None,
        should_stop: Callable[[], bool] | None = None,
    ) -> StreamResult:
        """Read one SSE answer. ``should_stop`` is checked on every line the server
        sends — keepalive comments too, every 15 s while it retrieves or the model
        thinks — so a cancelled caller is let go within that, not when the answer ends.
        """

        done: dict[str, Any] | None = None
        followups: list[str] = []
        for attempt in range(BUSY_MAX_ATTEMPTS):
            try:
                with self._http.stream("POST", path, json=body, headers=self._headers) as response:
                    if response.status_code == 429 and attempt + 1 < BUSY_MAX_ATTEMPTS:
                        # Turned away before the stream began (no slot, rate limit).
                        response.read()
                        self._sleep(_retry_after(response))
                        continue
                    if response.status_code >= 400:
                        response.read()
                        raise DocragError(_detail(response), response.status_code)
                    lines = response.iter_lines()
                    if should_stop is not None:
                        lines = _until(lines, should_stop)
                    for event in sse_events(lines):
                        kind = event.get("type")
                        if kind == "delta" and on_delta is not None:
                            on_delta(str(event.get("text", "")))
                        elif kind == "stage" and on_stage is not None:
                            on_stage(event)
                        elif kind == "done":
                            done = event
                        elif kind == "followups":
                            questions = event.get("questions", [])
                            if isinstance(questions, list):
                                followups = [q for q in questions if isinstance(q, str)]
                        elif kind == "error":
                            if done is not None:
                                break  # the answer is complete; only the follow-ups failed
                            raise DocragError(str(event.get("detail", "The answer failed.")))
            except httpx.HTTPError as exc:
                # After `done` the answer is whole: a reset while follow-ups were still
                # being written loses only those (the web UI's sawDone guard).
                if done is None:
                    raise self._unreachable(exc) from exc
            except DocragError:
                if done is None:
                    raise
            break
        if done is None:
            raise DocragError("The stream ended before the answer finished.")
        return StreamResult(done, followups)

    # -- changes -----------------------------------------------------------------------

    def upload(self, path: Path) -> dict[str, Any]:
        """Upload one file; waits out a full indexing queue (503) or a rate limit (429),
        each with its Retry-After."""

        try:
            content = path.read_bytes()
        except OSError as exc:
            raise DocragError(f"Cannot read {path}: {exc.strerror or exc}") from exc
        for attempt in range(UPLOAD_MAX_ATTEMPTS):
            try:
                response = self._http.post(
                    "/documents",
                    files={"file": (path.name, content, "application/octet-stream")},
                    headers=self._headers,
                )
            except httpx.HTTPError as exc:
                raise self._unreachable(exc) from exc
            if response.status_code in (429, 503) and attempt + 1 < UPLOAD_MAX_ATTEMPTS:
                self._sleep(_retry_after(response))
                continue
            if response.status_code >= 400:
                raise DocragError(_detail(response), response.status_code)
            result: dict[str, Any] = _response_json(response)
            return result
        raise DocragError("The indexing queue stayed full.")  # pragma: no cover

    def upload_url(self, url: str, filename: str | None = None) -> dict[str, Any]:
        """Have the server fetch and index ``url`` (URL_INGEST_ENABLED)."""

        body: dict[str, Any] = {"url": url}
        if filename:
            body["filename"] = filename
        result: dict[str, Any] = self._json(
            "POST", "/documents/from-url", json=body, attempts=BUSY_MAX_ATTEMPTS
        )
        return result

    def wait_for_job(
        self,
        job_id: str,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Poll a job until it is done or failed. Past ``timeout`` seconds (None: no
        limit) a DocragError says it is still running — it goes on on the server."""

        deadline = None if timeout is None else self._clock() + timeout
        while True:
            status: dict[str, Any] = self._json("GET", f"/documents/jobs/{_quote(job_id)}")
            if on_progress is not None:
                on_progress(status)
            if status.get("state") in ("done", "failed"):
                return status
            if deadline is not None and self._clock() >= deadline:
                raise DocragError(
                    f"Still {status.get('state', 'running')} after {timeout:g} s "
                    f"(job {job_id}); it goes on on the server."
                )
            self._sleep(JOB_POLL_SECONDS)

    def wait_for_jobs(
        self,
        job_ids: list[str],
        *,
        timeout: float | None = None,
        on_finished: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Poll several jobs together until each finishes or ``timeout`` seconds pass.

        Every job's last status by id: a finished one ``done``/``failed``, one the
        server no longer knows ``unknown`` (with the error), one still running at the
        timeout whatever state it was in. ``on_finished`` hears of each as it ends.
        """

        deadline = None if timeout is None else self._clock() + timeout
        pending = list(dict.fromkeys(job_ids))
        statuses: dict[str, dict[str, Any]] = {}
        while pending:
            for job_id, status in self._job_statuses(pending).items():
                statuses[job_id] = status
                if status.get("state") in FINISHED_JOB_STATES:
                    pending.remove(job_id)
                    if on_finished is not None:
                        on_finished(job_id, status)
            if not pending or (deadline is not None and self._clock() >= deadline):
                break
            self._sleep(JOB_POLL_SECONDS)
        return statuses

    def index_files(
        self,
        paths: list[Path],
        *,
        window: int = INGEST_WINDOW,
        timeout: float | None = None,
        on_finished: Callable[[Path, dict[str, Any]], None] | None = None,
    ) -> dict[Path, dict[str, Any]]:
        """Upload ``paths`` and wait for them, at most ``window`` indexing at once (the
        server indexes those side by side); the next goes up as soon as one finishes.

        Every path's last status: ``done``/``failed``/``unknown`` once finished (an
        upload the server refused is ``failed`` with its reason), else the state it was
        in when ``timeout`` seconds ran out — ``not_sent`` for one never uploaded.
        ``on_finished`` hears of each finished one as it ends.
        """

        deadline = None if timeout is None else self._clock() + timeout
        queue = list(paths)
        indexing: dict[str, Path] = {}
        statuses: dict[Path, dict[str, Any]] = {}

        def finish(path: Path, status: dict[str, Any]) -> None:
            statuses[path] = status
            if on_finished is not None:
                on_finished(path, status)

        while queue or indexing:
            timed_out = deadline is not None and self._clock() >= deadline
            while queue and len(indexing) < window and not timed_out:
                path = queue.pop(0)
                try:
                    indexing[self.upload(path)["job_id"]] = path
                except DocragError as exc:
                    finish(path, {"state": "failed", "error": str(exc)})
            finished = 0
            for job_id, status in self._job_statuses(list(indexing)).items():
                if status.get("state") in FINISHED_JOB_STATES:
                    finish(indexing.pop(job_id), status)
                    finished += 1
                else:
                    statuses[indexing[job_id]] = status
            if timed_out or (deadline is not None and self._clock() >= deadline):
                break
            # Straight on when a slot freed up and more are waiting to go.
            if indexing and not (finished and queue):
                self._sleep(JOB_POLL_SECONDS)
        for path in queue:
            statuses[path] = {"state": "not_sent"}
        return statuses

    def _job_statuses(self, job_ids: list[str]) -> dict[str, dict[str, Any]]:
        """One status per job; ``unknown`` (with the error) for one the server lost."""

        statuses: dict[str, dict[str, Any]] = {}
        for job_id in job_ids:
            try:
                statuses[job_id] = self._json("GET", f"/documents/jobs/{_quote(job_id)}")
            except DocragError as exc:
                statuses[job_id] = {"job_id": job_id, "state": "unknown", "error": str(exc)}
        return statuses

    def delete(self, filename: str) -> None:
        self._request("DELETE", f"/documents/{_quote(filename)}")

    def set_tags(self, filename: str, tags: list[str]) -> list[str]:
        """Replace a document's tags; the tags as stored."""

        result: dict[str, Any] = self._json(
            "PATCH", f"/documents/{_quote(filename)}/tags", json={"tags": tags}
        )
        return list(result.get("tags", []))

    def reindex(self, filename: str) -> dict[str, Any]:
        """Re-index one document from its stored original; the accepted job."""

        result: dict[str, Any] = self._json("POST", f"/documents/{_quote(filename)}/reindex")
        return result

    def reindex_stale(self) -> dict[str, Any]:
        """Re-index every document chunked under older settings: ``jobs`` and ``deferred``."""

        result: dict[str, Any] = self._json("POST", "/documents/reindex")
        return result

    def refresh(self, filename: str) -> dict[str, Any]:
        """Fetch a URL-ingested document again: the accepted job, or ``unchanged``."""

        result: dict[str, Any] = self._json("POST", f"/documents/{_quote(filename)}/refresh")
        return result

    def versions(self, filename: str) -> list[dict[str, Any]]:
        """Earlier versions of a document's original, newest first."""

        result: dict[str, Any] = self._json("GET", f"/documents/{_quote(filename)}/versions")
        return list(result.get("versions", []))

    def restore_version(self, filename: str, version: str) -> dict[str, Any]:
        result: dict[str, Any] = self._json(
            "POST", f"/documents/{_quote(filename)}/versions/{_quote(version)}/restore"
        )
        return result

    def query_log(self, limit: int = 200, uncovered: bool = False) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if uncovered:
            params["uncovered"] = "true"
        result: dict[str, Any] = self._json("GET", "/query-log", params=params)
        return list(result.get("queries", []))

    def active_jobs(self) -> list[dict[str, Any]]:
        """Queued and running indexing jobs, whoever started them."""

        result: dict[str, Any] = self._json("GET", "/documents/jobs")
        return list(result.get("jobs", []))

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._json("DELETE", f"/documents/jobs/{_quote(job_id)}")
        return result

    def workspaces(self) -> dict[str, Any]:
        """``enabled`` and the ``workspaces`` besides the default one."""

        result: dict[str, Any] = self._json("GET", "/workspaces")
        return result

    def create_workspace(self, name: str) -> None:
        self._request("POST", "/workspaces", json={"name": name})

    def delete_workspace(self, name: str) -> None:
        self._request("DELETE", f"/workspaces/{_quote(name)}")

    def watch_status(self) -> dict[str, Any]:
        result: dict[str, Any] = self._json("GET", "/watch")
        return result

    def watch_scan(self) -> None:
        self._request("POST", "/watch/scan")

    def export(self, destination: Path) -> int:
        """Stream ``GET /export`` into ``destination``; returns the bytes written.

        The download goes to a temporary file beside ``destination``, which replaces it
        only once the whole zip arrived: a failed or interrupted export never truncates
        an existing backup, and leaves nothing behind."""

        written = 0
        connected = False
        try:
            with self._http.stream("GET", "/export", headers=self._headers) as response:
                connected = True
                if response.status_code >= 400:
                    response.read()
                    raise DocragError(_detail(response), response.status_code)
                try:
                    fd, partial = tempfile.mkstemp(
                        dir=destination.parent, prefix=f".{destination.name}.", suffix=".part"
                    )
                except OSError as exc:
                    raise DocragError(f"Cannot write {destination}: {exc.strerror or exc}") from exc
                try:
                    with os.fdopen(fd, "wb") as out:
                        for block in response.iter_bytes():
                            out.write(block)
                            written += len(block)
                    os.replace(partial, destination)
                except BaseException:
                    Path(partial).unlink(missing_ok=True)
                    raise
        except httpx.HTTPError as exc:
            if connected:
                raise DocragError(f"The download was interrupted: {exc}") from exc
            raise self._unreachable(exc) from exc
        except OSError as exc:
            raise DocragError(f"Cannot write {destination}: {exc.strerror or exc}") from exc
        return written

    def restore(self, archive: Path) -> dict[str, Any]:
        try:
            handle = archive.open("rb")
        except OSError as exc:
            raise DocragError(f"Cannot read {archive}: {exc.strerror or exc}") from exc
        with handle:
            result: dict[str, Any] = self._json(
                "POST", "/import", files={"file": (archive.name, handle, "application/zip")}
            )
        return result


def _until(lines: Iterator[str], should_stop: Callable[[], bool]) -> Iterator[str]:
    for line in lines:
        if should_stop():
            raise Cancelled
        yield line


def _quote(segment: str) -> str:
    return quote(segment, safe="")
