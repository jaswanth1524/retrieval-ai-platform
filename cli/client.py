"""A small synchronous client for the DocRAG REST API (shared by the CLI and MCP server)."""

from __future__ import annotations

import json
import os
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
JOB_POLL_SECONDS = 1.0


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


def sse_events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """JSON events from an SSE body, by the SSE rules: ``data:`` lines (optional space,
    several per event) joined until a blank line; comments (``: ping``) ignored."""

    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if data:
                yield json.loads("\n".join(data))
                data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield json.loads("\n".join(data))


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
        self._owns_http = http is None
        self._http = http or httpx.Client(
            base_url=url,
            timeout=httpx.Timeout(30.0, read=READ_TIMEOUT_SECONDS),
            transport=transport,
        )
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> DocragClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, headers=self._headers, **kwargs)
        except httpx.HTTPError as exc:
            raise DocragError(f"Cannot reach DocRAG at {self._http.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise DocragError(_detail(response), response.status_code)
        return response

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._request(method, path, **kwargs).json()

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
        self, query: str, filenames: list[str] | None = None, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"query": query}
        if filenames:
            body["filenames"] = filenames
        if top_k:
            body["rerank_top_k"] = top_k
        results: list[dict[str, Any]] = self._json("POST", "/search", json=body)["results"]
        return results

    # -- answers -----------------------------------------------------------------------

    def ask(
        self,
        question: str,
        filenames: list[str] | None = None,
        provider: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_stage: Callable[[dict[str, Any]], None] | None = None,
    ) -> StreamResult:
        body: dict[str, Any] = {"question": question}
        if filenames:
            body["filenames"] = filenames
        if provider:
            body["llm_provider"] = provider
        return self._stream("/questions/stream", body, on_delta, on_stage)

    def summarize(
        self,
        filenames: list[str],
        instruction: str | None = None,
        provider: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_stage: Callable[[dict[str, Any]], None] | None = None,
    ) -> StreamResult:
        body: dict[str, Any] = {
            "mode": "compare" if len(filenames) == 2 else "summarize",
            "filenames": filenames,
        }
        if instruction:
            body["instruction"] = instruction
        if provider:
            body["llm_provider"] = provider
        return self._stream("/questions/summary/stream", body, on_delta, on_stage)

    def _stream(
        self,
        path: str,
        body: dict[str, Any],
        on_delta: Callable[[str], None] | None,
        on_stage: Callable[[dict[str, Any]], None] | None,
    ) -> StreamResult:
        done: dict[str, Any] | None = None
        followups: list[str] = []
        try:
            with self._http.stream("POST", path, json=body, headers=self._headers) as response:
                if response.status_code >= 400:
                    response.read()
                    raise DocragError(_detail(response), response.status_code)
                for event in sse_events(response.iter_lines()):
                    kind = event.get("type")
                    if kind == "delta" and on_delta is not None:
                        on_delta(str(event.get("text", "")))
                    elif kind == "stage" and on_stage is not None:
                        on_stage(event)
                    elif kind == "done":
                        done = event
                    elif kind == "followups":
                        followups = [q for q in event.get("questions", []) if isinstance(q, str)]
                    elif kind == "error":
                        raise DocragError(str(event.get("detail", "The answer failed.")))
        except httpx.HTTPError as exc:
            raise DocragError(f"Cannot reach DocRAG at {self._http.base_url}: {exc}") from exc
        if done is None:
            raise DocragError("The stream ended before the answer finished.")
        return StreamResult(done, followups)

    # -- changes -----------------------------------------------------------------------

    def upload(self, path: Path) -> dict[str, Any]:
        """Upload one file; waits out a full indexing queue (503 + Retry-After)."""

        content = path.read_bytes()
        for attempt in range(UPLOAD_MAX_ATTEMPTS):
            try:
                response = self._http.post(
                    "/documents",
                    files={"file": (path.name, content, "application/octet-stream")},
                    headers=self._headers,
                )
            except httpx.HTTPError as exc:
                raise DocragError(f"Cannot reach DocRAG at {self._http.base_url}: {exc}") from exc
            if response.status_code == 503 and attempt + 1 < UPLOAD_MAX_ATTEMPTS:
                self._sleep(_retry_after(response))
                continue
            if response.status_code >= 400:
                raise DocragError(_detail(response), response.status_code)
            result: dict[str, Any] = response.json()
            return result
        raise DocragError("The indexing queue stayed full.")  # pragma: no cover

    def upload_url(self, url: str, filename: str | None = None) -> dict[str, Any]:
        """Have the server fetch and index ``url`` (URL_INGEST_ENABLED)."""

        body: dict[str, Any] = {"url": url}
        if filename:
            body["filename"] = filename
        result: dict[str, Any] = self._json("POST", "/documents/from-url", json=body)
        return result

    def wait_for_job(
        self, job_id: str, on_progress: Callable[[dict[str, Any]], None] | None = None
    ) -> dict[str, Any]:
        while True:
            status: dict[str, Any] = self._json("GET", f"/documents/jobs/{_quote(job_id)}")
            if on_progress is not None:
                on_progress(status)
            if status.get("state") in ("done", "failed"):
                return status
            self._sleep(JOB_POLL_SECONDS)

    def delete(self, filename: str) -> None:
        self._request("DELETE", f"/documents/{_quote(filename)}")

    def export(self, destination: Path) -> int:
        """Stream ``GET /export`` into ``destination``; returns the bytes written."""

        written = 0
        try:
            with self._http.stream("GET", "/export", headers=self._headers) as response:
                if response.status_code >= 400:
                    response.read()
                    raise DocragError(_detail(response), response.status_code)
                with destination.open("wb") as out:
                    for block in response.iter_bytes():
                        out.write(block)
                        written += len(block)
        except httpx.HTTPError as exc:
            raise DocragError(f"Cannot reach DocRAG at {self._http.base_url}: {exc}") from exc
        return written

    def restore(self, archive: Path) -> dict[str, Any]:
        with archive.open("rb") as handle:
            result: dict[str, Any] = self._json(
                "POST", "/import", files={"file": (archive.name, handle, "application/zip")}
            )
        return result


def _quote(segment: str) -> str:
    return quote(segment, safe="")
