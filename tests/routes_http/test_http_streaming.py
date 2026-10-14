"""Streamed routes when the client leaves: nothing waits for garbage collection."""

from __future__ import annotations

import asyncio
import gc
import json
import threading
from collections.abc import Generator, Sequence
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.types import Message

from api.admission import QuestionSlots
from api.dependencies import get_generator, get_question_slots
from api.generation import ChatMessage
from api.settings import AppSettings
from tests.factories import wait_until
from tests.routes_http.support import ApiTestContext, raw_storage_client, upload_and_wait


class EndlessGenerator:
    """Streams tokens until it is closed; ``closed`` says it was."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        return "Alpha [1]."

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Generator[str]:
        try:
            yield "Alpha [1]."
            while True:
                threading.Event().wait(0.01)
                yield " more"
        finally:
            self.closed.set()


async def _call_until_bodies(
    app: FastAPI, method: str, path: str, body: bytes, bodies_before_leaving: int
) -> list[Message]:
    sent: list[Message] = []
    gone = asyncio.Event()
    request_sent = False

    async def receive() -> Message:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)
        count = sum(1 for item in sent if item["type"] == "http.response.body")
        if count >= bodies_before_leaving:
            gone.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "scheme": "http",
        "server": ("test", 80),
        "client": ("client", 1),
    }
    await app(scope, receive, send)
    return sent


def test_a_client_leaving_a_stream_stops_generation_and_frees_the_slot(
    api_context: ApiTestContext,
) -> None:
    assert upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
    generator = EndlessGenerator()
    slots = QuestionSlots(1)
    api_context.app.dependency_overrides[get_generator] = lambda: generator
    api_context.app.dependency_overrides[get_question_slots] = lambda: slots

    gc.disable()  # before the fix, only a collection ever stopped the endless answer
    try:
        sent = asyncio.run(
            _call_until_bodies(
                api_context.app,
                "POST",
                "/questions/stream",
                json.dumps({"question": "alpha"}).encode(),
                bodies_before_leaving=4,
            )
        )
        assert sent[0]["status"] == 200
        assert generator.closed.wait(5), "the answer kept generating after the client left"
        wait_until(slots.try_acquire, message="the question slot was never released")()
    finally:
        gc.enable()


def test_only_one_export_streams_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from api.routes import documents as document_routes

    # Held as if another export were mid-stream.
    assert document_routes._export_running.acquire(blocking=False)
    try:
        with raw_storage_client(tmp_path, monkeypatch) as client:
            busy = client.get("/export")
            assert busy.status_code == 409
            assert "already running" in busy.json()["detail"]
    finally:
        document_routes._export_running.release()

    with raw_storage_client(tmp_path, monkeypatch) as client:
        # Released when each response ends, so exports one after another both work.
        assert client.get("/export").status_code == 200
        assert client.get("/export").status_code == 200
