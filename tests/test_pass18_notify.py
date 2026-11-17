"""Pass 18: failure notifications (NOTIFY_URL)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api import notify as notify_module
from api.dependencies import clear_dependency_caches
from api.notify import Notifier, NotifyError, chosen_events, notify, set_notifier
from tests.factories import make_test_settings
from tests.routes_http.support import hermetic_app, make_settings


class Webhook:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.calls: list[dict[str, Any]] = []
        self.arrived = threading.Event()

    def __call__(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        self.arrived.set()
        return httpx.Response(self.status, request=httpx.Request("POST", url))


def _notifier(webhook: Webhook, clock: list[float], **settings: object) -> Notifier:
    return Notifier(
        make_test_settings(notify_url="https://hooks.example.com/T0/secret", **settings),
        post=webhook,
        clock=lambda: clock[0],
        check_address=lambda url: None,
    )


def test_an_event_is_posted_once_per_interval_and_repeats_are_counted() -> None:
    webhook, clock = Webhook(), [100.0]
    notifier = _notifier(webhook, clock)
    try:
        assert notifier.notify("backup_failed", "first") is True
        assert webhook.arrived.wait(5)
        assert notifier.notify("backup_failed", "second") is False
        assert notifier.notify("backup_failed", "third") is False
        # Another event has its own clock.
        assert notifier.notify("watch_gave_up", "a file") is True
        clock[0] += 3601
        webhook.arrived.clear()
        assert notifier.notify("backup_failed", "fourth") is True
        assert webhook.arrived.wait(5)
    finally:
        notifier.close()
    texts = [call["json"]["text"] for call in webhook.calls]
    assert texts[0] == "DocRAG: backup failed\nfirst"
    assert texts[-1].endswith("fourth (2 more like it since the last notice.)")
    assert all(call["follow_redirects"] is False for call in webhook.calls)


def test_only_chosen_events_are_sent_and_an_unknown_name_is_refused() -> None:
    webhook = Webhook()
    notifier = _notifier(webhook, [0.0], notify_events="backup_failed, watch_gave_up")
    try:
        assert notifier.notify("reindex_failed", "x") is False
    finally:
        notifier.close()
    assert webhook.calls == []
    assert "config_warning" in chosen_events("")
    with pytest.raises(ValueError, match="unknown events: backups"):
        chosen_events("backups")


def test_ntfy_gets_plain_text_and_a_failure_never_names_the_secret_path() -> None:
    webhook = Webhook(status=500)
    notifier = _notifier(webhook, [0.0], notify_format="text")
    try:
        with pytest.raises(NotifyError) as failure:
            notifier.send("test", "hello")
    finally:
        notifier.close()
    assert webhook.calls[0]["content"] == b"hello"
    assert webhook.calls[0]["headers"]["Title"] == "DocRAG: test"
    assert str(failure.value) == "hooks.example.com answered 500"


def test_a_private_webhook_address_is_refused_unless_allowed() -> None:
    webhook = Webhook()
    private = make_test_settings(notify_url="http://127.0.0.1:9/hook")
    refused = Notifier(private, post=webhook)
    allowed = Notifier(private.model_copy(update={"notify_allow_private": True}), post=webhook)
    try:
        with pytest.raises(NotifyError, match="NOTIFY_ALLOW_PRIVATE"):
            refused.send("test", "x")
        assert webhook.calls == []
        allowed.send("test", "x")
        assert len(webhook.calls) == 1
    finally:
        refused.close()
        allowed.close()


def test_notify_without_a_notifier_does_nothing_and_never_raises() -> None:
    set_notifier(None)
    notify("backup_failed", "nobody is listening")

    class Broken:
        def notify(self, event: str, text: str) -> bool:
            raise RuntimeError("boom")

        def close(self) -> None:
            pass

    notify_module._notifier = Broken()  # type: ignore[assignment]
    try:
        notify("backup_failed", "still fine")
    finally:
        notify_module._notifier = None


def test_a_failed_backup_and_a_file_the_watcher_gave_up_on_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api.scheduled_backup import BackupStatus, ScheduledBackup

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "api.scheduled_backup.notify", lambda event, text: sent.append((event, text))
    )
    backup = ScheduledBackup(
        make_test_settings(backup_dir=str(tmp_path)),
        repository=None,  # type: ignore[arg-type]
        targets=lambda: [],
        feedback_store=None,
        export_lock=threading.Lock(),
    )
    backup._note(BackupStatus(1.0, "default", True))
    backup._note(BackupStatus(2.0, "legal", False, error="disk full"))
    assert sent == [
        ("backup_failed", "The scheduled backup of workspace 'legal' failed: disk full")
    ]


def test_the_test_route_sends_now_and_says_how_it_went(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_dependency_caches()
    settings = make_settings()
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    webhook = Webhook()
    try:
        with TestClient(hermetic_app(settings, QdrantClient(":memory:"))) as client:
            assert client.post("/admin/notify-test").status_code == 404
            set_notifier(_notifier(webhook, [0.0]))
            assert client.post("/admin/notify-test").json() == {"sent": True, "error": None}
            webhook.status = 403
            assert client.post("/admin/notify-test").json() == {
                "sent": False,
                "error": "hooks.example.com answered 403",
            }
    finally:
        set_notifier(None)
        monkeypatch.undo()
        clear_dependency_caches()
