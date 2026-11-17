"""A message to the operator when unattended work fails (``NOTIFY_URL``, off by default).

Scheduled backups, the folder watcher, the stale re-indexer and the URL refresher run
with nobody watching: a failure used to be a log line and, for those who scrape them, a
metric. ``notify(event, text)`` posts it to a webhook — Slack-compatible JSON
(``{"text": …}``, which Mattermost, Discord's ``/slack`` endpoint and most relays take),
or plain text with a ``Title`` header for ntfy (``NOTIFY_FORMAT=text``).

It never blocks or fails the work it reports: the post runs on its own thread, errors
are logged, and each event is sent at most once per ``NOTIFY_MIN_INTERVAL_SECONDS`` —
repeats in between are counted and mentioned by the next one.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import httpx

from api.settings import AppSettings
from api.url_fetch import is_public_address

logger = logging.getLogger(__name__)

# What can be reported. NOTIFY_EVENTS picks among them (empty: all).
EVENTS = (
    "backup_failed",
    "watch_gave_up",
    "reindex_failed",
    "url_refresh_failed",
    "config_warning",
)
TEST_EVENT = "test"
_TIMEOUT_SECONDS = 10.0
_MAX_TEXT_CHARS = 1000

Post = Callable[..., httpx.Response]


class NotifyError(Exception):
    """The notification couldn't be sent; the message says why."""


def chosen_events(raw: str) -> frozenset[str]:
    """NOTIFY_EVENTS as a set; empty means every event. An unknown name is an error."""

    names = {name.strip() for name in raw.split(",") if name.strip()}
    unknown = sorted(names - set(EVENTS))
    if unknown:
        raise ValueError(
            f"NOTIFY_EVENTS names unknown events: {', '.join(unknown)} "
            f"(known: {', '.join(EVENTS)})."
        )
    return frozenset(names or EVENTS)


def _refuse_private(url: str) -> None:
    """Raise unless every address ``url``'s host resolves to is public: a webhook URL
    is posted to from inside the deployment, like a fetched document's."""

    host = urlsplit(url).hostname or ""
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        try:
            found = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise NotifyError(f"cannot resolve {host!r}: {exc.strerror or exc}") from exc
        addresses = [str(info[4][0]) for info in found]
    private = [address for address in addresses if not is_public_address(address)]
    if private:
        raise NotifyError(
            f"{host} is a private address ({private[0]}); set NOTIFY_ALLOW_PRIVATE=true "
            "to post to a service on your own network."
        )


class Notifier:
    def __init__(
        self,
        settings: AppSettings,
        *,
        post: Post = httpx.post,
        clock: Callable[[], float] = time.monotonic,
        check_address: Callable[[str], None] = _refuse_private,
    ) -> None:
        self._url = settings.notify_url.strip()
        self._events = chosen_events(settings.notify_events)
        self._format = settings.notify_format
        self._allow_private = bool(settings.notify_allow_private)
        self._min_interval = float(settings.notify_min_interval_seconds)
        self._post = post
        self._clock = clock
        self._check_address = check_address
        self._lock = threading.Lock()
        self._last_sent: dict[str, float] = {}
        self._held_back: dict[str, int] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docrag-notify")

    def notify(self, event: str, text: str) -> bool:
        """Queue ``text`` for sending; False when ``event`` isn't chosen or one went
        out too recently (it is counted for the next)."""

        if event not in self._events:
            return False
        now = self._clock()
        with self._lock:
            last = self._last_sent.get(event)
            if last is not None and now - last < self._min_interval:
                self._held_back[event] = self._held_back.get(event, 0) + 1
                return False
            self._last_sent[event] = now
            more = self._held_back.pop(event, 0)
        if more:
            text = f"{text} ({more} more like it since the last notice.)"
        try:
            self._pool.submit(self._send_logged, event, text)
        except RuntimeError:  # shutting down
            return False
        return True

    def _send_logged(self, event: str, text: str) -> None:
        try:
            self.send(event, text)
        except NotifyError as exc:
            logger.warning("Could not send the %s notification: %s", event, exc)

    def send(self, event: str, text: str) -> None:
        """Post now, on this thread; raises ``NotifyError`` with the reason."""

        text = text[:_MAX_TEXT_CHARS]
        title = f"DocRAG: {event.replace('_', ' ')}"
        if not self._allow_private:
            self._check_address(self._url)
        try:
            if self._format == "text":
                response = self._post(
                    self._url,
                    content=text.encode("utf-8"),
                    headers={"Title": title, "Content-Type": "text/plain; charset=utf-8"},
                    timeout=_TIMEOUT_SECONDS,
                    follow_redirects=False,
                )
            else:
                response = self._post(
                    self._url,
                    json={"text": f"{title}\n{text}", "event": event, "source": "docrag"},
                    timeout=_TIMEOUT_SECONDS,
                    follow_redirects=False,
                )
        except httpx.HTTPError as exc:
            raise NotifyError(f"posting to {self._host()} failed ({type(exc).__name__})") from exc
        if response.status_code >= 300:
            raise NotifyError(f"{self._host()} answered {response.status_code}")

    def _host(self) -> str:
        # Never the path: a webhook's path (or an ntfy topic) is its secret.
        return urlsplit(self._url).hostname or "the webhook"

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=False)


_notifier: Notifier | None = None


def set_notifier(notifier: Notifier | None) -> None:
    """The app's notifier for its lifetime (``main.lifespan``); None: notifications off."""

    global _notifier
    previous, _notifier = _notifier, notifier
    if previous is not None and previous is not notifier:
        previous.close()


def current_notifier() -> Notifier | None:
    return _notifier


def notify(event: str, text: str) -> None:
    """Tell the operator, when NOTIFY_URL is set and ``event`` is chosen. Never raises."""

    notifier = _notifier
    if notifier is None:
        return
    try:
        notifier.notify(event, text)
    except Exception:  # noqa: BLE001 - reporting a failure must not become one
        logger.warning("Could not queue the %s notification.", event, exc_info=True)
