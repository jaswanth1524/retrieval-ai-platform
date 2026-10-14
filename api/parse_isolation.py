"""Parse an upload in a child process, within PARSE_TIMEOUT_SECONDS.

The parsers (pypdf, pdfplumber, BeautifulSoup, python-docx, openpyxl) are pure Python
over untrusted bytes. In the API process a pathological file held one of the two ingest
workers for as long as it liked — a thread can't be stopped — and its parsing held the
GIL that question answering needs. In a child process it runs on its own interpreter,
its memory goes back to the system when it ends, and past the limit it is killed.

Children are forked from a ``forkserver`` that has already imported the parsers, so
each parse starts in milliseconds instead of re-importing them; ``fork`` from the API
process itself (threads, ONNX runtime) isn't safe. ``PARSE_TIMEOUT_SECONDS=0`` parses in
the API process with no limit (the tests do, to patch the parsers).
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
from multiprocessing.connection import Connection
from typing import Any

from api.documents import DocumentError, DocumentParseError, parse_document_bytes
from api.parsers.common import DocumentSection
from api.settings import AppSettings

logger = logging.getLogger(__name__)

_context_lock = threading.Lock()
_context: Any = None


def _forkserver_context() -> Any:
    global _context
    with _context_lock:
        if _context is None:
            context = multiprocessing.get_context("forkserver")
            # Imported once, in the server every child forks from.
            context.set_forkserver_preload(["api.documents"])
            _context = context
        return _context


def _parse_in_child(
    connection: Connection, filename: str, content: bytes, settings: AppSettings
) -> None:
    try:
        result: tuple[str, object] = ("ok", parse_document_bytes(filename, content, settings))
    except DocumentError as exc:
        result = ("document_error", exc)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, never raised here
        result = ("error", f"{type(exc).__name__}: {exc}")
    try:
        connection.send(result)
    except Exception as exc:  # an unpicklable exception or result
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def parse_with_limit(filename: str, content: bytes, settings: AppSettings) -> list[DocumentSection]:
    """``parse_document_bytes`` in a child process that is killed after
    ``settings.parse_timeout_seconds``; in this process when that is 0."""

    timeout = float(settings.parse_timeout_seconds)
    if timeout <= 0:
        return parse_document_bytes(filename, content, settings)

    context = _forkserver_context()
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_parse_in_child,
        args=(sender, filename, content, settings),
        name="docrag-parse",
        daemon=True,
    )
    process.start()
    sender.close()  # the child holds the only writing end now
    try:
        if not receiver.poll(timeout):
            raise DocumentParseError(
                f"Reading '{filename}' took longer than PARSE_TIMEOUT_SECONDS "
                f"({timeout:g} s) and was stopped. If the file is valid, raise the limit."
            )
        try:
            status, payload = receiver.recv()
        except EOFError:
            # The child died without a word: killed for memory, or a crash in native code.
            raise DocumentParseError(
                f"Reading '{filename}' failed: the parser process exited unexpectedly "
                f"(exit code {process.exitcode})."
            ) from None
    finally:
        receiver.close()
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
    if status == "ok" and isinstance(payload, list):
        return payload
    if status == "document_error" and isinstance(payload, DocumentError):
        raise payload
    logger.warning("Parsing %s failed in the parser process: %s", filename, payload)
    raise DocumentParseError(f"Could not read '{filename}'.")
