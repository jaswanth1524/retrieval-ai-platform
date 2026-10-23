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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
            # Imported once, in the server every child forks from: the parsers import
            # their libraries lazily, which cost each child ~100-200 ms per library.
            # (An unavailable one is skipped. Never the OCR engine: forking after ONNX
            # Runtime starts its threads isn't safe.)
            context.set_forkserver_preload(
                ["api.documents", "pdfplumber", "docx", "openpyxl", "pptx", "bs4"]
            )
            _context = context
        return _context


# A pathological PDF had pypdf log ten multi-kilobyte font dictionaries per parse.
_PARSER_LOGGERS = ("pypdf", "pdfminer")
_SAMPLE_CHARS = 200


class _CountingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0
        self.first = ""

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1
        if not self.first:
            self.first = record.getMessage()[:_SAMPLE_CHARS]


@contextmanager
def summarized_parser_warnings(filename: str) -> Iterator[None]:
    """Hold back the parser libraries' warnings while ``filename`` is parsed and log one
    line about them afterwards: how many, and the first (cut short)."""

    handler = _CountingHandler()
    saved = []
    for name in _PARSER_LOGGERS:
        library = logging.getLogger(name)
        saved.append((library, library.propagate))
        library.addHandler(handler)
        library.propagate = False
    try:
        yield
    finally:
        for library, propagate in saved:
            library.removeHandler(handler)
            library.propagate = propagate
        if handler.count:
            logger.warning(
                "Parsing %s: the parser reported %d problem(s) with the file; the first: %s",
                filename,
                handler.count,
                handler.first,
            )


def _parse(filename: str, content: bytes, settings: AppSettings) -> list[DocumentSection]:
    with summarized_parser_warnings(filename):
        return parse_document_bytes(filename, content, settings)


def _parse_in_child(
    connection: Connection, filename: str, content: bytes, settings: AppSettings
) -> None:
    try:
        # The child starts with no logging set up: its warnings bypassed LOG_FORMAT=json
        # and LOG_LEVEL.
        from api.logging_config import configure_logging

        configure_logging(settings.log_level, settings.log_format)
    except Exception:  # noqa: BLE001 - logging is no reason to fail a parse
        pass
    try:
        result: tuple[str, object] = ("ok", _parse(filename, content, settings))
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


def parse_with_limit(
    filename: str,
    content: bytes,
    settings: AppSettings,
    *,
    _target: Callable[[Connection, str, bytes, AppSettings], None] = _parse_in_child,
) -> list[DocumentSection]:
    """``parse_document_bytes`` in a child process that is killed after
    ``settings.parse_timeout_seconds``; in this process when that is 0.

    ``_target`` is the child's entry point (tests swap in one that crashes)."""

    timeout = float(settings.parse_timeout_seconds)
    if timeout <= 0:
        return _parse(filename, content, settings)

    context = _forkserver_context()
    receiver, sender = context.Pipe(duplex=False)
    try:
        process = context.Process(
            target=_target,
            args=(sender, filename, content, settings),
            name="docrag-parse",
            daemon=True,
        )
        process.start()
    except BaseException:
        # No child (the forkserver died, or no file descriptors left): nothing else
        # would close the pipe.
        receiver.close()
        sender.close()
        raise
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
            process.join(timeout=5)  # for its exit code
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
