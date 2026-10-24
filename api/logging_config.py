"""Logging configuration for the application's own loggers.

The root logger defaults to WARNING, which silences every ``logger.info`` call in
``api``/``eval`` (request logging, ingest progress, warmup notices). This module
configures only those two namespaces via ``dictConfig`` so uvicorn's own loggers
(``uvicorn``, ``uvicorn.error``, ``uvicorn.access``) are left untouched — with one
exception: under ``LOG_FORMAT=json`` uvicorn's plain-text access line is silenced and
``api.access`` (written by ``RequestIdMiddleware``) logs each request as JSON instead,
with its request id, which uvicorn's line can't carry.

``propagate`` is kept True: our StreamHandler lives on the ``api``/``eval`` loggers
themselves, and nothing in this app or uvicorn's default config attaches a handler
to the root logger, so propagation causes no double emit in production. Keeping it
True lets pytest's ``caplog`` (which captures via a root-attached handler) still see
``api.*`` records in tests.
"""

from __future__ import annotations

import json
import logging
import logging.config
from typing import Literal

from api.request_id import RequestIdFilter

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"


class JsonFormatter(logging.Formatter):
    """One JSON object per line (LOG_FORMAT=json), for log shippers."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        http = getattr(record, "http", None)
        if isinstance(http, dict):
            entry["http"] = http
        audit = getattr(record, "audit", None)
        if isinstance(audit, dict):
            entry["audit"] = audit
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(level: str, log_format: Literal["text", "json"] = "text") -> None:
    """Route the ``api`` and ``eval`` loggers to stderr at ``level``.

    Idempotent: ``dictConfig`` replaces the configured handlers on each call, so
    repeated invocations (multiple ``create_app`` calls in tests) never duplicate
    handlers. ``disable_existing_loggers`` is False so uvicorn's already-configured
    loggers keep working. Every line carries the request id (api/request_id.py).
    """

    formatter: dict[str, object] = (
        {"()": JsonFormatter} if log_format == "json" else {"format": _LOG_FORMAT}
    )
    json_logs = log_format == "json"
    # api.access replaces uvicorn's access line under JSON only; in text mode uvicorn's
    # line stays, and a second one per request would just repeat it.
    access_loggers: dict[str, object] = {
        "api.access": {"level": "INFO" if json_logs else "WARNING", "propagate": True},
    }
    if json_logs:
        access_loggers["uvicorn.access"] = {"level": "WARNING"}
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_id": {"()": RequestIdFilter}},
            "formatters": {"default": formatter},
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["request_id"],
                    "stream": "ext://sys.stderr",
                }
            },
            "loggers": {
                "api": {
                    "level": level,
                    "handlers": ["default"],
                    "propagate": True,
                },
                "eval": {
                    "level": level,
                    "handlers": ["default"],
                    "propagate": True,
                },
                **access_loggers,
                # Audit lines (api/audit.py) are kept at any LOG_LEVEL: a quieter log
                # must not also lose the record of what was deleted.
                "api.audit": {"level": "INFO", "propagate": True},
            },
        }
    )
