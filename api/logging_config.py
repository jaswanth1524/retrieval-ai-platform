"""Logging configuration for the application's own loggers.

The root logger defaults to WARNING, which silences every ``logger.info`` call in
``api``/``eval`` (request logging, ingest progress, warmup notices). This module
configures only those two namespaces via ``dictConfig`` so uvicorn's own loggers
(``uvicorn``, ``uvicorn.error``, ``uvicorn.access``) are left entirely untouched.

``propagate`` is kept True: our StreamHandler lives on the ``api``/``eval`` loggers
themselves, and nothing in this app or uvicorn's default config attaches a handler
to the root logger, so propagation causes no double emit in production. Keeping it
True lets pytest's ``caplog`` (which captures via a root-attached handler) still see
``api.*`` records in tests.
"""

from __future__ import annotations

import logging.config

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str) -> None:
    """Route the ``api`` and ``eval`` loggers to stderr at ``level``.

    Idempotent: ``dictConfig`` replaces the configured handlers on each call, so
    repeated invocations (multiple ``create_app`` calls in tests) never duplicate
    handlers. ``disable_existing_loggers`` is False so uvicorn's already-configured
    loggers keep working.
    """

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"default": {"format": _LOG_FORMAT}},
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
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
            },
        }
    )
