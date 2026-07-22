from __future__ import annotations

import logging

from api.logging_config import configure_logging


def test_configure_logging_sets_api_logger_level() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger("api").level == logging.DEBUG

    configure_logging("WARNING")
    assert logging.getLogger("api").level == logging.WARNING


def test_configure_logging_configures_eval_namespace() -> None:
    configure_logging("INFO")
    assert logging.getLogger("eval").level == logging.INFO


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("INFO")
    handlers = logging.getLogger("api").handlers
    assert len(handlers) == 1


def test_configure_logging_leaves_api_logger_propagating() -> None:
    # propagate must stay True so pytest's caplog (root-attached) still sees api.* records.
    configure_logging("INFO")
    assert logging.getLogger("api").propagate is True
