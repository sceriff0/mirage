"""bin/utils/logger.py — the logging singleton.

Every bin/ script calls configure_logging() then get_logger(__name__). The two
properties that matter are invisible when broken: configuring twice must not
duplicate handlers (every line then appears twice in a .command.log, which is
how a 200 MB log file happens), and get_logger before configure_logging must
auto-configure rather than return a logger that silently drops everything.
"""

from __future__ import annotations

import logging

import pytest
from utils import logger as mirage_logger


@pytest.fixture(autouse=True)
def _reset_logging():
    """The module is a singleton over the ROOT logger, so each test has to put
    the process back as it found it or the next one inherits its handlers."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    saved_flag = mirage_logger._LOGGING_CONFIGURED
    yield
    root.handlers = saved_handlers
    root.setLevel(saved_level)
    mirage_logger._LOGGING_CONFIGURED = saved_flag


def test_get_logger_auto_configures_rather_than_returning_a_silent_logger():
    mirage_logger._LOGGING_CONFIGURED = False
    logging.getLogger().handlers = []
    log = mirage_logger.get_logger("mirage.test.auto")
    assert logging.getLogger().handlers, (
        "get_logger returned without configuring: every message from this logger "
        "would go nowhere, and the script would look silent rather than broken"
    )
    assert log.name == "mirage.test.auto"


def test_configuring_twice_does_not_duplicate_handlers():
    """Two handlers means every line appears twice in .command.log. That is how a
    long-running task's log reaches hundreds of megabytes without anyone noticing
    anything is wrong."""
    mirage_logger._LOGGING_CONFIGURED = False
    logging.getLogger().handlers = []
    mirage_logger.configure_logging(level=logging.INFO)
    first = len(logging.getLogger().handlers)
    mirage_logger.configure_logging(level=logging.INFO)
    assert len(logging.getLogger().handlers) == first


def test_a_log_file_is_created_with_its_parent_directory(tmp_path):
    """The path a process passes is inside its own task directory, which may not
    exist yet -- mkdir(parents=True) is the difference between a log file and a
    FileNotFoundError at startup."""
    mirage_logger._LOGGING_CONFIGURED = False
    logging.getLogger().handlers = []
    target = tmp_path / "nested" / "deeper" / "run.log"
    mirage_logger.configure_logging(level=logging.INFO, log_file=target)
    mirage_logger.get_logger("mirage.test.file").info("hello from the test")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert target.exists()
    assert "hello from the test" in target.read_text()


def test_the_requested_level_is_applied():
    mirage_logger._LOGGING_CONFIGURED = False
    logging.getLogger().handlers = []
    mirage_logger.configure_logging(level=logging.DEBUG)
    assert logging.getLogger().level == logging.DEBUG


def test_log_progress_writes_to_stdout(capsys):
    """log_progress bypasses the logger on purpose (it is for direct console
    output), so it is the one function whose output does NOT move when logging is
    reconfigured -- which is why the lib probe's println analogue exists too."""
    mirage_logger.log_progress("a progress line")
    assert "a progress line" in capsys.readouterr().out
