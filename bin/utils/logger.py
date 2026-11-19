"""Pipeline-wide logging setup, configured once per process."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

__all__ = [
    "get_logger",
    "configure_logging",
    "log_progress",
]


_LOGGING_CONFIGURED = False
_LOG_LEVEL = logging.INFO
_LOG_FILE: Optional[Path] = None


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure root logging once; later calls are no-ops.

    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level.
    log_file : str or Path or None, optional
        Also log to this file when given.
    format_string : str or None, optional
        Log format; defaults to timestamp, name, level and message.
    """
    global _LOGGING_CONFIGURED, _LOG_LEVEL, _LOG_FILE

    if _LOGGING_CONFIGURED:
        logging.getLogger(__name__).debug(
            "Logging already configured, skipping reconfiguration"
        )
        return

    _LOG_LEVEL = level
    _LOG_FILE = Path(log_file) if log_file else None

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Drop existing handlers so messages are not duplicated.
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if _LOG_FILE:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(_LOG_FILE)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {_LOG_FILE}")

    _LOGGING_CONFIGURED = True
    root_logger.debug("Logging configuration complete")


def get_logger(name: str) -> logging.Logger:
    """Return the named logger, configuring logging with defaults if needed."""
    if not _LOGGING_CONFIGURED:
        configure_logging()

    return logging.getLogger(name)


def log_progress(message: str) -> None:
    """Print ``[YYYY-MM-DD HH:MM:SS] message`` to stdout and flush.

    Bypasses the logging framework, so log levels never filter it.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
