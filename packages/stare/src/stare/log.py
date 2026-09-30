"""Logging for the STARE stages: one root configuration, per-module loggers.

A copy of the two functions the stages use from the mirage pipeline's
``bin/utils/logger.py`` (``configure_logging``, ``get_logger``), so the package
imports nothing from the pipeline. The copy is kept identical to the original by
``tests/test_stare_package_copies_do_not_drift.py`` on the mirage side, which
compares the two definitions' ASTs (docstrings aside) -- edit both or neither.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

__all__ = ["get_logger", "configure_logging"]

# Global state for singleton pattern
_LOGGING_CONFIGURED = False
_LOG_LEVEL = logging.INFO
_LOG_FILE: Optional[Path] = None


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure the root logger once: console (stdout) and an optional file.

    A second call is a no-op, so a stage's ``main`` can call it unconditionally
    whether it runs as its own process or inside ``stare register``.
    """
    global _LOGGING_CONFIGURED, _LOG_LEVEL, _LOG_FILE

    if _LOGGING_CONFIGURED:
        # Already configured - this prevents multiple configuration
        logging.getLogger(__name__).debug(
            "Logging already configured, skipping reconfiguration"
        )
        return

    _LOG_LEVEL = level
    _LOG_FILE = Path(log_file) if log_file else None

    # Default format string follows best practices
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger (affects all loggers)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplication
    # This is important when running in notebooks or interactive environments
    root_logger.handlers.clear()

    # Console handler (always present for visibility)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional, for persistent logs)
    if _LOG_FILE:
        # Ensure parent directory exists
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
    """A logger named ``name``; configures logging with the defaults first if needed."""
    # Auto-configure with defaults if not configured
    if not _LOGGING_CONFIGURED:
        configure_logging()

    return logging.getLogger(name)
