"""Centralized logging configuration with singleton pattern.

This module provides a singleton logger factory that ensures consistent
logging configuration across the entire pipeline following Python best practices.

Examples
--------
>>> from logger import get_logger, configure_logging
>>> configure_logging(level=logging.DEBUG, log_file="pipeline.log")
>>> logger = get_logger(__name__)
>>> logger.info("Processing started")

Notes
-----
The singleton pattern ensures that:
- Logging is configured once globally
- All modules share the same logging configuration
- File handlers are not duplicated
- Log level can be changed globally at runtime
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Generator, Optional, TypeVar

__all__ = [
    "get_logger",
    "configure_logging",
    "log_progress",
    "log_timing",
    "timed",
]

# Type variable for preserving function signatures in decorators
F = TypeVar("F", bound=Callable)


# Global state for singleton pattern
_LOGGING_CONFIGURED = False
_LOG_LEVEL = logging.INFO
_LOG_FILE: Optional[Path] = None


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    format_string: Optional[str] = None
) -> None:
    """Configure global logging settings (call once at application startup).

    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level from the logging module
        (e.g., logging.DEBUG, logging.INFO, logging.WARNING).
    log_file : str or Path or None, optional
        Path to log file. If None, logs only to console.
    format_string : str or None, optional
        Custom format string for log messages.
        If None, uses default format with timestamp, name, level, and message.

    Returns
    -------
    None

    Notes
    -----
    This function should be called once at the start of your application.
    Subsequent calls will have no effect to prevent configuration conflicts.
    To reconfigure logging, restart the Python interpreter.

    The default format includes:
    - Timestamp in 'YYYY-MM-DD HH:MM:SS' format
    - Logger name (typically module __name__)
    - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Log message

    Examples
    --------
    Basic configuration (console only):

    >>> configure_logging(level=logging.INFO)

    With log file:

    >>> configure_logging(level=logging.DEBUG, log_file="pipeline.log")

    Custom format:

    >>> configure_logging(
    ...     level=logging.INFO,
    ...     format_string='%(levelname)s - %(message)s'
    ... )
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
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure root logger (affects all loggers)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplication
    # This is important when running in notebooks or interactive environments
    root_logger.handlers.clear()

    # Console handler (always present for visibility)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional, for persistent logs)
    if _LOG_FILE:
        # Ensure parent directory exists
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(_LOG_FILE)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {_LOG_FILE}")

    _LOGGING_CONFIGURED = True
    root_logger.debug("Logging configuration complete")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Parameters
    ----------
    name : str
        Logger name, typically `__name__` of the calling module.
        This allows log messages to show which module they came from.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    If `configure_logging()` hasn't been called yet, this will auto-configure
    logging with default settings (INFO level, console output only).

    The logger name should follow Python naming conventions:
    - Use `__name__` for module-level loggers
    - Use `__name__.ClassName` for class-specific loggers
    - Hierarchical names are separated by dots

    Examples
    --------
    Basic usage in a module:

    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
    >>> logger.debug("Detailed debug information")
    >>> logger.warning("Warning message")
    >>> logger.error("Error occurred")

    In a class:

    >>> class ImageProcessor:
    ...     def __init__(self):
    ...         self.logger = get_logger(f"{__name__}.ImageProcessor")
    ...     def process(self):
    ...         self.logger.info("Processing image")
    """
    # Auto-configure with defaults if not configured
    if not _LOGGING_CONFIGURED:
        configure_logging()

    return logging.getLogger(name)


def log_progress(message: str) -> None:
    """Print timestamped progress messages to stdout with flush.

    This is a simple utility function for scripts that need direct console
    output with timestamps. For more structured logging, use get_logger().

    Parameters
    ----------
    message : str
        Progress message to print.

    Returns
    -------
    None

    Notes
    -----
    This function prints directly to stdout with flush=True to ensure
    immediate visibility in log files and console output.

    The timestamp format is 'YYYY-MM-DD HH:MM:SS' for consistency with
    the main logging configuration.

    This function does NOT use the Python logging framework - it prints
    directly to stdout. Use this only for simple scripts or when you need
    guaranteed console output that won't be filtered by log levels.

    Examples
    --------
    >>> log_progress("Processing started")
    [2025-12-28 10:30:45] Processing started

    >>> log_progress("Completed 50% of task")
    [2025-12-28 10:31:12] Completed 50% of task

    See Also
    --------
    get_logger : Get a proper logger instance for structured logging
    configure_logging : Configure the logging system
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@contextmanager
def log_timing(
    operation: str,
    logger: Optional[logging.Logger] = None
) -> Generator[None, None, None]:
    """Context manager for timing operations with logging.

    Logs the start of an operation and its completion time. Useful for
    measuring and logging the duration of long-running operations.

    Parameters
    ----------
    operation : str
        Description of the operation being timed.
    logger : logging.Logger, optional
        Logger to use. If None, creates one for this module.

    Yields
    ------
    None

    Examples
    --------
    Basic usage:

    >>> with log_timing("Image registration"):
    ...     perform_registration()
    Starting: Image registration
    Completed: Image registration (45.23s)

    With custom logger:

    >>> logger = get_logger(__name__)
    >>> with log_timing("Loading data", logger=logger):
    ...     data = load_data()
    """
    _logger = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    _logger.info(f"Starting: {operation}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _logger.info(f"Completed: {operation} ({elapsed:.2f}s)")


def timed(func: F) -> F:
    """Decorator for timing function execution.

    Logs the execution time of the decorated function at DEBUG level.
    Useful for performance monitoring without cluttering INFO logs.

    Parameters
    ----------
    func : callable
        Function to decorate.

    Returns
    -------
    callable
        Wrapped function that logs its execution time.

    Notes
    -----
    The timing is logged at DEBUG level to avoid noise in normal output.
    Enable DEBUG logging to see timing information.

    Examples
    --------
    >>> @timed
    ... def process_image(path):
    ...     # processing logic
    ...     return result
    >>> result = process_image("image.tif")
    # At DEBUG level: "process_image completed in 2.34s"
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.getLogger(func.__module__).debug(
            f"{func.__name__} completed in {elapsed:.2f}s"
        )
        return result
    return wrapper  # type: ignore[return-value]
