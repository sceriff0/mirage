"""ATEIA pipeline utilities.

This package provides common utilities for the ATEIA image processing pipeline:

- **logger**: Centralized logging configuration with timing utilities
- **constants**: Pipeline-wide constants and exit codes
- **cli**: Command-line interface utilities
- **image_utils**: Image I/O and processing utilities
- **metadata**: OME-TIFF metadata handling
- **registration_utils**: Registration helper functions

Examples
--------
>>> from utils import get_logger, ExitCode, DEFAULT_FOV_SIZE
>>> logger = get_logger(__name__)
>>> logger.info(f"Using FOV size: {DEFAULT_FOV_SIZE}")
"""

from __future__ import annotations

# Logger utilities
from .logger import (
    get_logger,
    configure_logging,
    log_progress,
    log_timing,
    timed,
)

# Constants
from .constants import (
    ExitCode,
    DEFAULT_FOV_SIZE,
    DEFAULT_PIXEL_SIZE,
    DEFAULT_TILE_SIZE,
    LOG_SEPARATOR,
    TIFF_EXTENSIONS,
    OME_TIFF_EXTENSIONS,
)

# CLI utilities
from .cli import (
    create_base_parser,
    setup_logging_from_args,
    add_input_output_args,
)

__all__ = [
    # Logger
    "get_logger",
    "configure_logging",
    "log_progress",
    "log_timing",
    "timed",
    # Constants
    "ExitCode",
    "DEFAULT_FOV_SIZE",
    "DEFAULT_PIXEL_SIZE",
    "DEFAULT_TILE_SIZE",
    "LOG_SEPARATOR",
    "TIFF_EXTENSIONS",
    "OME_TIFF_EXTENSIONS",
    # CLI
    "create_base_parser",
    "setup_logging_from_args",
    "add_input_output_args",
]
