"""CLI utilities for consistent argument parsing.

This module provides base argument parser and logging setup utilities
to standardize command-line interfaces across all pipeline scripts.

Examples
--------
>>> from utils.cli import create_base_parser, setup_logging_from_args
>>>
>>> parser = create_base_parser("Process images with BaSiC correction")
>>> parser.add_argument("--image", required=True, help="Input image path")
>>> args = parser.parse_args()
>>> setup_logging_from_args(args)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

__all__ = [
    "create_base_parser",
    "setup_logging_from_args",
    "add_input_output_args",
]


def create_base_parser(
    description: str,
    epilog: Optional[str] = None
) -> argparse.ArgumentParser:
    """Create an argument parser with common options.

    Creates a parser pre-configured with standard options for verbosity,
    quiet mode, and log file output. Use this as the base for all CLI
    scripts to ensure consistent behavior.

    Parameters
    ----------
    description : str
        Brief description of the script's purpose.
    epilog : str, optional
        Additional text to display after the argument help.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with common options.

    Examples
    --------
    >>> parser = create_base_parser("Convert ND2 files to OME-TIFF")
    >>> parser.add_argument("--input", required=True)
    >>> args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Logging control group
    log_group = parser.add_argument_group("logging options")

    log_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging",
    )

    log_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error output (ERROR level only)",
    )

    log_group.add_argument(
        "--log-file",
        type=Path,
        metavar="PATH",
        help="Write logs to specified file in addition to console",
    )

    return parser


def setup_logging_from_args(
    args: argparse.Namespace,
    default_level: int = logging.INFO
) -> None:
    """Configure logging based on parsed command-line arguments.

    Reads the standard logging arguments (--verbose, --quiet, --log-file)
    from the parsed args and configures the logging system accordingly.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from argparse (must include verbose, quiet, log_file).
    default_level : int, default=logging.INFO
        Default logging level if neither --verbose nor --quiet is specified.

    Notes
    -----
    Precedence: --quiet > --verbose > default_level

    Examples
    --------
    >>> parser = create_base_parser("My script")
    >>> args = parser.parse_args()
    >>> setup_logging_from_args(args)
    """
    from .logger import configure_logging

    # Determine log level (quiet takes precedence over verbose)
    if getattr(args, "quiet", False):
        level = logging.ERROR
    elif getattr(args, "verbose", False):
        level = logging.DEBUG
    else:
        level = default_level

    # Get log file if specified
    log_file = getattr(args, "log_file", None)

    configure_logging(level=level, log_file=log_file)


def add_input_output_args(
    parser: argparse.ArgumentParser,
    input_help: str = "Input file or directory path",
    output_help: str = "Output directory path",
    input_required: bool = True,
    output_required: bool = True,
) -> argparse.ArgumentParser:
    """Add standard input/output arguments to a parser.

    Convenience function to add --input and --output-dir arguments
    with consistent naming and help text.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.
    input_help : str, default="Input file or directory path"
        Help text for the input argument.
    output_help : str, default="Output directory path"
        Help text for the output argument.
    input_required : bool, default=True
        Whether --input is required.
    output_required : bool, default=True
        Whether --output-dir is required.

    Returns
    -------
    argparse.ArgumentParser
        The same parser with arguments added (for chaining).

    Examples
    --------
    >>> parser = create_base_parser("Convert images")
    >>> add_input_output_args(parser, input_help="ND2 file to convert")
    >>> args = parser.parse_args()
    """
    io_group = parser.add_argument_group("input/output options")

    io_group.add_argument(
        "--input", "-i",
        type=Path,
        required=input_required,
        metavar="PATH",
        help=input_help,
    )

    io_group.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=output_required,
        metavar="DIR",
        help=output_help,
    )

    return parser
