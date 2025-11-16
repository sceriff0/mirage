"""Small common utilities to keep scripts KISS and DRY.

Place simple helpers here used by pipeline scripts (logging setup, dir creation,
file utilities). Keep this module minimal to avoid heavy dependencies.

The helpers in this module are intentionally small and typed so they can be
imported safely from other scripts and unit tested.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


__all__ = ["ensure_dir", "setup_file_logger", "write_text"]


def ensure_dir(path: Optional[str]) -> None:
    """Ensure a directory exists (no-op if exists).

    Parameters
    ----------
    path : str or None
        Path to create. If ``None``, the function is a no-op.
    """
    if path is None:
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_file_logger(log_file: Optional[str]) -> None:
    """Attach a simple file handler to the root logger if ``log_file`` is provided.

    Parameters
    ----------
    log_file : str or None
        Path to a log file. If ``None`` nothing is attached.
    """
    if not log_file:
        return
    fh = logging.FileHandler(log_file)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)


def write_text(path: str, text: str) -> None:
    """Write text to a file, ensuring the parent directory exists.

    Parameters
    ----------
    path : str
        Path of the file to write.
    text : str
        Text content to write.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, 'w') as f:
        f.write(text)
