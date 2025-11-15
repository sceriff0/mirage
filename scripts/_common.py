"""Small common utilities to keep scripts KISS and DRY.

Place simple helpers here used by pipeline scripts (logging setup, dir creation,
file utilities). Keep this module minimal to avoid heavy dependencies.
"""
import logging
import os
from pathlib import Path


def ensure_dir(path):
    """Ensure a directory exists (no-op if exists)."""
    if path is None:
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_file_logger(log_file: str):
    """Attach a simple file handler to the root logger if log_file is provided."""
    if not log_file:
        return
    fh = logging.FileHandler(log_file)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)


def write_text(path, text):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(text)
