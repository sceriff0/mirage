#!/usr/bin/env python3
"""Minimal registration script used by the pipeline.

This conservative implementation accepts multiple input preprocessed files and
creates a single merged output. Currently it copies the first available input to
the output path as a lightweight fallback. Replace with a real registration
implementation when available.

The functions are small and annotated so they can be unit tested or replaced
with an actual registration implementation later.
"""
from __future__ import annotations

import argparse
import shutil
import os
from typing import List, Optional

from scripts._common import ensure_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: input_files, out, qc_dir
    """
    parser = argparse.ArgumentParser(description='Minimal registration placeholder')
    parser.add_argument('--input-files', nargs='+', required=True, help='Preprocessed input files')
    parser.add_argument('--out', required=True, help='Output merged path')
    parser.add_argument('--qc-dir', required=False, help='QC directory for optional output')
    return parser.parse_args()


def merge_first_file(input_files: List[str], out: str, qc_dir: Optional[str] = None) -> int:
    """Copy the first available input file to ``out`` as a conservative merge.

    Parameters
    ----------
    input_files : list of str
        Paths to preprocessed input files (the first is used).
    out : str
        Destination path for merged output.
    qc_dir : str, optional
        Optional QC directory to create.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    ensure_dir(os.path.dirname(out) or '.')
    src = input_files[0]
    shutil.copy2(src, out)
    if qc_dir:
        ensure_dir(qc_dir)
    return 0


def main() -> int:
    args = parse_args()
    return merge_first_file(args.input_files, args.out, getattr(args, 'qc_dir', None))


if __name__ == '__main__':
    raise SystemExit(main())