#!/usr/bin/env python3
"""Minimal registration script used by the pipeline.

This conservative implementation accepts multiple input preprocessed files and
creates a single merged output. Currently it copies the first available input to
the output path as a lightweight fallback. Replace with a real registration
implementation when available.
"""
import argparse
import shutil
import os
from scripts._common import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Minimal registration placeholder')
    parser.add_argument('--input-files', nargs='+', required=True, help='Preprocessed input files')
    parser.add_argument('--out', required=True, help='Output merged path')
    parser.add_argument('--qc-dir', required=False, help='QC directory for optional output')
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(os.path.dirname(args.out) or '.')
    # Copy first file as merged output (conservative fallback)
    src = args.input_files[0]
    shutil.copy2(src, args.out)
    if args.qc_dir:
        ensure_dir(args.qc_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
# Placeholder for registration script
print('Registration placeholder')