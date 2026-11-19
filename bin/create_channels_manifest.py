#!/usr/bin/env python3
"""
Create a channels manifest JSON from registered OME-TIFF files.

Reads OME-XML metadata from each *_registered.ome.tiff file in the input
directory and writes a JSON manifest mapping filename to channel names.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from logger import configure_logging, get_logger
from metadata import extract_channel_names_from_ome

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract channel names from registered OME-TIFF files into a JSON manifest."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_registered.ome.tiff files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output JSON manifest.",
    )
    return parser.parse_args()


def main():
    """CLI entry point: extract channel names from registered OME-TIFFs into a JSON manifest."""
    configure_logging(level=logging.INFO)
    args = parse_args()

    manifest = {}
    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.endswith("_registered.ome.tiff"):
            continue
        filepath = os.path.join(args.input_dir, filename)
        names = extract_channel_names_from_ome(filepath)
        if not names:
            raise RuntimeError(f"No OME metadata in registered file: {filepath}")
        manifest[filename] = names

    with open(args.output, "w") as fp:
        # write(dumps(...)) rather than dump(...): dump issues many small writes and is
        # slower. Fine here because a channel manifest is tiny (see
        # tests/test_json_write_sizing.py for the per-cell writers).
        fp.write(json.dumps(manifest))

    logger.info(f"Channels manifest: {len(manifest)} files")
    for fname, chs in manifest.items():
        logger.info(f"  {fname}: {chs}")


if __name__ == "__main__":
    raise SystemExit(main())
