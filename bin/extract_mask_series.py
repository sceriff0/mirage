#!/usr/bin/env python3
"""Extract the mask series (Image:1) from a mask-carrying pyramid OME-TIFF.

Writes cell_mask.tif and nuclei_mask.tif (uint32). Fast-fails with a clear
message when the pyramid has no second series or it is not a 2-channel uint32
mask series (the incremental cyclic-IF mode relies on this).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from logger import configure_logging, get_logger  # noqa: E402
from ome_io import write_tiff  # noqa: E402

logger = get_logger(__name__)

#: TIFF tile size (px) for the cell/nuclei mask writes below -- a TIFF LAYOUT choice, not a
#: processing tile size. See ``bin/convert_image.py:35-49`` for why an untiled write forces
#: every downstream region read to decode the whole plane. Deliberately half of
#: CONVERT_TIFF_TILE/WRITE_TILE's 2048, not copied from it: measured directly on the mask's
#: own read pattern (many small windowed per-cell/per-region reads, not whole-plane decodes)
#: -- see docs/perf/2026-08-26-rss.md's tiled-vs-striped table.
MASK_TIFF_TILE = 1024


def main() -> None:
    """CLI entry point: extract the mask series from a pyramid OME-TIFF into cell/nuclei mask TIFFs."""
    configure_logging(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyramid", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    with tifffile.TiffFile(args.pyramid) as tif:
        if len(tif.series) < 2:
            logger.error(
                "%s has no mask series (found %d series). The prior run must set "
                "embed_masks=true with expanded compartment quantification.",
                args.pyramid,
                len(tif.series),
            )
            raise SystemExit(1)
        masks = tif.series[1].asarray()
    if masks.ndim != 3 or masks.shape[0] != 2:
        logger.error(
            "mask series has shape %s; expected (2, H, W) [cell, nuclei].", masks.shape
        )
        raise SystemExit(1)
    # Fast-fail on dtypes that cannot losslessly represent uint32 label IDs.
    # Accept unsigned-integer masks (uint8/uint16/uint32 -> lossless upcast) and
    # reject anything else (float probability maps, signed/int64 arrays) loudly
    # rather than silently corrupting downstream quantification.
    if masks.dtype not in (np.uint8, np.uint16, np.uint32):
        logger.error(
            "mask series in %s has dtype %s; expected an unsigned integer label mask "
            "(uint8/uint16/uint32). Refusing to coerce (silent-corruption risk).",
            args.pyramid,
            masks.dtype,
        )
        raise SystemExit(1)
    if masks.dtype != np.uint32:
        masks = masks.astype(np.uint32)  # lossless upcast from uint8/uint16

    args.outdir.mkdir(parents=True, exist_ok=True)
    # zlib matches every other mask writer here (segment.py, segment_cellsam.py,
    # segment_instantseg.py) and is close to free on label masks -- long runs of identical
    # integers. bigtiff is not an optimisation: a 40000x40000 uint32 mask is 6.4 GB, and classic
    # TIFF's 32-bit offsets overflow past 4 GB (struct.error: 'I' format requires
    # 0 <= number <= 4294967295), which is the same reason tiled_stitch.py:118 sets it.
    # Guarded by tests/test_mask_series_write_contract.py.
    for name, plane in (("cell_mask.tif", masks[0]), ("nuclei_mask.tif", masks[1])):
        write_tiff(
            args.outdir / name,
            plane,
            compression="zlib",
            bigtiff=True,
            tile=(MASK_TIFF_TILE, MASK_TIFF_TILE),
        )
    logger.info(
        "Extracted cell_mask + nuclei_mask from %s (series 1, %dx%d)",
        args.pyramid,
        masks.shape[1],
        masks.shape[2],
    )


if __name__ == "__main__":
    raise SystemExit(main())
