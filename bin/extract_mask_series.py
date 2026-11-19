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

#: TIFF tile size (px) for the mask writes below: a TIFF layout choice, so windowed
#: per-cell reads need not decode the whole plane. Half of the 2048 image tile, sized
#: for the mask's many small region reads.
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
    # Accept unsigned-integer masks (lossless upcast to uint32) and reject anything
    # else (float probability maps, signed/int64) rather than corrupting labels.
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
    # zlib, like every other mask writer here; cheap on label masks. bigtiff is
    # required: a 40000x40000 uint32 mask is 6.4 GB and classic TIFF offsets overflow
    # past 4 GB (tests/test_mask_series_write_contract.py).
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
