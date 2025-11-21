#!/usr/bin/env python3
"""Apply VALIS registration transforms to individual slides.

This script loads a pickled VALIS registrar and applies the computed transforms
to warp a single slide. Designed to be run in parallel for each slide.

The script handles both regular slides and the reference slide (which may need
special handling for DAPI extraction).
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from pathlib import Path
from typing import Optional

# Disable Numba caching to avoid file locator errors
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'

import tifffile
from valis import registration

from _common import ensure_dir

logger = logging.getLogger(__name__)

__all__ = [
    "load_registrar_pickle",
    "warp_and_save_slide",
]


def load_registrar_pickle(pickle_path: str) -> registration.Valis:
    """
    Load pickled VALIS registrar using official VALIS method.

    Parameters
    ----------
    pickle_path : str
        Path to registrar pickle file.

    Returns
    -------
    registrar : Valis
        Loaded VALIS registrar with all computed transforms.

    Notes
    -----
    Uses registration.load_registrar() which is the official VALIS method
    for loading pickled registrar objects.
    """
    logger.info(f"Loading registrar pickle: {pickle_path}")

    # Use official VALIS method to load registrar
    registrar = registration.load_registrar(pickle_path)

    logger.info(f"✓ Registrar loaded ({len(registrar.slide_dict)} slides)")
    logger.info(f"  Reference image: {registrar.reference_img_f}")

    return registrar


def warp_and_save_slide(
    registrar: registration.Valis,
    slide_name: str,
    output_path: str,
) -> None:
    """
    Warp and save a single slide using VALIS official method.

    Parameters
    ----------
    registrar : Valis
        Loaded VALIS registrar with computed transforms.
    slide_name : str
        Name of slide to warp (basename without path/extension).
    output_path : str
        Path to save registered OME-TIFF file.

    Notes
    -----
    Uses the official VALIS Slide.warp_and_save_slide() method which:
    - Applies all computed transforms (rigid + non-rigid + micro)
    - Saves directly as OME-TIFF with proper metadata
    - Handles reference slides automatically
    - Uses level 0 (full resolution) for warping
    """
    logger.info("=" * 80)
    logger.info(f"WARPING AND SAVING SLIDE: {slide_name}")
    logger.info("=" * 80)
    logger.info(f"Output path: {output_path}")

    start_time = time.time()

    # Get slide object from registrar using official method
    if slide_name not in registrar.slide_dict:
        raise KeyError(f"Slide '{slide_name}' not found in registrar")

    slide_obj = registrar.slide_dict[slide_name]

    # Get slide information
    channel_names = slide_obj.reader.metadata.channel_names
    logger.info(f"Channels: {channel_names} ({len(channel_names)} channels)")

    # Get slide dimensions before warping
    slide_dims = slide_obj.slide_dimensions_wh
    logger.info(f"Original dimensions: {slide_dims[0]} x {slide_dims[1]}")

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    ensure_dir(str(output_dir))

    # Warp and save using official VALIS method
    logger.info("Warping slide with computed transforms...")
    logger.info("  Applying: rigid + non-rigid + micro transforms")

    # Use official VALIS warp_and_save_slide method
    # This handles everything: warping, OME-TIFF format, metadata
    slide_obj.warp_and_save_slide(
        dst_f=output_path,
        level=0,              # Full resolution
        non_rigid=True,       # Apply non-rigid transforms
        crop=True,            # Crop to reference overlap
        interp_method="bicubic"  # High-quality interpolation
    )

    # Verify saved file
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time

    logger.info("")
    logger.info(f"✓ Slide warped and saved ({file_size_mb:.2f} MB)")
    logger.info(f"  Time elapsed: {elapsed:.2f}s")

    # Log output file info
    with tifffile.TiffFile(output_path) as tif:
        img = tif.asarray()
        logger.info(f"  Output shape: {img.shape}")
        logger.info(f"  Output dtype: {img.dtype}")
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
            logger.info(f"  ✓ OME metadata present")

    # Clean up
    gc.collect()

    logger.info("")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Apply VALIS registration to individual slide',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--registrar-pickle',
        type=str,
        required=True,
        help='Path to pickled VALIS registrar'
    )

    parser.add_argument(
        '--slide-name',
        type=str,
        required=True,
        help='Name of slide to register (basename without path/extension)'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to save registered OME-TIFF file'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    # Load registrar
    registrar = load_registrar_pickle(args.registrar_pickle)

    # Warp and save slide (single operation using official VALIS method)
    warp_and_save_slide(
        registrar=registrar,
        slide_name=args.slide_name,
        output_path=args.output_file,
    )

    logger.info("=" * 80)
    logger.info("✓ SLIDE REGISTRATION COMPLETE")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())
