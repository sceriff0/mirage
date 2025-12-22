#!/usr/bin/env python3
"""Create pyramidal OME-TIFF from merged image with masks.

This script takes an already-merged multichannel image and adds segmentation
and phenotype masks, then creates a pyramidal OME-TIFF using pyvips for
efficient visualization.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pyvips
import tifffile

logger = logging.getLogger(__name__)


def create_pyramidal_ometiff(
    merged_image_path: Path,
    seg_mask_path: Path,
    phenotype_mask_path: Path,
    output_path: Path,
    pyramid_resolutions: int = 3,
    pyramid_scale: int = 2
) -> None:
    """Create pyramidal OME-TIFF from merged image and masks using pyvips.

    Parameters
    ----------
    merged_image_path : Path
        Path to merged multichannel OME-TIFF file.
    seg_mask_path : Path
        Path to segmentation mask file (TIFF).
    phenotype_mask_path : Path
        Path to phenotype mask file (TIFF).
    output_path : Path
        Output pyramidal OME-TIFF file path.
    pyramid_resolutions : int, optional
        Number of pyramid levels. Default is 3.
    pyramid_scale : int, optional
        Scale factor between pyramid levels. Default is 2.
    """
    logger.info("=" * 80)
    logger.info("Creating Pyramidal OME-TIFF with pyvips")
    logger.info("=" * 80)

    # Load merged image using pyvips
    logger.info(f"\nLoading merged image: {merged_image_path}")
    merged_vips = pyvips.Image.new_from_file(str(merged_image_path), access='sequential')

    logger.info(f"  Shape: {merged_vips.width} x {merged_vips.height}")
    logger.info(f"  Bands: {merged_vips.bands}")
    logger.info(f"  Format: {merged_vips.format}")

    # Load masks with tifffile and convert to pyvips
    logger.info(f"\nLoading segmentation mask: {seg_mask_path}")
    seg_mask_np = tifffile.imread(seg_mask_path)
    if seg_mask_np.ndim > 2:
        seg_mask_np = seg_mask_np[0] if seg_mask_np.shape[0] < seg_mask_np.shape[-1] else seg_mask_np[..., 0]

    logger.info(f"  Shape: {seg_mask_np.shape}")
    logger.info(f"  Dtype: {seg_mask_np.dtype}")

    # Convert to uint32 if needed
    if seg_mask_np.dtype != np.uint32:
        seg_mask_np = seg_mask_np.astype(np.uint32)

    # Create pyvips image from numpy array
    seg_mask_vips = pyvips.Image.new_from_memory(
        seg_mask_np.tobytes(),
        seg_mask_np.shape[1],  # width
        seg_mask_np.shape[0],  # height
        1,  # bands
        'uint'
    )

    logger.info(f"\nLoading phenotype mask: {phenotype_mask_path}")
    phenotype_mask_np = tifffile.imread(phenotype_mask_path)
    if phenotype_mask_np.ndim > 2:
        phenotype_mask_np = phenotype_mask_np[0] if phenotype_mask_np.shape[0] < phenotype_mask_np.shape[-1] else phenotype_mask_np[..., 0]

    logger.info(f"  Shape: {phenotype_mask_np.shape}")
    logger.info(f"  Dtype: {phenotype_mask_np.dtype}")

    # Convert to uint32 if needed
    if phenotype_mask_np.dtype != np.uint32:
        phenotype_mask_np = phenotype_mask_np.astype(np.uint32)

    # Create pyvips image from numpy array
    phenotype_mask_vips = pyvips.Image.new_from_memory(
        phenotype_mask_np.tobytes(),
        phenotype_mask_np.shape[1],  # width
        phenotype_mask_np.shape[0],  # height
        1,  # bands
        'uint'
    )

    # Combine merged image with masks
    logger.info(f"\nCombining images:")
    logger.info(f"  Merged image: {merged_vips.bands} bands")
    logger.info(f"  Adding segmentation mask: 1 band")
    logger.info(f"  Adding phenotype mask: 1 band")

    # Bandjoin to combine all channels
    combined = merged_vips.bandjoin([seg_mask_vips, phenotype_mask_vips])

    total_bands = combined.bands
    logger.info(f"  Total bands: {total_bands}")

    # Create pyramid using pyvips
    logger.info(f"\nCreating pyramid with {pyramid_resolutions} levels, scale={pyramid_scale}")
    logger.info(f"Writing pyramidal OME-TIFF: {output_path}")

    # Write pyramidal TIFF with pyvips
    # This is equivalent to bfconvert with pyramid settings
    combined.tiffsave(
        str(output_path),
        compression='lzw',
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
        subifd=True,
        bigtiff=True,
        depth='onetile',
        # Set number of pyramid levels
        page_height=2048  # Page height for efficient streaming
    )

    logger.info(f"\n✓ Pyramidal OME-TIFF created successfully")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Size: {output_path.stat().st_size / 1e9:.2f} GB")
    logger.info("=" * 80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Create pyramidal OME-TIFF from merged image and masks using pyvips',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--merged-image',
        required=True,
        help='Path to merged multichannel OME-TIFF file'
    )
    parser.add_argument(
        '--seg-mask',
        required=True,
        help='Path to segmentation mask file (TIFF)'
    )
    parser.add_argument(
        '--phenotype-mask',
        required=True,
        help='Path to phenotype mask file (TIFF)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output pyramidal OME-TIFF file'
    )
    parser.add_argument(
        '--pyramid-resolutions',
        type=int,
        default=3,
        help='Number of pyramid levels'
    )
    parser.add_argument(
        '--pyramid-scale',
        type=int,
        default=2,
        help='Scale factor between pyramid levels'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    # Create pyramid
    create_pyramidal_ometiff(
        merged_image_path=Path(args.merged_image),
        seg_mask_path=Path(args.seg_mask),
        phenotype_mask_path=Path(args.phenotype_mask),
        output_path=Path(args.output),
        pyramid_resolutions=args.pyramid_resolutions,
        pyramid_scale=args.pyramid_scale
    )

    return 0


if __name__ == '__main__':
    exit(main())
