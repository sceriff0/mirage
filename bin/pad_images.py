#!/usr/bin/env python3
"""Pad multiple images to common maximum dimensions.

This script processes multiple OME-TIFF images and pads them all to the
maximum height and width found across all images. This ensures uniform
dimensions for downstream registration.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def find_max_dimensions(image_paths: List[Path]) -> Tuple[int, int, int]:
    """Find maximum dimensions across all images.

    Parameters:
        image_paths (list): List of paths to OME-TIFF images

    Returns:
        tuple: (max_channels, max_height, max_width)
    """
    max_c, max_h, max_w = 0, 0, 0

    logger.info(f"Scanning {len(image_paths)} images to find maximum dimensions...")

    for img_path in image_paths:
        img = tifffile.imread(str(img_path))

        # Ensure (C, H, W) format
        if img.ndim == 2:
            c, h, w = 1, img.shape[0], img.shape[1]
        elif img.ndim == 3:
            c, h, w = img.shape
        else:
            raise ValueError(f"Unexpected image dimensions: {img.shape} for {img_path.name}")

        logger.info(f"  {img_path.name}: ({c}, {h}, {w})")

        max_c = max(max_c, c)
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    logger.info(f"\nMaximum dimensions found: ({max_c}, {max_h}, {max_w})")
    return max_c, max_h, max_w


def pad_image_to_shape(
    img: np.ndarray,
    target_shape: Tuple[int, int, int],
    mode: str = 'constant'
) -> np.ndarray:
    """Pad image to target shape using symmetric padding.

    Parameters:
        img (ndarray): Input image in (C, H, W) format
        target_shape (tuple): Target (C, H, W) shape
        mode (str): Padding mode ('constant', 'edge', 'reflect', 'symmetric')

    Returns:
        ndarray: Padded image
    """
    c_img, h_img, w_img = img.shape
    c_target, h_target, w_target = target_shape

    # Validate channels match
    if c_img != c_target:
        raise ValueError(
            f"Channel mismatch: image has {c_img} channels, "
            f"target has {c_target} channels"
        )

    # Calculate padding
    pad_h = h_target - h_img
    pad_w = w_target - w_img

    # No padding needed
    if pad_h == 0 and pad_w == 0:
        return img

    # Symmetric padding (center the image)
    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before
    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before

    logger.info(f"    Padding: H={pad_h_before}+{pad_h_after}, W={pad_w_before}+{pad_w_after}")

    # Pad specification: ((C_before, C_after), (H_before, H_after), (W_before, W_after))
    pad_width = (
        (0, 0),
        (pad_h_before, pad_h_after),
        (pad_w_before, pad_w_after),
    )

    # Apply padding
    if mode == 'constant':
        padded = np.pad(img, pad_width, mode='constant', constant_values=0)
    else:
        padded = np.pad(img, pad_width, mode=mode)

    return padded


def pad_images(
    input_paths: List[Path],
    output_dir: Path,
    pad_mode: str = 'constant'
) -> None:
    """Pad all images to common maximum dimensions.

    Parameters:
        input_paths (list): List of input image paths
        output_dir (Path): Output directory for padded images
        pad_mode (str): Padding mode
    """
    # Find maximum dimensions
    max_c, max_h, max_w = find_max_dimensions(input_paths)
    target_shape = (max_c, max_h, max_w)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pad each image
    logger.info(f"\nPadding all images to ({max_c}, {max_h}, {max_w})...")
    logger.info(f"Padding mode: {pad_mode}\n")

    for idx, img_path in enumerate(input_paths, 1):
        logger.info(f"[{idx}/{len(input_paths)}] Processing: {img_path.name}")

        # Load image
        img = tifffile.imread(str(img_path))
        original_dtype = img.dtype

        # Ensure (C, H, W) format
        if img.ndim == 2:
            img = img[np.newaxis, ...]

        logger.info(f"  Original shape: {img.shape}")

        # Pad to target dimensions
        padded_img = pad_image_to_shape(img, target_shape, mode=pad_mode)
        logger.info(f"  Padded shape: {padded_img.shape}")

        # Preserve original dtype
        padded_img = padded_img.astype(original_dtype)

        # Save padded image
        output_path = output_dir / img_path.name
        logger.info(f"  Saving to: {output_path.name}")
        tifffile.imwrite(
            str(output_path),
            padded_img,
            photometric='minisblack',
            compression='zlib'
        )
        logger.info("  ✓ Saved\n")

    logger.info(f"✓ All {len(input_paths)} images padded successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Pad multiple images to common maximum dimensions"
    )
    parser.add_argument(
        '--input',
        nargs='+',
        required=True,
        help='Input image paths (multiple files)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for padded images'
    )
    parser.add_argument(
        '--pad-mode',
        type=str,
        default='constant',
        choices=['constant', 'edge', 'reflect', 'symmetric'],
        help='Padding mode (default: constant/zeros)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        input_paths = [Path(p) for p in args.input]
        output_dir = Path(args.output_dir)

        # Validate inputs
        for path in input_paths:
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")

        pad_images(input_paths, output_dir, args.pad_mode)
        return 0

    except Exception as e:
        logger.error(f"Padding failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
