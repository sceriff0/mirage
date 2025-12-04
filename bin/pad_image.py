#!/usr/bin/env python3
"""Pad a single image to specified target dimensions.

This script pads a single OME-TIFF image to target height and width,
ensuring uniform spatial dimensions for downstream registration.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def pad_image_to_shape(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    mode: str = 'constant'
) -> np.ndarray:
    """Pad image to target spatial dimensions using symmetric padding.

    Note: Only pads height and width - channel count is preserved.

    Parameters:
        img (ndarray): Input image in (C, H, W) format
        target_h (int): Target height
        target_w (int): Target width
        mode (str): Padding mode ('constant', 'edge', 'reflect', 'symmetric')

    Returns:
        ndarray: Padded image
    """
    c_img, h_img, w_img = img.shape

    # Calculate padding
    pad_h = target_h - h_img
    pad_w = target_w - w_img

    # No padding needed
    if pad_h == 0 and pad_w == 0:
        return img

    # Check if image is already larger
    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"Image ({h_img}x{w_img}) is larger than target ({target_h}x{target_w})"
        )

    # Symmetric padding (center the image)
    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before
    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before

    logger.info(f"    Padding: H={pad_h_before}+{pad_h_after}, W={pad_w_before}+{pad_w_after}")

    # Pad specification: ((C_before, C_after), (H_before, H_after), (W_before, W_after))
    pad_width = (
        (0, 0),                       # No channel padding
        (pad_h_before, pad_h_after),  # Height padding
        (pad_w_before, pad_w_after),  # Width padding
    )

    # Apply padding
    if mode == 'constant':
        padded = np.pad(img, pad_width, mode='constant', constant_values=0)
    else:
        padded = np.pad(img, pad_width, mode=mode)

    return padded


def pad_single_image(
    input_path: Path,
    output_path: Path,
    target_h: int,
    target_w: int,
    pad_mode: str = 'constant'
) -> None:
    """Pad single image to target spatial dimensions.

    Parameters:
        input_path (Path): Input image path
        output_path (Path): Output image path
        target_h (int): Target height
        target_w (int): Target width
        pad_mode (str): Padding mode
    """
    logger.info(f"Processing: {input_path.name}")

    # Load image
    img = tifffile.imread(str(input_path))
    original_dtype = img.dtype

    # Ensure (C, H, W) format
    if img.ndim == 2:
        img = img[np.newaxis, ...]

    logger.info(f"  Original shape: {img.shape}")
    logger.info(f"  Target dimensions: H={target_h}, W={target_w}")

    # Pad to target spatial dimensions (channels preserved)
    padded_img = pad_image_to_shape(img, target_h, target_w, mode=pad_mode)
    logger.info(f"  Padded shape: {padded_img.shape}")

    # Preserve original dtype
    padded_img = padded_img.astype(original_dtype)

    # Calculate file size to determine if BigTIFF is needed
    estimated_size = padded_img.nbytes
    use_bigtiff = estimated_size > 2**32  # 4GB limit

    if use_bigtiff:
        logger.info(f"  Using BigTIFF format (estimated size: {estimated_size / (1024**3):.2f} GB)")

    # Save padded image without compression (temporary file for GPU registration)
    logger.info(f"  Saving to: {output_path.name}")
    tifffile.imwrite(
        str(output_path),
        padded_img,
        photometric='minisblack',
        compression=None,  # No compression for faster I/O and lower memory
        bigtiff=use_bigtiff
    )
    logger.info("  ✓ Saved")


def main():
    parser = argparse.ArgumentParser(
        description="Pad image to target dimensions"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input image path'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output image path'
    )
    parser.add_argument(
        '--target-height',
        type=int,
        required=True,
        help='Target height in pixels'
    )
    parser.add_argument(
        '--target-width',
        type=int,
        required=True,
        help='Target width in pixels'
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
        input_path = Path(args.input)
        output_path = Path(args.output)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        pad_single_image(
            input_path,
            output_path,
            args.target_height,
            args.target_width,
            args.pad_mode
        )
        return 0

    except Exception as e:
        logger.error(f"Padding failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
