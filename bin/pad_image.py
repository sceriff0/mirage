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

    # Extract channel names from input file before loading
    channel_names = []
    try:
        with tifffile.TiffFile(str(input_path)) as tif:
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                channels = root.findall('.//ome:Channel', ns)
                channel_names = [ch.get('Name', '') for ch in channels]
                channel_names = [name for name in channel_names if name]
                if channel_names:
                    logger.info(f"  Extracted {len(channel_names)} channel names from OME metadata")
    except Exception as e:
        logger.warning(f"  Could not extract channel names from metadata: {e}")

    # Fallback: extract from filename
    if not channel_names:
        filename = input_path.stem
        name_part = filename.replace('_corrected', '').replace('_preprocessed', '').replace('_registered', '')
        parts = name_part.split('_')
        # Skip first part if it looks like a patient/sample ID
        if len(parts) > 1 and '-' in parts[0] and any(c.isdigit() for c in parts[0]):
            channel_names = parts[1:]
        else:
            channel_names = parts
        logger.info(f"  Inferred channel names from filename: {channel_names}")

    # Load image
    img = tifffile.imread(str(input_path))
    original_dtype = img.dtype

    # Ensure (C, H, W) format
    if img.ndim == 2:
        img = img[np.newaxis, ...]

    logger.info(f"  Original shape: {img.shape}")
    logger.info(f"  Target dimensions: H={target_h}, W={target_w}")

    # Ensure channel_names matches number of channels
    num_channels = img.shape[0]
    if len(channel_names) != num_channels:
        logger.warning(f"  Channel name count mismatch: {len(channel_names)} names vs {num_channels} channels")
        if len(channel_names) < num_channels:
            channel_names.extend([f"channel_{i}" for i in range(len(channel_names), num_channels)])
        else:
            channel_names = channel_names[:num_channels]

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

    # Generate OME-XML metadata with channel names
    num_channels, height, width = padded_img.shape
    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
    <Image ID="Image:0" Name="Padded">
        <Pixels ID="Pixels:0" Type="{padded_img.dtype.name}"
                SizeX="{width}" SizeY="{height}" SizeZ="1" SizeC="{num_channels}" SizeT="1"
                DimensionOrder="XYCZT"
                PhysicalSizeX="0.325" PhysicalSizeY="0.325" PhysicalSizeXUnit="µm" PhysicalSizeYUnit="µm">
            {chr(10).join(f'            <Channel ID="Channel:0:{i}" Name="{name}" SamplesPerPixel="1" />' for i, name in enumerate(channel_names))}
            <TiffData />
        </Pixels>
    </Image>
</OME>'''

    # Save padded image without compression (temporary file for GPU registration)
    logger.info(f"  Saving to: {output_path.name} with channel names: {channel_names}")
    tifffile.imwrite(
        str(output_path),
        padded_img,
        metadata={'axes': 'CYX'},
        description=ome_xml,
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
