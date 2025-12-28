#!/usr/bin/env python3
"""
Split multichannel TIFF images into individual single-channel TIFF files.
DAPI channel is only saved from the reference image.

Usage:
    python split_multichannel.py input.ome.tiff output_folder --is-reference
    python split_multichannel.py input.ome.tiff output_folder  # skips DAPI
"""

import argparse
import os
import tifffile
import numpy as np
import logging

# Add parent directory to path to import lib modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.logger import get_logger, configure_logging

logger = get_logger(__name__)


def get_ome_channel_names(tiff_path):
    """Try to extract channel names from OME-TIFF metadata."""
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            if tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

                channels = root.findall('.//ome:Channel', ns)
                if not channels:
                    # Try without namespace
                    channels = root.findall('.//{*}Channel')

                names = []
                for ch in channels:
                    name = ch.get('Name') or ch.get('ID', f'Channel_{len(names)}')
                    names.append(name)

                if names:
                    return names
    except Exception as e:
        logger.warning(f"Could not parse OME metadata: {e}")

    return None


def split_multichannel_tiff(input_path, output_dir, is_reference=False, channel_names=None):
    """
    Split a multichannel TIFF into single-channel TIFFs.
    DAPI is only saved if is_reference=True.

    Parameters:
    -----------
    input_path : str
        Path to the input multichannel TIFF file
    output_dir : str
        Directory to save the single-channel TIFFs
    is_reference : bool
        If True, saves DAPI channel. If False, skips DAPI.
    channel_names : list of str, optional
        Names for each channel. If None, tries to read from OME metadata.

    Returns:
    --------
    list : Paths to the saved files
    """
    # Read the image
    logger.info(f"Reading: {input_path}")
    logger.info(f"  Is reference: {is_reference}")

    img = tifffile.imread(input_path)
    logger.info(f"  Image shape: {img.shape}, dtype: {img.dtype}")

    # Determine array layout and number of channels
    if img.ndim == 2:
        # Single channel image
        logger.info("  Single-channel image")
        img = img[np.newaxis, ...]  # Add channel dimension
        n_channels = 1
    elif img.ndim == 3:
        # Could be (C, Y, X) or (Y, X, C)
        # Assume smallest dimension is channels
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            # (C, Y, X) format
            n_channels = img.shape[0]
            logger.info(f"  Format: (C, Y, X) with {n_channels} channels")
        elif img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
            # (Y, X, C) format - transpose to (C, Y, X)
            img = np.transpose(img, (2, 0, 1))
            n_channels = img.shape[0]
            logger.info(f"  Format: (Y, X, C) with {n_channels} channels - transposed")
        else:
            # Ambiguous - assume (C, Y, X)
            n_channels = img.shape[0]
            logger.info(f"  Assuming format: (C, Y, X) with {n_channels} channels")
    else:
        raise ValueError(f"Unexpected number of dimensions: {img.ndim}")

    # Get channel names
    if channel_names is None:
        channel_names = get_ome_channel_names(input_path)

    if channel_names is None:
        channel_names = [f"Channel_{i}" for i in range(n_channels)]
        logger.info(f"  Using default channel names")
    else:
        logger.info(f"  Channel names from metadata: {len(channel_names)} channels")

    # Validate channel count
    if len(channel_names) != n_channels:
        logger.warning(f"  Warning: {len(channel_names)} names vs {n_channels} channels")
        if len(channel_names) < n_channels:
            channel_names.extend([f"Channel_{i}" for i in range(len(channel_names), n_channels)])
        else:
            channel_names = channel_names[:n_channels]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save each channel (skip DAPI if not reference)
    saved_paths = []
    skipped_count = 0

    for i, name in enumerate(channel_names):
        # Check if this is DAPI channel
        is_dapi = name.upper() == 'DAPI'

        # Skip DAPI if this is not the reference image
        if is_dapi and not is_reference:
            logger.info(f"  Skipping: {name} (DAPI from non-reference)")
            skipped_count += 1
            continue

        # Clean the channel name for use as filename
        clean_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
        output_path = os.path.join(output_dir, f"{clean_name}.tiff")

        channel_data = img[i]
        tifffile.imwrite(output_path, channel_data, bigtiff=True, compression='zlib')

        saved_paths.append(output_path)
        status = " (DAPI from reference)" if is_dapi else ""
        logger.info(f"  Saved: {clean_name}.tiff{status}")

    logger.info(f"✓ Saved {len(saved_paths)} channels, skipped {skipped_count}")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description='Split multichannel TIFF into single-channel TIFFs (DAPI only from reference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input',
        type=str,
        help='Path to input multichannel TIFF file'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save single-channel TIFFs'
    )

    parser.add_argument(
        '--is-reference',
        action='store_true',
        help='This is the reference image (save DAPI)'
    )

    parser.add_argument(
        '--channels',
        nargs='+',
        default=None,
        help='Channel names (optional, will try to read from OME metadata)'
    )

    args = parser.parse_args()

    configure_logging(level=logging.INFO)

    logger.info("=" * 80)
    logger.info("SPLIT MULTICHANNEL TIFF")
    logger.info("=" * 80)

    split_multichannel_tiff(args.input, args.output_dir, args.is_reference, args.channels)

    logger.info("=" * 80)
    logger.info("✓ COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
