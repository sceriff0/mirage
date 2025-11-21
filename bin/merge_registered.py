#!/usr/bin/env python3
"""Merge registered slides with channel deduplication.

This script merges all individually registered slides into a single multichannel
OME-TIFF file, removing duplicate channels across slides.

The merging strategy:
1. Load all registered slides
2. Identify unique channels (by name)
3. For duplicate channels, keep the first occurrence
4. Stack all unique channels into final merged image
5. Save as OME-TIFF with proper metadata
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from _common import ensure_dir

logger = logging.getLogger(__name__)

__all__ = [
    "load_registered_slide",
    "deduplicate_channels",
    "merge_slides",
]


def load_registered_slide(slide_path: str) -> tuple[np.ndarray, list[str], dict]:
    """
    Load a registered OME-TIFF slide.

    Parameters
    ----------
    slide_path : str
        Path to registered OME-TIFF file.

    Returns
    -------
    image_data : ndarray, shape (C, Y, X)
        Multichannel image data.
    channel_names : list of str
        Channel names from OME metadata.
    metadata : dict
        Image metadata including physical pixel size.
    """
    logger.info(f"Loading: {Path(slide_path).name}")

    with tifffile.TiffFile(slide_path) as tif:
        image_data = tif.asarray()

        # Extract channel names from OME metadata
        channel_names = []
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
            # Parse OME-XML for channel names
            ome_xml = tif.ome_metadata
            # Simple parsing - look for Channel Name attributes
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(ome_xml)
                # Find all Channel elements
                for channel in root.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel'):
                    name = channel.get('Name')
                    if name:
                        channel_names.append(name)
            except Exception as e:
                logger.warning(f"  Could not parse channel names from OME-XML: {e}")

        # If no channel names found, use indices
        if not channel_names:
            n_channels = image_data.shape[0] if image_data.ndim == 3 else 1
            channel_names = [f"Channel_{i}" for i in range(n_channels)]
            logger.warning(f"  No channel names in metadata, using: {channel_names}")
        else:
            logger.info(f"  Channels: {channel_names}")

    # Ensure (C, Y, X) format
    if image_data.ndim == 2:
        image_data = image_data[np.newaxis, :, :]

    logger.info(f"  Shape: {image_data.shape}")
    logger.info(f"  Dtype: {image_data.dtype}")

    # Extract metadata (use first slide's metadata for physical size)
    metadata = {
        'physical_size_x': 0.325,  # Default from params
        'physical_size_y': 0.325,
        'physical_size_unit': 'µm',
    }

    return image_data, channel_names, metadata


def deduplicate_channels(
    slides_data: list[tuple[np.ndarray, list[str], str]],
) -> tuple[np.ndarray, list[str]]:
    """
    Merge slides and deduplicate channels by name.

    Parameters
    ----------
    slides_data : list of tuple
        List of (image_data, channel_names, slide_name) tuples.

    Returns
    -------
    merged_image : ndarray, shape (C_unique, Y, X)
        Merged image with deduplicated channels.
    unique_channel_names : list of str
        Unique channel names in order.

    Notes
    -----
    Deduplication strategy:
    - Keep first occurrence of each channel name
    - Log when duplicates are removed
    - All slides must have same (Y, X) dimensions
    """
    logger.info("=" * 80)
    logger.info("DEDUPLICATING CHANNELS")
    logger.info("=" * 80)

    # Check all slides have same spatial dimensions
    spatial_dims = [data.shape[1:] for data, _, _ in slides_data]
    if len(set(spatial_dims)) > 1:
        raise ValueError(f"All slides must have same spatial dimensions. Found: {set(spatial_dims)}")

    ref_shape = spatial_dims[0]
    logger.info(f"Spatial dimensions: {ref_shape[0]} x {ref_shape[1]}")
    logger.info("")

    # Collect all channels with their source slides
    all_channels = []
    for image_data, channel_names, slide_name in slides_data:
        for ch_idx, ch_name in enumerate(channel_names):
            all_channels.append({
                'name': ch_name,
                'slide': slide_name,
                'index': ch_idx,
                'data': image_data[ch_idx, :, :],
            })

    logger.info(f"Total channels before deduplication: {len(all_channels)}")

    # Deduplicate by channel name (keep first occurrence)
    seen_names = set()
    unique_channels = []
    duplicates_removed = []

    for channel in all_channels:
        ch_name = channel['name']
        if ch_name not in seen_names:
            seen_names.add(ch_name)
            unique_channels.append(channel)
        else:
            duplicates_removed.append(f"{ch_name} (from {channel['slide']})")

    logger.info(f"Unique channels after deduplication: {len(unique_channels)}")

    if duplicates_removed:
        logger.info("")
        logger.info(f"Removed {len(duplicates_removed)} duplicate channels:")
        for dup in duplicates_removed:
            logger.info(f"  - {dup}")
    else:
        logger.info("No duplicate channels found")

    logger.info("")
    logger.info("Final channel list:")
    for i, ch in enumerate(unique_channels):
        logger.info(f"  {i+1}. {ch['name']} (from {ch['slide']})")

    # Stack unique channels into merged image
    logger.info("")
    logger.info("Stacking channels into merged image...")
    merged_image = np.stack([ch['data'] for ch in unique_channels], axis=0)

    logger.info(f"  Merged shape: {merged_image.shape}")
    logger.info(f"  Merged dtype: {merged_image.dtype}")

    # Extract unique channel names in order
    unique_channel_names = [ch['name'] for ch in unique_channels]

    return merged_image, unique_channel_names


def merge_slides(
    slide_paths: list[str],
    output_path: str,
    reference_slide: Optional[str] = None,
) -> tuple[str, int]:
    """
    Merge registered slides with channel deduplication.

    Parameters
    ----------
    slide_paths : list of str
        Paths to registered OME-TIFF files.
    output_path : str
        Path to save merged OME-TIFF file.
    reference_slide : str, optional
        Name of reference slide (for logging).

    Returns
    -------
    output_path : str
        Path to saved merged file.
    n_unique_channels : int
        Number of unique channels in merged image.
    """
    logger.info("=" * 80)
    logger.info("MERGING REGISTERED SLIDES")
    logger.info("=" * 80)
    logger.info(f"Number of slides: {len(slide_paths)}")
    logger.info(f"Reference slide: {reference_slide}")
    logger.info(f"Output: {output_path}")
    logger.info("")

    start_time = time.time()

    # Load all registered slides
    logger.info("Loading registered slides...")
    slides_data = []
    for slide_path in slide_paths:
        image_data, channel_names, metadata = load_registered_slide(slide_path)
        slide_name = Path(slide_path).stem.replace('_registered', '')
        slides_data.append((image_data, channel_names, slide_name))

    logger.info("")

    # Deduplicate channels and merge
    merged_image, unique_channel_names = deduplicate_channels(slides_data)

    # Use metadata from first slide
    _, _, metadata = slides_data[0]

    # Save merged image
    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING MERGED IMAGE")
    logger.info("=" * 80)

    ensure_dir(str(Path(output_path).parent))

    ome_metadata = {
        'axes': 'CYX',
        'Channel': {'Name': unique_channel_names},
        'PhysicalSizeX': metadata['physical_size_x'],
        'PhysicalSizeXUnit': metadata['physical_size_unit'],
        'PhysicalSizeY': metadata['physical_size_y'],
        'PhysicalSizeYUnit': metadata['physical_size_unit'],
    }

    logger.info(f"Output path: {output_path}")
    logger.info(f"  Shape: {merged_image.shape}")
    logger.info(f"  Channels: {len(unique_channel_names)}")
    logger.info(f"  Dtype: {merged_image.dtype}")
    logger.info(f"  Physical size: {ome_metadata['PhysicalSizeX']} {ome_metadata['PhysicalSizeXUnit']}")

    tifffile.imwrite(
        output_path,
        merged_image,
        metadata=ome_metadata,
        photometric='minisblack',
        compression='zlib',
        ome=True,
    )

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time

    logger.info(f"✓ Merged image saved ({file_size_mb:.2f} MB)")
    logger.info(f"  Total time: {elapsed:.2f}s")

    # Clean up
    del merged_image, slides_data
    gc.collect()

    return output_path, len(unique_channel_names)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge registered slides with channel deduplication',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--input-files',
        type=str,
        nargs='+',
        required=True,
        help='Paths to registered OME-TIFF files'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to save merged OME-TIFF file'
    )

    # Optional arguments
    parser.add_argument(
        '--reference-slide',
        type=str,
        default=None,
        help='Name of reference slide (for logging)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    # Merge slides
    output_path, n_channels = merge_slides(
        slide_paths=args.input_files,
        output_path=args.output_file,
        reference_slide=args.reference_slide,
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ MERGE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Merged image: {output_path}")
    logger.info(f"Total unique channels: {n_channels}")

    return 0


if __name__ == '__main__':
    exit(main())
