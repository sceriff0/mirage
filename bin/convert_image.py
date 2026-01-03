#!/usr/bin/env python3
"""Convert microscopy images to standardized OME-TIFF format.

This script uses aicsimageio with BioformatsReader to support universal image formats
including .nd2, .lif, .ndpi, .tiff, .czi, and many others. It automatically extracts
physical pixel sizes from the image metadata and places DAPI in channel 0, following
the channel mapping specified in the input CSV.
"""

import logging
import argparse
from pathlib import Path
import os
import tempfile

# Add parent directory to path to import lib modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger, configure_logging
from typing import List, Optional, Tuple

import tifffile
import numpy as np

try:
    from aicsimageio import AICSImage
    from aicsimageio.readers import BioformatsReader
    AICSIMAGEIO_AVAILABLE = True
except ImportError:
    AICSIMAGEIO_AVAILABLE = False

from utils.image_utils import ensure_dir

logger = get_logger(__name__)

PIXEL_SIZE_UM = 0.325


def read_image_aics(file_path: Path, scene: int = 0) -> Tuple[np.ndarray, dict]:
    """
    Read image file using aicsimageio with BioformatsReader for universal format support.

    Parameters
    ----------
    file_path : Path
        Path to image file (supports .nd2, .lif, .ndpi, .tiff, etc.)
    scene : int
        Scene/series index to read (default: 0)

    Returns
    -------
    tuple
        (image_data, metadata_dict)

    Raises
    ------
    ImportError
        If aicsimageio is not available
    """
    if not AICSIMAGEIO_AVAILABLE:
        raise ImportError(
            "This script requires 'aicsimageio' and bioformats support. "
            "Install with: pip install aicsimageio bioformats_jar"
        )

    logger.info(f"Reading image with aicsimageio: {file_path.name}")

    # Read image with BioformatsReader for maximum compatibility
    img = AICSImage(file_path, reader=BioformatsReader)

    # Get physical pixel sizes
    pixel_sizes = img.physical_pixel_sizes  # (Z, Y, X) in microns
    logger.info(f"Physical pixel sizes (Z, Y, X): {pixel_sizes}")

    # Get dimension information
    dims = img.dims
    order = img.dims.order  # e.g., "TCZYX"
    shape = img.shape  # Tuple in TCZYX order
    scenes = img.scenes  # Available scenes/series

    logger.info(f"Dimension order: {order}")
    logger.info(f"Shape: {shape}")
    logger.info(f"Available scenes: {scenes} (using scene {scene})")

    # Set the scene to read
    if len(scenes) > 0:
        img.set_scene(scenes[scene])
        logger.info(f"Selected scene: {scenes[scene]}")

    # Determine the appropriate dimension string for extraction
    # We want to get data in a format with C (channels) dimension
    # Handle different cases based on what dimensions exist

    has_t = 'T' in order and shape[order.index('T')] > 1
    has_z = 'Z' in order and shape[order.index('Z')] > 1
    has_c = 'C' in order and shape[order.index('C')] > 1

    # Build dimension string for extraction
    if has_c:
        # Multi-channel image
        if has_z:
            if has_t:
                extract_dims = "TCZYX"
                target_order = "TCZYX"
            else:
                extract_dims = "CZYX"
                target_order = "CZYX"
        else:
            # No Z or single Z
            if has_t:
                extract_dims = "TCYX"
                target_order = "TCYX"
            else:
                extract_dims = "CYX"
                target_order = "CYX"
    else:
        # Single channel or RGB image (NDPI case)
        # For NDPI files, they appear as RGB but each channel is identical
        # We just need the YX plane
        if has_z:
            extract_dims = "ZYX"
            target_order = "ZYX"
        else:
            extract_dims = "YX"
            target_order = "YX"

    logger.info(f"Extracting dimensions: {extract_dims}")

    # Get the image data
    image_data = img.get_image_data(extract_dims)

    logger.info(f"Extracted image shape: {image_data.shape}")

    # Determine channel axis index and number of channels
    if 'C' in target_order:
        c_axis_index = target_order.index('C')
        num_channels = image_data.shape[c_axis_index]
    else:
        # Single channel image
        c_axis_index = None
        num_channels = 1
        # Add channel dimension at position 0 for consistency
        image_data = np.expand_dims(image_data, axis=0)
        target_order = 'C' + target_order
        c_axis_index = 0
        logger.info(f"Added channel dimension. New shape: {image_data.shape}, order: {target_order}")

    # Extract physical pixel size (use Y dimension, typically same as X)
    physical_pixel_size = pixel_sizes.Y if pixel_sizes.Y is not None else PIXEL_SIZE_UM

    # Try to extract channel names from metadata if available (informational only)
    # CSV channel names are always used as the authoritative source
    channel_names_from_file = None
    try:
        if hasattr(img, 'channel_names') and img.channel_names:
            channel_names_from_file = img.channel_names
    except Exception:
        pass  # Channel names from file are optional/informational

    metadata = {
        'axes': target_order,
        'c_axis_index': c_axis_index,
        'num_channels': num_channels,
        'physical_pixel_size': physical_pixel_size,
        'original_order': order,
        'original_shape': shape,
        'scenes': scenes,
        'channel_names_from_file': channel_names_from_file
    }

    logger.info(f"Final axes order: {target_order}")
    logger.info(f"Channel axis index: {c_axis_index}")
    logger.info(f"Number of channels: {num_channels}")
    logger.info(f"Physical pixel size: {physical_pixel_size} µm")

    return image_data, metadata


def rearrange_channels(
    image_data: np.ndarray,
    c_axis_index: int,
    channel_mapping: List[str],
    target_channels: List[str]
) -> np.ndarray:
    """
    Rearrange channels so DAPI is in position 0.

    Parameters
    ----------
    image_data : np.ndarray
        Input image data
    c_axis_index : int
        Index of channel axis
    channel_mapping : List[str]
        Current channel names in order
    target_channels : List[str]
        Target channel names in desired order (DAPI first)

    Returns
    -------
    np.ndarray
        Image with rearranged channels
    """
    if c_axis_index is None:
        logger.warning("No channel axis found, returning original image")
        return image_data

    # Build mapping from current to target positions
    indices = []
    for target_ch in target_channels:
        try:
            idx = channel_mapping.index(target_ch)
            indices.append(idx)
        except ValueError:
            raise ValueError(f"Channel '{target_ch}' not found in image channels: {channel_mapping}")

    # Reorder along channel axis
    reordered = np.take(image_data, indices, axis=c_axis_index)

    logger.info(f"Rearranged channels: {channel_mapping} -> {target_channels}")

    return reordered


def convert_to_ome_tiff(
    input_path: Path,
    output_dir: Path,
    patient_id: str,
    channel_names: List[str],
    pixel_size_um: float = PIXEL_SIZE_UM
) -> Tuple[Path, List[str]]:
    """
    Convert image to standardized OME-TIFF format with DAPI in channel 0.

    Uses aicsimageio with BioformatsReader for universal format support.
    Automatically extracts physical pixel sizes from image metadata.

    Parameters
    ----------
    input_path : Path
        Path to input image file (supports .nd2, .lif, .ndpi, .tiff, .czi, etc.)
    output_dir : Path
        Output directory
    patient_id : str
        Patient identifier
    channel_names : List[str]
        Channel names as specified in input (any order)
    pixel_size_um : float
        Default pixel size in micrometers (used if not found in image metadata)

    Returns
    -------
    tuple
        (Path to output OME-TIFF file, List of channel names in output order)
    """
    # Find DAPI channel and move it to position 0
    dapi_index = None
    for i, ch in enumerate(channel_names):
        if ch.upper() == 'DAPI':
            dapi_index = i
            break

    if dapi_index is None:
        raise ValueError(f"DAPI channel not found in: {channel_names}")

    # Reorder channels to put DAPI first
    output_channels = channel_names.copy()
    if dapi_index != 0:
        logger.info(f"Moving DAPI from position {dapi_index} to position 0")
        dapi_ch = output_channels.pop(dapi_index)
        output_channels.insert(0, dapi_ch)
    else:
        logger.info("DAPI already in position 0")

    # Output filename includes patient_id and channels in output order
    channels_str = '_'.join(output_channels)
    output_filename = output_dir / f"{patient_id}_{channels_str}.ome.tif"

    logger.info(f"Converting: {input_path.name}")
    logger.info(f"Patient ID: {patient_id}")
    logger.info(f"Input channels (as specified): {channel_names}")
    logger.info(f"Output channels (DAPI first): {output_channels}")

    # Read image using aicsimageio (supports all formats)
    image_data, metadata = read_image_aics(input_path)

    # Use channel names from CSV input (as-is, no flipping)
    original_channels = channel_names

    # Validate channel count
    if metadata['num_channels'] != len(channel_names):
        logger.warning(
            f"Channel count mismatch: image has {metadata['num_channels']}, "
            f"specified {len(channel_names)}"
        )

    # Log channel names from file metadata if available (for reference only)
    # CSV channel names are always used as the authoritative source
    if metadata.get('channel_names_from_file'):
        file_channels = metadata['channel_names_from_file']
        logger.info(f"Channel names from file metadata (for reference): {file_channels}")
        logger.info(f"Using CSV channel names as authoritative: {channel_names}")

    # Use physical pixel size from image metadata if available, otherwise use provided value
    actual_pixel_size = metadata.get('physical_pixel_size', pixel_size_um)
    if actual_pixel_size != pixel_size_um:
        logger.info(f"Using pixel size from image metadata: {actual_pixel_size} µm (overriding {pixel_size_um} µm)")

    # Rearrange channels to output order (DAPI first)
    if metadata['c_axis_index'] is not None and original_channels != output_channels:
        image_data = rearrange_channels(
            image_data,
            metadata['c_axis_index'],
            original_channels,
            output_channels
        )

    # Prepare OME metadata with output channel order
    ome_metadata = {
        'axes': metadata['axes'],
        'Channel': {'Name': output_channels},
        'PhysicalSizeX': actual_pixel_size,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': actual_pixel_size,
        'PhysicalSizeYUnit': 'µm'
    }

    logger.info(f"Writing OME-TIFF with metadata:")
    logger.info(f"  - Axes: {ome_metadata['axes']}")
    logger.info(f"  - Channels: {ome_metadata['Channel']['Name']}")
    logger.info(f"  - PhysicalSize: {ome_metadata['PhysicalSizeX']} {ome_metadata['PhysicalSizeXUnit']}")
    logger.info(f"  - Shape: {image_data.shape}")

    # Save as OME-TIFF
    tifffile.imwrite(
        output_filename,
        image_data,
        metadata=ome_metadata,
        photometric='minisblack',
        ome=True,
        bigtiff=True
    )

    logger.info(f"Saved: {output_filename.name}")

    # Verify
    with tifffile.TiffFile(output_filename) as tif:
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
            logger.info("  ✓ OME-XML metadata present")
        else:
            logger.warning("  ⚠ No OME metadata in saved file!")

    return output_filename, output_channels


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert microscopy images to standardized OME-TIFF using aicsimageio',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to input image file (supports .nd2, .lif, .ndpi, .tiff, .czi, etc.)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--patient_id',
        type=str,
        required=True,
        help='Patient identifier'
    )
    parser.add_argument(
        '--channels',
        type=str,
        required=True,
        help='Comma-separated channel names (must include DAPI, will be placed in channel 0)'
    )
    parser.add_argument(
        '--pixel_size',
        type=float,
        default=PIXEL_SIZE_UM,
        help='Pixel size in micrometers'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    configure_logging(level=logging.INFO)

    args = parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    channel_names = [ch.strip() for ch in args.channels.split(',')]

    ensure_dir(str(output_dir))

    logger.info("=" * 80)
    logger.info(f"Converting image: {input_path.name}")
    logger.info(f"Patient ID: {args.patient_id}")
    logger.info(f"Input channels: {channel_names}")
    logger.info(f"Pixel size: {args.pixel_size} µm")
    logger.info("=" * 80)

    output_path, output_channels = convert_to_ome_tiff(
        input_path,
        output_dir,
        args.patient_id,
        channel_names,
        args.pixel_size
    )

    logger.info("=" * 80)
    logger.info(f"✓ Conversion complete: {output_path.name}")
    logger.info(f"✓ Output channel order: {output_channels}")
    logger.info("=" * 80)

    # Write output channels to a file for Nextflow to parse
    channels_file = output_dir / f"{args.patient_id}_channels.txt"
    with open(channels_file, 'w') as f:
        f.write(','.join(output_channels))

    return 0


if __name__ == '__main__':
    exit(main())
