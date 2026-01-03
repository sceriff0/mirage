#!/usr/bin/env python3
"""Convert microscopy images (ND2, TIFF, etc.) to standardized OME-TIFF format.

This script handles multiple input formats and places DAPI in channel 0,
following the channel mapping specified in the input CSV.
"""

import logging
import argparse
from pathlib import Path

# Add parent directory to path to import lib modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger, configure_logging
from typing import List, Optional, Tuple

import tifffile
import numpy as np

try:
    import nd2
    ND2_AVAILABLE = True
except ImportError:
    ND2_AVAILABLE = False

from utils.image_utils import ensure_dir

logger = get_logger(__name__)

PIXEL_SIZE_UM = 0.325


def detect_format(file_path: Path) -> str:
    """
    Detect image format from file extension.

    Parameters
    ----------
    file_path : Path
        Path to image file

    Returns
    -------
    str
        Format identifier: 'nd2', 'tiff', 'ome_tiff'

    Raises
    ------
    ValueError
        If format is not supported
    """
    suffix = file_path.suffix.lower()

    if suffix == '.nd2':
        if not ND2_AVAILABLE:
            raise ImportError("ND2 support requires 'nd2' package")
        return 'nd2'
    elif suffix in ['.tif', '.tiff']:
        # Check if it's already OME-TIFF
        try:
            with tifffile.TiffFile(file_path) as tif:
                if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                    return 'ome_tiff'
        except Exception:
            pass
        return 'tiff'
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def read_nd2(file_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Read ND2 file and extract metadata.

    Parameters
    ----------
    file_path : Path
        Path to ND2 file

    Returns
    -------
    tuple
        (image_data, metadata_dict)
    """
    with nd2.ND2File(file_path) as f:
        image_data = f.asarray()
        sizes = f.sizes

        # Find channel axis
        axis_keys = list(sizes.keys())
        if 'c' in sizes:
            c_key = 'c'
        elif 'C' in sizes:
            c_key = 'C'
        else:
            raise ValueError(f"No channel dimension found in ND2. Axes: {sizes}")

        c_axis_index = axis_keys.index(c_key)
        num_channels = sizes[c_key]
        axes_string = "".join(axis_keys).upper()

        metadata = {
            'axes': axes_string,
            'c_axis_index': c_axis_index,
            'num_channels': num_channels
        }

        return image_data, metadata


def read_tiff(file_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Read TIFF file and extract metadata.

    Parameters
    ----------
    file_path : Path
        Path to TIFF file

    Returns
    -------
    tuple
        (image_data, metadata_dict)
    """
    with tifffile.TiffFile(file_path) as tif:
        image_data = tif.asarray()

        # Try to extract axes from OME metadata or assume standard order
        axes = None
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
            # Try to parse from OME metadata
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(tif.ome_metadata)
                pixels = root.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
                if pixels is not None:
                    axes = pixels.get('DimensionOrder', 'CYX')
            except Exception:
                pass

        # Default assumptions based on shape
        if axes is None:
            ndim = image_data.ndim
            if ndim == 2:
                axes = 'YX'
                c_axis_index = None
                num_channels = 1
            elif ndim == 3:
                # Assume CYX or YXC
                if image_data.shape[0] < image_data.shape[2]:
                    axes = 'CYX'
                    c_axis_index = 0
                    num_channels = image_data.shape[0]
                else:
                    axes = 'YXC'
                    c_axis_index = 2
                    num_channels = image_data.shape[2]
            elif ndim == 4:
                axes = 'ZCYX'
                c_axis_index = 1
                num_channels = image_data.shape[1]
            else:
                raise ValueError(f"Cannot infer axes for {ndim}D image")
        else:
            c_axis_index = axes.upper().index('C') if 'C' in axes.upper() else None
            num_channels = image_data.shape[c_axis_index] if c_axis_index is not None else 1

        metadata = {
            'axes': axes.upper(),
            'c_axis_index': c_axis_index,
            'num_channels': num_channels
        }

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

    Parameters
    ----------
    input_path : Path
        Path to input image file
    output_dir : Path
        Output directory
    patient_id : str
        Patient identifier
    channel_names : List[str]
        Channel names as specified in input (any order)
    pixel_size_um : float
        Pixel size in micrometers

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

    # Detect format and read image
    fmt = detect_format(input_path)
    logger.info(f"Detected format: {fmt}")

    if fmt == 'nd2':
        image_data, metadata = read_nd2(input_path)
        # Use channel names from CSV input (as-is, no flipping)
        original_channels = channel_names
    elif fmt in ['tiff', 'ome_tiff']:
        image_data, metadata = read_tiff(input_path)
        # Use channel names from CSV input
        original_channels = channel_names
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    # Validate channel count
    if metadata['num_channels'] != len(channel_names):
        logger.warning(
            f"Channel count mismatch: image has {metadata['num_channels']}, "
            f"specified {len(channel_names)}"
        )

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
        'PhysicalSizeX': pixel_size_um,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': pixel_size_um,
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
        description='Convert microscopy images to standardized OME-TIFF',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to input image file (ND2, TIFF, etc.)'
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
