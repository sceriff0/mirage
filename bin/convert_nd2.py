#!/usr/bin/env python3
"""Convert a single ND2 file to OME-TIFF format with reversed channel order.

This script converts one ND2 microscopy file to OME-TIFF format, reversing
the channel order and extracting channel names from the filename.
Designed to be called once per file by Nextflow.
"""

import logging
import argparse
from pathlib import Path

import nd2
import tifffile
import numpy as np

from _common import ensure_dir

logger = logging.getLogger(__name__)

PIXEL_SIZE_UM = 0.325


def get_channel_axis_info(sizes):
    """
    Find the channel axis key ('c' or 'C') and its index.

    Parameters
    ----------
    sizes : dict
        ND2 sizes dictionary

    Returns
    -------
    tuple
        (axis_key, axis_index)

    Raises
    ------
    ValueError
        If no channel dimension found
    """
    axis_keys = list(sizes.keys())

    if 'c' in sizes:
        key = 'c'
    elif 'C' in sizes:
        key = 'C'
    else:
        raise ValueError(f"No channel dimension ('c' or 'C') found. Axes: {sizes}")

    return key, axis_keys.index(key)


def convert_nd2_to_ome_tiff(
    nd2_path: Path,
    output_dir: Path,
    pixel_size_um: float = PIXEL_SIZE_UM
) -> Path:
    """
    Convert ND2 file to OME-TIFF with reversed channel order.

    Parameters
    ----------
    nd2_path : Path
        Path to input ND2 file
    output_dir : Path
        Directory for output OME-TIFF
    pixel_size_um : float
        Pixel size in micrometers

    Returns
    -------
    Path
        Path to generated OME-TIFF

    Raises
    ------
    ValueError
        If channel axis not found or other validation errors
    """
    output_filename = output_dir / (nd2_path.stem + '.ome.tif')

    logger.info(f"Converting: {nd2_path.name}")

    # Parse filename for channel names
    filename_stem = nd2_path.stem
    parts = filename_stem.split('_')
    channel_names = parts[1:]  # Skip patient ID
    logger.info(f"Detected channels: {channel_names}")

    # Read ND2 file
    with nd2.ND2File(nd2_path) as f:
        image_data = f.asarray()
        sizes = f.sizes

        # Identify channel axis
        c_key, c_axis_index = get_channel_axis_info(sizes)

        num_channels_image = sizes[c_key]
        if len(channel_names) != num_channels_image:
            logger.warning(
                f"Filename has {len(channel_names)} names, "
                f"image has {num_channels_image} channels"
            )

    # Reverse channel order
    reversed_image = np.flip(image_data, axis=c_axis_index)

    # Prepare metadata
    axis_order = list(sizes.keys())
    axes_string = "".join(axis_order).upper()

    metadata = {
        'axes': axes_string,
        'Channel': {'Name': channel_names},
        'PhysicalSizeX': pixel_size_um,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': pixel_size_um,
        'PhysicalSizeYUnit': 'µm'
    }

    # Save as OME-TIFF
    logger.info(f"Writing OME-TIFF with metadata:")
    logger.info(f"  - Axes: {metadata['axes']}")
    logger.info(f"  - Channels: {metadata['Channel']['Name']}")
    logger.info(f"  - PhysicalSizeX: {metadata['PhysicalSizeX']} {metadata['PhysicalSizeXUnit']}")
    logger.info(f"  - PhysicalSizeY: {metadata['PhysicalSizeY']} {metadata['PhysicalSizeYUnit']}")
    logger.info(f"  - Image shape: {reversed_image.shape}")

    tifffile.imwrite(
        output_filename,
        reversed_image,
        metadata=metadata,
        photometric='minisblack',
        ome=True
    )

    logger.info(f"Saved: {output_filename.name}")

    # Verify metadata was written correctly
    logger.info("Verifying saved metadata...")
    with tifffile.TiffFile(output_filename) as tif:
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
            logger.info(f"  ✓ OME-XML metadata present (length: {len(tif.ome_metadata)} chars)")
            # Check if physical size units are in the metadata
            if 'PhysicalSizeXUnit' in tif.ome_metadata or 'µm' in tif.ome_metadata:
                logger.info(f"  ✓ Physical size units found in metadata")
            else:
                logger.warning(f"  ⚠ Physical size units may not be properly saved")
        else:
            logger.warning(f"  ⚠ No OME metadata in saved file!")

        saved_img = tifffile.imread(output_filename)
        logger.info(f"  - Reloaded image shape: {saved_img.shape}")

    return output_filename


def verify_conversion(nd2_path: Path, ome_tiff_path: Path):
    """
    Verify that OME-TIFF matches reversed ND2 data.

    Parameters
    ----------
    nd2_path : Path
        Original ND2 file path
    ome_tiff_path : Path
        Generated OME-TIFF path

    Raises
    ------
    AssertionError
        If verification fails
    """
    logger.info(f"Verifying: {ome_tiff_path.name}")

    if not ome_tiff_path.exists():
        raise FileNotFoundError(f"Output file not found: {ome_tiff_path}")

    # Load and flip ND2 data
    with nd2.ND2File(nd2_path) as f:
        nd2_data = f.asarray()
        sizes = f.sizes

        c_key, c_axis_index = get_channel_axis_info(sizes)
        expected_data = np.flip(nd2_data, axis=c_axis_index)
        expected_channels = nd2_path.stem.split('_')[1:]

    # Load OME-TIFF
    tiff_data = tifffile.imread(ome_tiff_path)

    # Compare data
    if expected_data.shape != tiff_data.shape:
        raise AssertionError(
            f"Shape mismatch: ND2={expected_data.shape} vs TIFF={tiff_data.shape}"
        )

    if not np.array_equal(expected_data, tiff_data):
        raise AssertionError("Pixel values do not match")

    logger.info("✓ Pixel data verified")

    # Check metadata
    with tifffile.TiffFile(ome_tiff_path) as tif:
        ome_xml = tif.ome_metadata
        missing = [name for name in expected_channels if name not in ome_xml]

        if missing:
            logger.warning(f"Metadata missing names: {missing}")
        else:
            logger.info("✓ Metadata verified")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert a single ND2 file to OME-TIFF format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--nd2_file',
        type=str,
        required=True,
        help='Path to input ND2 file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory for OME-TIFF output'
    )
    parser.add_argument(
        '--pixel_size',
        type=float,
        default=PIXEL_SIZE_UM,
        help='Pixel size in micrometers'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify conversion'
    )

    return parser.parse_args()


def main():
    """Main entry point for single file conversion."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    nd2_path = Path(args.nd2_file)
    output_dir = Path(args.output_dir)

    ensure_dir(str(output_dir))

    logger.info("=" * 80)
    logger.info(f"Converting ND2 file: {nd2_path.name}")
    logger.info(f"Pixel size: {args.pixel_size} µm")
    logger.info("=" * 80)

    # Convert
    ome_tiff_path = convert_nd2_to_ome_tiff(nd2_path, output_dir, args.pixel_size)

    # Verify (optional)
    if args.verify:
        verify_conversion(nd2_path, ome_tiff_path)
    verify_conversion(nd2_path, ome_tiff_path) # Mandatory verification for the moment

    logger.info("=" * 80)
    logger.info(f"✓ Conversion complete: {ome_tiff_path.name}")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())
