#!/usr/bin/env python3
"""Convert microscopy images to OME-TIFF using bfio (Bio-Formats wrapper)."""

import logging
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile
from bfio import BioReader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PIXEL_SIZE_UM = 0.325


def read_image(file_path: Path) -> Tuple[np.ndarray, dict]:
    """Read image using bfio."""
    logger.info(f"Reading image: {file_path.name}")

    with BioReader(file_path) as reader:
        # Get dimensions
        logger.info(f"Dimensions: X={reader.X}, Y={reader.Y}, Z={reader.Z}, C={reader.C}, T={reader.T}")

        # Get pixel size
        pixel_size = reader.ps_x[0] if reader.ps_x else PIXEL_SIZE_UM
        logger.info(f"Pixel size: {pixel_size} µm")

        # Read all data
        image_data = reader[:, :, :, :, :]  # XYZCT order from bfio

        # bfio returns XYZCT, convert to CZYX for OME-TIFF
        # From (X, Y, Z, C, T) to (C, Z, Y, X) - drop T, transpose
        image_data = np.squeeze(image_data)  # Remove T if singleton

        if reader.Z > 1:
            # 3D: rearrange to CZYX
            image_data = np.transpose(image_data, (3, 2, 1, 0))  # XYZC -> CZYX
            axes = 'CZYX'
        else:
            # 2D: rearrange to CYX
            image_data = np.squeeze(image_data, axis=2)  # Remove Z
            image_data = np.transpose(image_data, (2, 1, 0))  # XYC -> CYX
            axes = 'CYX'

        logger.info(f"Output shape: {image_data.shape}, axes: {axes}")

        metadata = {
            'axes': axes,
            'num_channels': reader.C,
            'physical_pixel_size': pixel_size,
        }

    return image_data, metadata


def convert_to_ome_tiff(
    input_path: Path,
    output_dir: Path,
    patient_id: str,
    channel_names: List[str],
    pixel_size_um: float = PIXEL_SIZE_UM
) -> Tuple[Path, List[str]]:
    """Convert image to OME-TIFF with DAPI in channel 0."""

    # Find DAPI and move to position 0
    dapi_index = None
    for i, ch in enumerate(channel_names):
        if ch.upper() == 'DAPI':
            dapi_index = i
            break

    if dapi_index is None:
        raise ValueError(f"DAPI channel not found in: {channel_names}")

    output_channels = channel_names.copy()
    if dapi_index != 0:
        logger.info(f"Moving DAPI from position {dapi_index} to position 0")
        dapi_ch = output_channels.pop(dapi_index)
        output_channels.insert(0, dapi_ch)

    # Output filename
    channels_str = '_'.join(output_channels)
    output_filename = output_dir / f"{patient_id}_{channels_str}.ome.tif"

    logger.info(f"Converting: {input_path.name}")
    logger.info(f"Input channels: {channel_names}")
    logger.info(f"Output channels: {output_channels}")

    # Read image
    image_data, metadata = read_image(input_path)

    # Validate channel count
    if metadata['num_channels'] != len(channel_names):
        logger.warning(
            f"Channel count mismatch: image has {metadata['num_channels']}, "
            f"specified {len(channel_names)}"
        )

    actual_pixel_size = metadata.get('physical_pixel_size', pixel_size_um)

    # Rearrange channels if needed
    if channel_names != output_channels:
        c_axis = metadata['axes'].index('C')
        indices = [channel_names.index(ch) for ch in output_channels]
        image_data = np.take(image_data, indices, axis=c_axis)
        logger.info(f"Rearranged channels: {channel_names} -> {output_channels}")

    # Save as OME-TIFF
    output_dir.mkdir(parents=True, exist_ok=True)

    ome_metadata = {
        'axes': metadata['axes'],
        'Channel': {'Name': output_channels},
        'PhysicalSizeX': actual_pixel_size,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': actual_pixel_size,
        'PhysicalSizeYUnit': 'µm'
    }

    logger.info(f"Writing: {output_filename.name}")

    tifffile.imwrite(
        output_filename,
        image_data,
        metadata=ome_metadata,
        photometric='minisblack',
        ome=True,
        bigtiff=True
    )

    logger.info(f"Done: {output_filename.name}")

    return output_filename, output_channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--patient_id', required=True)
    parser.add_argument('--channels', required=True)
    parser.add_argument('--pixel_size', type=float, default=PIXEL_SIZE_UM)
    args = parser.parse_args()

    output_path, output_channels = convert_to_ome_tiff(
        Path(args.input_file),
        Path(args.output_dir),
        args.patient_id,
        [ch.strip() for ch in args.channels.split(',')],
        args.pixel_size
    )

    # Write channels file for Nextflow
    channels_file = Path(args.output_dir) / f"{args.patient_id}_channels.txt"
    channels_file.write_text(','.join(output_channels))

    return 0


if __name__ == '__main__':
    exit(main())