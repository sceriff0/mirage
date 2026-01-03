#!/usr/bin/env python3
"""Convert microscopy images to standardized OME-TIFF format using Bio-Formats.

Supports .nd2, .lif, .ndpi, .tiff, .czi, and all Bio-Formats supported formats.
"""

import logging
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile
import jpype
import jpype.imports

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PIXEL_SIZE_UM = 0.325


def start_jvm():
    """Start JVM with Bio-Formats JAR."""
    if jpype.isJVMStarted():
        return

    jar_path = os.environ.get('BIOFORMATS_JAR', '/opt/bioformats/bioformats_package.jar')

    if not Path(jar_path).exists():
        raise FileNotFoundError(
            f"Bio-Formats JAR not found at {jar_path}. "
            "Set BIOFORMATS_JAR environment variable to the correct path."
        )

    logger.info(f"Starting JVM with Bio-Formats: {jar_path}")
    jpype.startJVM(classpath=[jar_path])


def read_image_bioformats(file_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Read image file using Bio-Formats.

    Parameters
    ----------
    file_path : Path
        Path to image file (supports .nd2, .lif, .ndpi, .tiff, .czi, etc.)

    Returns
    -------
    tuple
        (image_data in CYX or CZYX order, metadata_dict)
    """
    start_jvm()

    from loci.formats import ImageReader, MetadataTools
    from loci.common import services, DebugTools
    from ome.units import UNITS

    # Suppress Bio-Formats debug output
    DebugTools.setRootLevel("ERROR")

    logger.info(f"Reading image: {file_path.name}")

    reader = ImageReader()
    meta = MetadataTools.createOMEXMLMetadata()
    reader.setMetadataStore(meta)
    reader.setId(str(file_path))

    try:
        # Get dimensions
        size_x = reader.getSizeX()
        size_y = reader.getSizeY()
        size_z = reader.getSizeZ()
        size_c = reader.getSizeC()
        size_t = reader.getSizeT()
        pixel_type = reader.getPixelType()
        is_little_endian = reader.isLittleEndian()

        logger.info(f"Dimensions: X={size_x}, Y={size_y}, Z={size_z}, C={size_c}, T={size_t}")

        # Get pixel size from metadata
        pixel_size = PIXEL_SIZE_UM
        try:
            phys_x = meta.getPixelsPhysicalSizeX(0)
            if phys_x is not None:
                pixel_size = float(phys_x.value(UNITS.MICROMETER))
                logger.info(f"Pixel size from metadata: {pixel_size} µm")
        except Exception as e:
            logger.warning(f"Could not read pixel size: {e}")

        # Get channel names from metadata
        channel_names_from_file = []
        try:
            for c in range(size_c):
                name = meta.getChannelName(0, c)
                if name:
                    channel_names_from_file.append(str(name))
        except Exception:
            pass

        if channel_names_from_file:
            logger.info(f"Channel names from file: {channel_names_from_file}")

        # Map Bio-Formats pixel type to numpy dtype
        dtype_map = {
            0: np.int8,    # INT8
            1: np.uint8,   # UINT8
            2: np.int16,   # INT16
            3: np.uint16,  # UINT16
            4: np.int32,   # INT32
            5: np.uint32,  # UINT32
            6: np.float32, # FLOAT
            7: np.float64, # DOUBLE
        }
        dtype = dtype_map.get(pixel_type, np.uint16)

        # Read all planes
        if size_z > 1:
            # 3D: CZYX
            image_data = np.zeros((size_c, size_z, size_y, size_x), dtype=dtype)
            axes = 'CZYX'
            for c in range(size_c):
                for z in range(size_z):
                    idx = reader.getIndex(z, c, 0)
                    plane = reader.openBytes(idx)
                    plane = np.frombuffer(plane, dtype=dtype).reshape(size_y, size_x)
                    image_data[c, z] = plane
        else:
            # 2D: CYX
            image_data = np.zeros((size_c, size_y, size_x), dtype=dtype)
            axes = 'CYX'
            for c in range(size_c):
                idx = reader.getIndex(0, c, 0)
                plane = reader.openBytes(idx)
                plane = np.frombuffer(plane, dtype=dtype).reshape(size_y, size_x)
                image_data[c] = plane

        logger.info(f"Loaded image shape: {image_data.shape}, axes: {axes}")

        metadata = {
            'axes': axes,
            'num_channels': size_c,
            'physical_pixel_size': pixel_size,
            'channel_names_from_file': channel_names_from_file or None,
        }

        return image_data, metadata

    finally:
        reader.close()


def rearrange_channels(
    image_data: np.ndarray,
    axes: str,
    channel_mapping: List[str],
    target_channels: List[str]
) -> np.ndarray:
    """Rearrange channels to target order (DAPI first)."""
    if 'C' not in axes:
        return image_data

    c_axis = axes.index('C')

    indices = []
    for target_ch in target_channels:
        try:
            idx = channel_mapping.index(target_ch)
            indices.append(idx)
        except ValueError:
            raise ValueError(f"Channel '{target_ch}' not found in: {channel_mapping}")

    reordered = np.take(image_data, indices, axis=c_axis)
    logger.info(f"Rearranged channels: {channel_mapping} -> {target_channels}")

    return reordered


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
    logger.info(f"Patient ID: {patient_id}")
    logger.info(f"Input channels: {channel_names}")
    logger.info(f"Output channels: {output_channels}")

    # Read image
    image_data, metadata = read_image_bioformats(input_path)

    # Validate channel count
    if metadata['num_channels'] != len(channel_names):
        logger.warning(
            f"Channel count mismatch: image has {metadata['num_channels']}, "
            f"specified {len(channel_names)}"
        )

    # Use pixel size from metadata if available
    actual_pixel_size = metadata.get('physical_pixel_size', pixel_size_um)

    # Rearrange channels
    if channel_names != output_channels:
        image_data = rearrange_channels(
            image_data,
            metadata['axes'],
            channel_names,
            output_channels
        )

    # Prepare OME metadata
    ome_metadata = {
        'axes': metadata['axes'],
        'Channel': {'Name': output_channels},
        'PhysicalSizeX': actual_pixel_size,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': actual_pixel_size,
        'PhysicalSizeYUnit': 'µm'
    }

    logger.info(f"Writing OME-TIFF: {output_filename.name}")
    logger.info(f"  Shape: {image_data.shape}")
    logger.info(f"  Pixel size: {actual_pixel_size} µm")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        output_filename,
        image_data,
        metadata=ome_metadata,
        photometric='minisblack',
        ome=True,
        bigtiff=True
    )

    logger.info(f"Saved: {output_filename.name}")

    return output_filename, output_channels


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert microscopy images to OME-TIFF using Bio-Formats'
    )
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--patient_id', type=str, required=True)
    parser.add_argument('--channels', type=str, required=True,
                        help='Comma-separated channel names (must include DAPI)')
    parser.add_argument('--pixel_size', type=float, default=PIXEL_SIZE_UM)
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    channel_names = [ch.strip() for ch in args.channels.split(',')]

    logger.info("=" * 60)
    logger.info(f"Input: {input_path.name}")
    logger.info(f"Channels: {channel_names}")
    logger.info("=" * 60)

    output_path, output_channels = convert_to_ome_tiff(
        input_path,
        output_dir,
        args.patient_id,
        channel_names,
        args.pixel_size
    )

    logger.info("=" * 60)
    logger.info(f"✓ Output: {output_path.name}")
    logger.info(f"✓ Channel order: {output_channels}")
    logger.info("=" * 60)

    # Write channels file for Nextflow
    channels_file = output_dir / f"{args.patient_id}_channels.txt"
    with open(channels_file, 'w') as f:
        f.write(','.join(output_channels))

    return 0


if __name__ == '__main__':
    exit(main())