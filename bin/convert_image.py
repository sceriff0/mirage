#!/usr/bin/env python3
"""Convert microscopy images to standardized OME-TIFF format.

Supports:
- ND2 (Nikon) via bioio-nd2
- CZI (Zeiss) via bioio-czi  
- LIF (Leica) via bioio-lif
- TIFF/OME-TIFF via bioio-tifffile/bioio-ome-tiff
- NDPI/NDPIS (Hamamatsu) via bioio-bioformats

REQUIRES JAVA for NDPI/NDPIS support.
"""

import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import tifffile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PIXEL_SIZE_UM = 0.325

# Format detection
BIOIO_NATIVE_FORMATS = {'.nd2', '.czi', '.lif', '.tif', '.tiff'}
BIOFORMATS_FORMATS = {'.ndpi', '.ndpis', '.svs', '.mrxs', '.scn', '.vms', '.vmu', '.bif'}


def get_file_format(file_path: Path) -> str:
    """Determine file format and appropriate reader."""
    name_lower = file_path.name.lower()
    suffix = file_path.suffix.lower()
    
    if name_lower.endswith('.ome.tif') or name_lower.endswith('.ome.tiff'):
        return 'bioio'
    
    if suffix in BIOFORMATS_FORMATS:
        return 'bioformats'
    
    if suffix in BIOIO_NATIVE_FORMATS:
        return 'bioio'
    
    return 'bioio'


def read_image_bioio(file_path: Path) -> Tuple[np.ndarray, dict]:
    """Read image using BioIO (ND2, CZI, LIF, TIFF)."""
    from bioio import BioImage
    
    logger.info(f"Reading with BioIO: {file_path.name}")
    
    img = BioImage(file_path)
    
    logger.info(f"Dimensions: {img.dims}")
    logger.info(f"Shape: {img.shape}")
    logger.info(f"Dimension order: {img.dims.order}")
    
    ps = img.physical_pixel_sizes
    pixel_size_x = ps.X if ps.X is not None else PIXEL_SIZE_UM
    pixel_size_y = ps.Y if ps.Y is not None else PIXEL_SIZE_UM
    pixel_size_z = ps.Z
    
    logger.info(f"Pixel sizes - X: {pixel_size_x}, Y: {pixel_size_y}, Z: {pixel_size_z}")
    logger.info(f"Channel names from file: {img.channel_names}")
    
    image_data = img.data  # TCZYX order
    
    metadata = {
        'num_channels': img.dims.C,
        'physical_pixel_size_x': pixel_size_x,
        'physical_pixel_size_y': pixel_size_y,
        'physical_pixel_size_z': pixel_size_z,
        'channel_names_from_file': img.channel_names,
        'original_dims': img.dims.order,
    }
    
    return image_data, metadata


def read_image_bioformats(file_path: Path) -> Tuple[np.ndarray, dict]:
    """Read image using Bio-Formats (NDPI, NDPIS, SVS, etc.)."""
    from bioio import BioImage
    import bioio_bioformats
    
    logger.info(f"Reading with Bio-Formats: {file_path.name}")
    
    img = BioImage(file_path, reader=bioio_bioformats.Reader)
    
    logger.info(f"Dimensions: {img.dims}")
    logger.info(f"Shape: {img.shape}")
    logger.info(f"Dimension order: {img.dims.order}")
    logger.info(f"Scenes available: {img.scenes}")
    
    ps = img.physical_pixel_sizes
    pixel_size_x = ps.X if ps.X is not None else PIXEL_SIZE_UM
    pixel_size_y = ps.Y if ps.Y is not None else PIXEL_SIZE_UM
    pixel_size_z = ps.Z
    
    logger.info(f"Pixel sizes - X: {pixel_size_x}, Y: {pixel_size_y}, Z: {pixel_size_z}")
    logger.info(f"Channel names from file: {img.channel_names}")
    
    # Bio-Formats returns TCZYX order
    image_data = img.data
    
    metadata = {
        'num_channels': img.dims.C,
        'physical_pixel_size_x': pixel_size_x,
        'physical_pixel_size_y': pixel_size_y,
        'physical_pixel_size_z': pixel_size_z,
        'channel_names_from_file': img.channel_names,
        'original_dims': img.dims.order,
    }
    
    return image_data, metadata


def read_image(file_path: Path) -> Tuple[np.ndarray, dict]:
    """Read image using appropriate reader."""
    format_type = get_file_format(file_path)
    logger.info(f"Detected format type: {format_type}")
    
    if format_type == 'bioformats':
        return read_image_bioformats(file_path)
    else:
        return read_image_bioio(file_path)


def convert_to_ome_tiff(
    input_path: Path,
    output_dir: Path,
    patient_id: str,
    channel_names: Optional[List[str]] = None,
    pixel_size_um: float = PIXEL_SIZE_UM
) -> Tuple[Path, List[str]]:
    """Convert image to OME-TIFF with DAPI in channel 0."""
    
    # Read image first to get metadata
    image_data, metadata = read_image(input_path)
    original_dims = metadata.get('original_dims', 'TCZYX')
    
    # Use channel names from file if not specified
    if channel_names is None:
        channel_names = metadata.get('channel_names_from_file')
        if channel_names is None:
            raise ValueError("No channel names provided and none found in file metadata")
        logger.info(f"Using channel names from file: {channel_names}")
    
    # Validate channel count
    num_channels_in_image = metadata['num_channels']
    if num_channels_in_image != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: image has {num_channels_in_image}, "
            f"specified {len(channel_names)}: {channel_names}"
        )
    
    # Find DAPI
    dapi_index = None
    for i, ch in enumerate(channel_names):
        if 'DAPI' in ch.upper():
            dapi_index = i
            break
    
    if dapi_index is None:
        raise ValueError(f"DAPI channel not found in: {channel_names}")
    
    # Reorder: DAPI first
    output_channels = channel_names.copy()
    if dapi_index != 0:
        logger.info(f"Moving DAPI from position {dapi_index} to position 0")
        dapi_ch = output_channels.pop(dapi_index)
        output_channels.insert(0, dapi_ch)
    else:
        logger.info("DAPI already in position 0")
    
    channels_str = '_'.join(output_channels)
    output_filename = output_dir / f"{patient_id}_{channels_str}.ome.tif"
    
    logger.info(f"Converting: {input_path.name}")
    logger.info(f"Input channels: {channel_names}")
    logger.info(f"Output channels: {output_channels}")
    
    px_x = metadata.get('physical_pixel_size_x', pixel_size_um)
    px_y = metadata.get('physical_pixel_size_y', pixel_size_um)
    px_z = metadata.get('physical_pixel_size_z')
    
    # Normalize dimensions
    if original_dims == 'TCZYX':
        c_axis = 1
        if image_data.shape[0] == 1:
            image_data = image_data[0]
            original_dims = 'CZYX'
            c_axis = 0
    elif original_dims in ('CZYX', 'CYX'):
        c_axis = 0
    else:
        c_axis = 0
    
    # Squeeze Z if singleton
    if original_dims == 'CZYX' and image_data.shape[1] == 1:
        image_data = image_data[:, 0, :, :]
        original_dims = 'CYX'
    
    # Rearrange channels
    if channel_names != output_channels:
        indices = [channel_names.index(ch) for ch in output_channels]
        image_data = np.take(image_data, indices, axis=c_axis)
        logger.info(f"Rearranged channels: {channel_names} -> {output_channels}")
    
    logger.info(f"Final shape: {image_data.shape}, axes: {original_dims}")
    logger.info(f"Pixel size: X={px_x}, Y={px_y}, Z={px_z}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build OME metadata for tifffile
    ome_metadata = {
        'axes': original_dims,
        'Channel': {'Name': output_channels},
        'PhysicalSizeX': px_x,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': px_y,
        'PhysicalSizeYUnit': 'µm',
    }
    
    if px_z is not None:
        ome_metadata['PhysicalSizeZ'] = px_z
        ome_metadata['PhysicalSizeZUnit'] = 'µm'
    
    # Write OME-TIFF using tifffile
    logger.info(f"Writing: {output_filename.name}")
    
    tifffile.imwrite(
        output_filename,
        image_data,
        metadata=ome_metadata,
        photometric='minisblack',
        ome=True,
        bigtiff=True,
    )
    
    logger.info(f"Saved: {output_filename.name}")
    
    # Verify
    with tifffile.TiffFile(output_filename) as tif:
        if tif.ome_metadata:
            logger.info("✓ OME-XML metadata present")
        else:
            logger.warning("⚠ No OME metadata in saved file")
    
    return output_filename, output_channels


def main():
    parser = argparse.ArgumentParser(
        description='Convert microscopy images to OME-TIFF'
    )
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--patient_id', type=str, required=True)
    parser.add_argument('--channels', type=str, default=None,
                        help='Comma-separated channel names (optional, will read from file if not specified)')
    parser.add_argument('--pixel_size', type=float, default=PIXEL_SIZE_UM)
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    channel_names = None
    if args.channels:
        channel_names = [ch.strip() for ch in args.channels.split(',')]
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    logger.info("=" * 70)
    logger.info("Image Converter (Bio-Formats enabled)")
    logger.info("=" * 70)
    logger.info(f"Input: {input_path}")
    if channel_names:
        logger.info(f"Channels (override): {channel_names}")
    else:
        logger.info("Channels: will read from file metadata")
    logger.info("=" * 70)
    
    try:
        output_path, output_channels = convert_to_ome_tiff(
            input_path, output_dir, args.patient_id, channel_names, args.pixel_size
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise
    
    logger.info("=" * 70)
    logger.info(f"✓ Output: {output_path}")
    logger.info(f"✓ Channel order: {output_channels}")
    logger.info("=" * 70)
    
    # Write channels file for Nextflow
    channels_file = output_dir / f"{args.patient_id}_channels.txt"
    channels_file.write_text(','.join(output_channels))
    
    return 0


if __name__ == '__main__':
    exit(main())