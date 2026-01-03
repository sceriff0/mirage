#!/usr/bin/env python3
"""Convert microscopy images to standardized OME-TIFF format.

Supports:
- ND2 (Nikon) via bioio-nd2
- CZI (Zeiss) via bioio-czi  
- LIF (Leica) via bioio-lif
- TIFF/OME-TIFF via bioio-tifffile/bioio-ome-tiff
- NDPI (Hamamatsu) via OpenSlide (RGB only, see note below)

NOTE ON NDPI: OpenSlide reads NDPI as RGB. If your NDPI contains 
multichannel fluorescence data, you need Bio-Formats (Java) instead.
"""

import logging
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PIXEL_SIZE_UM = 0.325

# Format detection
BIOIO_FORMATS = {'.nd2', '.czi', '.lif', '.tif', '.tiff'}
OPENSLIDE_FORMATS = {'.ndpi', '.svs', '.mrxs', '.scn', '.vms', '.vmu', '.bif'}


def get_file_format(file_path: Path) -> str:
    """Determine file format and appropriate reader."""
    name_lower = file_path.name.lower()
    suffix = file_path.suffix.lower()
    
    # Check for OME-TIFF first
    if name_lower.endswith('.ome.tif') or name_lower.endswith('.ome.tiff'):
        return 'bioio'
    
    if suffix in OPENSLIDE_FORMATS:
        return 'openslide'
    
    if suffix in BIOIO_FORMATS:
        return 'bioio'
    
    # Default to bioio
    return 'bioio'


def read_image_bioio(file_path: Path) -> Tuple[np.ndarray, dict]:
    """Read image using BioIO (ND2, CZI, LIF, TIFF)."""
    from bioio import BioImage
    
    logger.info(f"Reading with BioIO: {file_path.name}")
    
    img = BioImage(file_path)
    
    logger.info(f"Dimensions: {img.dims}")
    logger.info(f"Shape: {img.shape}")
    logger.info(f"Dimension order: {img.dims.order}")
    
    # Get pixel sizes
    ps = img.physical_pixel_sizes
    pixel_size_x = ps.X if ps.X is not None else PIXEL_SIZE_UM
    pixel_size_y = ps.Y if ps.Y is not None else PIXEL_SIZE_UM
    pixel_size_z = ps.Z  # Can be None
    
    logger.info(f"Pixel sizes - X: {pixel_size_x}, Y: {pixel_size_y}, Z: {pixel_size_z}")
    
    # Get channel names
    channel_names = img.channel_names
    logger.info(f"Channel names from file: {channel_names}")
    
    # BioIO returns TCZYX order
    image_data = img.data
    
    logger.info(f"Loaded data shape: {image_data.shape}")
    
    metadata = {
        'num_channels': img.dims.C,
        'physical_pixel_size_x': pixel_size_x,
        'physical_pixel_size_y': pixel_size_y,
        'physical_pixel_size_z': pixel_size_z,
        'channel_names_from_file': channel_names,
        'original_dims': img.dims.order,
    }
    
    return image_data, metadata


def read_image_openslide(file_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Read image using OpenSlide (NDPI, SVS, etc.).
    
    WARNING: OpenSlide reads slides as RGB. For multichannel fluorescence,
    you need Bio-Formats instead.
    """
    import openslide
    
    logger.info(f"Reading with OpenSlide: {file_path.name}")
    logger.warning("OpenSlide reads as RGB. For multichannel fluorescence NDPI, use Bio-Formats.")
    
    slide = openslide.OpenSlide(str(file_path))
    
    # Get dimensions
    width, height = slide.dimensions
    level_count = slide.level_count
    
    logger.info(f"Dimensions: {width} x {height}")
    logger.info(f"Levels: {level_count}")
    
    # Get pixel size from properties
    pixel_size_x = PIXEL_SIZE_UM
    pixel_size_y = PIXEL_SIZE_UM
    
    props = dict(slide.properties)
    if 'openslide.mpp-x' in props:
        pixel_size_x = float(props['openslide.mpp-x'])
    if 'openslide.mpp-y' in props:
        pixel_size_y = float(props['openslide.mpp-y'])
    
    logger.info(f"Pixel sizes - X: {pixel_size_x}, Y: {pixel_size_y}")
    
    # Check if image is too large
    max_pixels = 100000 * 100000 # 10 billion pixels
    if width * height > max_pixels:
        raise MemoryError(
            f"Image too large to load into memory: {width}x{height}. "
            f"Consider using tiled reading or downsampling."
        )
    
    logger.info("Reading full slide into memory...")
    
    # Read full image at level 0
    pil_image = slide.read_region((0, 0), 0, (width, height))
    image_array = np.array(pil_image)
    
    # Drop alpha channel if RGBA
    if image_array.ndim == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    
    # Transpose from (Y, X, C) to (C, Y, X)
    if image_array.ndim == 3:
        image_array = np.transpose(image_array, (2, 0, 1))
    
    num_channels = image_array.shape[0] if image_array.ndim == 3 else 1
    
    logger.info(f"Loaded array shape: {image_array.shape} (CYX)")
    
    slide.close()
    
    metadata = {
        'num_channels': num_channels,
        'physical_pixel_size_x': pixel_size_x,
        'physical_pixel_size_y': pixel_size_y,
        'physical_pixel_size_z': None,
        'channel_names_from_file': None,
        'original_dims': 'CYX',
    }
    
    return image_array, metadata


def read_image(file_path: Path) -> Tuple[np.ndarray, dict]:
    """Read image using appropriate reader based on format."""
    format_type = get_file_format(file_path)
    logger.info(f"Detected format type: {format_type}")
    
    if format_type == 'openslide':
        return read_image_openslide(file_path)
    else:
        return read_image_bioio(file_path)


def convert_to_ome_tiff(
    input_path: Path,
    output_dir: Path,
    patient_id: str,
    channel_names: List[str],
    pixel_size_um: float = PIXEL_SIZE_UM
) -> Tuple[Path, List[str]]:
    """Convert image to OME-TIFF with DAPI in channel 0."""
    from bioio.writers import OmeTiffWriter
    from bioio_base.types import PhysicalPixelSizes
    
    # Find DAPI and create output channel order
    dapi_index = None
    for i, ch in enumerate(channel_names):
        if ch.upper() == 'DAPI':
            dapi_index = i
            break
    
    if dapi_index is None:
        raise ValueError(f"DAPI channel not found in: {channel_names}")
    
    # Reorder channels: DAPI first
    output_channels = channel_names.copy()
    if dapi_index != 0:
        logger.info(f"Moving DAPI from position {dapi_index} to position 0")
        dapi_ch = output_channels.pop(dapi_index)
        output_channels.insert(0, dapi_ch)
    else:
        logger.info("DAPI already in position 0")
    
    # Generate output filename
    channels_str = '_'.join(output_channels)
    output_filename = output_dir / f"{patient_id}_{channels_str}.ome.tif"
    
    logger.info(f"Converting: {input_path.name}")
    logger.info(f"Patient ID: {patient_id}")
    logger.info(f"Input channels: {channel_names}")
    logger.info(f"Output channels: {output_channels}")
    
    # Read image
    image_data, metadata = read_image(input_path)
    original_dims = metadata.get('original_dims', 'TCZYX')
    
    # Validate channel count
    num_channels_in_image = metadata['num_channels']
    if num_channels_in_image != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: image has {num_channels_in_image} channels, "
            f"but {len(channel_names)} channel names provided: {channel_names}"
        )
    
    # Get pixel sizes from metadata
    px_x = metadata.get('physical_pixel_size_x', pixel_size_um)
    px_y = metadata.get('physical_pixel_size_y', pixel_size_um)
    px_z = metadata.get('physical_pixel_size_z')
    
    # Determine dimension order and channel axis
    if original_dims == 'TCZYX':
        # BioIO standard: TCZYX
        c_axis = 1
        # Squeeze singleton T
        if image_data.shape[0] == 1:
            image_data = image_data[0]  # Now CZYX
            dim_order = 'CZYX'
            c_axis = 0
        else:
            dim_order = 'TCZYX'
    elif original_dims == 'CZYX':
        c_axis = 0
        dim_order = 'CZYX'
    elif original_dims == 'CYX':
        c_axis = 0
        dim_order = 'CYX'
    else:
        # Fallback
        c_axis = 0
        dim_order = original_dims
    
    # Further squeeze singleton Z if present
    if dim_order == 'CZYX' and image_data.shape[1] == 1:
        image_data = image_data[:, 0, :, :]  # Now CYX
        dim_order = 'CYX'
    elif dim_order == 'TCZYX' and image_data.shape[2] == 1:
        image_data = np.squeeze(image_data, axis=2)  # Now TCYX
        dim_order = 'TCYX'
    
    # Rearrange channels if needed
    if channel_names != output_channels:
        indices = [channel_names.index(ch) for ch in output_channels]
        image_data = np.take(image_data, indices, axis=c_axis)
        logger.info(f"Rearranged channels: {channel_names} -> {output_channels}")
    
    logger.info(f"Final data shape: {image_data.shape}")
    logger.info(f"Final dimension order: {dim_order}")
    logger.info(f"Pixel size: X={px_x} µm, Y={px_y} µm, Z={px_z}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build PhysicalPixelSizes object (Z, Y, X order)
    physical_sizes = PhysicalPixelSizes(Z=px_z, Y=px_y, X=px_x)
    
    # Write OME-TIFF
    logger.info(f"Writing: {output_filename.name}")
    
    OmeTiffWriter.save(
        data=image_data,
        uri=str(output_filename),  # Convert Path to string
        dim_order=dim_order,
        channel_names=output_channels,
        physical_pixel_sizes=physical_sizes,
    )
    
    logger.info(f"Successfully saved: {output_filename.name}")
    
    return output_filename, output_channels


def main():
    parser = argparse.ArgumentParser(
        description='Convert microscopy images to OME-TIFF (ND2, CZI, LIF, TIFF, NDPI)'
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input image file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--patient_id', type=str, required=True,
                        help='Patient identifier')
    parser.add_argument('--channels', type=str, required=True,
                        help='Comma-separated channel names (must include DAPI)')
    parser.add_argument('--pixel_size', type=float, default=PIXEL_SIZE_UM,
                        help=f'Default pixel size in µm (default: {PIXEL_SIZE_UM})')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    channel_names = [ch.strip() for ch in args.channels.split(',')]
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    logger.info("=" * 70)
    logger.info("Image Converter (Pure Python/C - NO JAVA)")
    logger.info("Supported: ND2, CZI, LIF, TIFF, OME-TIFF, NDPI (RGB only)")
    logger.info("=" * 70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Channels: {channel_names}")
    logger.info("=" * 70)
    
    try:
        output_path, output_channels = convert_to_ome_tiff(
            input_path,
            output_dir,
            args.patient_id,
            channel_names,
            args.pixel_size
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