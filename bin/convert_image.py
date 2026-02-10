#!/usr/bin/env python3
"""Convert microscopy images to standardized OME-TIFF format.

Supports:
- ND2 (Nikon) via bioio-nd2
- CZI (Zeiss) via bioio-czi
- LIF (Leica) via bioio-lif
- TIFF/OME-TIFF via bioio-tifffile/bioio-ome-tiff
- NDPI/NDPIS (Hamamatsu) via tifffile
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from logger import configure_logging, get_logger

logger = get_logger(__name__)

PIXEL_SIZE_UM = 0.325

# Format detection
BIOIO_NATIVE_FORMATS = {'.nd2', '.czi', '.lif', '.tif', '.tiff'}
TIFFFILE_FORMATS = {'.ndpi', '.ndpis'}  # Hamamatsu formats readable by tifffile


def get_file_format(file_path: Path) -> str:
    """Determine file format and appropriate reader."""
    name_lower = file_path.name.lower()
    suffix = file_path.suffix.lower()

    if name_lower.endswith('.ome.tif') or name_lower.endswith('.ome.tiff'):
        return 'bioio'

    if suffix in TIFFFILE_FORMATS:
        return 'tifffile'

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

    # Determine channel count - handle 'S' (Samples) as channels if 'C' is not used
    dim_order = img.dims.order
    num_channels = img.dims.C

    # Check if 'S' dimension should be treated as channels
    # This happens when image has 'S' but 'C' is 1 or not meaningful
    image_data = None  # Will be set below
    if 'S' in dim_order and (num_channels == 1 or 'C' not in dim_order):
        s_count = img.dims.S
        if s_count > 1:
            logger.info(f"Detected 'S' dimension ({s_count}) used as channels instead of 'C' ({num_channels})")
            num_channels = s_count

            # If there's a singleton C dimension, squeeze it out to avoid
            # having two 'C' characters in the dimension order
            if 'C' in dim_order and img.dims.C == 1:
                c_pos = dim_order.index('C')
                image_data = np.squeeze(img.data, axis=c_pos)
                dim_order = dim_order[:c_pos] + dim_order[c_pos+1:]
                logger.info(f"Squeezed singleton C dimension at position {c_pos}")
            else:
                image_data = img.data

            # Remap dimension order for downstream processing (treat S as C)
            dim_order = dim_order.replace('S', 'C')
            logger.info(f"Remapped dimension order: {dim_order}")

    # Load data if not already loaded during S dimension handling
    if image_data is None:
        image_data = img.data

    ps = img.physical_pixel_sizes
    pixel_size_x = ps.X if ps.X is not None else PIXEL_SIZE_UM
    pixel_size_y = ps.Y if ps.Y is not None else PIXEL_SIZE_UM
    pixel_size_z = ps.Z

    logger.info(f"Pixel sizes - X: {pixel_size_x}, Y: {pixel_size_y}, Z: {pixel_size_z}")
    logger.info(f"Channel names from file: {img.channel_names}")

    metadata = {
        'num_channels': num_channels,
        'physical_pixel_size_x': pixel_size_x,
        'physical_pixel_size_y': pixel_size_y,
        'physical_pixel_size_z': pixel_size_z,
        'channel_names_from_file': img.channel_names,
        'original_dims': dim_order,
    }

    return image_data, metadata


def parse_ndpis(ndpis_path: Path) -> List[Path]:
    """Parse .ndpis manifest file to get list of NDPI files.

    NDPIS format:
    [NanoZoomer Digital Pathology Image Set]
    NoImages=4
    Image0=2025-10-22 10.53.32-CY5.ndpi
    Image1=2025-10-22 10.53.32-TRITC.ndpi
    ...
    """
    ndpi_files = []

    # Resolve symlinks to get the real directory where NDPI files are located
    real_ndpis_path = ndpis_path.resolve()
    ndpi_parent = real_ndpis_path.parent

    with open(ndpis_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('Image') and '=' in line:
            filename = line.split('=', 1)[1].strip()
            ndpi_path = ndpi_parent / filename
            ndpi_files.append(ndpi_path)

    logger.info(f"Parsed NDPIS: {len(ndpi_files)} images (from {ndpi_parent})")
    return ndpi_files


def read_single_ndpi(file_path: Path) -> Tuple[np.ndarray, float, float]:
    """Read a single NDPI file using tifffile, return image and pixel sizes."""
    with tifffile.TiffFile(file_path) as tif:
        series = tif.series[0]
        image_data = series.asarray()

        logger.info(f"  {file_path.name}: shape={image_data.shape}, axes={series.axes}")

        # Extract pixel size from TIFF tags
        pixel_size_x = PIXEL_SIZE_UM
        pixel_size_y = PIXEL_SIZE_UM

        page = tif.pages[0]
        if page.tags.get('XResolution') and page.tags.get('YResolution'):
            x_res = page.tags['XResolution'].value
            y_res = page.tags['YResolution'].value
            res_unit = page.tags.get('ResolutionUnit')

            if x_res and y_res:
                x_res_val = x_res[0] / x_res[1] if isinstance(x_res, tuple) else x_res
                y_res_val = y_res[0] / y_res[1] if isinstance(y_res, tuple) else y_res

                if res_unit and res_unit.value == 3:  # centimeters
                    pixel_size_x = 10000.0 / x_res_val
                    pixel_size_y = 10000.0 / y_res_val
                elif res_unit and res_unit.value == 2:  # inches
                    pixel_size_x = 25400.0 / x_res_val
                    pixel_size_y = 25400.0 / y_res_val

        # Convert to grayscale if RGB (NDPI fluorescence channels are often stored as RGB)
        if image_data.ndim == 3 and image_data.shape[-1] == 3:
            # Check if R, G, B channels are identical
            if np.array_equal(image_data[..., 0], image_data[..., 1]) and \
               np.array_equal(image_data[..., 1], image_data[..., 2]):
                logger.info(f"    RGB channels identical, taking first channel")
            else:
                logger.warning(f"    RGB channels differ! Taking first channel anyway")
            image_data = image_data[..., 0]
        elif image_data.ndim == 3 and image_data.shape[-1] == 4:
            # RGBA - check RGB channels
            if np.array_equal(image_data[..., 0], image_data[..., 1]) and \
               np.array_equal(image_data[..., 1], image_data[..., 2]):
                logger.info(f"    RGBA channels identical, taking first channel")
            else:
                logger.warning(f"    RGBA channels differ! Taking first channel anyway")
            image_data = image_data[..., 0]

        return image_data, pixel_size_x, pixel_size_y


def read_image_tifffile(file_path: Path) -> Tuple[np.ndarray, dict]:
    """Read NDPI/NDPIS image using tifffile."""
    logger.info(f"Reading with tifffile: {file_path.name}")

    suffix = file_path.suffix.lower()

    if suffix == '.ndpis':
        # NDPIS is a manifest file pointing to multiple NDPI files
        ndpi_files = parse_ndpis(file_path)

        if not ndpi_files:
            raise ValueError(f"No NDPI files found in NDPIS manifest: {file_path}")

        # Read all NDPI files and stack them
        channel_images = []
        pixel_size_x = PIXEL_SIZE_UM
        pixel_size_y = PIXEL_SIZE_UM

        for ndpi_path in ndpi_files:
            if not ndpi_path.exists():
                raise FileNotFoundError(f"NDPI file not found: {ndpi_path}")
            img, px_x, px_y = read_single_ndpi(ndpi_path)
            channel_images.append(img)
            pixel_size_x = px_x
            pixel_size_y = px_y

        # Stack channels: each image becomes a channel (CYX)
        image_data = np.stack(channel_images, axis=0)

    else:
        # Single NDPI file
        image_data, pixel_size_x, pixel_size_y = read_single_ndpi(file_path)

        # Add channel dimension
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, ...]

    num_channels = image_data.shape[0]
    axes = 'CYX'

    logger.info(f"Final shape: {image_data.shape}, axes: {axes}")
    logger.info(f"Pixel sizes - X: {pixel_size_x}, Y: {pixel_size_y}")

    metadata = {
        'num_channels': num_channels,
        'physical_pixel_size_x': pixel_size_x,
        'physical_pixel_size_y': pixel_size_y,
        'physical_pixel_size_z': None,
        'channel_names_from_file': None,  # Channel names must come from args
        'original_dims': axes,
    }

    return image_data, metadata


def read_image(file_path: Path) -> Tuple[np.ndarray, dict]:
    """Read image using appropriate reader."""
    format_type = get_file_format(file_path)
    logger.info(f"Detected format type: {format_type}")

    if format_type == 'tifffile':
        return read_image_tifffile(file_path)
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
            logger.info("OME-XML metadata present")
        else:
            logger.warning("No OME metadata found in saved file")
    
    return output_filename, output_channels


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Convert microscopy images to OME-TIFF'
    )
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--patient_id', type=str, required=True)
    parser.add_argument(
        '--channels',
        type=str,
        default=None,
        help='Comma-separated channel names (optional, reads metadata when omitted)'
    )
    parser.add_argument('--pixel_size', type=float, default=PIXEL_SIZE_UM)
    return parser.parse_args()


def main() -> int:
    """Run conversion CLI."""
    configure_logging()
    args = parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    channel_names = None
    if args.channels:
        channel_names = [ch.strip() for ch in args.channels.split(',')]

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    logger.info("=" * 70)
    logger.info("Image Converter")
    logger.info("=" * 70)
    logger.info(f"Input: {input_path}")
    if channel_names:
        logger.info(f"Channels (override): {channel_names}")
    else:
        logger.info("Channels: read from image metadata")
    logger.info("=" * 70)

    try:
        output_path, output_channels = convert_to_ome_tiff(
            input_path, output_dir, args.patient_id, channel_names, args.pixel_size
        )
    except Exception as exc:
        logger.error(f"Conversion failed: {exc}")
        return 1

    logger.info("=" * 70)
    logger.info(f"Output: {output_path}")
    logger.info(f"Channel order: {output_channels}")
    logger.info("=" * 70)

    # Write channels file for Nextflow metadata propagation
    channels_file = output_dir / f"{args.patient_id}_channels.txt"
    channels_file.write_text(','.join(output_channels))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
