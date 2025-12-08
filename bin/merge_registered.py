#!/usr/bin/env python3
"""
Merge individually registered slides into a single OME-TIFF file.

This script takes a directory of registered slides (output from VALIS warp_and_save_slide)
and merges them into a single multi-channel OME-TIFF. Keeps all channels from all slides,
but for DAPI only retains it from the reference image.

Uses VALIS slide_io for robust image reading.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import tifffile
import os
import gc
import tempfile

os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_CACHING'] = '1'


# Import only slide_io to avoid heavy VALIS dependencies
import valis.slide_io as slide_io


def log(msg: str):
    """Print timestamped message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def get_channel_names_from_ome(filepath: str) -> list:
    """Extract channel names from OME-TIFF metadata. Returns empty list if fails."""
    try:
        with tifffile.TiffFile(filepath) as tif:
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                channels = root.findall('.//ome:Channel', ns)
                if channels:
                    return [ch.get('Name', '') for ch in channels]
    except:
        pass
    return []


def extract_markers_from_filename(filename: str) -> list:
    """Extract marker names from filename like 'B19-10215_DAPI_SMA_panck_corrected'."""
    # Remove _registered, _corrected, and _padded suffixes
    name = filename.replace('_registered.ome.tif', '').replace('_registered.ome.tiff', '')
    name = name.replace('_corrected', '').replace('_padded', '')

    # Split by underscore and filter out sample ID
    parts = name.split('_')

    # Assume first part is sample ID (starts with letter+numbers), rest are markers
    markers = [p for p in parts if not (len(p) > 0 and p[0].isalpha() and any(c.isdigit() for c in p) and '-' in p)]

    return markers if markers else [name]


def is_reference_slide(slide_name: str, reference_markers: list) -> bool:
    """Check if slide contains reference markers."""
    slide_name_upper = slide_name.upper()
    # Slide is reference if it contains ALL reference markers
    return all(marker.upper() in slide_name_upper for marker in reference_markers)


def read_slide(filepath: str) -> tuple:
    """Read slide using tifffile (RAM-efficient). Returns (numpy array (C, H, W), channel_names)."""
    # Read with tifffile which handles large TIFFs efficiently
    img = tifffile.imread(str(filepath))

    # Ensure (C, H, W) format
    if img.ndim == 2:
        # Single channel: (H, W) -> (1, H, W)
        img = img[np.newaxis, ...]
    elif img.ndim == 3:
        # Multi-channel: could be (H, W, C) or (C, H, W)
        # Detect based on typical image dimensions
        if img.shape[0] < img.shape[2]:
            # Already (C, H, W) format
            pass
        else:
            # (H, W, C) format - transpose to (C, H, W)
            img = np.transpose(img, (2, 0, 1))

    # Try to get channel names from OME metadata first
    channel_names = get_channel_names_from_ome(str(filepath))

    # If metadata fails, extract from filename
    if not channel_names or len(channel_names) != img.shape[0]:
        filename = Path(filepath).name
        markers = extract_markers_from_filename(filename)

        # Match number of channels
        if len(markers) == img.shape[0]:
            channel_names = markers
        elif len(markers) < img.shape[0]:
            # Pad with generic names
            channel_names = markers + [f"Channel_{i}" for i in range(len(markers), img.shape[0])]
        else:
            # Use first N markers
            channel_names = markers[:img.shape[0]]

    return img, channel_names


def merge_slides(input_dir: str, output_path: str, reference_markers: list = None):
    """
    Merge all registered slides into a single OME-TIFF.

    Keeps all channels from all slides, except DAPI which is only kept from the reference slide.

    Memory-efficient: Uses memory-mapped arrays and writes channels incrementally.

    Args:
        input_dir: Directory containing registered slides
        output_path: Output OME-TIFF path
        reference_markers: List of markers that identify the reference slide (e.g., ['DAPI', 'SMA'])
    """
    if reference_markers is None:
        reference_markers = ['DAPI', 'SMA']  # Default

    log(f"Merging slides from: {input_dir}")
    log(f"Output: {output_path}")
    log(f"Reference markers: {reference_markers}")

    # Find all registered slides
    slide_files = sorted(
        list(Path(input_dir).glob("*.ome.tif")) +
        list(Path(input_dir).glob("*.ome.tiff"))
    )

    if not slide_files:
        raise ValueError(f"No registered slides found in {input_dir}")

    log(f"Found {len(slide_files)} slides")

    # Identify reference slide
    reference_slide = None
    for slide_file in slide_files:
        if is_reference_slide(slide_file.name, reference_markers):
            reference_slide = slide_file
            log(f"Reference slide: {slide_file.name}")
            break

    if not reference_slide:
        log(f"WARNING: No reference slide found with markers {reference_markers}")
        log(f"Will keep DAPI from first slide")
        reference_slide = slide_files[0]

    # PASS 1: Collect metadata and determine final dimensions
    log("Pass 1: Scanning slides for metadata...")
    slide_metadata = []
    channel_names = []
    height, width, dtype = None, None, None

    for slide_file in slide_files:
        log(f"  Scanning: {slide_file.name}")
        is_reference = (slide_file == reference_slide)

        # Read metadata only (first page shape and dtype)
        with tifffile.TiffFile(str(slide_file)) as tif:
            page = tif.pages[0]
            if height is None:
                height, width = page.shape[-2:]  # Get H, W from last two dims
                dtype = page.dtype
                log(f"  Image dimensions: {height} x {width}, dtype: {dtype}")
            else:
                # Verify all slides have same dimensions
                h, w = page.shape[-2:]
                if (h, w) != (height, width):
                    raise ValueError(f"Slide {slide_file.name} has different dimensions: {h}x{w} vs {height}x{width}")

            # Get number of channels
            if len(page.shape) == 2:
                num_channels = 1
            elif page.axes == 'YX':
                num_channels = 1
            elif 'C' in page.axes:
                c_idx = page.axes.index('C')
                num_channels = page.shape[c_idx]
            else:
                # Assume first dimension is channels if 3D
                num_channels = page.shape[0] if len(page.shape) == 3 else 1

        # Get channel names from OME or filename
        names = get_channel_names_from_ome(str(slide_file))
        if not names or len(names) != num_channels:
            filename = Path(slide_file).name
            markers = extract_markers_from_filename(filename)
            if len(markers) == num_channels:
                names = markers
            elif len(markers) < num_channels:
                names = markers + [f"Channel_{i}" for i in range(len(markers), num_channels)]
            else:
                names = markers[:num_channels]

        # Decide which channels to keep
        channels_to_keep = []
        for i, name in enumerate(names):
            is_dapi = name.upper() == 'DAPI'
            if is_dapi:
                if is_reference:
                    channels_to_keep.append((i, name))
                    log(f"    Channel {i}: {name} (DAPI from reference) ✓")
                else:
                    log(f"    Channel {i}: {name} (DAPI, skipped) ⊗")
            else:
                channels_to_keep.append((i, name))
                log(f"    Channel {i}: {name} ✓")

        slide_metadata.append({
            'file': slide_file,
            'channels_to_keep': channels_to_keep,
            'is_reference': is_reference
        })

        # Add to global channel list
        channel_names.extend([name for _, name in channels_to_keep])

    num_output_channels = len(channel_names)
    log(f"Total output channels: {num_output_channels}")
    log(f"Output dimensions: {num_output_channels} x {height} x {width}")

    # Calculate memory requirements
    bytes_per_pixel = np.dtype(dtype).itemsize
    total_bytes = num_output_channels * height * width * bytes_per_pixel
    total_gb = total_bytes / (1024**3)
    log(f"Output size: {total_gb:.2f} GB")

    # PASS 2: Create output file and write channels incrementally
    log("Pass 2: Writing channels to output file...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Create OME-XML metadata
    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
    <Image ID="Image:0" Name="Merged">
        <Pixels ID="Pixels:0" Type="{dtype}"
                SizeX="{width}" SizeY="{height}" SizeZ="1" SizeC="{num_output_channels}" SizeT="1"
                DimensionOrder="XYCZT"
                PhysicalSizeX="0.325" PhysicalSizeY="0.325" PhysicalSizeXUnit="um" PhysicalSizeYUnit="um">
            {chr(10).join(f'<Channel ID="Channel:0:{i}" Name="{name}" SamplesPerPixel="1" />' for i, name in enumerate(channel_names))}
            <TiffData />
        </Pixels>
    </Image>
</OME>'''

    # Create temporary memmap to accumulate all channels
    # This is memory-efficient: only one channel loaded at a time
    log("Creating temporary output buffer...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="merge_registered_"))
    output_memmap_path = tmp_dir / "merged.npy"

    output_shape = (num_output_channels, height, width)
    output_memmap = np.memmap(str(output_memmap_path), dtype=dtype, mode='w+', shape=output_shape)

    output_channel_idx = 0

    for slide_meta in slide_metadata:
        slide_file = slide_meta['file']
        channels_to_keep = slide_meta['channels_to_keep']

        if not channels_to_keep:
            continue

        log(f"Processing: {slide_file.name}")

        # Memory-map the input file (doesn't load into RAM)
        with tifffile.TiffFile(str(slide_file)) as tif:
            # Get memmap of the entire array
            img_memmap = tif.asarray(out='memmap')

            # Determine shape format
            if img_memmap.ndim == 2:
                # Single channel (H, W)
                is_chw = True
            elif img_memmap.ndim == 3:
                # Multi-channel - determine if (C, H, W) or (H, W, C)
                if img_memmap.shape[0] < img_memmap.shape[2]:
                    # Already (C, H, W)
                    is_chw = True
                else:
                    # (H, W, C) format - need transpose
                    is_chw = False
            else:
                raise ValueError(f"Unexpected array shape: {img_memmap.shape}")

            # Extract and write each channel to memmap
            for channel_idx, channel_name in channels_to_keep:
                # Read single channel from memmap (only loads this slice into RAM)
                if img_memmap.ndim == 2:
                    # Single channel image
                    channel_data = np.array(img_memmap)
                elif is_chw:
                    # (C, H, W) format - direct slice
                    channel_data = np.array(img_memmap[channel_idx, :, :])
                else:
                    # (H, W, C) format - need to extract and transpose
                    channel_data = np.array(img_memmap[:, :, channel_idx])

                # Write to output memmap
                output_memmap[output_channel_idx, :, :] = channel_data
                log(f"  Channel {output_channel_idx}: {channel_name}")

                output_channel_idx += 1

                # Clean up channel data immediately
                del channel_data
                gc.collect()

            # Clean up input memmap
            del img_memmap
            gc.collect()

    # Flush memmap to disk
    log("Flushing temporary buffer...")
    output_memmap.flush()

    # Write final output file from memmap
    log(f"Writing final output file: {output_path}")
    tifffile.imwrite(
        output_path,
        output_memmap,
        tile=(256, 256),
        metadata={'axes': 'CYX'},
        description=ome_xml,
        photometric='minisblack',
        compression='lzw',
        bigtiff=True
    )

    # Clean up
    del output_memmap
    gc.collect()

    # Remove temporary directory
    try:
        import shutil
        shutil.rmtree(tmp_dir)
        log(f"Cleaned up temporary files: {tmp_dir}")
    except Exception as e:
        log(f"Warning: Could not remove temporary directory {tmp_dir}: {e}")

    file_size = Path(output_path).stat().st_size / (1024 * 1024 * 1024)
    log(f"✓ Saved: {output_path} ({file_size:.2f} GB, {num_output_channels} channels)")
    log(f"✓ Channel list: {', '.join(channel_names)}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge registered slides into single OME-TIFF (keeps all channels except duplicate DAPI)'
    )
    parser.add_argument('--input-dir', required=True, help='Directory with registered slides')
    parser.add_argument('--output', required=True, help='Output OME-TIFF path')
    parser.add_argument(
        '--reference-markers',
        nargs='+',
        default=['DAPI'],
        help='Markers that identify reference slide (default: DAPI)'
    )

    args = parser.parse_args()

    try:
        merge_slides(args.input_dir, args.output, reference_markers=args.reference_markers)
        slide_io.kill_jvm()
        log("✓ Complete!")
    except Exception as e:
        log(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
