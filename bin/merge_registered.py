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
    # Remove _registered and _corrected suffixes
    name = filename.replace('_registered.ome.tif', '').replace('_registered.ome.tiff', '')
    name = name.replace('_corrected', '')

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

    Args:
        input_dir: Directory containing registered slides
        output_path: Output OME-TIFF path
        reference_markers: List of markers that identify the reference slide (e.g., ['DAPI', 'SMA'])
    """
    if reference_markers is None:
        reference_markers = ['DAPI']  # Default

    log(f"Merging slides from: {input_dir}")
    log(f"Output: {output_path}")
    log(f"Reference markers: {reference_markers}")

    # Find all registered slides
    slide_files = sorted(
        list(Path(input_dir).glob("*_registered.ome.tif")) +
        list(Path(input_dir).glob("*_registered.ome.tiff"))
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

    # Load all channels
    channels = []
    channel_names = []

    for slide_file in slide_files:
        log(f"Loading: {slide_file.name}")
        is_reference = (slide_file == reference_slide)

        # Read using VALIS slide_io (returns image and channel names)
        img, names = read_slide(str(slide_file))

        log(f"  Shape: {img.shape}, channels: {names}")
        log(f"  Is reference: {is_reference}")

        # Add each channel
        for i, name in enumerate(names):
            # Check if this is DAPI channel
            is_dapi = name.upper() == 'DAPI'

            if is_dapi:
                if is_reference:
                    # Keep DAPI from reference slide
                    channels.append(img[i])
                    channel_names.append(name)
                    log(f"  ✓ Added DAPI (from reference)")
                else:
                    # Skip DAPI from non-reference slides
                    log(f"  ⊗ Skipping DAPI (not reference)")
            else:
                # Keep all non-DAPI channels
                channels.append(img[i])
                channel_names.append(name)
                log(f"  ✓ Added: {name}")

    # Stack all channels
    log(f"Stacking {len(channels)} channels...")
    merged = np.stack(channels, axis=0)  # (C, H, W)
    log(f"  Final shape: {merged.shape}")

    # Create OME-XML metadata
    num_channels, height, width = merged.shape
    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
    <Image ID="Image:0" Name="Merged">
        <Pixels ID="Pixels:0" Type="uint16"
                SizeX="{width}" SizeY="{height}" SizeZ="1" SizeC="{num_channels}" SizeT="1"
                DimensionOrder="XYCZT"
                PhysicalSizeX="0.325" PhysicalSizeY="0.325" PhysicalSizeXUnit="um" PhysicalSizeYUnit="um">
            {chr(10).join(f'<Channel ID="Channel:0:{i}" Name="{name}" SamplesPerPixel="1" />' for i, name in enumerate(channel_names))}
            <TiffData />
        </Pixels>
    </Image>
</OME>'''

    # Save
    log("Writing OME-TIFF...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(
        output_path,
        merged,
        tile=(256, 256),
        metadata={'axes': 'CYX'},
        description=ome_xml,
    )

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    log(f"✓ Saved: {output_path} ({file_size:.1f} MB, {num_channels} channels)")
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
