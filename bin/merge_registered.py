#!/usr/bin/env python3
"""
Merge individually registered slides into a single OME-TIFF file.

This script takes a directory of registered slides (output from VALIS warp_and_save_slide)
and merges them into a single multi-channel OME-TIFF, skipping duplicate channels.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import pyvips
import numpy as np
from typing import List, Tuple, Dict


def log_progress(msg: str):
    """Print timestamped progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def get_channel_name_from_filename(filename: str) -> str:
    """
    Extract channel name from registered slide filename.

    Expected format: {original_name}_registered.ome.tif
    Where original_name typically contains the marker name.
    """
    # Remove _registered.ome.tif suffix
    name = filename.replace('_registered.ome.tif', '')
    name = name.replace('_registered.ome.tiff', '')

    # Extract marker name (typically last part before any suffix)
    # Example: "B19-10215_DAPI_SMA_panck_corrected" -> we want the marker part
    parts = name.split('_')

    # Return the base name for now - this will be the channel identifier
    return name


def load_registered_slide(slide_path: str) -> Tuple[pyvips.Image, List[str]]:
    """
    Load a registered slide and extract its channel names.

    Returns:
        (image, channel_names)
    """
    img = pyvips.Image.new_from_file(slide_path)

    # Try to get channel names from metadata
    channel_names = []

    # Check if image has channel names in metadata
    if 'image-description' in img.get_fields():
        desc = img.get('image-description')
        # Parse OME-XML if present - simplified for now
        # In practice, you might want to use xml.etree for proper parsing
        pass

    # If no metadata, generate names from filename and band count
    base_name = get_channel_name_from_filename(Path(slide_path).name)

    if img.bands == 1:
        channel_names = [base_name]
    else:
        # Multi-channel image - add band index
        channel_names = [f"{base_name}_C{i}" for i in range(img.bands)]

    return img, channel_names


def merge_slides(
    input_dir: str,
    output_path: str,
    skip_duplicates: bool = True
) -> None:
    """
    Merge all registered slides in input_dir into a single multi-channel OME-TIFF.

    Args:
        input_dir: Directory containing *_registered.ome.tif files
        output_path: Output path for merged OME-TIFF
        skip_duplicates: If True, skip channels with duplicate names
    """
    log_progress(f"Merging registered slides from: {input_dir}")
    log_progress(f"Output: {output_path}")
    log_progress(f"Skip duplicates: {skip_duplicates}")
    log_progress("")

    # Find all registered slide files
    input_path = Path(input_dir)
    slide_files = sorted(
        list(input_path.glob("*_registered.ome.tif")) +
        list(input_path.glob("*_registered.ome.tiff"))
    )

    if not slide_files:
        raise ValueError(f"No registered slides found in {input_dir}")

    log_progress(f"Found {len(slide_files)} registered slides:")
    for f in slide_files:
        log_progress(f"  - {f.name}")
    log_progress("")

    # Load all slides and their channel names
    all_images: List[pyvips.Image] = []
    all_channel_names: List[str] = []
    seen_channels: set = set()

    for idx, slide_file in enumerate(slide_files, 1):
        log_progress(f"[{idx}/{len(slide_files)}] Loading: {slide_file.name}")

        img, channel_names = load_registered_slide(str(slide_file))

        log_progress(f"  Channels: {channel_names}")
        log_progress(f"  Size: {img.width}x{img.height}, {img.bands} bands")

        # Process each channel/band
        for band_idx, channel_name in enumerate(channel_names):
            if skip_duplicates and channel_name in seen_channels:
                log_progress(f"  ⊗ Skipping duplicate channel: {channel_name}")
                continue

            # Extract single band
            if img.bands == 1:
                band_img = img
            else:
                band_img = img.extract_band(band_idx)

            all_images.append(band_img)
            all_channel_names.append(channel_name)
            seen_channels.add(channel_name)

            log_progress(f"  ✓ Added channel: {channel_name}")

        log_progress("")

    # Merge all channels
    log_progress(f"Merging {len(all_images)} channels into single image...")

    if len(all_images) == 0:
        raise ValueError("No channels to merge!")

    if len(all_images) == 1:
        merged = all_images[0]
    else:
        merged = pyvips.Image.bandjoin(all_images)

    log_progress(f"  Final size: {merged.width}x{merged.height}, {merged.bands} bands")
    log_progress(f"  Channel names: {all_channel_names}")
    log_progress("")

    # Save merged image as OME-TIFF with channel names
    log_progress(f"Saving merged image to: {output_path}")

    # Create OME-XML metadata
    ome_xml = create_ome_metadata(
        width=merged.width,
        height=merged.height,
        channels=all_channel_names,
        pixel_size=0.325  # TODO: get from params or metadata
    )

    # Ensure output directory exists
    os.makedirs(Path(output_path).parent, exist_ok=True)

    # Save with OME-TIFF format
    merged.write_to_file(
        output_path,
        compression='lzw',
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
        bigtiff=True,
        properties=True,
        xres=merged.get('xres') if 'xres' in merged.get_fields() else 3076.923,
        yres=merged.get('yres') if 'yres' in merged.get_fields() else 3076.923,
        **{'image-description': ome_xml}
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log_progress(f"✓ Merged image saved: {output_path} ({file_size_mb:.2f} MB)")
    log_progress(f"✓ Total channels: {len(all_channel_names)}")


def create_ome_metadata(
    width: int,
    height: int,
    channels: List[str],
    pixel_size: float = 0.325
) -> str:
    """
    Create OME-XML metadata for the merged image.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        channels: List of channel names
        pixel_size: Physical pixel size in microns

    Returns:
        OME-XML string
    """
    num_channels = len(channels)

    # Create channel XML entries
    channel_entries = []
    for idx, name in enumerate(channels):
        channel_entries.append(
            f'<Channel ID="Channel:0:{idx}" Name="{name}" SamplesPerPixel="1" />'
        )

    channels_xml = '\n        '.join(channel_entries)

    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
    <Image ID="Image:0" Name="Merged">
        <Pixels ID="Pixels:0"
                Type="uint16"
                SizeX="{width}"
                SizeY="{height}"
                SizeZ="1"
                SizeC="{num_channels}"
                SizeT="1"
                DimensionOrder="XYCZT"
                PhysicalSizeX="{pixel_size}"
                PhysicalSizeY="{pixel_size}"
                PhysicalSizeXUnit="µm"
                PhysicalSizeYUnit="µm">
        {channels_xml}
        <TiffData />
        </Pixels>
    </Image>
</OME>'''

    return ome_xml


def main():
    parser = argparse.ArgumentParser(
        description='Merge registered slides into single OME-TIFF'
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing registered slides (*_registered.ome.tif)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output path for merged OME-TIFF'
    )
    parser.add_argument(
        '--keep-duplicates',
        action='store_true',
        help='Keep duplicate channel names instead of skipping them'
    )

    args = parser.parse_args()

    try:
        merge_slides(
            input_dir=args.input_dir,
            output_path=args.output,
            skip_duplicates=not args.keep_duplicates
        )
        log_progress("✓ Merge complete!")
        sys.exit(0)

    except Exception as e:
        log_progress(f"✗ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
