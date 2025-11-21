#!/usr/bin/env python3
"""
Merge individually registered slides into a single OME-TIFF file.

This script takes a directory of registered slides (output from VALIS warp_and_save_slide)
and merges them into a single multi-channel OME-TIFF, skipping duplicate channels.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import tifffile


def log(msg: str):
    """Print timestamped message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def get_channel_name(filename: str) -> str:
    """Extract channel name from filename."""
    name = filename.replace('_registered.ome.tif', '').replace('_registered.ome.tiff', '')
    return name.replace('_corrected', '')


def merge_slides(input_dir: str, output_path: str, skip_duplicates: bool = True):
    """Merge all registered slides into a single OME-TIFF."""

    log(f"Merging slides from: {input_dir}")
    log(f"Output: {output_path}")

    # Find all registered slides
    slide_files = sorted(
        list(Path(input_dir).glob("*_registered.ome.tif")) +
        list(Path(input_dir).glob("*_registered.ome.tiff"))
    )

    if not slide_files:
        raise ValueError(f"No registered slides found in {input_dir}")

    log(f"Found {len(slide_files)} slides")

    # Load all channels
    channels = []
    channel_names = []
    seen = set()

    for slide_file in slide_files:
        log(f"Loading: {slide_file.name}")

        # Read image
        img = tifffile.imread(str(slide_file))

        # Get channel name
        base_name = get_channel_name(slide_file.name)

        # Handle single or multi-channel images
        if img.ndim == 2:
            # Single channel: (H, W)
            img = img[np.newaxis, ...]  # Add channel dimension: (1, H, W)
            names = [base_name]
        elif img.ndim == 3:
            # Multi-channel: could be (H, W, C) or (C, H, W)
            # Detect based on typical image dimensions
            if img.shape[0] < img.shape[2]:
                # Likely (C, H, W) format
                names = [f"{base_name}_C{i}" for i in range(img.shape[0])]
            else:
                # Likely (H, W, C) format - transpose to (C, H, W)
                img = np.transpose(img, (2, 0, 1))
                names = [f"{base_name}_C{i}" for i in range(img.shape[0])]
        else:
            raise ValueError(f"Unsupported image dimensions: {img.shape}")

        log(f"  Shape: {img.shape}, channels: {len(names)}")

        # Add each channel
        for i, name in enumerate(names):
            if skip_duplicates and name in seen:
                log(f"  ⊗ Skipping duplicate: {name}")
                continue

            channels.append(img[i])
            channel_names.append(name)
            seen.add(name)
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
                PhysicalSizeX="0.325" PhysicalSizeY="0.325" PhysicalSizeXUnit="µm" PhysicalSizeYUnit="µm">
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
        compression='lzw',
        tile=(256, 256),
        metadata={'axes': 'CYX'},
        description=ome_xml,
    )

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    log(f"✓ Saved: {output_path} ({file_size:.1f} MB, {num_channels} channels)")


def main():
    parser = argparse.ArgumentParser(description='Merge registered slides into single OME-TIFF')
    parser.add_argument('--input-dir', required=True, help='Directory with registered slides')
    parser.add_argument('--output', required=True, help='Output OME-TIFF path')
    parser.add_argument('--keep-duplicates', action='store_true', help='Keep duplicate channels')

    args = parser.parse_args()

    try:
        merge_slides(args.input_dir, args.output, skip_duplicates=not args.keep_duplicates)
        log("✓ Complete!")
    except Exception as e:
        log(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
