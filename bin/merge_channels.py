#!/usr/bin/env python3
"""
Merge single-channel TIFF files into a single multi-channel OME-TIFF.

This script takes a directory of single-channel TIFF files (output from SPLIT_CHANNELS)
and merges them into a single multi-channel OME-TIFF. DAPI filtering is already handled
by the split_multichannel.py script, so this just combines all channels.

Optionally appends segmentation and phenotype masks as additional channels.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import tifffile
import os
import gc
import tempfile

os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_CACHING'] = '1'

# Import from lib modules for DRY principle
sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from logger import log_progress as log


def merge_channels(input_dir: str, output_path: str, segmentation_mask: str = None, phenotype_mask: str = None, phenotype_mapping: str = None):
    """
    Merge all single-channel TIFF files into a single OME-TIFF.

    DAPI filtering is already handled by split_multichannel.py, so all channels
    in the input directory are included.

    Memory-efficient: Uses memory-mapped arrays and writes channels incrementally.

    Args:
        input_dir: Directory containing single-channel TIFF files
        output_path: Output OME-TIFF path
        segmentation_mask: Path to segmentation mask file (optional)
        phenotype_mask: Path to phenotype mask file with different colors per phenotype (optional)
        phenotype_mapping: Path to JSON file mapping phenotype numbers to names (optional)
    """
    log(f"Merging channels from: {input_dir}")
    log(f"Output: {output_path}")

    # Find all single-channel TIFF files
    channel_files = sorted(
        list(Path(input_dir).glob("*.tif")) +
        list(Path(input_dir).glob("*.tiff"))
    )

    if not channel_files:
        raise ValueError(f"No TIFF files found in {input_dir}")

    log(f"Found {len(channel_files)} channel files")

    # PASS 1: Collect metadata and determine final dimensions
    log("Pass 1: Scanning channels for metadata...")
    channel_names = []
    height, width, dtype = None, None, None

    for channel_file in channel_files:
        # Extract channel name from filename (without extension)
        channel_name = channel_file.stem
        channel_names.append(channel_name)
        log(f"  Channel: {channel_name}")

        # Read metadata only (shape and dtype)
        with tifffile.TiffFile(str(channel_file)) as tif:
            page = tif.pages[0]
            if height is None:
                # Get shape - handle both 2D and 3D
                if len(page.shape) == 2:
                    height, width = page.shape
                else:
                    height, width = page.shape[-2:]  # Get H, W from last two dims
                dtype = page.dtype
                log(f"  Image dimensions: {height} x {width}, dtype: {dtype}")
            else:
                # Verify all channels have same dimensions
                if len(page.shape) == 2:
                    h, w = page.shape
                else:
                    h, w = page.shape[-2:]
                if (h, w) != (height, width):
                    raise ValueError(f"Channel {channel_name} has different dimensions: {h}x{w} vs {height}x{width}")

    # Load phenotype mapping if provided
    pheno_label_map = None
    if phenotype_mapping:
        log(f"Loading phenotype mapping: {phenotype_mapping}")
        import json
        with open(phenotype_mapping, 'r') as f:
            pheno_label_map = json.load(f)
        # Convert keys to integers if they're strings
        pheno_label_map = {int(k): v for k, v in pheno_label_map.items()}
        log(f"  Loaded {len(pheno_label_map)} phenotype labels: {pheno_label_map}")

    # Check if masks should be added
    masks_to_add = []
    if segmentation_mask:
        log(f"Will append segmentation mask: {segmentation_mask}")
        channel_names.append("Segmentation")
        masks_to_add.append(('segmentation', segmentation_mask, None))

    if phenotype_mask:
        log(f"Will append phenotype mask: {phenotype_mask}")
        channel_names.append("Phenotype")
        masks_to_add.append(('phenotype', phenotype_mask, pheno_label_map))

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
    log("Creating temporary output buffer...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="merge_channels_"))
    output_memmap_path = tmp_dir / "merged.npy"

    output_shape = (num_output_channels, height, width)
    output_memmap = np.memmap(str(output_memmap_path), dtype=dtype, mode='w+', shape=output_shape)

    output_channel_idx = 0

    # Process each channel file
    for channel_file, channel_name in zip(channel_files, channel_names[:len(channel_files)]):
        log(f"Processing: {channel_file.name}")

        # Read single channel
        channel_data = tifffile.imread(str(channel_file))

        # Ensure 2D
        if channel_data.ndim > 2:
            channel_data = channel_data.squeeze()

        # Verify dimensions match
        if channel_data.shape != (height, width):
            raise ValueError(f"Channel {channel_name} shape {channel_data.shape} doesn't match expected ({height}, {width})")

        # Write to output memmap
        output_memmap[output_channel_idx, :, :] = channel_data
        log(f"  Channel {output_channel_idx}: {channel_name}")

        output_channel_idx += 1

        # Clean up channel data immediately
        del channel_data
        gc.collect()

    # Store phenotype colormap and labels for later
    phenotype_lut = None
    phenotype_n_categories = 0
    phenotype_labels = None

    # Append mask channels if provided
    for mask_type, mask_path, label_map in masks_to_add:
        log(f"Appending {mask_type} mask: {mask_path}")
        mask_data = tifffile.imread(mask_path)

        # Ensure 2D
        if mask_data.ndim > 2:
            mask_data = mask_data.squeeze()

        # Verify dimensions match
        if mask_data.shape != (height, width):
            raise ValueError(f"{mask_type} mask shape {mask_data.shape} doesn't match image shape ({height}, {width})")

        # Convert phenotype mask to categorical format
        if mask_type == 'phenotype':
            log(f"  Converting phenotype mask to categorical format...")
            # Convert int64 to int32 or smaller for compatibility
            if mask_data.dtype == np.int64:
                pheno_min = mask_data.min()
                pheno_max = mask_data.max()
                log(f"    Phenotype range: {pheno_min} to {pheno_max}")

                # Shift negative values to start from 0 for categorical LUT
                if pheno_min < 0:
                    log(f"    Shifting values by {-pheno_min} to make non-negative")
                    mask_data = mask_data - pheno_min
                    pheno_max = pheno_max - pheno_min

                # Convert to smallest compatible unsigned type
                if pheno_max <= 255:
                    mask_data = mask_data.astype(np.uint8)
                    log(f"    Converted to uint8 for categorical display")
                elif pheno_max <= 65535:
                    mask_data = mask_data.astype(np.uint16)
                    log(f"    Converted to uint16 for categorical display")
                else:
                    mask_data = mask_data.astype(np.int32)
                    log(f"    Converted to int32 (too many categories for uint16)")

            # Create categorical colormap for QuPath
            phenotype_n_categories = int(mask_data.max() + 1)
            log(f"  Creating categorical LUT for {phenotype_n_categories} phenotypes...")

            # Store labels if provided
            if label_map:
                phenotype_labels = label_map
                log(f"  Using phenotype labels: {phenotype_labels}")

            # Distinctive colors for categorical display (RGB)
            base_colors = [
                [0, 0, 0],         # 0: Background (black)
                [255, 0, 0],       # 1: Red
                [0, 255, 0],       # 2: Green
                [0, 0, 255],       # 3: Blue
                [255, 255, 0],     # 4: Yellow
                [255, 0, 255],     # 5: Magenta
                [0, 255, 255],     # 6: Cyan
                [255, 128, 0],     # 7: Orange
                [128, 0, 255],     # 8: Purple
                [0, 255, 128],     # 9: Spring green
                [255, 0, 128],     # 10: Rose
                [128, 255, 0],     # 11: Lime
                [0, 128, 255],     # 12: Sky blue
                [255, 128, 128],   # 13: Light red
                [128, 255, 128],   # 14: Light green
                [128, 128, 255],   # 15: Light blue
                [192, 192, 0],     # 16: Olive
                [192, 0, 192],     # 17: Dark magenta
                [0, 192, 192],     # 18: Teal
                [255, 192, 128],   # 19: Peach
            ]

            # Extend with random colors if needed
            import random
            colors = base_colors.copy()
            for i in range(len(colors), phenotype_n_categories):
                random.seed(i)
                colors.append([random.randint(50, 255) for _ in range(3)])

            # Create LUT for tifffile
            if mask_data.dtype == np.uint8:
                lut_size = 256
            elif mask_data.dtype == np.uint16:
                lut_size = 65536
            else:
                lut_size = 256

            phenotype_lut = np.zeros((3, lut_size), dtype=np.uint16)
            for i, color in enumerate(colors[:min(phenotype_n_categories, lut_size)]):
                phenotype_lut[0, i] = color[0] * 256  # R (scale to uint16 range)
                phenotype_lut[1, i] = color[1] * 256  # G
                phenotype_lut[2, i] = color[2] * 256  # B

        # Write to output memmap
        output_memmap[output_channel_idx, :, :] = mask_data
        log(f"  Channel {output_channel_idx}: {mask_type.capitalize()}")
        output_channel_idx += 1

        # Clean up
        del mask_data
        gc.collect()

    # Flush memmap to disk
    log("Flushing temporary buffer...")
    output_memmap.flush()

    # Write final output file from memmap
    log(f"Writing final output file: {output_path}")

    # Prepare ImageJ metadata with colormap if phenotype mask is present
    imagej_metadata = None
    if phenotype_lut is not None:
        log(f"Adding ImageJ metadata with {phenotype_n_categories} phenotype colors for QuPath...")
        imagej_metadata = {
            'axes': 'CYX',
            'LUTs': [None] * (num_output_channels - 1) + [phenotype_lut.T]  # Only apply LUT to phenotype channel
        }

        # Add phenotype labels if available
        if phenotype_labels:
            # Create labels array for ImageJ
            labels_array = [''] * phenotype_n_categories
            for idx, label in phenotype_labels.items():
                if idx < phenotype_n_categories:
                    labels_array[idx] = label
            imagej_metadata['Labels'] = labels_array
            log(f"  Added phenotype labels to ImageJ metadata: {labels_array}")

    # Write OME-TIFF with proper metadata
    tifffile.imwrite(
        output_path,
        output_memmap,
        tile=(256, 256),
        metadata={'axes': 'CYX'} if imagej_metadata is None else imagej_metadata,
        description=ome_xml,
        photometric='minisblack',
        compression='lzw',
        bigtiff=True,
        ome=True,  # Write proper OME-TIFF format
        imagej=True if phenotype_lut is not None else False
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
        description='Merge single-channel TIFFs into multi-channel OME-TIFF'
    )
    parser.add_argument('--input-dir', required=True, help='Directory with single-channel TIFF files')
    parser.add_argument('--output', required=True, help='Output OME-TIFF path')
    parser.add_argument('--segmentation-mask', help='Path to segmentation mask TIFF')
    parser.add_argument('--phenotype-mask', help='Path to phenotype mask TIFF')
    parser.add_argument('--phenotype-mapping', help='Path to phenotype mapping JSON (phenotype number to name)')

    args = parser.parse_args()

    try:
        merge_channels(
            args.input_dir,
            args.output,
            segmentation_mask=args.segmentation_mask,
            phenotype_mask=args.phenotype_mask,
            phenotype_mapping=args.phenotype_mapping,
        )
        log("✓ Complete!")
    except Exception as e:
        log(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
