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
from metadata import extract_channel_names_from_ome as get_channel_names_from_ome
from metadata import extract_markers_from_filename
from image_utils import normalize_image_dimensions

# Import only slide_io to avoid heavy VALIS dependencies
import valis.slide_io as slide_io

# Function definitions removed - now imported from lib modules:
# - log() -> imported from lib.logger.log_progress
# - get_channel_names_from_ome() -> imported from lib.metadata
# - extract_markers_from_filename() -> imported from lib.metadata


def is_reference_slide(slide_name: str, reference_markers: list) -> bool:
    """Check if slide contains reference markers."""
    slide_name_upper = slide_name.upper()
    # Slide is reference if it contains ALL reference markers
    return all(marker.upper() in slide_name_upper for marker in reference_markers)


def read_slide(filepath: str) -> tuple:
    """Read slide using tifffile (RAM-efficient). Returns (numpy array (C, H, W), channel_names)."""
    # Read with tifffile which handles large TIFFs efficiently
    img = tifffile.imread(str(filepath))

    # Ensure (C, H, W) format using shared utility function
    img = normalize_image_dimensions(img)

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


def merge_slides(input_dir: str, output_path: str, reference_markers: list = None, segmentation_mask: str = None, phenotype_mask: str = None, phenotype_mapping: str = None):
    """
    Merge all registered slides into a single OME-TIFF.

    Keeps all channels from all slides, except DAPI which is only kept from the reference slide.
    Optionally appends segmentation and phenotype masks as additional channels.

    Memory-efficient: Uses memory-mapped arrays and writes channels incrementally.

    Args:
        input_dir: Directory containing registered slides
        output_path: Output OME-TIFF path
        reference_markers: List of markers that identify the reference slide (e.g., ['DAPI', 'SMA'])
        segmentation_mask: Path to segmentation mask file (optional)
        phenotype_mask: Path to phenotype mask file with different colors per phenotype (optional)
        phenotype_mapping: Path to JSON file mapping phenotype numbers to names (optional)
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
    # Important: Use ome=True to ensure proper OME-XML metadata structure
    # This allows bfconvert to correctly read and preserve metadata when creating pyramids
    tifffile.imwrite(
        output_path,
        output_memmap,
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
    parser.add_argument('--segmentation-mask', help='Path to segmentation mask TIFF')
    parser.add_argument('--phenotype-mask', help='Path to phenotype mask TIFF')
    parser.add_argument('--phenotype-mapping', help='Path to phenotype mapping JSON (phenotype number to name)')

    args = parser.parse_args()

    try:
        merge_slides(
            args.input_dir,
            args.output,
            reference_markers=args.reference_markers,
            segmentation_mask=args.segmentation_mask,
            phenotype_mask=args.phenotype_mask,
            phenotype_mapping=args.phenotype_mapping,
        )
        slide_io.kill_jvm()
        log("✓ Complete!")
    except Exception as e:
        log(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
