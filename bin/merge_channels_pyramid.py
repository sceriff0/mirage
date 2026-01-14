#!/usr/bin/env python3
"""
Merge single-channel TIFF files into a single multi-channel PYRAMIDAL OME-TIFF.

This script takes a directory of single-channel TIFF files (output from SPLIT_CHANNELS)
and merges them into a single multi-channel pyramidal OME-TIFF. DAPI filtering is already
handled by the split_multichannel.py script, so this just combines all channels.

Optionally appends segmentation and phenotype masks as additional channels.
Phenotype masks are stored as single-channel integer label images with embedded
colormap metadata for visualization in QuPath and other OME-TIFF viewers.

FEATURES:
- Generates pyramidal OME-TIFF directly (no need for bfconvert)
- Proper OME-XML with channel names, colors, and pixel sizes
- Embedded colormap for phenotype label visualization
- QuPath compatible output
- Memory-efficient processing for large images
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import tifffile
import os
import gc
import tempfile
import json
import colorsys
from typing import Dict, List, Tuple, Optional
from xml.sax.saxutils import escape as xml_escape

os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = tempfile.gettempdir() + '/numba_cache'
os.environ['NUMBA_DISABLE_CACHING'] = '1'

# Import from lib modules for DRY principle
sys.path.insert(0, str(Path(__file__).parent / 'utils'))
try:
    from logger import log_progress as log
except ImportError:
    def log(msg):
        print(f"[INFO] {msg}")


# =============================================================================
# PHENOTYPE COLOR PALETTE - Distinctive colors for categorical visualization
# =============================================================================
PHENOTYPE_COLORS = {
    # Index 0 is always background
    0: {"name": "Background", "rgb": (0, 0, 0)},
    # Immune cells - cool colors
    1: {"name": "Immune", "rgb": (0, 255, 0)},           # Green
    2: {"name": "T helper", "rgb": (255, 255, 0)},        # Yellow
    3: {"name": "T cytotoxic", "rgb": (0, 255, 255)},     # Cyan
    4: {"name": "activated T cytotoxic", "rgb": (0, 200, 255)},  # Light cyan
    5: {"name": "CD4 T regulatory", "rgb": (255, 200, 0)}, # Gold
    6: {"name": "CD8 T regulatory", "rgb": (200, 255, 0)}, # Lime
    # Macrophages - orange/red tones
    7: {"name": "Macrophages", "rgb": (255, 128, 0)},     # Orange
    8: {"name": "M1", "rgb": (255, 80, 80)},              # Light red
    9: {"name": "M2", "rgb": (255, 160, 80)},             # Light orange
    # Tumor - red/magenta
    10: {"name": "PANCK+ Tumor", "rgb": (255, 0, 0)},     # Red
    11: {"name": "VIM+ Tumor", "rgb": (255, 0, 255)},     # Magenta
    # Stroma - neutral
    12: {"name": "Stroma", "rgb": (128, 128, 128)},       # Gray
    13: {"name": "Unknown", "rgb": (64, 64, 64)},         # Dark gray
}

# Standard marker colors for fluorescence channels
MARKER_COLORS = {
    'DAPI': (0, 0, 255),        # Blue
    'CD45': (0, 255, 0),        # Green
    'CD3': (255, 255, 0),       # Yellow
    'CD8': (255, 0, 255),       # Magenta
    'CD4': (0, 255, 255),       # Cyan
    'CD14': (255, 128, 0),      # Orange
    'CD163': (255, 0, 128),     # Pink
    'FOXP3': (128, 255, 0),     # Lime
    'PANCK': (255, 0, 0),       # Red
    'VIMENTIN': (128, 0, 255),  # Purple
    'SMA': (0, 128, 255),       # Sky blue
    'GZMB': (255, 128, 128),    # Light red
    'PD1': (128, 255, 128),     # Light green
    'PDL1': (255, 200, 100),    # Peach
    'L1CAM': (100, 200, 255),   # Light blue
    'PAX2': (200, 100, 255),    # Light purple
    'CD74': (255, 255, 128),    # Light yellow
    'Segmentation': (255, 255, 255),  # White
}


def rgb_to_ome_color(r: int, g: int, b: int, a: int = 255) -> int:
    """
    Convert RGB(A) to OME-XML Color attribute (signed 32-bit ARGB).

    OME uses signed 32-bit integer: if high bit set, interpreted as negative.
    Format: ARGB packed as (A << 24) | (R << 16) | (G << 8) | B

    Returns signed int32 as required by OME-XML spec.
    """
    # Pack as unsigned first
    value = (a << 24) | (r << 16) | (g << 8) | b
    # Convert to signed 32-bit (Python handles arbitrary precision)
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def generate_channel_color(name: str, index: int) -> Tuple[int, int, int]:
    """Generate a color for a channel based on name or index."""
    # Check if we have a predefined color
    if name in MARKER_COLORS:
        return MARKER_COLORS[name]

    # Generate color using golden ratio for good distribution
    h = (index * 0.618033988749895) % 1.0
    s = 0.7 + (index % 3) * 0.1  # Vary saturation slightly
    v = 0.85 + (index % 2) * 0.1  # Vary value slightly

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def create_phenotype_colormap(label_map: Dict[int, str], n_categories: int) -> Dict[int, Tuple[str, Tuple[int, int, int]]]:
    """
    Create a colormap for phenotype categories.

    Returns dict: {index: (name, (r, g, b))}
    """
    colormap = {}

    # First, try to match by phenotype name
    name_to_color = {v["name"]: v["rgb"] for v in PHENOTYPE_COLORS.values()}

    for idx in range(n_categories):
        name = label_map.get(idx, f"Phenotype_{idx}")

        # Try to find color by name
        if name in name_to_color:
            rgb = name_to_color[name]
        elif idx < len(PHENOTYPE_COLORS):
            # Use indexed color
            rgb = list(PHENOTYPE_COLORS.values())[idx]["rgb"]
        else:
            # Generate a random but deterministic color
            np.random.seed(idx + 42)
            rgb = tuple(np.random.randint(50, 255, 3).tolist())

        colormap[idx] = (name, rgb)

    return colormap


def build_ome_xml(
    width: int,
    height: int,
    num_channels: int,
    dtype: np.dtype,
    channel_names: List[str],
    channel_colors: List[Tuple[int, int, int]],
    phenotype_colormap: Optional[Dict[int, Tuple[str, Tuple[int, int, int]]]] = None,
    physical_size_x: float = 0.325,
    physical_size_y: float = 0.325,
    physical_size_unit: str = "um"
) -> str:
    """
    Build complete OME-XML with proper channel metadata and phenotype annotations.

    This creates OME-XML that:
    1. Defines all channels with names and colors
    2. Includes StructuredAnnotations for phenotype colormap (QuPath can read this)
    3. Is fully compliant with OME-XML 2016-06 schema
    """

    # Map numpy dtype to OME pixel type
    dtype_map = {
        'uint8': 'uint8',
        'uint16': 'uint16',
        'uint32': 'uint32',
        'int8': 'int8',
        'int16': 'int16',
        'int32': 'int32',
        'float32': 'float',
        'float64': 'double',
    }
    ome_dtype = dtype_map.get(str(dtype), 'uint16')

    # Build channel elements
    channel_elements = []
    for i, (name, color) in enumerate(zip(channel_names, channel_colors)):
        ome_color = rgb_to_ome_color(*color)
        safe_name = xml_escape(name)
        channel_elements.append(
            f'            <Channel ID="Channel:0:{i}" Name="{safe_name}" '
            f'Color="{ome_color}" SamplesPerPixel="1"/>'
        )

    channels_xml = "\n".join(channel_elements)

    # Build StructuredAnnotations for phenotype colormap
    annotations_xml = ""
    if phenotype_colormap:
        map_entries = []
        for idx, (name, rgb) in sorted(phenotype_colormap.items()):
            safe_name = xml_escape(name)
            r, g, b = rgb
            # Store as key-value pairs that QuPath/other tools can parse
            map_entries.append(f'                <M K="phenotype_{idx}_name">{safe_name}</M>')
            map_entries.append(f'                <M K="phenotype_{idx}_color">#{r:02x}{g:02x}{b:02x}</M>')
            map_entries.append(f'                <M K="phenotype_{idx}_rgb">{r},{g},{b}</M>')

        # Also add a summary JSON for easy parsing
        colormap_json = json.dumps({
            str(k): {"name": v[0], "rgb": list(v[1])}
            for k, v in phenotype_colormap.items()
        })

        annotations_xml = f'''
    <StructuredAnnotations>
        <MapAnnotation ID="Annotation:Phenotypes" Namespace="phenotype.colormap">
            <Description>Phenotype label colormap for categorical visualization</Description>
            <Value>
{chr(10).join(map_entries)}
                <M K="colormap_json">{xml_escape(colormap_json)}</M>
                <M K="n_categories">{len(phenotype_colormap)}</M>
            </Value>
        </MapAnnotation>
    </StructuredAnnotations>'''

    # Assemble full OME-XML
    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
    <Image ID="Image:0" Name="MultiplexImage">
        <Description>Multiplex imaging data with segmentation and phenotype masks</Description>
        <Pixels ID="Pixels:0"
                Type="{ome_dtype}"
                SizeX="{width}"
                SizeY="{height}"
                SizeZ="1"
                SizeC="{num_channels}"
                SizeT="1"
                DimensionOrder="XYCZT"
                PhysicalSizeX="{physical_size_x}"
                PhysicalSizeY="{physical_size_y}"
                PhysicalSizeXUnit="{physical_size_unit}"
                PhysicalSizeYUnit="{physical_size_unit}"
                Interleaved="false"
                BigEndian="false">
{channels_xml}
            <TiffData/>
        </Pixels>
    </Image>{annotations_xml}
</OME>'''

    return ome_xml


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a 2D image by a given factor using area averaging.
    
    This is more memory-efficient than cv2.resize for large images
    and produces good quality results for pyramid generation.
    
    Args:
        image: 2D numpy array
        factor: Downsampling factor (2 = half size)
    
    Returns:
        Downsampled 2D array
    """
    # Try to use cv2 if available (faster and better quality)
    try:
        import cv2
        h, w = image.shape
        new_h, new_w = h // factor, w // factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except ImportError:
        pass
    
    # Fallback: block averaging (works well for integer factors)
    h, w = image.shape
    new_h, new_w = h // factor, w // factor
    
    # Trim to exact multiple of factor
    trimmed = image[:new_h * factor, :new_w * factor]
    
    # Reshape and average
    if image.dtype in [np.float32, np.float64]:
        reshaped = trimmed.reshape(new_h, factor, new_w, factor)
        return reshaped.mean(axis=(1, 3)).astype(image.dtype)
    else:
        # For integer types, use float for averaging then convert back
        reshaped = trimmed.reshape(new_h, factor, new_w, factor).astype(np.float32)
        return reshaped.mean(axis=(1, 3)).astype(image.dtype)


def calculate_pyramid_levels(
    height: int, 
    width: int, 
    min_size: int = 256,
    max_levels: int = 10,
    scale_factor: int = 2
) -> List[Tuple[int, int]]:
    """
    Calculate pyramid level dimensions.
    
    Args:
        height: Base image height
        width: Base image width
        min_size: Minimum dimension for smallest level
        max_levels: Maximum number of pyramid levels
        scale_factor: Downscaling factor between levels
    
    Returns:
        List of (height, width) tuples for each level
    """
    levels = [(height, width)]
    h, w = height, width
    
    for _ in range(max_levels - 1):
        h = h // scale_factor
        w = w // scale_factor
        
        if h < min_size or w < min_size:
            break
            
        levels.append((h, w))
    
    return levels


def write_pyramidal_ome_tiff(
    data_source,  # Can be memmap or callable that returns channel data
    output_path: str,
    ome_xml: str,
    num_channels: int,
    height: int,
    width: int,
    dtype: np.dtype,
    pyramid_resolutions: int = 5,
    pyramid_scale: int = 2,
    tile_size: int = 256,
    compression: str = 'lzw'
):
    """
    Write a pyramidal OME-TIFF with proper metadata.
    
    This function writes each channel with its pyramid levels as SubIFDs,
    which is the format expected by Bio-Formats and QuPath.
    
    Args:
        data_source: Either a memmap array (C, Y, X) or callable(channel_idx) -> 2D array
        output_path: Output file path
        ome_xml: Complete OME-XML string
        num_channels: Number of channels
        height: Image height
        width: Image width
        dtype: Data type
        pyramid_resolutions: Number of pyramid levels (including base)
        pyramid_scale: Downscaling factor between levels
        tile_size: Tile size for efficient access
        compression: Compression algorithm ('lzw', 'zlib', 'jpeg', None)
    """
    # Calculate pyramid levels
    levels = calculate_pyramid_levels(
        height, width, 
        min_size=tile_size, 
        max_levels=pyramid_resolutions,
        scale_factor=pyramid_scale
    )
    
    log(f"Writing pyramidal OME-TIFF with {len(levels)} resolution levels:")
    for i, (h, w) in enumerate(levels):
        log(f"  Level {i}: {w} x {h}")
    
    # Determine if data_source is indexable or callable
    is_callable = callable(data_source)
    
    # Write pyramidal TIFF
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        options = dict(
            tile=(tile_size, tile_size),
            compression=compression,
            photometric='minisblack',
        )
        
        for c in range(num_channels):
            # Get channel data
            if is_callable:
                channel_data = data_source(c)
            else:
                channel_data = np.asarray(data_source[c])
            
            # Ensure 2D
            if channel_data.ndim > 2:
                channel_data = channel_data.squeeze()
            
            # Write base resolution (level 0)
            # First channel gets the OME-XML description
            tif.write(
                channel_data,
                subifds=len(levels) - 1,  # Number of sub-resolution levels
                description=ome_xml if c == 0 else None,
                metadata=None,  # Don't let tifffile add its own metadata
                **options
            )
            
            # Write pyramid levels as SubIFDs
            current_data = channel_data
            for level_idx in range(1, len(levels)):
                # Downsample from previous level
                downsampled = downsample_image(current_data, pyramid_scale)
                
                tif.write(
                    downsampled,
                    subfiletype=1,  # Reduced resolution image
                    **options
                )
                
                current_data = downsampled
            
            # Clean up
            del current_data
            if is_callable:
                del channel_data
            
            # Progress logging
            if (c + 1) % 5 == 0 or c == num_channels - 1:
                log(f"  Written channel {c + 1}/{num_channels}")
            
            gc.collect()
    
    log(f"Pyramidal OME-TIFF complete: {output_path}")


def merge_channels(
    input_dir: str,
    output_path: str,
    segmentation_mask: str = None,
    phenotype_mask: str = None,
    phenotype_mapping: str = None,
    physical_size_x: float = 0.325,
    physical_size_y: float = 0.325,
    pyramid_resolutions: int = 5,
    pyramid_scale: int = 2,
    tile_size: int = 256,
    compression: str = 'lzw'
):
    """
    Merge all single-channel TIFF files into a single pyramidal OME-TIFF.

    DAPI filtering is already handled by split_multichannel.py, so all channels
    in the input directory are included.

    Memory-efficient: Uses memory-mapped arrays and writes channels incrementally.

    Args:
        input_dir: Directory containing single-channel TIFF files
        output_path: Output OME-TIFF path
        segmentation_mask: Path to segmentation mask file (optional)
        phenotype_mask: Path to phenotype mask with integer labels (optional)
        phenotype_mapping: Path to JSON file mapping phenotype numbers to names (optional)
        physical_size_x: Pixel size in X (micrometers)
        physical_size_y: Pixel size in Y (micrometers)
        pyramid_resolutions: Number of pyramid levels to generate
        pyramid_scale: Downscaling factor between pyramid levels
        tile_size: Tile size for pyramid (default 256)
        compression: Compression algorithm ('lzw', 'zlib', 'jpeg', or None)
    """
    log(f"=" * 70)
    log(f"MERGE CHANNELS - Pyramidal OME-TIFF (QuPath Compatible)")
    log(f"=" * 70)
    log(f"Input directory: {input_dir}")
    log(f"Output: {output_path}")
    log(f"Pyramid: {pyramid_resolutions} levels, scale factor {pyramid_scale}")

    # Find all single-channel TIFF files
    channel_files = sorted(
        list(Path(input_dir).glob("*.tif")) +
        list(Path(input_dir).glob("*.tiff"))
    )

    if not channel_files:
        raise ValueError(f"No TIFF files found in {input_dir}")

    log(f"Found {len(channel_files)} channel files")

    # PASS 1: Collect metadata and determine final dimensions
    log("-" * 50)
    log("Pass 1: Scanning channels for metadata...")
    channel_names = []
    channel_colors = []
    height, width, dtype = None, None, None

    for i, channel_file in enumerate(channel_files):
        # Extract channel name from filename (without extension)
        channel_name = channel_file.stem
        channel_names.append(channel_name)

        # Assign color
        color = generate_channel_color(channel_name, i)
        channel_colors.append(color)

        log(f"  [{i}] {channel_name}: RGB{color}")

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
    pheno_label_map = {}
    phenotype_colormap = None

    if phenotype_mapping:
        log(f"Loading phenotype mapping: {phenotype_mapping}")
        with open(phenotype_mapping, 'r') as f:
            pheno_label_map = json.load(f)
        # Convert keys to integers if they're strings
        pheno_label_map = {int(k): v for k, v in pheno_label_map.items()}
        log(f"  Loaded {len(pheno_label_map)} phenotype labels:")
        for idx, name in sorted(pheno_label_map.items()):
            log(f"    {idx}: {name}")

    # Determine which masks to add
    masks_info = []

    if segmentation_mask:
        log(f"Will append segmentation mask: {segmentation_mask}")
        channel_names.append("Segmentation")
        channel_colors.append(MARKER_COLORS.get('Segmentation', (255, 255, 255)))
        masks_info.append(('segmentation', segmentation_mask))

    if phenotype_mask:
        log(f"Will append phenotype mask: {phenotype_mask}")
        channel_names.append("Phenotype")
        # Use a neutral color for the label channel itself
        channel_colors.append((200, 200, 200))
        masks_info.append(('phenotype', phenotype_mask))

    num_output_channels = len(channel_names)
    log("-" * 50)
    log(f"Total output channels: {num_output_channels}")
    log(f"Output dimensions: C={num_output_channels}, H={height}, W={width}")

    # Calculate memory requirements
    bytes_per_pixel = np.dtype(dtype).itemsize
    total_bytes = num_output_channels * height * width * bytes_per_pixel
    total_gb = total_bytes / (1024**3)
    log(f"Estimated uncompressed size: {total_gb:.2f} GB")

    # PASS 2: Create temporary memmap and accumulate all channels
    log("-" * 50)
    log("Pass 2: Accumulating channels...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Create temporary memmap to accumulate all channels
    tmp_dir = Path(tempfile.mkdtemp(prefix="merge_channels_"))
    output_memmap_path = tmp_dir / "merged.npy"

    try:
        output_shape = (num_output_channels, height, width)
        output_memmap = np.memmap(str(output_memmap_path), dtype=dtype, mode='w+', shape=output_shape)

        output_channel_idx = 0

        # Process each channel file
        for channel_file in channel_files:
            channel_name = channel_file.stem
            log(f"  Loading channel {output_channel_idx}: {channel_name}")

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
            output_channel_idx += 1

            # Clean up channel data immediately
            del channel_data
            gc.collect()

        # Variables to store phenotype data for RGB generation
        phenotype_data = None

        # Append mask channels if provided
        for mask_type, mask_path in masks_info:
            log(f"  Processing {mask_type} mask: {mask_path}")
            mask_data = tifffile.imread(mask_path)

            # Ensure 2D
            if mask_data.ndim > 2:
                mask_data = mask_data.squeeze()

            # Verify dimensions match
            if mask_data.shape != (height, width):
                raise ValueError(f"{mask_type} mask shape {mask_data.shape} doesn't match image shape ({height}, {width})")

            # Process phenotype mask specially
            if mask_type == 'phenotype':
                log(f"    Processing phenotype labels...")

                # Get range
                pheno_min = int(mask_data.min())
                pheno_max = int(mask_data.max())
                n_categories = pheno_max + 1
                log(f"    Label range: {pheno_min} to {pheno_max} ({n_categories} categories)")

                # Handle negative values (shift to 0-based)
                if pheno_min < 0:
                    log(f"    Shifting values by {-pheno_min} to make non-negative")
                    mask_data = mask_data - pheno_min
                    pheno_max = pheno_max - pheno_min
                    n_categories = pheno_max + 1

                # Convert to appropriate dtype
                if pheno_max <= 255:
                    mask_data = mask_data.astype(np.uint8)
                    log(f"    Converted to uint8")
                elif pheno_max <= 65535:
                    mask_data = mask_data.astype(np.uint16)
                    log(f"    Converted to uint16")
                else:
                    mask_data = mask_data.astype(np.uint32)
                    log(f"    Converted to uint32")

                # Create colormap
                phenotype_colormap = create_phenotype_colormap(pheno_label_map, n_categories)
                log(f"    Created colormap for {len(phenotype_colormap)} phenotypes:")
                for idx, (name, rgb) in sorted(phenotype_colormap.items()):
                    log(f"      {idx}: {name} -> RGB{rgb}")

                # Store for RGB generation
                phenotype_data = mask_data.copy()

            # Write to output memmap
            log(f"  Writing channel {output_channel_idx}: {mask_type.capitalize()}")
            output_memmap[output_channel_idx, :, :] = mask_data
            output_channel_idx += 1

            # Clean up
            del mask_data
            gc.collect()

        # Clean up phenotype data if it exists
        if phenotype_data is not None:
            del phenotype_data
            gc.collect()

        # Flush memmap to disk
        log("Flushing temporary buffer...")
        output_memmap.flush()

        # Build OME-XML
        log("-" * 50)
        log("Building OME-XML metadata...")
        ome_xml = build_ome_xml(
            width=width,
            height=height,
            num_channels=num_output_channels,
            dtype=dtype,
            channel_names=channel_names,
            channel_colors=channel_colors,
            phenotype_colormap=phenotype_colormap,
            physical_size_x=physical_size_x,
            physical_size_y=physical_size_y
        )

        # Log OME-XML summary
        log(f"  Channels defined: {num_output_channels}")
        log(f"  Pixel size: {physical_size_x} x {physical_size_y} µm")
        if phenotype_colormap:
            log(f"  Phenotype annotations: {len(phenotype_colormap)} categories")

        # Write final pyramidal output file
        log("-" * 50)
        log(f"Writing pyramidal OME-TIFF: {output_path}")
        
        write_pyramidal_ome_tiff(
            data_source=output_memmap,
            output_path=output_path,
            ome_xml=ome_xml,
            num_channels=num_output_channels,
            height=height,
            width=width,
            dtype=dtype,
            pyramid_resolutions=pyramid_resolutions,
            pyramid_scale=pyramid_scale,
            tile_size=tile_size,
            compression=compression
        )

        # Clean up memmap
        del output_memmap
        gc.collect()

    finally:
        # Remove temporary directory (always cleanup)
        try:
            import shutil
            shutil.rmtree(tmp_dir)
            log(f"Cleaned up temporary files")
        except Exception as e:
            log(f"Warning: Could not remove temporary directory {tmp_dir}: {e}")

    # Final summary
    file_size = Path(output_path).stat().st_size / (1024 * 1024 * 1024)
    log("=" * 70)
    log(f"SUCCESS: {output_path}")
    log(f"  Size: {file_size:.2f} GB")
    log(f"  Channels: {num_output_channels}")
    log(f"  Channel list: {', '.join(channel_names)}")
    log(f"  Pyramid levels: {pyramid_resolutions}")
    log(f"  Pixel size: {physical_size_x} x {physical_size_y} µm")
    if phenotype_colormap:
        log(f"  Phenotype categories: {len(phenotype_colormap)}")
    log("=" * 70)

    # Also save a standalone colormap JSON for QuPath import
    if phenotype_colormap:
        colormap_output = str(Path(output_path).with_suffix('.phenotype_colors.json'))
        colormap_data = {
            "description": "Phenotype colormap for QuPath visualization",
            "format": "index -> {name, rgb}",
            "categories": {
                str(k): {"name": v[0], "rgb": list(v[1]), "hex": f"#{v[1][0]:02x}{v[1][1]:02x}{v[1][2]:02x}"}
                for k, v in phenotype_colormap.items()
            }
        }
        with open(colormap_output, 'w') as f:
            json.dump(colormap_data, f, indent=2)
        log(f"Saved colormap: {colormap_output}")

    return channel_names, phenotype_colormap


def main():
    parser = argparse.ArgumentParser(
        description='Merge single-channel TIFFs into pyramidal multi-channel OME-TIFF (QuPath compatible)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge with default pyramid settings
  %(prog)s --input-dir ./channels --output merged.ome.tiff

  # With custom pixel size and pyramid settings
  %(prog)s --input-dir ./channels --output merged.ome.tiff \\
      --physical-size-x 0.5 --physical-size-y 0.5 \\
      --pyramid-resolutions 6 --pyramid-scale 2

  # With segmentation and phenotype masks
  %(prog)s --input-dir ./channels --output merged.ome.tiff \\
      --segmentation-mask seg.tiff \\
      --phenotype-mask pheno.tiff \\
      --phenotype-mapping phenotypes.json
"""
    )
    parser.add_argument('--input-dir', required=True, 
                        help='Directory with single-channel TIFF files')
    parser.add_argument('--output', required=True, 
                        help='Output OME-TIFF path')
    parser.add_argument('--segmentation-mask', 
                        help='Path to segmentation mask TIFF')
    parser.add_argument('--phenotype-mask', 
                        help='Path to phenotype mask TIFF')
    parser.add_argument('--phenotype-mapping',
                        help='Path to phenotype mapping JSON (phenotype number to name)')
    parser.add_argument('--physical-size-x', type=float, default=0.325,
                        help='Pixel size in X (micrometers, default: 0.325)')
    parser.add_argument('--physical-size-y', type=float, default=0.325,
                        help='Pixel size in Y (micrometers, default: 0.325)')
    
    # Pyramid-specific arguments
    parser.add_argument('--pyramid-resolutions', type=int, default=5,
                        help='Number of pyramid resolution levels (default: 5)')
    parser.add_argument('--pyramid-scale', type=int, default=2,
                        help='Downscaling factor between pyramid levels (default: 2)')
    parser.add_argument('--tile-size', type=int, default=256,
                        help='Tile size for pyramid (default: 256)')
    parser.add_argument('--compression', type=str, default='lzw',
                        choices=['lzw', 'zlib', 'jpeg', 'none'],
                        help='Compression algorithm (default: lzw)')

    args = parser.parse_args()
    
    # Handle compression='none'
    compression = None if args.compression == 'none' else args.compression

    try:
        merge_channels(
            args.input_dir,
            args.output,
            segmentation_mask=args.segmentation_mask,
            phenotype_mask=args.phenotype_mask,
            phenotype_mapping=args.phenotype_mapping,
            physical_size_x=args.physical_size_x,
            physical_size_y=args.physical_size_y,
            pyramid_resolutions=args.pyramid_resolutions,
            pyramid_scale=args.pyramid_scale,
            tile_size=args.tile_size,
            compression=compression,
        )
        log("Complete!")
    except Exception as e:
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
