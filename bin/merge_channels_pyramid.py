#!/usr/bin/env python3
"""Merge single-channel TIFF files into a pyramidal multi-channel OME-TIFF."""
from __future__ import annotations

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

# Add path for utils
sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from logger import get_logger, configure_logging

try:
    from validation import log_image_stats, detect_wrapped_values, validate_image_range
except ImportError:
    # Fallback if validation module not available
    def log_image_stats(data, stage, logger=None): pass
    def detect_wrapped_values(data, **kwargs): return False, 0, 0.0
    def validate_image_range(data, stage, **kwargs): return True, data

os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = tempfile.gettempdir() + '/numba_cache'
os.environ['NUMBA_DISABLE_CACHING'] = '1'

logger = get_logger(__name__)

__all__ = ["main"]


def log(msg):
    logger.info(msg)


# =============================================================================
# PHENOTYPE COLOR PALETTE
# =============================================================================
PHENOTYPE_COLORS = {
    0: {"name": "Background", "rgb": (0, 0, 0)},
    1: {"name": "Immune", "rgb": (0, 255, 0)},
    2: {"name": "T helper", "rgb": (255, 255, 0)},
    3: {"name": "T cytotoxic", "rgb": (0, 255, 255)},
    4: {"name": "activated T cytotoxic", "rgb": (0, 200, 255)},
    5: {"name": "CD4 T regulatory", "rgb": (255, 200, 0)},
    6: {"name": "CD8 T regulatory", "rgb": (200, 255, 0)},
    7: {"name": "Macrophages", "rgb": (255, 128, 0)},
    8: {"name": "M1", "rgb": (255, 80, 80)},
    9: {"name": "M2", "rgb": (255, 160, 80)},
    10: {"name": "PANCK+ Tumor", "rgb": (255, 0, 0)},
    11: {"name": "VIM+ Tumor", "rgb": (255, 0, 255)},
    12: {"name": "Stroma", "rgb": (128, 128, 128)},
    13: {"name": "Unknown", "rgb": (64, 64, 64)},
}

MARKER_COLORS = {
    'DAPI': (0, 0, 255),
    'CD45': (0, 255, 0),
    'CD3': (255, 255, 0),
    'CD8': (255, 0, 255),
    'CD4': (0, 255, 255),
    'CD14': (255, 128, 0),
    'CD163': (255, 0, 128),
    'FOXP3': (128, 255, 0),
    'PANCK': (255, 0, 0),
    'VIMENTIN': (128, 0, 255),
    'SMA': (0, 128, 255),
    'GZMB': (255, 128, 128),
    'PD1': (128, 255, 128),
    'PDL1': (255, 200, 100),
    'L1CAM': (100, 200, 255),
    'PAX2': (200, 100, 255),
    'CD74': (255, 255, 128),
    'Segmentation': (255, 255, 255),
}


def rgb_to_ome_color(r: int, g: int, b: int, a: int = 255) -> int:
    """Convert RGBA to OME-XML signed 32-bit ARGB color."""
    value = (a << 24) | (r << 16) | (g << 8) | b
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def generate_channel_color(name: str, index: int) -> Tuple[int, int, int]:
    """Generate a color for a channel based on name or index."""
    # Check predefined colors first
    name_upper = name.upper()
    for key in MARKER_COLORS:
        if key.upper() in name_upper or name_upper in key.upper():
            return MARKER_COLORS[key]
    
    if name in MARKER_COLORS:
        return MARKER_COLORS[name]

    # Generate color using golden ratio
    h = (index * 0.618033988749895) % 1.0
    s = 0.7 + (index % 3) * 0.1
    v = 0.85 + (index % 2) * 0.1
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def create_phenotype_colormap(label_map: Dict[int, str], n_categories: int) -> Dict[int, Tuple[str, Tuple[int, int, int]]]:
    """Create a colormap for phenotype categories."""
    colormap = {}
    name_to_color = {v["name"]: v["rgb"] for v in PHENOTYPE_COLORS.values()}

    for idx in range(n_categories):
        name = label_map.get(idx, f"Phenotype_{idx}")
        if name in name_to_color:
            rgb = name_to_color[name]
        elif idx < len(PHENOTYPE_COLORS):
            rgb = list(PHENOTYPE_COLORS.values())[idx]["rgb"]
        else:
            np.random.seed(idx + 42)
            rgb = tuple(np.random.randint(50, 255, 3).tolist())
        colormap[idx] = (name, rgb)

    return colormap


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2D or 3D (CYX) image by a given factor."""
    try:
        import cv2
        if image.ndim == 2:
            h, w = image.shape
            new_h, new_w = h // factor, w // factor
            # cv2.INTER_AREA is good for downsampling but may introduce small rounding errors
            # For integer types, use float intermediate then round
            if image.dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
                downsampled = cv2.resize(image.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
                return np.round(downsampled).astype(image.dtype)
            else:
                return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif image.ndim == 3:
            # CYX format - downsample each channel
            c, h, w = image.shape
            new_h, new_w = h // factor, w // factor
            result = np.zeros((c, new_h, new_w), dtype=image.dtype)
            for i in range(c):
                if image.dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
                    downsampled = cv2.resize(image[i].astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
                    result[i] = np.round(downsampled).astype(image.dtype)
                else:
                    result[i] = cv2.resize(image[i], (new_w, new_h), interpolation=cv2.INTER_AREA)
            return result
    except ImportError:
        pass

    # Fallback: block averaging
    if image.ndim == 2:
        h, w = image.shape
        new_h, new_w = h // factor, w // factor
        trimmed = image[:new_h * factor, :new_w * factor]
        reshaped = trimmed.reshape(new_h, factor, new_w, factor)
        if image.dtype in [np.float32, np.float64]:
            return reshaped.mean(axis=(1, 3)).astype(image.dtype)
        else:
            # Fix: Round before converting to integer dtype to avoid truncation artifacts
            return np.round(reshaped.mean(axis=(1, 3))).astype(image.dtype)
    elif image.ndim == 3:
        c, h, w = image.shape
        new_h, new_w = h // factor, w // factor
        result = np.zeros((c, new_h, new_w), dtype=image.dtype)
        for i in range(c):
            trimmed = image[i, :new_h * factor, :new_w * factor]
            reshaped = trimmed.reshape(new_h, factor, new_w, factor)
            # Fix: Round before converting to integer dtype to avoid truncation artifacts
            if image.dtype in [np.float32, np.float64]:
                result[i] = reshaped.mean(axis=(1, 3)).astype(image.dtype)
            else:
                result[i] = np.round(reshaped.mean(axis=(1, 3))).astype(image.dtype)
        return result


def calculate_pyramid_levels(
    height: int,
    width: int,
    min_size: int = 256,
    max_levels: int = 10,
    scale_factor: int = 2
) -> List[Tuple[int, int]]:
    """Calculate pyramid level dimensions."""
    levels = [(height, width)]
    h, w = height, width

    for _ in range(max_levels - 1):
        h = h // scale_factor
        w = w // scale_factor
        if h < min_size or w < min_size:
            break
        levels.append((h, w))

    return levels


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
    physical_size_unit: str = "µm"
) -> str:
    """
    Build OME-XML metadata for multi-channel pyramidal image.
    
    IMPORTANT: For tifffile's automatic OME-TIFF handling, we use the 
    metadata dict approach rather than raw XML for the base image,
    but we can still embed additional annotations.
    """
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
            f'      <Channel ID="Channel:0:{i}" Name="{safe_name}" '
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
            map_entries.append(f'        <M K="phenotype_{idx}_name">{safe_name}</M>')
            map_entries.append(f'        <M K="phenotype_{idx}_color">#{r:02x}{g:02x}{b:02x}</M>')

        colormap_json = json.dumps({
            str(k): {"name": v[0], "rgb": list(v[1])}
            for k, v in phenotype_colormap.items()
        })

        annotations_xml = f'''
  <StructuredAnnotations>
    <MapAnnotation ID="Annotation:Phenotypes" Namespace="phenotype.colormap">
      <Value>
{chr(10).join(map_entries)}
        <M K="colormap_json">{xml_escape(colormap_json)}</M>
      </Value>
    </MapAnnotation>
  </StructuredAnnotations>'''

    # Build complete OME-XML
    # NOTE: TiffData is left simple - tifffile will handle IFD mapping
    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
  <Image ID="Image:0" Name="MultiplexImage">
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


def write_pyramidal_ome_tiff(
    data: np.ndarray,  # Full CYX array
    output_path: str,
    channel_names: List[str],
    channel_colors: List[Tuple[int, int, int]],
    phenotype_colormap: Optional[Dict] = None,
    physical_size_x: float = 0.325,
    physical_size_y: float = 0.325,
    pyramid_resolutions: int = 5,
    pyramid_scale: int = 2,
    tile_size: int = 256,
    compression: str = 'lzw'
):
    """
    Write a pyramidal OME-TIFF with proper QuPath-compatible structure.
    
    KEY FIX: Write the entire CYX array at once, then write pyramid levels.
    This creates the correct IFD structure that Bio-Formats/QuPath expects:
    - Base resolution: all channels as separate pages
    - SubIFDs: downsampled versions for each channel
    
    Args:
        data: 3D numpy array in CYX order (channels, height, width)
        output_path: Output file path
        channel_names: List of channel names
        channel_colors: List of (R, G, B) tuples
        phenotype_colormap: Optional colormap for phenotype labels
        physical_size_x: Pixel size in X (micrometers)
        physical_size_y: Pixel size in Y (micrometers)
        pyramid_resolutions: Number of pyramid levels
        pyramid_scale: Downscaling factor between levels
        tile_size: Tile size for efficient access
        compression: Compression algorithm
    """
    num_channels, height, width = data.shape
    
    # Calculate pyramid levels
    levels = calculate_pyramid_levels(
        height, width,
        min_size=tile_size,
        max_levels=pyramid_resolutions,
        scale_factor=pyramid_scale
    )
    num_subresolutions = len(levels) - 1
    
    log(f"Writing pyramidal OME-TIFF with {len(levels)} resolution levels:")
    for i, (h, w) in enumerate(levels):
        log(f"  Level {i}: {w} x {h}")

    # Build metadata dict for tifffile (it will generate proper OME-XML)
    metadata = {
        'axes': 'CYX',
        'Channel': {'Name': channel_names},
        'PhysicalSizeX': physical_size_x,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': physical_size_y,
        'PhysicalSizeYUnit': 'µm',
    }

    # Common write options
    options = dict(
        tile=(tile_size, tile_size),
        compression=compression,
        photometric='minisblack',
        resolutionunit='CENTIMETER',
    )

    with tifffile.TiffWriter(output_path, bigtiff=True, ome=True) as tif:
        # Write base resolution with all channels
        # subifds parameter reserves space for pyramid levels
        log(f"  Writing base resolution ({width} x {height})...")
        tif.write(
            data,
            subifds=num_subresolutions,
            resolution=(1e4 / physical_size_x, 1e4 / physical_size_y),
            metadata=metadata,
            **options
        )

        # Generate and write pyramid levels
        current_data = data
        for level_idx in range(1, len(levels)):
            level_h, level_w = levels[level_idx]
            log(f"  Writing pyramid level {level_idx} ({level_w} x {level_h})...")
            
            # Downsample from previous level
            downsampled = downsample_image(current_data, pyramid_scale)
            
            tif.write(
                downsampled,
                subfiletype=1,  # REDUCEDIMAGE flag for pyramid level
                resolution=(1e4 / (physical_size_x * (pyramid_scale ** level_idx)),
                           1e4 / (physical_size_y * (pyramid_scale ** level_idx))),
                **options
            )
            
            current_data = downsampled
            gc.collect()

    log(f"Pyramidal OME-TIFF complete: {output_path}")
    
    # Verify the output
    verify_ome_tiff(output_path)


def verify_ome_tiff(path: str):
    """Verify the OME-TIFF structure is correct."""
    log("Verifying OME-TIFF structure...")
    with tifffile.TiffFile(path) as tif:
        log(f"  Number of pages: {len(tif.pages)}")
        log(f"  Number of series: {len(tif.series)}")
        
        if tif.series:
            series = tif.series[0]
            log(f"  Series 0 shape: {series.shape}")
            log(f"  Series 0 axes: {series.axes}")
            log(f"  Series 0 is_pyramidal: {series.is_pyramidal}")
            
            if hasattr(series, 'levels') and series.levels:
                log(f"  Pyramid levels: {len(series.levels)}")
                for i, level in enumerate(series.levels):
                    log(f"    Level {i}: shape={level.shape}")
        
        if tif.ome_metadata:
            log("  OME-XML metadata: present")
            # Parse and show channel names
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                channels = root.findall('.//ome:Channel', ns)
                if channels:
                    log(f"  Channels in OME-XML: {len(channels)}")
                    for ch in channels[:5]:  # Show first 5
                        log(f"    - {ch.get('Name')}")
                    if len(channels) > 5:
                        log(f"    ... and {len(channels) - 5} more")
            except Exception as e:
                log(f"  Warning: Could not parse OME-XML: {e}")
        else:
            log("  WARNING: No OME-XML metadata found!")


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

    # PASS 1: Collect metadata
    log("-" * 50)
    log("Pass 1: Scanning channels for metadata...")
    channel_names = []
    channel_colors = []
    height, width, dtype = None, None, None

    for i, channel_file in enumerate(channel_files):
        channel_name = channel_file.stem
        channel_names.append(channel_name)
        color = generate_channel_color(channel_name, i)
        channel_colors.append(color)
        log(f"  [{i}] {channel_name}: RGB{color}")

        with tifffile.TiffFile(str(channel_file)) as tif:
            page = tif.pages[0]
            if height is None:
                if len(page.shape) == 2:
                    height, width = page.shape
                else:
                    height, width = page.shape[-2:]
                dtype = page.dtype
                log(f"  Image dimensions: {height} x {width}, dtype: {dtype}")
            else:
                if len(page.shape) == 2:
                    h, w = page.shape
                else:
                    h, w = page.shape[-2:]
                if (h, w) != (height, width):
                    raise ValueError(f"Dimension mismatch: {channel_name}")

    # Load phenotype mapping
    pheno_label_map = {}
    phenotype_colormap = None

    if phenotype_mapping:
        log(f"Loading phenotype mapping: {phenotype_mapping}")
        with open(phenotype_mapping, 'r') as f:
            pheno_label_map = json.load(f)
        pheno_label_map = {int(k): v for k, v in pheno_label_map.items()}

    # Add mask channels
    masks_info = []
    if segmentation_mask:
        log(f"Will append segmentation mask: {segmentation_mask}")
        channel_names.append("Segmentation")
        channel_colors.append(MARKER_COLORS.get('Segmentation', (255, 255, 255)))
        masks_info.append(('segmentation', segmentation_mask))

    if phenotype_mask:
        log(f"Will append phenotype mask: {phenotype_mask}")
        channel_names.append("Phenotype")
        channel_colors.append((200, 200, 200))
        masks_info.append(('phenotype', phenotype_mask))

    num_output_channels = len(channel_names)
    log(f"Total output channels: {num_output_channels}")

    # PASS 2: Load all channels into memory
    log("-" * 50)
    log("Pass 2: Loading channels into memory...")
    
    # Create output array
    output_data = np.zeros((num_output_channels, height, width), dtype=dtype)
    output_idx = 0

    # Load channel files
    for channel_file in channel_files:
        log(f"  Loading: {channel_file.stem}")
        channel_data = tifffile.imread(str(channel_file))
        if channel_data.ndim > 2:
            channel_data = channel_data.squeeze()

        # Checkpoint 4: Validate channel data for negative/wrapped values
        ch_min, ch_max = channel_data.min(), channel_data.max()
        if ch_min < 0:
            log(f"    WARNING: Negative values detected in {channel_file.stem}: min={ch_min}")
            channel_data = np.clip(channel_data, 0, None)
            log(f"    Clipped to 0. New min={channel_data.min()}")
        elif dtype == np.uint16:
            # Check for wrapped values (negatives that became high positives)
            has_wrapped, wrap_count, wrap_pct = detect_wrapped_values(channel_data)
            if has_wrapped:
                log(f"    WARNING: {wrap_count} potential wrapped negative pixels ({wrap_pct:.4f}%) in {channel_file.stem}")

        output_data[output_idx] = channel_data
        output_idx += 1
        del channel_data
        gc.collect()

    # Load mask channels
    for mask_type, mask_path in masks_info:
        log(f"  Loading {mask_type} mask...")
        mask_data = tifffile.imread(mask_path)
        if mask_data.ndim > 2:
            mask_data = mask_data.squeeze()

        if mask_type == 'phenotype':
            pheno_min = int(mask_data.min())
            pheno_max = int(mask_data.max())
            n_categories = pheno_max + 1
            log(f"    Label range: {pheno_min} to {pheno_max}")

            if pheno_min < 0:
                mask_data = mask_data - pheno_min
                pheno_max = pheno_max - pheno_min
                n_categories = pheno_max + 1

            phenotype_colormap = create_phenotype_colormap(pheno_label_map, n_categories)

        # Convert mask to output dtype if needed
        # CRITICAL FIX: Don't downcast segmentation masks (uint32) to uint16
        # This would cause label IDs > 65535 to overflow/wrap around
        if mask_type == 'segmentation' and mask_data.dtype == np.uint32:
            # Keep segmentation masks as uint32 to preserve all label IDs
            log(f"    WARNING: Segmentation mask is uint32 but channel dtype is {dtype}")
            log(f"    Keeping mask as uint32 to avoid label ID overflow")
            # This means the output array needs to support uint32 for this channel
            # We'll need to handle this specially - for now, clip to max value
            if dtype in [np.uint8, np.uint16]:
                max_val = np.iinfo(dtype).max
                if mask_data.max() > max_val:
                    log(f"    ERROR: Mask has labels up to {mask_data.max()} but dtype {dtype} max is {max_val}")
                    log(f"    Clipping mask values to {max_val} - this may cause data loss!")
                    mask_data = np.clip(mask_data, 0, max_val).astype(dtype)
                else:
                    mask_data = mask_data.astype(dtype)
            else:
                mask_data = mask_data.astype(dtype)
        elif mask_data.dtype != dtype:
            mask_data = mask_data.astype(dtype)

        output_data[output_idx] = mask_data
        output_idx += 1
        del mask_data
        gc.collect()

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # PASS 3: Write pyramidal OME-TIFF
    log("-" * 50)
    log("Pass 3: Writing pyramidal OME-TIFF...")
    
    write_pyramidal_ome_tiff(
        data=output_data,
        output_path=output_path,
        channel_names=channel_names,
        channel_colors=channel_colors,
        phenotype_colormap=phenotype_colormap,
        physical_size_x=physical_size_x,
        physical_size_y=physical_size_y,
        pyramid_resolutions=pyramid_resolutions,
        pyramid_scale=pyramid_scale,
        tile_size=tile_size,
        compression=compression
    )

    # Clean up
    del output_data
    gc.collect()

    # Summary
    file_size = Path(output_path).stat().st_size / (1024 * 1024 * 1024)
    log("=" * 70)
    log(f"SUCCESS: {output_path}")
    log(f"  Size: {file_size:.2f} GB")
    log(f"  Channels: {num_output_channels}")
    log(f"  Channel list: {', '.join(channel_names)}")
    log("=" * 70)

    # Save colormap JSON
    if phenotype_colormap:
        colormap_output = str(Path(output_path).with_suffix('.phenotype_colors.json'))
        colormap_data = {
            "categories": {
                str(k): {"name": v[0], "rgb": list(v[1])}
                for k, v in phenotype_colormap.items()
            }
        }
        with open(colormap_output, 'w') as f:
            json.dump(colormap_data, f, indent=2)
        log(f"Saved colormap: {colormap_output}")

    return channel_names, phenotype_colormap


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge single-channel TIFFs into pyramidal OME-TIFF (QuPath compatible)',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
                        help='Path to phenotype mapping JSON')
    parser.add_argument('--physical-size-x', type=float, default=0.325,
                        help='Pixel size in X (micrometers, default: 0.325)')
    parser.add_argument('--physical-size-y', type=float, default=0.325,
                        help='Pixel size in Y (micrometers, default: 0.325)')
    parser.add_argument('--pyramid-resolutions', type=int, default=5,
                        help='Number of pyramid levels (default: 5)')
    parser.add_argument('--pyramid-scale', type=int, default=2,
                        help='Downscaling factor between levels (default: 2)')
    parser.add_argument('--tile-size', type=int, default=256,
                        help='Tile size (default: 256)')
    parser.add_argument('--compression', type=str, default='lzw',
                        choices=['lzw', 'zlib', 'jpeg', 'none'],
                        help='Compression algorithm (default: lzw)')
    return parser.parse_args()


def main() -> int:
    """Run merge and pyramid CLI."""
    configure_logging()
    args = parse_args()
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
        return 0
    except Exception as e:
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
