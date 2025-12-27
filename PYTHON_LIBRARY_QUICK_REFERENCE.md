# Python Library Quick Reference

Quick reference guide for using the refactored `lib/` modules in your Python scripts.

---

## Table of Contents

1. [Logging](#logging)
2. [Metadata](#metadata)
3. [Registration Utilities](#registration-utilities)
4. [QC Generation](#qc-generation)
5. [Migration Guide](#migration-guide)

---

## Logging

### Basic Setup

```python
from lib.logger import get_logger, configure_logging
import logging

def main():
    # Configure once at script startup
    configure_logging(level=logging.INFO)

    # Get logger in your module
    logger = get_logger(__name__)

    logger.info("Processing started")
    logger.debug("Debug details")
    logger.warning("Warning message")
    logger.error("Error occurred")
```

### With Log File

```python
configure_logging(
    level=logging.DEBUG,
    log_file="pipeline.log",
    format_string='%(asctime)s - %(levelname)s - %(message)s'
)
```

### In Multiple Modules

```python
# main.py
from lib.logger import configure_logging
configure_logging(level=logging.INFO)  # Once!

# module1.py
from lib.logger import get_logger
logger = get_logger(__name__)  # Will use global config

# module2.py
from lib.logger import get_logger
logger = get_logger(__name__)  # Will use same global config
```

---

## Metadata

### Extract Channel Names from Filename

```python
from lib.metadata import get_channel_names

# Standard pipeline naming: SampleID_Channel1_Channel2...
channels = get_channel_names("B19-10215_DAPI_SMA_panCK.tif")
# Returns: ['DAPI', 'SMA', 'panCK']

# Works with suffixes
channels = get_channel_names("Sample01_DAPI_CD3_corrected.ome.tiff")
# Returns: ['DAPI', 'CD3']
```

### Extract from OME-XML Metadata

```python
from lib.metadata import extract_channel_names_from_ome

channels = extract_channel_names_from_ome("registered.ome.tif")
# Returns: ['DAPI', 'SMA', 'panCK', 'CD3'] or [] if not OME-TIFF
```

### Create OME-XML Metadata

```python
from lib.metadata import create_ome_xml
import numpy as np

xml = create_ome_xml(
    channel_names=['DAPI', 'SMA', 'panCK'],
    dtype=np.uint16,
    width=2048,
    height=2048,
    pixel_size_um=0.325
)

# Use when saving TIFF:
tifffile.imwrite("output.ome.tif", image, description=xml)
```

### Get All Metadata

```python
from lib.metadata import get_ome_metadata

meta = get_ome_metadata("image.ome.tif")
# Returns dict with:
# {
#     'channel_names': ['DAPI', 'SMA'],
#     'num_channels': 2,
#     'width': 2048,
#     'height': 2048,
#     'dtype': 'uint16',
#     'pixel_size_um': 0.325,
#     'ome_xml': '<?xml ...'
# }
```

---

## Registration Utilities

### Autoscale Image (ImageJ Style)

```python
from lib.registration_utils import autoscale
import numpy as np

# Load 16-bit image
img = tifffile.imread("image.tif")  # dtype=uint16

# Scale to 0-255 using percentiles
scaled = autoscale(img, low_p=1.0, high_p=99.0)
# Returns: uint8 array

# Conservative scaling (wider range)
scaled = autoscale(img, low_p=0.1, high_p=99.9)
```

### Generate Crop Coordinates

```python
from lib.registration_utils import extract_crop_coords

# For 3-channel image
shape = (3, 2048, 2048)
coords = extract_crop_coords(
    image_shape=shape,
    crop_size=1000,
    overlap=100
)

# Returns list of dicts:
# [
#     {'y': 0, 'x': 0, 'h': 1000, 'w': 1000},
#     {'y': 0, 'x': 900, 'h': 1000, 'w': 1000},
#     ...
# ]

# Process each crop
for coord in coords:
    y, x, h, w = coord['y'], coord['x'], coord['h'], coord['w']
    crop = image[:, y:y+h, x:x+w]
    # ... process crop ...
```

### Place Crops Back into Image

```python
from lib.registration_utils import place_crop_hardcutoff

# Create target array
target = np.zeros((3, 2048, 2048), dtype=np.float32)

# Place processed crop
place_crop_hardcutoff(
    target_mem=target,
    crop=processed_crop,
    y=0, x=0, h=1000, w=1000,
    img_height=2048,
    img_width=2048,
    overlap=100
)
```

### Memory-Mapped Arrays

```python
from lib.registration_utils import create_memmaps_for_merge, cleanup_memmaps

# Create large arrays on disk
merged, weights, tmp_dir = create_memmaps_for_merge(
    output_shape=(3, 2048, 2048),
    dtype=np.float32,
    prefix="my_process_"
)

# Use like normal arrays
merged[:] = 0
merged[0, :100, :100] = processed_crop

# Clean up when done
merged.flush()
del merged, weights
cleanup_memmaps(tmp_dir)
```

---

## QC Generation

### Generate Registration QC

```python
from lib.qc import create_registration_qc

# Full-featured QC generation
create_registration_qc(
    reference_path="reference.tif",
    registered_path="registered.tif",
    output_path="qc_output.tif",
    scale_factor=0.25,       # 4x downsample for preview
    save_fullres=True,       # Save full-res compressed TIFF
    save_png=True,           # Save PNG preview
    save_tiff=True           # Save downsampled TIFF
)

# Creates 3 files:
# - qc_output_fullres.tif (full resolution, compressed)
# - qc_output.tif (downsampled, ImageJ-compatible)
# - qc_output.png (downsampled preview)
```

### Custom QC Generation

```python
from lib.qc import create_dapi_overlay
import tifffile

# Load DAPI channels
ref_dapi = tifffile.imread("ref.tif")[0]  # First channel
reg_dapi = tifffile.imread("reg.tif")[0]

# Create overlay
rgb_bgr, rgb_cyx = create_dapi_overlay(
    reference_dapi=ref_dapi,
    registered_dapi=reg_dapi,
    scale_factor=0.5  # 2x downsample
)

# Save
import cv2
cv2.imwrite("overlay.png", rgb_bgr)  # BGR for PNG
tifffile.imwrite("overlay.tif", rgb_cyx)  # CYX for TIFF
```

### Autoscale for Display

```python
from lib.qc import autoscale_for_display

# Min-max scaling
scaled = autoscale_for_display(image, method='minmax')

# Percentile-based (robust to outliers)
scaled = autoscale_for_display(image, method='percentile')
```

---

## Migration Guide

### Replace Duplicated Functions

#### Logger

**Before:**
```python
import logging
logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO, format='...')
    logger.info("Starting")
```

**After:**
```python
from lib.logger import get_logger, configure_logging

def main():
    configure_logging(level=logging.INFO)
    logger = get_logger(__name__)
    logger.info("Starting")
```

#### Channel Names

**Before:**
```python
def get_channel_names(filename):
    base = os.path.basename(filename)
    name_part = base.replace('_corrected', '').split('.')[0]
    parts = name_part.split('_')
    return parts[1:]
```

**After:**
```python
from lib.metadata import get_channel_names
channels = get_channel_names(filename)
```

#### Autoscale

**Before:**
```python
def autoscale(img, low_p=1.0, high_p=99.0):
    lo = np.percentile(img, low_p)
    hi = np.percentile(img, high_p)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)
```

**After:**
```python
from lib.registration_utils import autoscale
scaled = autoscale(img, low_p=1.0, high_p=99.0)
```

#### QC Generation

**Before:**
```python
def create_qc_rgb_composite(ref_path, reg_path, output_path):
    # ... 140 lines of code ...
    ref_img = tifffile.imread(ref_path)
    # ... find DAPI ...
    # ... autoscale ...
    # ... create RGB ...
    # ... save ...
```

**After:**
```python
from lib.qc import create_registration_qc
create_registration_qc(ref_path, reg_path, output_path)
```

---

## Complete Example Script

```python
#!/usr/bin/env python3
"""Example script using lib modules."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Import from library modules
from lib.logger import get_logger, configure_logging
from lib.metadata import get_channel_names, create_ome_xml
from lib.registration_utils import autoscale, extract_crop_coords
from lib.qc import create_registration_qc

import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Example script')
    parser.add_argument('--input', required=True, help='Input image')
    parser.add_argument('--output', required=True, help='Output image')
    parser.add_argument('--verbose', action='store_true', help='Debug logging')
    return parser.parse_args()


def process_image(input_path: Path, output_path: Path) -> None:
    """Process image with library utilities."""
    logger = get_logger(__name__)

    # Extract metadata
    channels = get_channel_names(input_path.name)
    logger.info(f"Detected channels: {channels}")

    # Load and process
    img = tifffile.imread(str(input_path))
    logger.info(f"Loaded image: {img.shape}, {img.dtype}")

    # Autoscale
    scaled = autoscale(img[0] if img.ndim == 3 else img)
    logger.info(f"Autoscaled to uint8")

    # Generate crops
    coords = extract_crop_coords(img.shape, crop_size=1000, overlap=100)
    logger.info(f"Generated {len(coords)} crops")

    # Save with OME metadata
    xml = create_ome_xml(
        channel_names=channels,
        dtype=img.dtype,
        width=img.shape[-1],
        height=img.shape[-2]
    )

    tifffile.imwrite(
        str(output_path),
        img,
        description=xml,
        compression='zlib'
    )
    logger.info(f"Saved: {output_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Configure logging once
    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(level=log_level)

    logger = get_logger(__name__)
    logger.info("Processing started")

    try:
        process_image(Path(args.input), Path(args.output))
        logger.info("✓ Complete")
        return 0
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        logger.debug("Traceback:", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
```

---

## Tips & Best Practices

### 1. Always Configure Logging Once

```python
# ✓ Good - configure at startup
def main():
    configure_logging(level=logging.INFO)
    # ... rest of code ...

# ✗ Bad - configure multiple times
configure_logging(...)  # In global scope
configure_logging(...)  # In function
```

### 2. Use Type Hints

```python
from pathlib import Path
from typing import List

def process(paths: List[Path]) -> int:
    """Process with type safety."""
    ...
```

### 3. Follow NumPy Docstring Style

See any function in `lib/` for examples!

### 4. Import What You Need

```python
# ✓ Good - explicit imports
from lib.metadata import get_channel_names, create_ome_xml

# ✗ Less good - import everything
from lib.metadata import *
```

---

## Next Scripts to Refactor

High priority (lots of duplication):
1. **`register_gpu.py`** - ~300 lines of duplication
2. **`register_cpu.py`** - ~300 lines of duplication
3. **`merge_registered.py`** - ~50 lines of duplication

---

*Quick reference for ATEIA Python library modules*
