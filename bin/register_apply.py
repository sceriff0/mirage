#!/usr/bin/env python3
"""Apply VALIS registration transforms to individual slides.

This script loads a pickled VALIS registrar and applies the computed transforms
to warp a single slide. Designed to be run in parallel for each slide.

The script handles both regular slides and the reference slide (which may need
special handling for DAPI extraction).
"""

from __future__ import annotations

import argparse
import gc
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from valis import registration, slide_tools

from _common import ensure_dir

logger = logging.getLogger(__name__)

__all__ = [
    "load_registrar_pickle",
    "apply_registration_to_slide",
    "save_registered_slide",
]


def load_registrar_pickle(pickle_path: str) -> registration.Valis:
    """
    Load pickled VALIS registrar.

    Parameters
    ----------
    pickle_path : str
        Path to registrar pickle file.

    Returns
    -------
    registrar : Valis
        Loaded VALIS registrar with all computed transforms.
    """
    logger.info(f"Loading registrar pickle: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        registrar = pickle.load(f)

    logger.info(f"✓ Registrar loaded ({len(registrar.slide_dict)} slides)")
    logger.info(f"  Reference image: {registrar.reference_img_f}")

    return registrar


def apply_registration_to_slide(
    registrar: registration.Valis,
    slide_name: str,
    is_reference: bool = False,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Apply registration transforms to warp a single slide.

    Parameters
    ----------
    registrar : Valis
        Loaded VALIS registrar with computed transforms.
    slide_name : str
        Name of slide to warp (basename without path/extension).
    is_reference : bool, optional
        Whether this is the reference slide. Default is False.

    Returns
    -------
    warped_image : ndarray, shape (C, Y, X)
        Warped multichannel image.
    channel_names : list of str
        Channel names for this slide.
    metadata : dict
        Slide metadata including physical pixel size.

    Notes
    -----
    - Uses level 0 (full resolution) for warping
    - Converts pyvips.Image to numpy array
    - Handles reference slide specially (no transformation needed)
    """
    logger.info("=" * 80)
    logger.info(f"APPLYING REGISTRATION: {slide_name}")
    logger.info("=" * 80)
    logger.info(f"Is reference slide: {is_reference}")

    start_time = time.time()

    # Get slide object from registrar
    if slide_name not in registrar.slide_dict:
        raise KeyError(f"Slide '{slide_name}' not found in registrar")

    slide_obj = registrar.slide_dict[slide_name]

    # Get channel names
    channel_names = slide_obj.reader.metadata.channel_names
    logger.info(f"Channels: {channel_names} ({len(channel_names)} channels)")

    # Warp slide using computed transforms
    logger.info("Warping slide with computed transforms...")
    if is_reference:
        logger.info("  Using reference slide (identity transform)")
    else:
        logger.info("  Applying rigid + non-rigid + micro transforms")

    # Warp at full resolution (level 0)
    warped_vips = slide_obj.warp_slide(
        level=0,
        non_rigid=True,
        crop=True,
    )

    logger.info(f"  Warped image size: {warped_vips.width} x {warped_vips.height}")
    logger.info(f"  Warped image bands: {warped_vips.bands}")

    # Convert pyvips.Image to numpy array (C, Y, X)
    logger.info("Converting to numpy array...")
    warped_image = warped_vips.numpy()

    # Ensure correct shape (C, Y, X)
    if warped_image.ndim == 2:
        # Single channel - add channel dimension
        warped_image = warped_image[np.newaxis, :, :]
    elif warped_image.ndim == 3 and warped_image.shape[2] == warped_vips.bands:
        # (Y, X, C) format - transpose to (C, Y, X)
        warped_image = np.transpose(warped_image, (2, 0, 1))

    logger.info(f"  Final numpy shape: {warped_image.shape}")
    logger.info(f"  Data type: {warped_image.dtype}")
    logger.info(f"  Value range: [{warped_image.min()}, {warped_image.max()}]")

    # Get metadata
    metadata = {
        'physical_size_x': slide_obj.reader.metadata.pixel_physical_size_xyu[0],
        'physical_size_y': slide_obj.reader.metadata.pixel_physical_size_xyu[1],
        'physical_size_unit': slide_obj.reader.metadata.pixel_physical_size_xyu[2],
    }

    logger.info(f"  Physical size: {metadata['physical_size_x']:.4f} {metadata['physical_size_unit']}")

    # Clean up
    del warped_vips
    gc.collect()

    elapsed = time.time() - start_time
    logger.info(f"Registration applied in {elapsed:.2f}s")
    logger.info("")

    return warped_image, channel_names, metadata


def save_registered_slide(
    warped_image: np.ndarray,
    channel_names: list[str],
    metadata: dict,
    output_path: str,
) -> None:
    """
    Save warped slide as OME-TIFF.

    Parameters
    ----------
    warped_image : ndarray, shape (C, Y, X)
        Warped multichannel image.
    channel_names : list of str
        Channel names.
    metadata : dict
        Slide metadata with physical_size_x, physical_size_y, physical_size_unit.
    output_path : str
        Path to save registered OME-TIFF file.
    """
    logger.info(f"Saving registered slide: {output_path}")

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    ensure_dir(str(output_dir))

    # Prepare OME-TIFF metadata
    ome_metadata = {
        'axes': 'CYX',
        'Channel': {'Name': channel_names},
        'PhysicalSizeX': metadata['physical_size_x'],
        'PhysicalSizeXUnit': metadata['physical_size_unit'],
        'PhysicalSizeY': metadata['physical_size_y'],
        'PhysicalSizeYUnit': metadata['physical_size_unit'],
    }

    logger.info(f"  Axes: {ome_metadata['axes']}")
    logger.info(f"  Channels: {ome_metadata['Channel']['Name']}")
    logger.info(f"  Physical size: {ome_metadata['PhysicalSizeX']} {ome_metadata['PhysicalSizeXUnit']}")

    # Save as OME-TIFF with compression
    tifffile.imwrite(
        output_path,
        warped_image,
        metadata=ome_metadata,
        photometric='minisblack',
        compression='zlib',
        ome=True,
    )

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"✓ Registered slide saved ({file_size_mb:.2f} MB)")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Apply VALIS registration to individual slide',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--registrar-pickle',
        type=str,
        required=True,
        help='Path to pickled VALIS registrar'
    )

    parser.add_argument(
        '--slide-name',
        type=str,
        required=True,
        help='Name of slide to register (basename without path/extension)'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to save registered OME-TIFF file'
    )

    # Optional arguments
    parser.add_argument(
        '--is-reference',
        action='store_true',
        help='This is the reference slide'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    # Load registrar
    registrar = load_registrar_pickle(args.registrar_pickle)

    # Apply registration to slide
    warped_image, channel_names, metadata = apply_registration_to_slide(
        registrar=registrar,
        slide_name=args.slide_name,
        is_reference=args.is_reference,
    )

    # Save registered slide
    save_registered_slide(
        warped_image=warped_image,
        channel_names=channel_names,
        metadata=metadata,
        output_path=args.output_file,
    )

    logger.info("=" * 80)
    logger.info("✓ SLIDE REGISTRATION COMPLETE")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())
