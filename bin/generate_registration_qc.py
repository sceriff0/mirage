#!/usr/bin/env python3
"""Generate QC RGB overlay images for registration quality assessment.

Creates RGB composite images comparing registered slides to reference:
- RED = Registered slide DAPI
- GREEN = Reference slide DAPI
- BLUE = zeros

Based on register.py save_qc_dapi_rgb() function.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path

# Disable Numba caching
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'

import numpy as np
import tifffile
from valis import registration

from _common import ensure_dir

logger = logging.getLogger(__name__)

__all__ = [
    "autoscale",
    "get_channel_names",
    "generate_qc_images",
]


def autoscale(img: np.ndarray, low_p: float = 1.0, high_p: float = 99.0) -> np.ndarray:
    """
    Auto brightness/contrast similar to ImageJ's 'Auto'.

    Parameters
    ----------
    img : ndarray
        Input image
    low_p : float
        Lower percentile for clipping
    high_p : float
        Upper percentile for clipping

    Returns
    -------
    ndarray, dtype uint8
        Autoscaled image in [0, 255] range
    """
    lo = np.percentile(img, low_p)
    hi = np.percentile(img, high_p)
    img_scaled = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (img_scaled * 255).astype(np.uint8)


def get_channel_names(filename: str) -> list[str]:
    """
    Parse channel names from filename.

    Expected format: PatientID_DAPI_Marker1_Marker2_corrected.ome.tif

    Parameters
    ----------
    filename : str
        Base filename (not full path)

    Returns
    -------
    list of str
        Channel names extracted from filename (excludes Patient ID)
    """
    base = os.path.basename(filename)
    name_part = base.split('_corrected')[0]  # Remove suffix
    parts = name_part.split('_')
    channels = parts[1:]  # Exclude Patient ID
    return channels


def generate_qc_images(
    registrar_pickle: str,
    registered_dir: str,
    output_dir: str,
) -> int:
    """
    Generate QC RGB overlay images for all registered slides.

    Parameters
    ----------
    registrar_pickle : str
        Path to pickled VALIS registrar
    registered_dir : str
        Directory containing registered OME-TIFF files
    output_dir : str
        Output directory for QC images

    Returns
    -------
    int
        Number of QC images generated

    Notes
    -----
    Creates RGB composites where:
    - RED = Registered slide DAPI (warped)
    - GREEN = Reference slide DAPI (static)
    - BLUE = zeros

    Perfect overlay appears yellow; misalignment shows as red/green fringing.
    """
    logger.info("=" * 80)
    logger.info("GENERATING REGISTRATION QC IMAGES")
    logger.info("=" * 80)
    logger.info(f"Registrar pickle: {registrar_pickle}")
    logger.info(f"Registered slides: {registered_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Create output directory
    ensure_dir(output_dir)

    # Load registrar using official VALIS method
    logger.info("Loading registrar...")
    registrar = registration.load_registrar(registrar_pickle)
    logger.info(f"  ✓ Loaded ({len(registrar.slide_dict)} slides)")
    logger.info(f"  Reference image: {registrar.reference_img_f}")

    # Find reference slide in slide_dict
    ref_image = registrar.reference_img_f
    ref_image_no_ext = ref_image.replace('.ome.tif', '').replace('.ome.tiff', '')

    logger.info("")
    logger.info(f"Searching for reference slide: {ref_image_no_ext}")

    ref_slide_key = None
    for key in registrar.slide_dict.keys():
        if key == ref_image_no_ext:
            ref_slide_key = key
            logger.info(f"  ✓ Found: {ref_slide_key}")
            break

    if ref_slide_key is None:
        raise KeyError(
            f"Reference '{ref_image_no_ext}' not found in slide_dict. "
            f"Available: {list(registrar.slide_dict.keys())}"
        )

    # Extract reference DAPI channel
    logger.info("")
    logger.info("Extracting reference DAPI channel...")
    ref_slide = registrar.slide_dict[ref_slide_key]
    ref_basename = os.path.basename(ref_slide_key)
    ref_channels = get_channel_names(ref_basename)

    logger.info(f"  Reference channels: {ref_channels}")

    # Find DAPI channel index
    ref_dapi_idx = next((i for i, ch in enumerate(ref_channels) if "DAPI" in ch.upper()), 0)
    logger.info(f"  DAPI channel index: {ref_dapi_idx}")

    # Load reference DAPI using slide2vips (memory efficient)
    ref_vips = ref_slide.slide2vips(level=0)
    logger.info(f"  Reference dimensions: {ref_vips.width} x {ref_vips.height}, bands: {ref_vips.bands}")

    # Extract only DAPI channel to save memory
    ref_dapi = ref_vips.extract_band(ref_dapi_idx).numpy()
    logger.info(f"  Reference DAPI shape: {ref_dapi.shape}, dtype: {ref_dapi.dtype}")

    # Process each slide (except reference)
    logger.info("")
    logger.info(f"Processing {len(registrar.slide_dict) - 1} slides for QC...")
    logger.info("")

    qc_count = 0
    for idx, (slide_name, slide_obj) in enumerate(registrar.slide_dict.items(), 1):
        # Skip reference slide
        if slide_name == ref_slide_key:
            logger.info(f"[{idx}/{len(registrar.slide_dict)}] Skipping reference: {slide_name}")
            continue

        logger.info(f"[{idx}/{len(registrar.slide_dict)}] Processing: {slide_name}")

        # Get slide channel information
        slide_basename = os.path.basename(slide_name)
        slide_channels = get_channel_names(slide_basename)
        logger.info(f"  Channels: {slide_channels}")

        # Find DAPI channel
        slide_dapi_idx = next((i for i, ch in enumerate(slide_channels) if "DAPI" in ch.upper()), 0)
        logger.info(f"  DAPI channel index: {slide_dapi_idx}")

        # Warp slide using computed transforms
        logger.info(f"  Warping slide (non_rigid=True, crop={registrar.crop})...")
        warped_vips = slide_obj.warp_slide(level=0, non_rigid=True, crop=registrar.crop)
        logger.info(f"  Warped dimensions: {warped_vips.width} x {warped_vips.height}, bands: {warped_vips.bands}")

        # Extract DAPI channel only (memory efficient)
        logger.info(f"  Extracting DAPI channel...")
        reg_dapi = warped_vips.extract_band(slide_dapi_idx).numpy()
        logger.info(f"  Registered DAPI shape: {reg_dapi.shape}, dtype: {reg_dapi.dtype}")

        # Autoscale both images
        logger.info(f"  Applying autoscale...")
        ref_dapi_scaled = autoscale(ref_dapi)
        reg_dapi_scaled = autoscale(reg_dapi)

        # Free memory immediately
        del reg_dapi

        # Create RGB composite (R=registered, G=reference, B=zeros)
        logger.info(f"  Creating RGB composite...")
        h, w = reg_dapi_scaled.shape
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = reg_dapi_scaled  # Red = registered
        rgb[:, :, 1] = ref_dapi_scaled  # Green = reference
        rgb[:, :, 2] = 0                # Blue = zeros

        # Free memory
        del reg_dapi_scaled

        # Save QC image
        out_filename = f"{slide_basename}_QC_RGB.tif"
        out_path = os.path.join(output_dir, out_filename)
        logger.info(f"  Saving: {out_path}")

        tifffile.imwrite(out_path, rgb, photometric='rgb')

        file_size_mb = Path(out_path).stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Saved ({file_size_mb:.2f} MB)")

        del rgb
        gc.collect()

        qc_count += 1
        logger.info("")

    logger.info("=" * 80)
    logger.info(f"✓ QC GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Generated {qc_count} QC images")
    logger.info(f"Output directory: {output_dir}")

    return qc_count


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate QC RGB overlay images for registration quality assessment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--registrar-pickle',
        type=str,
        required=True,
        help='Path to pickled VALIS registrar'
    )

    parser.add_argument(
        '--registered-dir',
        type=str,
        required=True,
        help='Directory containing registered OME-TIFF files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for QC images'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    # Generate QC images
    qc_count = generate_qc_images(
        registrar_pickle=args.registrar_pickle,
        registered_dir=args.registered_dir,
        output_dir=args.output_dir,
    )

    return 0 if qc_count > 0 else 1


if __name__ == '__main__':
    exit(main())
