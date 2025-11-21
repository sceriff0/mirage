#!/usr/bin/env python3
"""Step 3: Warp slides, merge, and generate QC images.

This script loads the final pickle from Step 2, warps all slides,
merges them with channel deduplication, and generates QC RGB overlays.

Based on register.py working implementation.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from datetime import datetime
from pathlib import Path

# Disable Numba caching
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'

import numpy as np
import tifffile
from valis import registration

from _common import ensure_dir

logger = logging.getLogger(__name__)


def log_progress(message: str) -> None:
    """Print timestamped progress messages."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def get_channel_names(filename: str) -> list[str]:
    """Parse channel names from filename.

    Expected format: PatientID_DAPI_Marker1_Marker2_corrected.ome.tif
    """
    base = os.path.basename(filename)
    name_part = base.split('_corrected')[0]
    parts = name_part.split('_')
    channels = parts[1:]  # Exclude Patient ID
    return channels


def autoscale(img, low_p=1, high_p=99):
    """Auto brightness/contrast similar to ImageJ's 'Auto'."""
    lo = np.percentile(img, low_p)
    hi = np.percentile(img, high_p)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def save_qc_dapi_rgb(registrar, qc_dir: str, ref_image: str):
    """Create QC RGB composites: ref DAPI (GREEN) + reg DAPI (RED)."""
    log_progress(f"\nCreating QC directory: {qc_dir}")
    os.makedirs(qc_dir, exist_ok=True)

    # Find reference slide
    ref_image_no_ext = ref_image.replace('.ome.tif', '').replace('.ome.tiff', '')
    log_progress(f"  Looking for reference: {ref_image_no_ext}")

    ref_slide_key = None
    for key in registrar.slide_dict.keys():
        if key == ref_image_no_ext:
            ref_slide_key = key
            break

    if ref_slide_key is None:
        log_progress(f"  ⚠ Reference not found, skipping QC")
        return

    # Extract reference DAPI
    log_progress(f"  Extracting reference DAPI...")
    ref_slide = registrar.slide_dict[ref_slide_key]
    ref_basename = os.path.basename(ref_slide_key)
    ref_channels = get_channel_names(ref_basename)
    ref_dapi_idx = next((i for i, ch in enumerate(ref_channels) if "DAPI" in ch.upper()), 0)

    ref_vips = ref_slide.slide2vips(level=0)
    ref_dapi = ref_vips.extract_band(ref_dapi_idx).numpy()
    log_progress(f"  Reference DAPI shape: {ref_dapi.shape}")

    # Process each slide
    log_progress(f"\nProcessing {len(registrar.slide_dict) - 1} slides for QC...")
    for idx, (slide_name, slide_obj) in enumerate(registrar.slide_dict.items(), 1):
        if slide_name == ref_slide_key:
            log_progress(f"\n[{idx}/{len(registrar.slide_dict)}] Skipping reference: {slide_name}")
            continue

        log_progress(f"\n[{idx}/{len(registrar.slide_dict)}] Processing: {slide_name}")

        slide_basename = os.path.basename(slide_name)
        slide_channels = get_channel_names(slide_basename)
        slide_dapi_idx = next((i for i, ch in enumerate(slide_channels) if "DAPI" in ch.upper()), 0)

        # Warp slide
        warped_vips = slide_obj.warp_slide(level=0, non_rigid=True, crop=registrar.crop)
        reg_dapi = warped_vips.extract_band(slide_dapi_idx).numpy()

        # Autoscale
        ref_dapi_scaled = autoscale(ref_dapi)
        reg_dapi_scaled = autoscale(reg_dapi)

        del reg_dapi

        # Create RGB composite (R=reg, G=ref, B=zeros)
        h, w = reg_dapi_scaled.shape
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = reg_dapi_scaled  # Red
        rgb[:, :, 1] = ref_dapi_scaled  # Green
        rgb[:, :, 2] = 0                # Blue

        del reg_dapi_scaled

        # Save
        out_filename = f"{slide_basename}_QC_RGB.tif"
        out_path = os.path.join(qc_dir, out_filename)
        tifffile.imwrite(out_path, rgb, photometric='rgb')
        log_progress(f"  ✓ Saved: {out_filename}")

        del rgb
        gc.collect()

    log_progress("\n✓ All QC RGB composites saved")


def main():
    parser = argparse.ArgumentParser(description='Step 3: Warp, merge, and generate QC')
    parser.add_argument('--input-pickle', required=True, help='Registrar pickle from Step 2')
    parser.add_argument('--output-merged', required=True, help='Path for merged OME-TIFF output')
    parser.add_argument('--preprocessed-dir', required=True, help='Directory with preprocessed files')
    parser.add_argument('--qc-dir', default=None, help='QC output directory (optional)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Change to preprocessed directory so registrar can find files
    os.chdir(args.preprocessed_dir)
    log_progress(f"Changed to directory: {args.preprocessed_dir}")

    # Initialize JVM
    log_progress("\nInitializing JVM...")
    registration.init_jvm()

    log_progress("=" * 80)
    log_progress("STEP 3: WARP, MERGE, AND GENERATE QC")
    log_progress("=" * 80)
    log_progress(f"Input pickle: {args.input_pickle}")
    log_progress(f"Output merged: {args.output_merged}")
    log_progress(f"QC directory: {args.qc_dir}")
    log_progress("")

    start_time = time.time()

    # Load registrar using official VALIS method
    log_progress("Loading registrar from Step 2...")
    registrar = registration.load_registrar(args.input_pickle)

    log_progress(f"✓ Loaded registrar ({len(registrar.slide_dict)} slides)")
    log_progress(f"  Reference: {registrar.reference_img_f}")
    log_progress(f"  Slides: {list(registrar.slide_dict.keys())}")

    # Build channel_name_dict (same logic as register.py)
    log_progress("")
    log_progress("=" * 70)
    log_progress("BUILDING CHANNEL NAME DICTIONARY")
    log_progress("=" * 70)
    channel_name_dict = {}

    for f in registrar.original_img_list:
        basename = os.path.basename(f)
        slide_name = basename.replace('.ome.tif', '').replace('.ome.tiff', '')

        log_progress(f"\n  {slide_name}")

        if slide_name not in registrar.slide_dict:
            log_progress(f"    ⚠ Not found in slide_dict, skipping")
            continue

        expected_names = get_channel_names(basename)
        slide_obj = registrar.slide_dict[slide_name]

        # Get actual channel count
        vips_img = slide_obj.slide2vips(level=0)
        actual_channels = vips_img.bands
        log_progress(f"    Expected: {expected_names} ({len(expected_names)} names)")
        log_progress(f"    Actual: {actual_channels} channels")
        del vips_img

        # Use only as many names as there are channels
        channel_names_to_use = expected_names[:actual_channels]

        # Pad if needed
        if actual_channels > len(expected_names):
            for i in range(len(expected_names), actual_channels):
                channel_names_to_use.append(f"Channel_{i}")

        channel_name_dict[f] = channel_names_to_use
        log_progress(f"    Final: {channel_names_to_use}")

    log_progress(f"\n✓ Channel dictionary built ({len(channel_name_dict)} entries)")

    # Merge and warp slides
    log_progress("")
    log_progress("=" * 70)
    log_progress("WARPING AND MERGING SLIDES")
    log_progress("=" * 70)
    log_progress(f"Output: {args.output_merged}")
    log_progress(f"Drop duplicates: True")
    log_progress("")

    ensure_dir(os.path.dirname(args.output_merged))

    merge_start = time.time()
    merged_img, channel_names, _ = registrar.warp_and_merge_slides(
        args.output_merged,
        channel_name_dict=channel_name_dict,
        drop_duplicates=True,
    )
    merge_elapsed = time.time() - merge_start

    log_progress("")
    log_progress("✓ Merge completed!")
    log_progress(f"  Time elapsed: {merge_elapsed:.2f}s ({merge_elapsed/60:.1f} min)")
    log_progress(f"  Merged dimensions: {merged_img.width} x {merged_img.height}")
    log_progress(f"  Channels: {merged_img.bands}")
    log_progress(f"  Channel names ({len(channel_names)}): {channel_names}")

    file_size_mb = Path(args.output_merged).stat().st_size / (1024 * 1024)
    log_progress(f"  File size: {file_size_mb:.2f} MB")

    del merged_img
    del channel_name_dict
    gc.collect()

    # Generate QC images
    if args.qc_dir:
        log_progress("")
        log_progress("=" * 70)
        log_progress("GENERATING QC IMAGES")
        log_progress("=" * 70)

        ensure_dir(args.qc_dir)

        qc_start = time.time()
        save_qc_dapi_rgb(registrar, args.qc_dir, registrar.reference_img_f)
        qc_elapsed = time.time() - qc_start

        log_progress(f"\n✓ QC generation completed")
        log_progress(f"  Time elapsed: {qc_elapsed:.2f}s ({qc_elapsed/60:.1f} min)")

    total_elapsed = time.time() - start_time

    log_progress("")
    log_progress("=" * 80)
    log_progress("✓ STEP 3 COMPLETE")
    log_progress("=" * 80)
    log_progress(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)")
    log_progress(f"Merged: {args.output_merged}")
    if args.qc_dir:
        log_progress(f"QC: {args.qc_dir}")
    log_progress("")

    # Cleanup
    registration.kill_jvm()
    return 0


if __name__ == '__main__':
    exit(main())
