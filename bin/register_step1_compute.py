#!/usr/bin/env python3
"""Step 1: Compute base VALIS registration (rigid + non-rigid) and save pickle.

This script performs the initial registration without micro-registration,
saves the registrar as a pickle file for the next step.

Based on register.py working implementation.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

# Disable Numba caching
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_DISABLE_CACHING'] = '1'

from valis import registration
from valis.micro_rigid_registrar import MicroRigidRegistrar
from valis import feature_detectors
from valis import feature_matcher

from _common import ensure_dir

logger = logging.getLogger(__name__)


def log_progress(message: str) -> None:
    """Print timestamped progress messages."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def find_reference_image(directory: str, required_markers: list[str]) -> str:
    """Find image file containing all required markers in filename."""
    all_files = os.listdir(directory)
    image_files = [f for f in all_files if f.lower().endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff'))]

    log_progress(f"Found {len(image_files)} image files in {directory}")

    matching_files = []
    for filename in image_files:
        filename_upper = filename.upper()
        if all(marker.upper() in filename_upper for marker in required_markers):
            matching_files.append(filename)

    if len(matching_files) == 0:
        raise FileNotFoundError(f"No image found containing all markers: {required_markers}")
    elif len(matching_files) == 1:
        log_progress(f"✓ Found reference image: {matching_files[0]}")
        return matching_files[0]
    else:
        raise ValueError(f"Found {len(matching_files)} images with markers {required_markers}")


def main():
    parser = argparse.ArgumentParser(description='Step 1: Compute base VALIS registration')
    parser.add_argument('--input-dir', required=True, help='Directory with preprocessed OME-TIFF files')
    parser.add_argument('--output-pickle', required=True, help='Path to save registrar pickle')
    parser.add_argument('--reference-markers', nargs='+', default=['DAPI', 'SMA'], help='Reference markers')
    parser.add_argument('--max-processed-dim', type=int, default=1800, help='Max dimension for rigid registration')
    parser.add_argument('--max-non-rigid-dim', type=int, default=3500, help='Max dimension for non-rigid registration')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    log_progress("=" * 80)
    log_progress("STEP 1: COMPUTE BASE REGISTRATION (RIGID + NON-RIGID)")
    log_progress("=" * 80)
    log_progress(f"Input directory: {args.input_dir}")
    log_progress(f"Output pickle: {args.output_pickle}")
    log_progress(f"Reference markers: {args.reference_markers}")
    log_progress("")

    start_time = time.time()

    # Find reference image
    log_progress(f"Searching for reference image with markers: {args.reference_markers}")
    try:
        ref_image = find_reference_image(args.input_dir, args.reference_markers)
    except (FileNotFoundError, ValueError) as e:
        log_progress(f"WARNING: {e}")
        log_progress("Falling back to first available image")
        files = sorted(glob.glob(os.path.join(args.input_dir, '*.ome.tif')))
        if not files:
            raise FileNotFoundError(f"No .ome.tif files in {args.input_dir}")
        ref_image = os.path.basename(files[0])

    log_progress(f"Using reference image: {ref_image}")

    # Initialize JVM
    log_progress("\nInitializing JVM...")
    registration.init_jvm()

    # Initialize registrar
    log_progress("\n" + "=" * 70)
    log_progress("INITIALIZING VALIS REGISTRAR")
    log_progress("=" * 70)
    log_progress("Configuration:")
    log_progress(f"  - Reference: {ref_image}")
    log_progress(f"  - Max processed dim: {args.max_processed_dim}px")
    log_progress(f"  - Max non-rigid dim: {args.max_non_rigid_dim}px")
    log_progress(f"  - Feature detector: SuperPoint + SuperGlue")
    log_progress(f"  - Micro-rigid: MicroRigidRegistrar (for later)")
    log_progress("")

    registrar = registration.Valis(
        args.input_dir,
        ".",
        reference_img_f=ref_image,
        align_to_reference=True,
        crop="reference",
        max_processed_image_dim_px=args.max_processed_dim,
        max_non_rigid_registration_dim_px=args.max_non_rigid_dim,
        feature_detector_cls=feature_detectors.SuperPointFD,
        matcher=feature_matcher.SuperGlueMatcher(),
        micro_rigid_registrar_cls=MicroRigidRegistrar,
        create_masks=True,
    )

    log_progress(f"✓ Registrar initialized ({len(registrar.slide_dict)} slides)")
    log_progress(f"  Slides: {list(registrar.slide_dict.keys())}")

    # Perform base registration (rigid + non-rigid)
    log_progress("\n" + "=" * 70)
    log_progress("PERFORMING BASE REGISTRATION")
    log_progress("=" * 70)
    log_progress("Starting rigid and non-rigid registration...")
    log_progress("This may take 15-45 minutes...")
    log_progress("")

    reg_start = time.time()
    _, _, error_df = registrar.register()
    reg_elapsed = time.time() - reg_start

    log_progress("")
    log_progress("✓ Base registration completed")
    log_progress(f"  Time elapsed: {reg_elapsed:.2f}s ({reg_elapsed/60:.1f} min)")
    log_progress("")
    log_progress("Registration errors:")
    log_progress(f"{error_df}")

    # Save registrar pickle
    log_progress("\n" + "=" * 70)
    log_progress("SAVING REGISTRAR PICKLE")
    log_progress("=" * 70)

    ensure_dir(str(Path(args.output_pickle).parent))

    with open(args.output_pickle, 'wb') as f:
        pickle.dump(registrar, f, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_size_mb = Path(args.output_pickle).stat().st_size / (1024 * 1024)
    log_progress(f"✓ Registrar pickle saved: {args.output_pickle}")
    log_progress(f"  Size: {pickle_size_mb:.2f} MB")

    # Verify
    with open(args.output_pickle, 'rb') as f:
        test_reg = pickle.load(f)
    log_progress(f"  ✓ Verified ({len(test_reg.slide_dict)} slides)")

    total_elapsed = time.time() - start_time

    log_progress("")
    log_progress("=" * 80)
    log_progress("✓ STEP 1 COMPLETE")
    log_progress("=" * 80)
    log_progress(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)")
    log_progress(f"Pickle: {args.output_pickle}")
    log_progress("")

    return 0


if __name__ == '__main__':
    exit(main())
