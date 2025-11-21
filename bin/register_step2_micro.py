#!/usr/bin/env python3
"""Step 2: Compute micro-registration and update pickle.

This script loads the pickle from Step 1 using VALIS official method,
computes micro-registration, and saves the updated registrar pickle.

Based on register.py working implementation and VALIS documentation.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from pathlib import Path

# Disable Numba caching
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'

import numpy as np
from valis import registration

logger = logging.getLogger(__name__)


def log_progress(message: str) -> None:
    """Print timestamped progress messages."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Step 2: Compute micro-registration')
    parser.add_argument('--input-pickle', required=True, help='Registrar pickle from Step 1')
    parser.add_argument('--output-pickle', required=True, help='Path to save updated pickle')
    parser.add_argument('--preprocessed-dir', required=True, help='Directory with preprocessed files')
    parser.add_argument('--micro-reg-fraction', type=float, default=0.25, help='Micro-reg fraction')

    args = parser.parse_args()

    # Change to preprocessed directory so registrar can find files
    os.chdir(args.preprocessed_dir)
    log_progress(f"Changed to directory: {args.preprocessed_dir}")

    # Initialize JVM
    log_progress("\nInitializing JVM...")
    registration.init_jvm()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    log_progress("=" * 80)
    log_progress("STEP 2: COMPUTE MICRO-REGISTRATION")
    log_progress("=" * 80)
    log_progress(f"Input pickle: {args.input_pickle}")
    log_progress(f"Output pickle: {args.output_pickle}")
    log_progress(f"Micro-reg fraction: {args.micro_reg_fraction}")
    log_progress("")

    start_time = time.time()

    # Load registrar from Step 1 using official VALIS method
    log_progress("Loading registrar from Step 1...")
    log_progress("  Using registration.load_registrar() (official VALIS method)")
    registrar = registration.load_registrar(args.input_pickle)

    log_progress(f"✓ Loaded registrar ({len(registrar.slide_dict)} slides)")
    log_progress(f"  Reference: {registrar.reference_img_f}")
    log_progress(f"  Slides: {list(registrar.slide_dict.keys())}")

    # Compute micro-registration
    log_progress("")
    log_progress("=" * 70)
    log_progress("PERFORMING MICRO-REGISTRATION")
    log_progress("=" * 70)
    log_progress("NOTE: This may fail if SimpleElastix is not properly installed")
    log_progress("")

    try:
        # Calculate micro_reg_size (same logic as register.py)
        img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
        min_max_size = np.min([np.max(d) for d in img_dims])
        micro_reg_size = int(np.floor(min_max_size * args.micro_reg_fraction))

        log_progress("Micro-registration parameters:")
        log_progress(f"  - Minimum max dimension: {min_max_size}px")
        log_progress(f"  - Micro-reg fraction: {args.micro_reg_fraction}")
        log_progress(f"  - Micro-reg size: {micro_reg_size}px")
        log_progress(f"  - Reference: {registrar.reference_img_f}")
        log_progress("")
        log_progress("Starting micro-registration (may take 30-120 minutes)...")
        log_progress("")

        micro_start = time.time()
        _, micro_error = registrar.register_micro(
            max_non_rigid_registration_dim_px=micro_reg_size,
            reference_img_f=registrar.reference_img_f,
            align_to_reference=True,
        )
        micro_elapsed = time.time() - micro_start

        log_progress("")
        log_progress("✓ Micro-registration completed")
        log_progress(f"  Time elapsed: {micro_elapsed:.2f}s ({micro_elapsed/60:.1f} min)")
        log_progress("")
        log_progress("Micro-registration errors:")
        log_progress(f"{micro_error}")
        log_progress("")

    except Exception as e:
        log_progress("")
        log_progress("=" * 70)
        log_progress("⚠ MICRO-REGISTRATION FAILED")
        log_progress("=" * 70)
        log_progress(f"Error: {e}")
        log_progress("")
        log_progress("Continuing without micro-registration...")
        log_progress("This is usually caused by:")
        log_progress("  - SimpleElastix not being available")
        log_progress("  - Memory constraints")
        log_progress("  - Image size issues")
        log_progress("")

    # Save updated registrar pickle
    log_progress("=" * 70)
    log_progress("SAVING UPDATED REGISTRAR PICKLE")
    log_progress("=" * 70)

    with open(args.output_pickle, 'wb') as f:
        pickle.dump(registrar, f, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_size_mb = Path(args.output_pickle).stat().st_size / (1024 * 1024)
    log_progress(f"✓ Updated registrar pickle saved: {args.output_pickle}")
    log_progress(f"  Size: {pickle_size_mb:.2f} MB")

    # Verify using official method
    test_reg = registration.load_registrar(args.output_pickle)
    log_progress(f"  ✓ Verified with load_registrar() ({len(test_reg.slide_dict)} slides)")

    total_elapsed = time.time() - start_time

    log_progress("")
    log_progress("=" * 80)
    log_progress("✓ STEP 2 COMPLETE")
    log_progress("=" * 80)
    log_progress(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)")
    log_progress(f"Pickle: {args.output_pickle}")
    log_progress("")

    # Cleanup
    registration.kill_jvm()
    return 0


if __name__ == '__main__':
    exit(main())
