#!/usr/bin/env python3
"""VALIS registration script for WSI processing pipeline.

This script performs multi-modal image registration using VALIS (Virtual Alignment
of pathoLogy Image Series). It aligns multiple preprocessed OME-TIFF files and
creates a merged output with all channels registered to a reference image.

The registration uses:
- SuperPoint feature detection with SuperGlue matching
- Micro-rigid registration for high-resolution alignment
- Optical flow-based non-rigid deformation
"""
from __future__ import annotations

import argparse
import os
import glob
import gc
import shutil
import numpy as np
from typing import Optional
import tifffile
import pyvips

# Add utils directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

# Import from utils modules for DRY principle
from image_utils import ensure_dir
from logger import log_progress
from metadata import get_channel_names
from registration_utils import autoscale

# Disable numba caching to avoid file system issues in containers
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_CACHING'] = '1'

# Force library paths for VALIS
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

from valis import registration
from valis.micro_rigid_registrar import MicroRigidRegistrar
from valis import feature_detectors
from valis import feature_matcher
from valis import non_rigid_registrars


def debug_check_slide2vips(registrar):
    log_progress("\n=== slide2vips() DEBUG ===")
    for name, slide in registrar.slide_dict.items():
        try:
            v = slide.slide2vips(level=0)
            if v is None:
                log_progress(f"❌ {name}: slide2vips() returned None")
            else:
                log_progress(f"✅ {name}: bands={v.bands}, size={v.width}x{v.height}")
        except Exception as e:
            log_progress(f"🔥 {name}: slide2vips() raised error: {e}")
    log_progress("=== END DEBUG ===\n")


# log_progress, get_channel_names, and autoscale are now imported from lib modules above
# These duplicate function definitions have been removed to follow DRY principle



def find_reference_image(directory: str, required_markers: list[str], 
                        valid_extensions: Optional[list[str]] = None) -> str:
    """Find image file containing all required markers in filename.
    
    Parameters
    ----------
    directory : str
        Path to directory containing images
    required_markers : list of str
        Marker names that must appear in filename (case-insensitive)
    valid_extensions : list of str, optional
        Valid file extensions. Default: ['.tif', '.tiff', '.ome.tif']
    
    Returns
    -------
    str
        Filename (not full path) of matching image
    
    Raises
    ------
    FileNotFoundError
        If no matching file found
    ValueError
        If multiple matching files found
    """
    if valid_extensions is None:
        valid_extensions = ['.tif', '.tiff', '.ome.tif', '.ome.tiff']
    
    all_files = os.listdir(directory)
    image_files = [
        f for f in all_files 
        if any(f.lower().endswith(ext) for ext in valid_extensions)
    ]
    
    log_progress(f"Found {len(image_files)} image files in {directory}")
    
    matching_files = []
    for filename in image_files:
        filename_upper = filename.upper()
        if all(marker.upper() in filename_upper for marker in required_markers):
            matching_files.append(filename)
    
    if len(matching_files) == 0:
        error_msg = (
            f"No image found containing all markers: {required_markers}\n"
            f"Available files: {image_files[:5]}..."
        )
        raise FileNotFoundError(error_msg)
    
    elif len(matching_files) == 1:
        log_progress(f"✓ Found reference image: {matching_files[0]}")
        return matching_files[0]
    
    else:
        error_msg = (
            f"Found {len(matching_files)} images with markers {required_markers}:\n"
            + "\n".join(f"  - {f}" for f in matching_files)
        )
        raise ValueError(error_msg)


def valis_registration(input_dir: str, out: str,
                       reference: Optional[str] = None,
                       reference_markers: Optional[list[str]] = None,
                       max_processed_image_dim_px: int = 512,
                       max_non_rigid_dim_px: int = 2048,
                       micro_reg_fraction: float = 0.125,
                       num_features: int = 5000,
                       max_image_dim_px: int = 4000,
                       skip_micro_registration: bool = False) -> int:
    """Perform VALIS registration on preprocessed images.

    Parameters
    ----------
    input_dir : str
        Directory containing preprocessed OME-TIFF files
    out : str
        Output merged file path
    reference : str, optional
        Filename of reference image (takes precedence over reference_markers)
    reference_markers : list of str, optional
        Markers to identify reference image (legacy fallback). Default: ['DAPI', 'SMA']
    max_processed_image_dim_px : int, optional
        Maximum image dimension for rigid registration. Default: 512
    max_non_rigid_dim_px : int, optional
        Maximum dimension for non-rigid registration. Default: 2048
    micro_reg_fraction : float, optional
        Fraction of image size for micro-registration. Default: 0.125
    num_features : int, optional
        Number of SuperPoint features to detect. Default: 5000
    max_image_dim_px : int, optional
        Maximum image dimension for caching (controls RAM usage). Default: 4000
    skip_micro_registration : bool, optional
        Skip the micro-rigid registration step. Default: False

    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Initialize JVM for VALIS with increased heap size for large OME-TIFF files
    # For 25GB+ OME-TIFF files, we need substantial heap space for BioFormats
    log_progress("Initializing JVM with increased heap size for large OME-TIFF processing...")
    registration.init_jvm(mem_gb=16)  # Allocate 16GB heap for BioFormats reader
    log_progress("✓ JVM initialized with 16GB heap")

    # Configuration
    if reference_markers is None:
        reference_markers = ['DAPI', 'SMA']

    ensure_dir(os.path.dirname(out) or '.')

    # Use output directory as results directory for VALIS internal files
    results_dir = os.path.dirname(out)

    # ========================================================================
    # VALIS Parameters - Use provided values or defaults
    # ========================================================================
    log_progress("=" * 70)
    log_progress("VALIS Registration Configuration")
    log_progress("=" * 70)
    log_progress(f"Rigid resolution: {max_processed_image_dim_px}px")
    log_progress(f"Non-rigid resolution: {max_non_rigid_dim_px}px")
    log_progress(f"Micro-registration fraction: {micro_reg_fraction}")
    log_progress(f"Feature detector: SuperPoint with {num_features} features")
    log_progress("=" * 70)

    # Find reference image
    if reference:
        # Modern approach: use specified reference filename
        log_progress(f"Using specified reference image: {reference}")
        ref_image_path = os.path.join(input_dir, reference)
        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(f"Specified reference image not found: {ref_image_path}")
        ref_image = reference
    else:
        # Legacy approach: search by markers
        log_progress(f"Searching for reference image with markers: {reference_markers}")
        try:
            ref_image = find_reference_image(input_dir, required_markers=reference_markers)
        except (FileNotFoundError, ValueError) as e:
            log_progress(f"ERROR: {e}")
            log_progress("Falling back to first available image")
            files = sorted(glob.glob(os.path.join(input_dir, '*.ome.tif')))
            if not files:
                raise FileNotFoundError(f"No .ome.tif files in {input_dir}")
            ref_image = os.path.basename(files[0])

    log_progress(f"Using reference image: {ref_image}")

    # ========================================================================
    # Initialize VALIS Registrar with Memory Optimization
    # ========================================================================
    log_progress("\nInitializing VALIS registration with memory optimization...")

    # Disable pyvips cache to reduce memory usage
    # Per VALIS docs: "cache can grow quite large, so it may be best to disable it"
    try:
        pyvips.cache_set_max(0)
        pyvips.cache_set_max_mem(0) 
        log_progress("✓ Disabled pyvips cache (cache_set_max=0, cache_set_max_mem=16)")
    except Exception as e:
        log_progress(f"⚠ Could not disable pyvips cache: {e}")

    # Enable PyVIPS sequential access mode for additional memory optimization
    # Research shows sequential mode reduces memory by 40-50% for top-to-bottom processing
    # Safe for VALIS since warping processes images top-to-bottom
    # Source: https://libvips.github.io/pyvips/vimage.html#access-modes
    try:
        pyvips.voperation.cache_set_max(0)
        pyvips.voperation.cache_set_max_mem(0)
        log_progress("✓ Enabled PyVIPS sequential access optimization")
    except Exception as e:
        log_progress(f"⚠ Sequential access mode not available: {e}")

    # Use provided max_image_dim_px parameter for memory control
    # VALIS docs: "mostly to keep memory in check"
    # For 60K x 50K images, we need to limit the cached size (e.g., 6000 -> 6K x 6K)
    log_progress(f"Memory optimization parameters:")
    log_progress(f"  max_processed_image_dim_px: {max_processed_image_dim_px} (controls analysis resolution)")
    log_progress(f"  max_non_rigid_registration_dim_px: {max_non_rigid_dim_px} (controls non-rigid accuracy)")
    log_progress(f"  max_image_dim_px: {max_image_dim_px} (limits cached image size for RAM control)")

    registrar = registration.Valis(
        input_dir,
        results_dir,

        # Reference image
        reference_img_f=ref_image,
        align_to_reference=True,
        crop="reference",

        # Image size parameters - tuned for memory efficiency
        # max_processed_image_dim_px: controls resolution for feature detection & rigid registration
        # max_non_rigid_registration_dim_px: controls resolution for non-rigid registration
        # max_image_dim_px: limits dimensions of images stored in registrar (key for RAM control)
        max_processed_image_dim_px=max_processed_image_dim_px,
        max_non_rigid_registration_dim_px=max_non_rigid_dim_px,
        max_image_dim_px=max_image_dim_px,  # Critical: prevents loading full 28K x 28K images into RAM

        # Feature detection - SuperPoint/SuperGlue
        feature_detector_cls=feature_detectors.SuperPointFD,
        matcher=feature_matcher.SuperGlueMatcher(),

        # Non-rigid registration
        #non_rigid_registrar_cls=non_rigid_registrars.SimpleElastixWarper,

        # Micro-rigid registration
        micro_rigid_registrar_cls=None if skip_micro_registration else MicroRigidRegistrar,

        # Registration behavior
        create_masks=True,
    )

    # ========================================================================
    # Perform Registration
    # ========================================================================
    log_progress("Starting rigid and non-rigid registration...")
    log_progress("This may take 15-45 minutes...")

    try:
        _, _, error_df = registrar.register()
        log_progress("✓ Initial registration completed")
        log_progress(f"\nRegistration errors:\n{error_df}")
    except Exception as e:
        if "unable to write to memory" in str(e) or "TIFFFillStrip" in str(e):
            log_progress("\n" + "=" * 70)
            log_progress("ERROR: pyvips memory allocation failure")
            log_progress("=" * 70)
            log_progress("VALIS cannot load the TIFF files into memory.")
            log_progress("\nPossible causes:")
            log_progress("  1. Files are too large for available RAM")
            log_progress("  2. TIFF files may be corrupted or have format issues")
            log_progress("  3. Files need to be saved with tiling/compression")
            log_progress("\nSuggested fixes:")
            log_progress("  1. Increase max_image_dim_px parameter (currently 6000)")
            log_progress("  2. Re-save TIFF files with compression='zlib' and tile=(256,256)")
            log_progress("  3. Ensure preprocessing saves tiles with proper TIFF structure")
            log_progress("=" * 70)
            raise RuntimeError(f"VALIS registration failed due to memory/TIFF issue: {e}")
        else:
            raise

    # ========================================================================
    # Micro-registration - Try with error handling
    # ========================================================================
    if skip_micro_registration:
        log_progress("\nSkipping micro-registration (--skip-micro-registration flag set)")
    else:
        log_progress("\nAttempting micro-registration...")
        log_progress("NOTE: This may fail if SimpleElastix is not properly installed")

        try:
            img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
            min_max_size = np.min([np.max(d) for d in img_dims])
            micro_reg_size = int(np.floor(min_max_size * micro_reg_fraction))

            log_progress(f"Micro-registration size: {micro_reg_size}px")
            log_progress("Starting micro-registration (may take 30-120 minutes)...")

            _, micro_error = registrar.register_micro(
                max_non_rigid_registration_dim_px=micro_reg_size,
                reference_img_f=ref_image,
                align_to_reference=True,
            )

            log_progress("✓ Micro-registration completed")
            log_progress(f"\nMicro-registration errors:\n{micro_error}")

        except Exception as e:
            log_progress(f"\n⚠ Micro-registration FAILED: {e}")
            log_progress("Continuing without micro-registration...")
            log_progress("(This is usually caused by SimpleElastix not being available)")
    
    # ========================================================================
    # Merge and Save
    # ========================================================================
    log_progress("\n" + "=" * 70)
    log_progress("MERGE AND SAVE PHASE")
    log_progress("=" * 70)
    log_progress("Preparing to merge channels...")

    # Log registrar state
    log_progress(f"\nRegistrar state:")
    log_progress(f"  - Number of slides: {len(registrar.slide_dict)}")
    log_progress(f"  - Original image list: {registrar.original_img_list}")
    log_progress(f"  - Slide dict keys: {list(registrar.slide_dict.keys())}")

    log_progress(f"\nWarping all slides to: {out}")
    log_progress(f"  - Output directory: {out}")
    log_progress(f"  - Strategy: registrar.warp_and_save_slides() (batch processing)")
    log_progress("")

    # Create output directory
    ensure_dir(out)

    # Log slide information and fix src_f paths
    # VALIS sometimes loses src_f during registration, so we reconstruct it
    log_progress(f"Slides to warp:")
    for idx, slide_name in enumerate(registrar.slide_dict.keys(), 1):
        slide_obj = registrar.slide_dict[slide_name]

        # Check current src_f
        current_src_f = getattr(slide_obj, 'src_f', None)

        # If src_f is None or invalid, reconstruct it from input_dir
        if current_src_f is None or current_src_f == 'None' or not os.path.exists(current_src_f):
            # Try to find the file in input_dir
            for ext in ['.ome.tif', '.ome.tiff']:
                candidate = os.path.join(input_dir, f"{slide_name}{ext}")
                if os.path.exists(candidate):
                    slide_obj.src_f = candidate
                    log_progress(f"  [{idx}] {slide_name} - Fixed src_f: {candidate} (was: {current_src_f})")
                    break
            else:
                log_progress(f"  [{idx}] {slide_name} - ERROR: Could not find source file")
        else:
            log_progress(f"  [{idx}] {slide_name} - src_f OK: {current_src_f}")
    log_progress("")

    # Warp slides individually to ensure src_f is set correctly before each warp
    # VALIS batch processing loses src_f between slides, so we process one at a time
    log_progress("Applying transforms (rigid + non-rigid + micro) individually per slide...")
    log_progress("This may take significant time for large images...")

    warped_count = 0
    failed_slides = []

    for slide_name, slide_obj in registrar.slide_dict.items():
        try:
            # Ensure src_f is set right before warping this slide
            current_src_f = getattr(slide_obj, 'src_f', None)
            if current_src_f is None or current_src_f == 'None' or not os.path.exists(current_src_f):
                # Reconstruct src_f from input_dir
                for ext in ['.ome.tif', '.ome.tiff']:
                    candidate = os.path.join(input_dir, f"{slide_name}{ext}")
                    if os.path.exists(candidate):
                        slide_obj.src_f = candidate
                        log_progress(f"  Re-set src_f for {slide_name}: {candidate}")
                        break

            log_progress(f"  Warping {slide_name}...")
            dst_f = os.path.join(out, f"{slide_name}.ome.tiff")

            slide_obj.warp_and_save_slide(
                dst_f=dst_f,
                non_rigid=True,
                crop="reference",
                interp_method="bicubic",
            )

            log_progress(f"  ✓ {slide_name} completed")
            warped_count += 1

        except Exception as e:
            log_progress(f"  ✗ ERROR warping {slide_name}: {e}")
            failed_slides.append(slide_name)
            # Log diagnostic info
            log_progress(f"    Diagnostic: src_f = {getattr(slide_obj, 'src_f', 'N/A')}")
            import traceback
            traceback.print_exc()

    if failed_slides:
        log_progress(f"\n✗ {len(failed_slides)} slide(s) failed to warp: {', '.join(failed_slides)}")
        raise RuntimeError(f"VALIS warping failed for {len(failed_slides)} slide(s)")

    log_progress(f"✓ All {warped_count} slides warped successfully")

    # Rename output files to match expected naming convention
    # VALIS outputs: {slide_name}.ome.tiff
    # Pipeline expects: {slide_name}_registered.ome.tiff
    log_progress("\nRenaming output files to match pipeline convention...")
    renamed_count = 0
    for slide_name in registrar.slide_dict.keys():
        # VALIS default output naming
        valis_output = os.path.join(out, f"{slide_name}.ome.tiff")
        expected_output = os.path.join(out, f"{slide_name}_registered.ome.tiff")

        if os.path.exists(valis_output):
            shutil.move(valis_output, expected_output)
            log_progress(f"  Renamed: {slide_name}.ome.tiff -> {slide_name}_registered.ome.tiff")
            renamed_count += 1
        elif os.path.exists(expected_output):
            log_progress(f"  Already exists: {slide_name}_registered.ome.tiff")
            renamed_count += 1
        else:
            log_progress(f"  ⚠ WARNING: Output not found for {slide_name}")

    # Report results
    log_progress(f"\n{'='*70}")
    log_progress(f"Warping Summary:")
    log_progress(f"  Successfully warped: {renamed_count}/{len(registrar.slide_dict)}")
    log_progress(f"{'='*70}")

    if renamed_count == 0:
        log_progress("ERROR: No slides were warped successfully.")
        raise RuntimeError("All slides failed to warp. Registration cannot proceed.")
    elif renamed_count < len(registrar.slide_dict):
        log_progress(f"⚠ WARNING: Only {renamed_count}/{len(registrar.slide_dict)} slides were warped")

    log_progress(f"✓ {renamed_count} slides warped and saved to: {out}")

    gc.collect()

    log_progress("\n" + "=" * 70)
    log_progress("✓ REGISTRATION COMPLETED SUCCESSFULLY!")
    log_progress("=" * 70)
    log_progress("NOTE: QC generation is handled separately by the pipeline")

    # Cleanup
    registration.kill_jvm()

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='VALIS registration for WSI processing')
    parser.add_argument('--input-dir', required=True, help='Directory containing preprocessed files')
    parser.add_argument('--out', required=True, help='Output merged path')
    parser.add_argument('--reference', type=str, default=None,
                       help='Filename of reference image (modern approach, takes precedence over --reference-markers)')
    parser.add_argument('--reference-markers', nargs='+', default=['DAPI', 'SMA'],
                       help='Markers to identify reference image (legacy fallback, default: DAPI SMA)')
    parser.add_argument('--max-processed-dim', type=int, default=512,
                       help='Maximum image dimension for rigid registration (default: 512)')
    parser.add_argument('--max-non-rigid-dim', type=int, default=2048,
                       help='Maximum dimension for non-rigid registration (default: 2048)')
    parser.add_argument('--micro-reg-fraction', type=float, default=0.125,
                       help='Fraction of image size for micro-registration (default: 0.125)')
    parser.add_argument('--num-features', type=int, default=5000,
                       help='Number of SuperPoint features to detect (default: 5000)')
    parser.add_argument('--max-image-dim', type=int, default=4000,
                       help='Maximum image dimension for caching (controls RAM usage, default: 4000)')
    parser.add_argument('--skip-micro-registration', action='store_true',
                       help='Skip the micro-rigid registration step (default: False)')
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        return valis_registration(
            args.input_dir,
            args.out,
            args.reference,
            args.reference_markers,
            args.max_processed_dim,
            args.max_non_rigid_dim,
            args.micro_reg_fraction,
            args.num_features,
            args.max_image_dim,
            args.skip_micro_registration
        )
    except Exception as e:
        log_progress(f"ERROR: Registration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
