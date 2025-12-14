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
import numpy as np
from datetime import datetime
from typing import Optional
import tifffile
import pyvips

from _common import ensure_dir

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


def log_progress(message: str) -> None:
    """Print timestamped progress messages."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def get_channel_names(filename: str) -> list[str]:
    """Parse channel names from filename.

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
    # Remove all suffixes that might be present
    name_part = base.replace('_corrected', '').replace('_padded', '').replace('_preprocessed', '').replace('_registered', '').split('.')[0]
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
    """Create QC RGB composites with registered (red) and reference (green) DAPI channels."""
    from pathlib import Path
    from skimage.transform import rescale
    import cv2

    qc_path = Path(qc_dir)
    qc_path.mkdir(parents=True, exist_ok=True)
    log_progress(f"Creating QC outputs in: {qc_path}")

    # Find reference slide key (slide_dict keys are slide names without extension)
    ref_image_no_ext = ref_image.replace('.ome.tif', '').replace('.ome.tiff', '')
    ref_slide_key = next((k for k in registrar.slide_dict.keys() if k == ref_image_no_ext), None)

    if ref_slide_key is None:
        raise KeyError(f"Reference '{ref_image_no_ext}' not found. Available: {list(registrar.slide_dict.keys())}")

    # Extract reference DAPI
    ref_slide = registrar.slide_dict[ref_slide_key]
    ref_channels = get_channel_names(os.path.basename(ref_slide_key))
    ref_dapi_idx = next((i for i, ch in enumerate(ref_channels) if "DAPI" in ch.upper()), 0)

    ref_vips = ref_slide.slide2vips(level=0)
    ref_dapi = ref_vips.extract_band(ref_dapi_idx).numpy()
    ref_dapi_scaled = autoscale(ref_dapi)

    log_progress(f"Processing {len(registrar.slide_dict) - 1} slides for QC...")

    for slide_name, slide_obj in registrar.slide_dict.items():
        if slide_name == ref_slide_key:
            continue

        log_progress(f"Creating QC for: {slide_name}")

        # Extract registered DAPI
        slide_channels = get_channel_names(os.path.basename(slide_name))
        slide_dapi_idx = next((i for i, ch in enumerate(slide_channels) if "DAPI" in ch.upper()), 0)

        warped_vips = slide_obj.warp_slide(level=0, non_rigid=True, crop=registrar.crop)
        reg_dapi = warped_vips.extract_band(slide_dapi_idx).numpy()
        reg_dapi_scaled = autoscale(reg_dapi)

        # Downsample by 0.25 scale factor
        ref_down = rescale(ref_dapi_scaled, scale=0.25, anti_aliasing=True, preserve_range=True).astype(np.uint8)
        reg_down = rescale(reg_dapi_scaled, scale=0.25, anti_aliasing=True, preserve_range=True).astype(np.uint8)

        del reg_dapi, reg_dapi_scaled

        # Create RGB composite: Red = registered, Green = reference
        h, w = reg_down.shape
        rgb_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_bgr[:, :, 2] = reg_down  # Blue channel (will appear red in BGR)
        rgb_bgr[:, :, 1] = ref_down  # Green channel
        rgb_bgr[:, :, 0] = 0         # Red channel (will appear blue in BGR)

        # Save as PNG (OpenCV uses BGR order)
        base_name = os.path.basename(slide_name)
        png_path = qc_path / f"{base_name}_QC_RGB.png"
        cv2.imwrite(str(png_path), rgb_bgr)
        log_progress(f"  Saved QC PNG: {png_path.name}")

        # Save as ImageJ-compatible TIFF (CYX order)
        rgb_stack = np.stack([
            reg_down,   # Red channel
            ref_down,   # Green channel
            np.zeros_like(ref_down, dtype=np.uint8)  # Blue channel
        ], axis=0)

        tiff_path = qc_path / f"{base_name}_QC_RGB.tif"
        tifffile.imwrite(
            str(tiff_path),
            rgb_stack,
            imagej=True,
            metadata={'axes': 'CYX', 'mode': 'composite'}
        )
        log_progress(f"  Saved QC TIFF: {tiff_path.name}")

        del rgb_bgr, rgb_stack, reg_down
        gc.collect()

    log_progress("QC generation complete")



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


def valis_registration(input_dir: str, out: str, qc_dir: Optional[str] = None,
                       reference_markers: Optional[list[str]] = None,
                       max_processed_image_dim_px: int = 1800,
                       max_non_rigid_dim_px: int = 3500,
                       micro_reg_fraction: float = 0.5,
                       num_features: int = 5000) -> int:
    """Perform VALIS registration on preprocessed images.

    Parameters
    ----------
    input_dir : str
        Directory containing preprocessed OME-TIFF files
    out : str
        Output merged file path
    qc_dir : str, optional
        QC directory for registration outputs
    reference_markers : list of str, optional
        Markers to identify reference image. Default: ['DAPI', 'SMA']
    max_processed_image_dim_px : int, optional
        Maximum image dimension for rigid registration. Default: 1800
    max_non_rigid_dim_px : int, optional
        Maximum dimension for non-rigid registration. Default: 3500
    micro_reg_fraction : float, optional
        Fraction of image size for micro-registration. Default: 0.5
    num_features : int, optional
        Number of SuperPoint features to detect. Default: 5000

    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Initialize JVM for VALIS
    registration.init_jvm()

    # Configuration
    if reference_markers is None:
        reference_markers = ['DAPI', 'SMA']

    ensure_dir(os.path.dirname(out) or '.')
    if qc_dir:
        ensure_dir(qc_dir)

    # Use qc_dir as results directory if available, otherwise use output directory
    results_dir = qc_dir if qc_dir else os.path.dirname(out)

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
    # Check input file metadata and validate with pyvips
    # ========================================================================
    log_progress("\nValidating input files...")
    input_files = sorted(glob.glob(os.path.join(input_dir, '*.ome.tif')))
    log_progress(f"Found {len(input_files)} OME-TIFF files")

    valid_files = []
    for idx, fpath in enumerate(input_files, 1):
        fname = os.path.basename(fpath)
        log_progress(f"\n[{idx}/{len(input_files)}] Validating: {fname}")

        try:
            # First check with tifffile for basic metadata
            with tifffile.TiffFile(fpath) as tif:
                # Get shape without loading full array
                if tif.series:
                    shape = tif.series[0].shape
                    dtype = tif.series[0].dtype
                    log_progress(f"  - Shape: {shape}, dtype: {dtype}")

                if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                    log_progress(f"  ✓ Has OME metadata")
                else:
                    log_progress(f"  ⚠ NO OME metadata")

            # Now validate with pyvips (this is what VALIS will use)
            try:
                vips_img = pyvips.Image.new_from_file(fpath)
                log_progress(f"  ✓ pyvips can read: {vips_img.width}x{vips_img.height}, {vips_img.bands} bands")
                valid_files.append(fpath)
                del vips_img
            except Exception as pyvips_err:
                log_progress(f"  ✗ pyvips FAILED: {pyvips_err}")
                log_progress(f"  ⚠ Skipping this file - VALIS will not be able to process it")

        except Exception as e:
            log_progress(f"  ✗ Error validating file: {e}")
            log_progress(f"  ⚠ Skipping this file")

    log_progress(f"\nValidation complete: {len(valid_files)}/{len(input_files)} files can be processed")

    if len(valid_files) == 0:
        raise RuntimeError("No valid files found! All files failed pyvips validation.")

    if len(valid_files) < len(input_files):
        log_progress(f"⚠ WARNING: {len(input_files) - len(valid_files)} files will be skipped due to validation errors")
        log_progress("Consider re-generating these files or checking for corruption")

    # ========================================================================
    # Initialize VALIS Registrar with Memory Optimization
    # ========================================================================
    log_progress("\nInitializing VALIS registration with memory optimization...")

    # Disable pyvips cache to reduce memory usage
    # Per VALIS docs: "cache can grow quite large, so it may be best to disable it"
    try:
        pyvips.cache_set_max(0)
        pyvips.cache_set_max_mem(0)
        log_progress("✓ Disabled pyvips cache (cache_set_max=0, cache_set_max_mem=0)")
    except Exception as e:
        log_progress(f"⚠ Could not disable pyvips cache: {e}")

    # Calculate max_image_dim_px based on available memory
    # VALIS docs: "mostly to keep memory in check"
    # For 60K x 50K images, we need to limit the cached size
    max_image_dim = 6000  # Limit cached images to 6K x 6K (vs full 60K x 50K)

    log_progress(f"Memory optimization parameters:")
    log_progress(f"  max_processed_image_dim_px: {max_processed_image_dim_px} (controls analysis resolution)")
    log_progress(f"  max_non_rigid_registration_dim_px: {max_non_rigid_dim_px} (controls non-rigid accuracy)")
    log_progress(f"  max_image_dim_px: {max_image_dim} (limits cached image size for RAM control)")

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
        #max_image_dim_px=max_image_dim,  # Critical: prevents loading full 60K x 50K images into RAM

        # Feature detection - SuperPoint/SuperGlue
        feature_detector_cls=feature_detectors.SuperPointFD,
        matcher=feature_matcher.SuperGlueMatcher(),

        # Non-rigid registration
        #non_rigid_registrar_cls=non_rigid_registrars.SimpleElastixWarper,

        # Micro-rigid registration
        micro_rigid_registrar_cls=MicroRigidRegistrar,

        # Registration behavior
        create_masks=True,
    )

    # ========================================================================
    # Perform Registration
    # ========================================================================
    log_progress("Starting rigid and non-rigid registration...")
    log_progress("This may take 15-45 minutes...")

    _, _, error_df = registrar.register()

    log_progress("✓ Initial registration completed")
    log_progress(f"\nRegistration errors:\n{error_df}")

    # ========================================================================
    # Micro-registration - Try with error handling
    # ========================================================================
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

    log_progress(f"\nWarping slides individually to: {out}")
    log_progress(f"  - Output directory: {out}")
    log_progress(f"  - Strategy: Individual warp_and_save_slide() for low RAM")
    log_progress("")

    # Create output directory
    ensure_dir(out)

    # Build mapping from slide name to original file path
    slide_name_to_path = {}
    for f in registrar.original_img_list:
        basename = os.path.basename(f)
        slide_name = basename.replace('.ome.tif', '').replace('.ome.tiff', '')
        slide_name_to_path[slide_name] = f

    log_progress(f"\nSlide name to path mapping:")
    for name, path in slide_name_to_path.items():
        log_progress(f"  '{name}' -> '{path}'")
    log_progress("")

    # Warp each slide individually and save
    warped_count = 0
    for idx, (slide_name, slide_obj) in enumerate(registrar.slide_dict.items(), 1):
        log_progress(f"[{idx}/{len(registrar.slide_dict)}] Warping: {slide_name}")

        # Get original file path
        if slide_name not in slide_name_to_path:
            log_progress(f"  ✗ ERROR: Cannot find original path for '{slide_name}'")
            log_progress(f"  Available names: {list(slide_name_to_path.keys())}")
            continue

        src_path = slide_name_to_path[slide_name]
        log_progress(f"  Source: {src_path}")

        # Output path for this slide
        out_path = os.path.join(out, f"{slide_name}_registered.ome.tiff")

        # Warp and save using official VALIS method with tiled processing
        # Tile size chosen to balance memory vs I/O overhead
        # For 60K x 50K images, 2048x2048 tiles = ~900 tiles per image
        log_progress(f"  Applying transforms (rigid + non-rigid + micro) with tiled processing...")

        # Note: tile_wh must be integer (single value), not tuple

        slide_obj.warp_and_save_slide(
            src_f=src_path,
            dst_f=out_path,
            level=0,              # Full resolution
            non_rigid=True,       # Apply non-rigid transforms
            crop=True,            # Crop to reference overlap
            interp_method="bicubic",
            tile_wh=1024,         # Process in 2K tiles to reduce RAM (must be int, not tuple)
        )

        warped_count += 1
        log_progress(f"  ✓ Saved: {out_path}")

        # Aggressive memory cleanup after each slide
        # Close slide reader if possible
        if hasattr(slide_obj, 'slide_reader') and hasattr(slide_obj.slide_reader, 'close'):
            try:
                slide_obj.slide_reader.close()
            except:
                pass

        # Force garbage collection
        gc.collect()

        log_progress(f"  ✓ Memory cleanup completed")

    log_progress(f"✓ All {warped_count} slides warped and saved to: {out}")

    gc.collect()

    # Save individual registered slides to QC directory with reference DAPI first
    if qc_dir:
        log_progress(f"\n" + "=" * 70)
        log_progress("QC IMAGE GENERATION")
        log_progress("=" * 70)
        log_progress(f"QC directory: {qc_dir}")
        log_progress(f"Reference image: {ref_image}")
        save_qc_dapi_rgb(registrar, qc_dir, ref_image)

    log_progress("\n" + "=" * 70)
    log_progress("✓ REGISTRATION COMPLETED SUCCESSFULLY!")
    log_progress("=" * 70)

    # Cleanup
    registration.kill_jvm()

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='VALIS registration for WSI processing')
    parser.add_argument('--input-dir', required=True, help='Directory containing preprocessed files')
    parser.add_argument('--out', required=True, help='Output merged path')
    parser.add_argument('--qc-dir', required=False, help='QC directory for registration outputs')
    parser.add_argument('--reference-markers', nargs='+', default=['DAPI', 'SMA'],
                       help='Markers to identify reference image (default: DAPI SMA)')
    parser.add_argument('--max-processed-dim', type=int, default=1800,
                       help='Maximum image dimension for rigid registration (default: 1800)')
    parser.add_argument('--max-non-rigid-dim', type=int, default=3500,
                       help='Maximum dimension for non-rigid registration (default: 3500)')
    parser.add_argument('--micro-reg-fraction', type=float, default=0.5,
                       help='Fraction of image size for micro-registration (default: 0.5)')
    parser.add_argument('--num-features', type=int, default=5000,
                       help='Number of SuperPoint features to detect (default: 5000)')
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    try:
        return valis_registration(
            args.input_dir,
            args.out,
            args.qc_dir,
            args.reference_markers,
            args.max_processed_dim,
            args.max_non_rigid_dim,
            args.micro_reg_fraction,
            args.num_features
        )
    except Exception as e:
        log_progress(f"ERROR: Registration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
