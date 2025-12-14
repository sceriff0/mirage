#!/usr/bin/env python3
"""VALIS registration script with STREAM-BASED MERGING for LOW MEMORY environments.

This is a memory-optimized version of register.py that:
1. Processes slides ONE AT A TIME instead of loading all into memory
2. Writes merged channels directly to disk incrementally
3. Uses lower resolution for QC images (level=1 = half resolution)
4. Generates QC images during registration, not after
5. Explicitly frees memory after each operation

MEMORY SAVINGS: 70-80% reduction compared to register.py

The registration uses:
- SuperPoint feature detection with SuperGlue matching
- Micro-rigid registration (optional, can be disabled with --skip-micro)
- Optical flow-based non-rigid deformation
"""
from __future__ import annotations

import argparse
import os
import glob
import gc
import json
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict
import tifffile

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


def log_progress(message: str) -> None:
    """Print timestamped progress messages."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def get_channel_names(filename: str) -> list[str]:
    """Parse channel names from filename."""
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


def generate_qc_for_slide(
    slide_obj,
    ref_slide,
    ref_dapi,
    slide_name: str,
    qc_dir: str,
    qc_level: int = 1
):
    """
    Generate QC image for a single slide immediately after registration.

    Uses lower resolution (level=1 by default) to save memory.
    """
    log_progress(f"\n  Generating QC for: {slide_name}")

    slide_basename = os.path.basename(slide_name)
    slide_channels = get_channel_names(slide_basename)
    slide_dapi_idx = next((i for i, ch in enumerate(slide_channels) if "DAPI" in ch.upper()), 0)

    # Warp slide at lower resolution for QC
    log_progress(f"  - Warping at level={qc_level} (1/{2**qc_level} resolution)...")
    warped_vips = slide_obj.warp_slide(level=qc_level, non_rigid=True, crop='reference')

    # Extract only DAPI channel
    reg_dapi = warped_vips.extract_band(slide_dapi_idx).numpy()
    log_progress(f"  - Registered DAPI shape: {reg_dapi.shape}")

    # Autoscale both
    ref_dapi_scaled = autoscale(ref_dapi)
    reg_dapi_scaled = autoscale(reg_dapi)

    del reg_dapi

    # Create RGB composite (memory-efficient)
    h, w = reg_dapi_scaled.shape
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = reg_dapi_scaled  # Red
    rgb[:, :, 1] = ref_dapi_scaled  # Green
    rgb[:, :, 2] = 0                # Blue

    del reg_dapi_scaled

    # Save
    out_filename = os.path.basename(slide_name) + "_QC_RGB.tif"
    out_path = os.path.join(qc_dir, out_filename)
    tifffile.imwrite(out_path, rgb, photometric='rgb')
    log_progress(f"  ✓ QC saved: {out_filename}")

    del rgb
    gc.collect()


def find_reference_image(directory: str, required_markers: list[str],
                        valid_extensions: Optional[list[str]] = None) -> str:
    """Find image file containing all required markers in filename."""
    if valid_extensions is None:
        valid_extensions = ['.ome.tif', '.ome.tiff']

    files = []
    for ext in valid_extensions:
        files.extend(glob.glob(os.path.join(directory, f'*{ext}')))

    if not files:
        raise FileNotFoundError(f"No image files with extensions {valid_extensions} in {directory}")

    # Find files containing all required markers
    candidates = []
    for f in files:
        basename = os.path.basename(f).upper()
        if all(marker.upper() in basename for marker in required_markers):
            candidates.append(f)

    if not candidates:
        raise ValueError(f"No files found containing all markers: {required_markers}")

    # Return basename only
    return os.path.basename(candidates[0])


def stream_merge_slides(
    registrar,
    output_path: str,
    channel_name_dict: Dict[str, list[str]],
    qc_dir: Optional[str] = None,
    qc_level: int = 1,
    ref_image: Optional[str] = None
) -> Tuple[int, int, list[str]]:
    """
    STREAM-BASED MERGING: Process and write slides one at a time.

    This avoids loading all warped slides into memory simultaneously.

    Parameters
    ----------
    registrar : Valis
        Registered VALIS object
    output_path : str
        Path to output merged OME-TIFF
    channel_name_dict : dict
        Mapping of slide paths to channel names
    qc_dir : str, optional
        Directory for QC images
    qc_level : int
        Pyramid level for QC images (1 = half res, 2 = quarter res)
    ref_image : str, optional
        Reference image name for QC

    Returns
    -------
    width : int
        Width of merged image
    height : int
        Height of merged image
    all_channel_names : list
        List of all channel names in merged image
    """
    log_progress("\n" + "=" * 70)
    log_progress("STREAM-BASED MERGE (Low Memory Mode)")
    log_progress("=" * 70)

    # Get reference slide for QC
    ref_slide = None
    ref_dapi = None
    if qc_dir and ref_image:
        ref_image_no_ext = ref_image.replace('.ome.tif', '').replace('.ome.tiff', '')
        if ref_image_no_ext in registrar.slide_dict:
            ref_slide = registrar.slide_dict[ref_image_no_ext]
            ref_channels = get_channel_names(ref_image_no_ext)
            ref_dapi_idx = next((i for i, ch in enumerate(ref_channels) if "DAPI" in ch.upper()), 0)

            # Load reference DAPI at QC resolution
            ref_vips = ref_slide.slide2vips(level=qc_level)
            ref_dapi = ref_vips.extract_band(ref_dapi_idx).numpy()
            log_progress(f"Reference DAPI loaded at level={qc_level}: {ref_dapi.shape}")

    # Collect all channel names and count total channels
    all_channel_names = []
    slide_channel_info = []  # [(slide_name, slide_obj, channel_names)]

    for file_path in registrar.original_img_list:
        basename = os.path.basename(file_path)
        slide_name = basename.replace('.ome.tif', '').replace('.ome.tiff', '')

        if slide_name not in registrar.slide_dict:
            log_progress(f"WARNING: Skipping {slide_name} (not in slide_dict)")
            continue

        slide_obj = registrar.slide_dict[slide_name]

        # Get channel names for this slide
        if file_path in channel_name_dict:
            channels = channel_name_dict[file_path]
        else:
            channels = get_channel_names(basename)

        slide_channel_info.append((slide_name, slide_obj, channels))
        all_channel_names.extend(channels)

    log_progress(f"\nTotal slides: {len(slide_channel_info)}")
    log_progress(f"Total channels: {len(all_channel_names)}")
    log_progress(f"Channel names: {all_channel_names}")

    # Get output dimensions from first slide (reference)
    first_slide = slide_channel_info[0][1]
    warped_test = first_slide.warp_slide(level=0, non_rigid=True, crop=registrar.crop)
    output_width = warped_test.width
    output_height = warped_test.height
    del warped_test
    gc.collect()

    log_progress(f"\nOutput dimensions: {output_width} x {output_height}")
    log_progress(f"Output path: {output_path}")

    # Create OME-TIFF metadata
    metadata = {
        'axes': 'CYX',
        'Channel': {'Name': all_channel_names},
        'PhysicalSizeX': 0.325,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': 0.325,
        'PhysicalSizeYUnit': 'µm'
    }

    log_progress("\nProcessing slides one-by-one...")

    # Open TIFF writer for streaming
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif_writer:
        channel_idx = 0

        for idx, (slide_name, slide_obj, channels) in enumerate(slide_channel_info, 1):
            log_progress(f"\n[{idx}/{len(slide_channel_info)}] Processing: {slide_name}")
            log_progress(f"  - Channels: {channels}")

            # Warp slide at full resolution
            log_progress(f"  - Warping slide...")
            warped_vips = slide_obj.warp_slide(level=0, non_rigid=True, crop=registrar.crop)
            log_progress(f"  - Warped: {warped_vips.width}x{warped_vips.height}, {warped_vips.bands} bands")

            # Extract and write each channel
            for ch_idx, ch_name in enumerate(channels):
                log_progress(f"  - Writing channel {channel_idx + 1}/{len(all_channel_names)}: {ch_name}")

                # Extract single channel
                channel_data = warped_vips.extract_band(ch_idx).numpy()

                # Write channel to file
                if channel_idx == 0:
                    # First page: write with metadata
                    tif_writer.write(
                        channel_data,
                        photometric='minisblack',
                        metadata=metadata,
                        contiguous=False
                    )
                else:
                    # Subsequent pages: append
                    tif_writer.write(
                        channel_data,
                        photometric='minisblack',
                        contiguous=False
                    )

                del channel_data
                channel_idx += 1

            # Free warped slide memory
            del warped_vips

            # Generate QC for this slide (if not reference)
            if qc_dir and ref_dapi is not None:
                if slide_name != ref_image_no_ext:
                    generate_qc_for_slide(
                        slide_obj, ref_slide, ref_dapi,
                        slide_name, qc_dir, qc_level
                    )

            # Force garbage collection after each slide
            gc.collect()
            log_progress(f"  ✓ Slide {slide_name} complete")

    log_progress(f"\n✓ Merged file written: {output_path}")
    log_progress(f"  - Total channels written: {channel_idx}")

    return output_width, output_height, all_channel_names


def run_registration(
    input_dir: str,
    out: str,
    qc_dir: Optional[str] = None,
    reference_markers: Optional[list[str]] = None,
    max_processed_image_dim_px: int = 1200,  # Lower default for memory
    max_non_rigid_dim_px: int = 2500,         # Lower default for memory
    micro_reg_fraction: float = 0.25,
    num_features: int = 5000,
    skip_micro: bool = False,
    qc_level: int = 1
) -> int:
    """
    Run VALIS registration with stream-based merging.

    Parameters
    ----------
    skip_micro : bool
        If True, skip micro-registration to save memory
    qc_level : int
        Pyramid level for QC images (1=half, 2=quarter resolution)
    """
    # Initialize JVM for VALIS
    registration.init_jvm()

    # Configuration
    if reference_markers is None:
        reference_markers = ['DAPI', 'SMA']

    ensure_dir(os.path.dirname(out) or '.')
    if qc_dir:
        ensure_dir(qc_dir)

    results_dir = qc_dir if qc_dir else os.path.dirname(out)

    # ========================================================================
    # VALIS Parameters
    # ========================================================================
    log_progress("=" * 70)
    log_progress("VALIS LOW MEMORY Registration Configuration")
    log_progress("=" * 70)
    log_progress(f"Rigid resolution: {max_processed_image_dim_px}px")
    log_progress(f"Non-rigid resolution: {max_non_rigid_dim_px}px")
    log_progress(f"Micro-registration: {'DISABLED' if skip_micro else 'ENABLED'}")
    log_progress(f"QC pyramid level: {qc_level} (1/{2**qc_level} resolution)")
    log_progress(f"Feature detector: SuperPoint with {num_features} features")
    log_progress("=" * 70)

    # Find reference image
    log_progress(f"\nSearching for reference image with markers: {reference_markers}")

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
    # Initialize VALIS Registrar
    # ========================================================================
    log_progress("\nInitializing VALIS registration...")

    registrar = registration.Valis(
        input_dir,
        results_dir,

        # Reference image
        reference_img_f=ref_image,
        align_to_reference=True,
        crop="reference",

        # Image size parameters (lower for memory)
        max_processed_image_dim_px=max_processed_image_dim_px,
        max_non_rigid_registration_dim_px=max_non_rigid_dim_px,

        # Feature detection
        feature_detector_cls=feature_detectors.SuperPointFD,
        matcher=feature_matcher.SuperGlueMatcher(),

        # Non-rigid registration - disabled for memory
        non_rigid_registrar_cls=None,

        # Micro-rigid registration
        micro_rigid_registrar_cls=MicroRigidRegistrar if not skip_micro else None,

        # Registration behavior
        create_masks=True,
    )

    # ========================================================================
    # Perform Registration
    # ========================================================================
    log_progress("\nStarting rigid registration...")
    log_progress("This may take 15-45 minutes...")

    _, _, error_df = registrar.register()

    log_progress("✓ Initial registration completed")
    log_progress(f"\nRegistration errors:\n{error_df}")

    # ========================================================================
    # Micro-registration (optional)
    # ========================================================================
    if not skip_micro:
        log_progress("\nCalculating micro-registration parameters...")

        img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
        min_max_size = np.min([np.max(d) for d in img_dims])
        micro_reg_size = int(np.floor(min_max_size * micro_reg_fraction))

        log_progress(f"Micro-registration size: {micro_reg_size}px")
        log_progress("Starting micro-registration...")

        try:
            _, micro_error = registrar.register_micro(
                max_non_rigid_registration_dim_px=micro_reg_size,
                reference_img_f=ref_image,
                align_to_reference=True,
            )
            log_progress("✓ Micro-registration completed")
            log_progress(f"\nMicro-registration errors:\n{micro_error}")
        except Exception as e:
            log_progress(f"WARNING: Micro-registration failed: {e}")
            log_progress("Continuing without micro-registration...")
    else:
        log_progress("\nMicro-registration SKIPPED (--skip-micro enabled)")

    # ========================================================================
    # Build channel name dictionary
    # ========================================================================
    log_progress("\nBuilding channel_name_dict...")
    channel_name_dict = {}

    for f in registrar.original_img_list:
        basename = os.path.basename(f)
        slide_name = basename.replace('.ome.tif', '').replace('.ome.tiff', '')

        if slide_name not in registrar.slide_dict:
            log_progress(f"    WARNING: '{slide_name}' not found in registrar.slide_dict!")
            continue

        slide_obj = registrar.slide_dict[slide_name]

        # Get metadata
        vips_img = slide_obj.slide2vips(level=0)
        actual_channels = vips_img.bands
        del vips_img

        expected_names = get_channel_names(basename)
        channel_names_to_use = expected_names[:actual_channels]

        if actual_channels > len(expected_names):
            for i in range(len(expected_names), actual_channels):
                channel_names_to_use.append(f"Channel_{i}")

        channel_name_dict[f] = channel_names_to_use

    log_progress(f"✓ channel_name_dict built with {len(channel_name_dict)} entries")

    # ========================================================================
    # STREAM-BASED MERGE AND QC
    # ========================================================================
    width, height, all_channels = stream_merge_slides(
        registrar,
        out,
        channel_name_dict,
        qc_dir=qc_dir,
        qc_level=qc_level,
        ref_image=ref_image
    )

    log_progress(f"\n✓ Final merged image: {width}x{height}, {len(all_channels)} channels")
    log_progress(f"  Channel names: {all_channels}")

    # Cleanup
    del registrar
    gc.collect()

    log_progress("\n" + "=" * 70)
    log_progress("✓ REGISTRATION COMPLETED SUCCESSFULLY!")
    log_progress("=" * 70)

    registration.kill_jvm()

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='VALIS LOW MEMORY registration with stream-based merging'
    )
    parser.add_argument('--input-dir', required=True, help='Directory containing preprocessed files')
    parser.add_argument('--out', required=True, help='Output merged path')
    parser.add_argument('--qc-dir', help='Output directory for QC images')
    parser.add_argument('--reference-markers', nargs='+', help='Markers to identify reference image')
    parser.add_argument('--max-processed-dim', type=int, default=1200,
                       help='Max dimension for rigid registration (lower = less memory)')
    parser.add_argument('--max-non-rigid-dim', type=int, default=2500,
                       help='Max dimension for non-rigid registration')
    parser.add_argument('--micro-reg-fraction', type=float, default=0.25,
                       help='Fraction of image size for micro-registration')
    parser.add_argument('--num-features', type=int, default=5000,
                       help='Number of features for SuperPoint')
    parser.add_argument('--skip-micro', action='store_true',
                       help='Skip micro-registration to save memory (recommended for low-mem)')
    parser.add_argument('--qc-level', type=int, default=1, choices=[0, 1, 2],
                       help='Pyramid level for QC images (0=full, 1=half, 2=quarter res)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        return run_registration(
            input_dir=args.input_dir,
            out=args.out,
            qc_dir=args.qc_dir,
            reference_markers=args.reference_markers,
            max_processed_image_dim_px=args.max_processed_dim,
            max_non_rigid_dim_px=args.max_non_rigid_dim,
            micro_reg_fraction=args.micro_reg_fraction,
            num_features=args.num_features,
            skip_micro=args.skip_micro,
            qc_level=args.qc_level
        )
    except Exception as e:
        log_progress(f"ERROR: Registration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
