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
import numpy as np
from datetime import datetime
from typing import Optional

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
    name_part = base.split('_corrected')[0]
    parts = name_part.split('_')
    channels = parts[1:]  # Exclude Patient ID
    return channels


def save_qc_with_reference_dapi(registrar, qc_dir: str, ref_image: str) -> None:
    """Save QC images with reference DAPI and registered DAPI only.

    Each QC image contains only 2 channels:
    - Channel 0: Reference DAPI (unwarped, from reference image)
    - Channel 1: Registered DAPI (warped, from the current image)

    The reference image itself is not saved to QC.

    Parameters
    ----------
    registrar : registration.Valis
        VALIS registrar object after registration
    qc_dir : str
        Directory to save QC images
    ref_image : str
        Reference image filename
    """
    import tifffile

    log_progress("\nSaving QC images with reference DAPI + registered DAPI...")

    # Get reference slide
    ref_slide = registrar.slide_dict[ref_image]
    ref_channels = get_channel_names(ref_image)

    # Find DAPI channel index in reference image
    ref_dapi_idx = None
    for idx, ch in enumerate(ref_channels):
        if 'DAPI' in ch.upper():
            ref_dapi_idx = idx
            break

    if ref_dapi_idx is None:
        log_progress("Warning: No DAPI channel found in reference image, using first channel")
        ref_dapi_idx = 0

    # Get reference DAPI channel (unwarped, from original reference)
    ref_dapi = ref_slide.slide2vips().numpy()[..., ref_dapi_idx]

    log_progress(f"Reference DAPI channel shape: {ref_dapi.shape}")
    log_progress(f"Reference image: {ref_image}, DAPI channel index: {ref_dapi_idx}")

    # Process each slide (except reference)
    for slide_name, slide_obj in registrar.slide_dict.items():
        if slide_name == ref_image:
            log_progress(f"Skipping reference image: {slide_name}")
            continue

        log_progress(f"Processing {slide_name}")

        # Get channels for this slide
        slide_channels = get_channel_names(slide_name)

        # Find DAPI channel index in this slide
        slide_dapi_idx = None
        for idx, ch in enumerate(slide_channels):
            if 'DAPI' in ch.upper():
                slide_dapi_idx = idx
                break

        if slide_dapi_idx is None:
            log_progress(f"  Warning: No DAPI channel found in {slide_name}, using first channel")
            slide_dapi_idx = 0

        # Get warped/registered version of this slide
        warped_img = slide_obj.warp_slide(
            non_rigid=True,
            crop=registrar.crop
        )

        # Extract only the registered DAPI channel
        registered_dapi = warped_img[..., slide_dapi_idx]

        # Combine: reference DAPI + registered DAPI (2 channels only)
        qc_img = np.dstack([ref_dapi, registered_dapi])

        log_progress(f"  QC image shape: {qc_img.shape} (ref DAPI + registered DAPI)")

        # Save with original filename to preserve cycle information
        output_path = os.path.join(qc_dir, slide_name)
        tifffile.imwrite(output_path, qc_img, photometric='minisblack')

    log_progress("✓ QC images saved (reference DAPI + registered DAPI only)")


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
    # Initialize VALIS Registrar
    # ========================================================================
    log_progress("Initializing VALIS registration...")
    
    registrar = registration.Valis(
        input_dir,
        results_dir,
        
        # Reference image
        reference_img_f=ref_image,
        align_to_reference=True,
        crop="reference",
        
        # Image size parameters
        max_processed_image_dim_px=max_processed_image_dim_px,
        max_non_rigid_registration_dim_px=max_non_rigid_dim_px,
        
        # Feature detection - SuperPoint/SuperGlue
        feature_detector_cls=feature_detectors.SuperPointFD,
        matcher=feature_matcher.SuperGlueMatcher(),
        
        # Non-rigid registration
        non_rigid_registrar_cls=non_rigid_registrars.OpticalFlowWarper,
        
        # Micro-rigid registration
        micro_rigid_registrar_cls=MicroRigidRegistrar,
        
        # Registration behavior
        compose_non_rigid=False,
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
    # Micro-registration
    # ========================================================================
    log_progress("\nCalculating micro-registration parameters...")
    
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
    
    # ========================================================================
    # Merge and Save
    # ========================================================================
    log_progress("\nPreparing to merge channels...")
    
    # Parse channel names from filenames
    channel_name_dict = {
        f: get_channel_names(f)
        for f in registrar.original_img_list
    }
    
    log_progress("Channel mapping detected:")
    for filename, channels in channel_name_dict.items():
        log_progress(f"  {filename}: {channels}")
    
    log_progress(f"\nMerging and warping slides to: {out}")
    
    merged_img, channel_names, _ = registrar.warp_and_merge_slides(
        out,
        channel_name_dict=channel_name_dict,
        drop_duplicates=True,
    )
    
    log_progress(f"✓ Merged image shape: {merged_img.shape}")
    log_progress(f"✓ Channel names: {channel_names}")
    
    # Save individual registered slides to QC directory with reference DAPI first
    if qc_dir:
        save_qc_with_reference_dapi(registrar, qc_dir, ref_image)
    
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
