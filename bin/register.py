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
    name_part = base.split('_corrected')[0] # Remove suffix
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
    """
    Create QC RGB composites equivalently to ImageJ:
      - ref DAPI  → GREEN
      - reg DAPI  → RED
      - optional BLUE = zeros
    """
    log_progress(f"Creating QC directory: {qc_dir}")
    os.makedirs(qc_dir, exist_ok=True)

    # Find the full path key for ref_image in slide_dict
    log_progress(f"\nSearching for reference image in slide_dict...")
    log_progress(f"  - Looking for: {ref_image}")
    log_progress(f"  - slide_dict keys: {list(registrar.slide_dict.keys())}")

    ref_slide_key = None
    for key in registrar.slide_dict.keys():
        basename = os.path.basename(key)
        log_progress(f"  - Checking key: {key} (basename: {basename})")
        if basename == ref_image or key == ref_image:
            ref_slide_key = key
            log_progress(f"  ✓ Match found: {ref_slide_key}")
            break

    if ref_slide_key is None:
        raise KeyError(f"Reference image '{ref_image}' not found in slide_dict. Available keys: {list(registrar.slide_dict.keys())}")

    # -------- Get REFERENCE DAPI --------
    log_progress(f"\nExtracting reference DAPI channel...")
    ref_slide = registrar.slide_dict[ref_slide_key]
    ref_basename = os.path.basename(ref_slide_key)
    log_progress(f"  - Reference slide basename: {ref_basename}")

    ref_channels = get_channel_names(ref_basename)
    log_progress(f"  - Reference channels: {ref_channels}")

    ref_dapi_idx = next((i for i,ch in enumerate(ref_channels) if "DAPI" in ch.upper()), 0)
    log_progress(f"  - Reference DAPI index: {ref_dapi_idx}")

    ref_vips = ref_slide.slide2vips(level=0)
    log_progress(f"  - Reference slide dimensions: {ref_vips.width}x{ref_vips.height}, bands: {ref_vips.bands}")

    ref_dapi = ref_vips.numpy()[..., ref_dapi_idx]
    log_progress(f"  - Reference DAPI shape: {ref_dapi.shape}, dtype: {ref_dapi.dtype}")

    log_progress(f"\nProcessing {len(registrar.slide_dict) - 1} slides for QC...")

    for idx, (slide_name, slide_obj) in enumerate(registrar.slide_dict.items(), 1):
        if slide_name == ref_slide_key:
            log_progress(f"\n[{idx}/{len(registrar.slide_dict)}] Skipping reference: {slide_name}")
            continue

        log_progress(f"\n[{idx}/{len(registrar.slide_dict)}] Processing: {slide_name}")

        slide_basename = os.path.basename(slide_name)
        log_progress(f"  - Basename: {slide_basename}")

        slide_channels = get_channel_names(slide_basename)
        log_progress(f"  - Channels: {slide_channels}")

        slide_dapi_idx = next((i for i,ch in enumerate(slide_channels) if "DAPI" in ch.upper()), 0)
        log_progress(f"  - DAPI index: {slide_dapi_idx}")

        # Registered (warped) image
        log_progress(f"  - Warping slide (non_rigid=True, crop={registrar.crop})...")
        warped_vips = slide_obj.warp_slide(level=0, non_rigid=True, crop=registrar.crop)
        log_progress(f"  - Warped dimensions: {warped_vips.width}x{warped_vips.height}, bands: {warped_vips.bands}")

        # Convert to numpy
        log_progress(f"  - Converting warped slide to numpy...")
        warped = warped_vips.numpy()
        log_progress(f"  - Warped numpy shape: {warped.shape}, dtype: {warped.dtype}")

        # Registered DAPI channel
        log_progress(f"  - Extracting DAPI channel at index {slide_dapi_idx}...")
        reg_dapi = warped[..., slide_dapi_idx]
        log_progress(f"  - Registered DAPI shape: {reg_dapi.shape}, dtype: {reg_dapi.dtype}")

        # ----- Auto brightness/contrast -----
        log_progress(f"  - Applying autoscale...")
        ref_dapi_scaled = autoscale(ref_dapi)
        reg_dapi_scaled = autoscale(reg_dapi)
        log_progress(f"  - Scaled shapes: ref={ref_dapi_scaled.shape}, reg={reg_dapi_scaled.shape}")

        # ----- Merge channels (ImageJ-style RGB) -----
        # R = registered DAPI
        # G = reference DAPI
        # B = zero
        log_progress(f"  - Creating RGB composite (R=reg, G=ref, B=zeros)...")
        rgb = np.dstack([
            reg_dapi_scaled,    # Red channel
            ref_dapi_scaled,    # Green channel
            np.zeros_like(ref_dapi_scaled)  # Blue channel
        ]).astype(np.uint8)
        log_progress(f"  - RGB shape: {rgb.shape}, dtype: {rgb.dtype}")

        # Save RGB composite
        out_filename = os.path.basename(slide_name) + "_QC_RGB.tif"
        out_path = os.path.join(qc_dir, out_filename)
        log_progress(f"  - Saving to: {out_path}")
        tifffile.imwrite(out_path, rgb, photometric='rgb')
        log_progress(f"  ✓ Saved")
        del rgb

    log_progress("\n✓ All QC RGB composites saved.")



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
    # Check input file metadata
    # ========================================================================
    log_progress("\nChecking input files metadata...")
    input_files = sorted(glob.glob(os.path.join(input_dir, '*.ome.tif')))
    log_progress(f"Found {len(input_files)} OME-TIFF files")

    for idx, fpath in enumerate(input_files[:3], 1):  # Check first 3 files
        fname = os.path.basename(fpath)
        log_progress(f"\n[{idx}] Checking: {fname}")
        try:
            with tifffile.TiffFile(fpath) as tif:
                img = tif.asarray()
                log_progress(f"  - Image shape: {img.shape}")
                log_progress(f"  - Image dtype: {img.dtype}")

                if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                    log_progress(f"  ✓ Has OME metadata ({len(tif.ome_metadata)} chars)")
                    if 'PhysicalSizeX' in tif.ome_metadata:
                        log_progress(f"  ✓ Has PhysicalSizeX")
                    if 'PhysicalSizeXUnit' in tif.ome_metadata:
                        log_progress(f"  ✓ Has PhysicalSizeXUnit")
                    if 'µm' in tif.ome_metadata or 'um' in tif.ome_metadata:
                        log_progress(f"  ✓ Has micrometer units")
                else:
                    log_progress(f"  ⚠ NO OME metadata!")
        except Exception as e:
            log_progress(f"  ⚠ Error reading file: {e}")

    if len(input_files) > 3:
        log_progress(f"\n... and {len(input_files) - 3} more files")

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

        # Image size parameters
        max_processed_image_dim_px=max_processed_image_dim_px,
        max_non_rigid_registration_dim_px=max_non_rigid_dim_px,

        # Feature detection - SuperPoint/SuperGlue
        feature_detector_cls=feature_detectors.BriskFD,
        #matcher=feature_matcher.SuperGlueMatcher(),

        # Non-rigid registration
        non_rigid_registrar_cls=None,#non_rigid_registrars.OpticalFlowWarper,

        # Micro-rigid registration
        #micro_rigid_registrar_cls=MicroRigidRegistrar,

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
    # Micro-registration
    # ========================================================================
    log_progress("\nCalculating micro-registration parameters...")

    img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
    min_max_size = np.min([np.max(d) for d in img_dims])
    micro_reg_size = int(np.floor(min_max_size * micro_reg_fraction))

    log_progress(f"Micro-registration size: {micro_reg_size}px")
    log_progress("Starting micro-registration (may take 30-120 minutes)...")

    '''
    _, micro_error = registrar.register_micro(
        max_non_rigid_registration_dim_px=micro_reg_size,
        reference_img_f=ref_image,
        align_to_reference=True,
    )

    log_progress("✓ Micro-registration completed")
    log_progress(f"\nMicro-registration errors:\n{micro_error}")
    '''
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

    # Parse channel names from filenames and match with actual channel counts
    log_progress("\nBuilding channel_name_dict...")
    channel_name_dict = {}

    for f in registrar.original_img_list:
        log_progress(f"\n  Processing file: {f}")

        # Get expected channel names from filename
        basename = os.path.basename(f)
        log_progress(f"    - Basename: {basename}")

        # Get slide name (basename without extension)
        slide_name = basename.replace('.ome.tif', '').replace('.ome.tiff', '')
        log_progress(f"    - Slide name (no extension): {slide_name}")

        expected_names = get_channel_names(basename)
        log_progress(f"    - Expected channel names from filename: {expected_names}")

        # Get actual number of channels in the slide
        # Use slide_name to look up in slide_dict
        if slide_name not in registrar.slide_dict:
            log_progress(f"    - ERROR: '{slide_name}' not found in registrar.slide_dict!")
            log_progress(f"    - Available keys: {list(registrar.slide_dict.keys())}")
            continue

        slide_obj = registrar.slide_dict[slide_name]
        log_progress(f"    - Slide object name: {slide_obj.name}")

        vips_img = slide_obj.slide2vips(level=0)
        actual_channels = vips_img.bands
        log_progress(f"    - Actual channels in slide: {actual_channels}")
        log_progress(f"    - Slide dimensions: {vips_img.width}x{vips_img.height}")

        # Only use as many names as there are actual channels
        channel_names_to_use = expected_names[:actual_channels]

        # If we have more channels than names, pad with generic names
        if actual_channels > len(expected_names):
            log_progress(f"    - WARNING: More channels ({actual_channels}) than names ({len(expected_names)}), padding...")
            for i in range(len(expected_names), actual_channels):
                channel_names_to_use.append(f"Channel_{i}")

        channel_name_dict[f] = channel_names_to_use
        log_progress(f"    - Final channel names to use: {channel_names_to_use}")

    log_progress(f"\n✓ channel_name_dict built with {len(channel_name_dict)} entries")
    log_progress(f"\nFull channel_name_dict:")
    for key, value in channel_name_dict.items():
        log_progress(f"  '{key}': {value}")

    log_progress(f"\nMerging and warping slides to: {out}")
    log_progress(f"  - Output file: {out}")
    log_progress(f"  - Output directory: {os.path.dirname(out)}")
    log_progress(f"  - drop_duplicates: True")

    merged_img, channel_names, _ = registrar.warp_and_merge_slides(
        out,
        channel_name_dict=channel_name_dict,
        drop_duplicates=True,
    )

    log_progress(f"\n✓ Merge completed!")
    log_progress(f"  - Merged image shape: {merged_img.width}x{merged_img.height}")
    log_progress(f"  - Number of channels: {merged_img.bands}")
    log_progress(f"  - Channel names ({len(channel_names)}): {channel_names}")

    del merged_img  # Free memory

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
