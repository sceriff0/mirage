#!/usr/bin/env python3
"""VALIS registration script for WSI processing pipeline.

This script performs multi-modal image registration using VALIS (Virtual Alignment
of pathoLogy Image Series). It aligns multiple preprocessed OME-TIFF files and
creates registered outputs for each slide.

Features:
- SuperPoint feature detection with SuperGlue matching
- Optional micro-rigid registration for high-resolution alignment
- Structured error handling with retry logic for transient failures
- Optional parallel slide warping for improved performance
- Memory-optimized processing for large images
- Progress tracking with ETA estimation

Usage:
    python register.py --input-dir /path/to/preprocessed --out /path/to/output

Example:
    python register.py \\
        --input-dir ./preprocessed \\
        --out ./registered \\
        --reference panel1.ome.tif \\
        --max-image-dim 6000 \\
        --parallel-warping \\
        --n-workers 4
"""
from __future__ import annotations

# Standard library
import argparse
import gc
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np
import tifffile

# Add utils directory to path before local imports
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

# Local utilities
from image_utils import ensure_dir
from logger import log_progress
from metadata import get_channel_names
from progress import PhaseReporter, ProgressTracker
from registration_errors import (
    ErrorSeverity,
    RegistrationErrorContext,
    RegistrationResult,
)
from registration_utils import autoscale
from retry import RetryContext, default_cleanup

# Environment configuration (must be before VALIS imports that use numba)
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_CACHING'] = '1'
os.environ['LD_LIBRARY_PATH'] = (
    '/usr/local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:'
    + os.environ.get('LD_LIBRARY_PATH', '')
)

# VALIS library imports
from valis import registration
from valis.micro_rigid_registrar import MicroRigidRegistrar
from valis import feature_detectors
from valis import feature_matcher
from valis import preprocessing

# Non-rigid registrars - OpticalFlowWarper is default, NonRigidTileRegistrar for large images
from valis_lib.non_rigid_registrars import OpticalFlowWarper, NonRigidTileRegistrar

# Affine optimizer for post-registration refinement
from valis_lib.affine_optimizer import AffineOptimizerMattesMI

# VALIS exception hierarchy
from valis_lib.exceptions import (
    ValisError,
    SlideReadError,
    RegistrationError,
    FeatureDetectionError,
    WarpingError,
    ResourceError,
    MemoryError as ValisMemoryError,
    wrap_exception,
)



def estimate_jvm_memory(input_dir: str, default_gb: int = 16) -> int:
    """Estimate JVM memory based on input file sizes.

    Parameters
    ----------
    input_dir : str
        Directory containing input files
    default_gb : int
        Default memory allocation in GB

    Returns
    -------
    int
        Recommended JVM heap size in GB
    """
    total_size_gb = 0.0
    try:
        for f in os.listdir(input_dir):
            if f.lower().endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff')):
                fpath = os.path.join(input_dir, f)
                total_size_gb += os.path.getsize(fpath) / (1024 ** 3)

        # Rule of thumb: JVM needs ~2x the largest file size for processing
        # Minimum 8GB, maximum 32GB
        recommended = max(8, min(32, int(total_size_gb * 2 + 4)))
        log_progress(f"Input files total: {total_size_gb:.1f} GB, recommending {recommended} GB JVM heap")
        return recommended
    except Exception as e:
        log_progress(f"Could not estimate JVM memory: {e}, using default {default_gb} GB")
        return default_gb


def validate_input_slides(input_dir: str) -> Tuple[List[str], List[str]]:
    """Validate input slides before registration.

    Parameters
    ----------
    input_dir : str
        Directory containing input slides

    Returns
    -------
    valid_slides : list of str
        Paths to valid slide files
    invalid_slides : list of str
        Paths to invalid/empty slides with error messages
    """
    valid_slides = []
    invalid_slides = []

    for f in os.listdir(input_dir):
        if not f.lower().endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff')):
            continue

        fpath = os.path.join(input_dir, f)
        try:
            # Quick validation using tifffile
            with tifffile.TiffFile(fpath) as tif:
                if len(tif.pages) == 0:
                    log_progress(f"  WARNING: {f} has no pages, skipping")
                    invalid_slides.append((fpath, "No pages"))
                    continue

                # Check if image is essentially empty
                page = tif.pages[0]
                if page.shape[0] < 10 or page.shape[1] < 10:
                    log_progress(f"  WARNING: {f} is too small ({page.shape}), skipping")
                    invalid_slides.append((fpath, f"Too small: {page.shape}"))
                    continue

            valid_slides.append(fpath)
        except Exception as e:
            log_progress(f"  WARNING: Cannot read {f}: {e}")
            invalid_slides.append((fpath, str(e)))

    return valid_slides, invalid_slides


@dataclass
class WarpResult:
    """Result of a single slide warping operation."""

    slide_name: str
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None


def warp_single_slide(
    slide_name: str,
    slide_obj,
    src_path: str,
    out_path: str,
    use_non_rigid: bool,
) -> WarpResult:
    """Warp a single slide (thread-safe).

    Parameters
    ----------
    slide_name : str
        Name of the slide
    slide_obj : Slide
        VALIS Slide object with registration parameters
    src_path : str
        Path to source slide file
    out_path : str
        Path for output registered slide
    use_non_rigid : bool
        Whether to apply non-rigid transforms

    Returns
    -------
    WarpResult
        Result with success status and any error message
    """
    try:
        slide_obj.warp_and_save_slide(
            src_f=src_path,
            dst_f=out_path,
            level=0,
            non_rigid=use_non_rigid,
            crop=True,
            interp_method="bicubic",
        )
        return WarpResult(slide_name, success=True, output_path=out_path)
    except Exception as e:
        return WarpResult(slide_name, success=False, error=str(e))
    finally:
        gc.collect()


def find_reference_image(
    directory: str,
    required_markers: List[str],
    valid_extensions: Optional[List[str]] = None,
) -> str:
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


def valis_registration(
    input_dir: str,
    out: str,
    reference: Optional[str] = None,
    reference_markers: Optional[List[str]] = None,
    max_processed_image_dim_px: int = 512,
    max_non_rigid_dim_px: int = 2048,
    micro_reg_fraction: float = 0.125,
    num_features: int = 5000,
    max_image_dim_px: int = 4000,
    skip_micro_registration: bool = False,
    parallel_warping: bool = False,
    n_workers: int = 2,
    use_tiled_registration: bool = False,
    tile_size: int = 512,
    use_affine_optimizer: bool = True,
    image_type: str = "auto",
) -> int:
    """Perform VALIS registration on preprocessed images.

    Parameters
    ----------
    input_dir : str
        Directory containing preprocessed OME-TIFF files
    out : str
        Output directory for registered slides
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
    parallel_warping : bool, optional
        Enable parallel slide warping using ThreadPoolExecutor. Default: False
    n_workers : int, optional
        Number of parallel workers for warping. Default: 2
    use_tiled_registration : bool, optional
        Use NonRigidTileRegistrar for very large images. Processes tiles in parallel
        which improves performance without sacrificing accuracy. Default: False
    tile_size : int, optional
        Tile size for tiled registration. Default: 512
    use_affine_optimizer : bool, optional
        Use AffineOptimizerMattesMI to refine rigid transforms using mutual information.
        Improves accuracy for multi-modal registration. Default: True
    image_type : str, optional
        Image type for preprocessing: "brightfield", "fluorescence", or "auto".
        "auto" attempts to detect based on image characteristics. Default: "auto"

    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Initialize phase reporter for structured progress tracking
    reporter = PhaseReporter()

    # ========================================================================
    # Phase 1: Initialization
    # ========================================================================
    reporter.enter_phase("init")

    # Validate input slides early
    log_progress("Validating input slides...")
    valid_slides, invalid_slides = validate_input_slides(input_dir)
    if not valid_slides:
        raise FileNotFoundError(f"No valid slides found in {input_dir}")
    log_progress(f"  Valid: {len(valid_slides)}, Invalid: {len(invalid_slides)}")

    # Initialize JVM with adaptive memory sizing
    jvm_mem_gb = estimate_jvm_memory(input_dir, default_gb=16)
    log_progress(f"Initializing JVM with {jvm_mem_gb}GB heap...")
    registration.init_jvm(mem_gb=jvm_mem_gb)
    log_progress(f"JVM initialized with {jvm_mem_gb}GB heap")

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
    log_progress("\nInitializing VALIS registration...")
    # Note: pyvips cache is already disabled at module level in valis_lib/registration.py

    log_progress(f"Memory optimization parameters:")
    log_progress(f"  max_processed_image_dim_px: {max_processed_image_dim_px} (controls analysis resolution)")
    log_progress(f"  max_non_rigid_registration_dim_px: {max_non_rigid_dim_px} (controls non-rigid accuracy)")
    log_progress(f"  max_image_dim_px: {max_image_dim_px} (limits cached image size for RAM control)")

    # ========================================================================
    # Configure Non-Rigid Registration Strategy
    # ========================================================================
    # NonRigidTileRegistrar: Processes tiles in parallel, same accuracy as OpticalFlowWarper
    # but better performance for large images. Each tile is registered independently and
    # displacement fields are stitched together.
    if use_tiled_registration:
        log_progress(f"  Non-rigid registrar: NonRigidTileRegistrar (tile_size={tile_size}px)")
        non_rigid_registrar = NonRigidTileRegistrar(tile_wh=tile_size, tile_buffer=100)
    else:
        log_progress(f"  Non-rigid registrar: OpticalFlowWarper (default)")
        non_rigid_registrar = OpticalFlowWarper()

    # ========================================================================
    # Configure Affine Optimizer for Multi-Modal Registration
    # ========================================================================
    # AffineOptimizerMattesMI uses Mattes Mutual Information which is excellent
    # for multi-modal registration (e.g., brightfield + fluorescence).
    # It refines the initial rigid transforms for better accuracy.
    if use_affine_optimizer:
        log_progress(f"  Affine optimizer: AffineOptimizerMattesMI (Mattes MI)")
        affine_optimizer = AffineOptimizerMattesMI
    else:
        log_progress(f"  Affine optimizer: None (using initial transforms only)")
        affine_optimizer = None

    # ========================================================================
    # Configure Image Preprocessing Based on Image Type
    # ========================================================================
    # ColorfulStandardizer: Best for brightfield/IHC images - normalizes staining
    # ChannelGetter: Best for fluorescence - extracts specific channel with adaptive equalization
    if image_type == "brightfield":
        log_progress(f"  Image preprocessing: ColorfulStandardizer (brightfield/IHC)")
        processing_cls = preprocessing.ColorfulStandardizer
        processing_kwargs = {}
    elif image_type == "fluorescence":
        log_progress(f"  Image preprocessing: ChannelGetter (fluorescence)")
        processing_cls = preprocessing.ChannelGetter
        processing_kwargs = {"channel": "dapi", "adaptive_eq": True}
    else:
        # Auto: let VALIS detect and use defaults
        log_progress(f"  Image preprocessing: Auto-detect")
        processing_cls = None
        processing_kwargs = None

    # Build registrar kwargs
    registrar_kwargs = {
        # Reference image
        "reference_img_f": ref_image,
        "align_to_reference": True,
        "crop": "reference",

        # Image size parameters - tuned for memory efficiency
        "max_processed_image_dim_px": max_processed_image_dim_px,
        "max_non_rigid_registration_dim_px": max_non_rigid_dim_px,
        "max_image_dim_px": max_image_dim_px,

        # Feature detection - SuperPoint/SuperGlue (best for multi-modal)
        "feature_detector_cls": feature_detectors.SuperPointFD,
        "matcher": feature_matcher.SuperGlueMatcher(),

        # Non-rigid registration - handles local deformations after rigid alignment
        "non_rigid_registrar_cls": type(non_rigid_registrar),

        # Affine optimizer - refines rigid transforms using mutual information
        "affine_optimizer_cls": affine_optimizer,

        # Micro-rigid registration - high-resolution local alignment
        "micro_rigid_registrar_cls": None if skip_micro_registration else MicroRigidRegistrar,

        # Registration behavior
        "create_masks": True,
    }

    # Add image preprocessing if specified
    if processing_cls is not None:
        registrar_kwargs["img_processing_cls"] = processing_cls
    if processing_kwargs is not None:
        registrar_kwargs["img_processing_kwargs"] = processing_kwargs

    registrar = registration.Valis(input_dir, results_dir, **registrar_kwargs)

    # ========================================================================
    # Perform Registration
    # ========================================================================
    reporter.enter_phase("rigid")
    log_progress("Starting rigid and non-rigid registration...")
    log_progress("This may take 15-45 minutes...")

    try:
        _, _, error_df = registrar.register()
        log_progress("Initial registration completed")
        log_progress(f"\nRegistration errors:\n{error_df}")
    except ValisMemoryError as e:
        log_progress(f"\nERROR: Memory exhausted during registration: {e}")
        raise
    except FeatureDetectionError as e:
        log_progress(f"\nERROR: Feature detection failed: {e}")
        log_progress("Consider adjusting num_features or max_processed_dim parameters")
        raise
    except RegistrationError as e:
        log_progress(f"\nERROR: Registration algorithm failed: {e}")
        raise
    except ValisError as e:
        log_progress(f"\nERROR: VALIS error: {e}")
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "unable to write to memory" in error_msg or "tifffillstrip" in error_msg:
            log_progress("\n" + "=" * 70)
            log_progress("ERROR: pyvips memory allocation failure")
            log_progress("=" * 70)
            log_progress("VALIS cannot load the TIFF files into memory.")
            log_progress("\nPossible causes:")
            log_progress("  1. Files are too large for available RAM")
            log_progress("  2. TIFF files may be corrupted or have format issues")
            log_progress("  3. Files need to be saved with tiling/compression")
            log_progress("\nSuggested fixes:")
            log_progress(f"  1. Increase max_image_dim_px parameter (currently {max_image_dim_px})")
            log_progress("  2. Re-save TIFF files with compression='zlib' and tile=(256,256)")
            log_progress("  3. Ensure preprocessing saves tiles with proper TIFF structure")
            log_progress("=" * 70)
            raise wrap_exception(
                e, ValisMemoryError,
                "VALIS registration failed due to memory/TIFF issue",
                suggestion="Increase max_image_dim_px or re-save TIFFs with compression"
            )
        raise wrap_exception(e, RegistrationError, f"Registration failed: {e}")

    # ========================================================================
    # Micro-registration - Try with error handling
    # ========================================================================
    if skip_micro_registration:
        log_progress("\nSkipping micro-registration (--skip-micro-registration flag set)")
    else:
        reporter.enter_phase("micro")
        log_progress("Attempting micro-registration...")
        log_progress("NOTE: This may fail if SimpleElastix is not properly installed")

        try:
            img_dims = np.array([s.slide_dimensions_wh[0] for s in registrar.slide_dict.values()])
            min_max_size = np.min([np.max(d) for d in img_dims])
            micro_reg_size = int(np.floor(min_max_size * micro_reg_fraction))

            log_progress(f"Micro-registration size: {micro_reg_size}px")
            log_progress("Starting micro-registration (may take 30-120 minutes)...")

            _, micro_error = registrar.register_micro(
                max_non_rigid_registration_dim_px=micro_reg_size,
                reference_img_f=ref_image,
                align_to_reference=True,
            )

            log_progress("Micro-registration completed")
            log_progress(f"\nMicro-registration errors:\n{micro_error}")

        except Exception as e:
            log_progress(f"\nMicro-registration FAILED: {e}")
            log_progress("Continuing without micro-registration...")
            log_progress("(This is usually caused by SimpleElastix not being available)")
    
    # ========================================================================
    # Warp and Save Phase
    # ========================================================================
    reporter.enter_phase("warp")
    log_progress("Preparing to warp slides...")

    # Log registrar state
    log_progress(f"\nRegistrar state:")
    log_progress(f"  - Number of slides: {len(registrar.slide_dict)}")
    log_progress(f"  - Slide dict keys: {list(registrar.slide_dict.keys())}")

    # Check if non-rigid registration succeeded by examining displacement fields
    non_rigid_available = False
    for slide_name, slide_obj in registrar.slide_dict.items():
        has_bk = hasattr(slide_obj, 'bk_dxdy') and slide_obj.bk_dxdy is not None
        has_fwd = hasattr(slide_obj, 'fwd_dxdy') and slide_obj.fwd_dxdy is not None
        has_stored = hasattr(slide_obj, 'stored_dxdy') and slide_obj.stored_dxdy

        if has_bk or has_fwd or has_stored:
            non_rigid_available = True
            log_progress(f"  Non-rigid displacement fields found for: {slide_name}")
            break

    if non_rigid_available:
        log_progress("  Non-rigid registration succeeded - will apply full transforms")
        use_non_rigid = True
    else:
        log_progress("  No non-rigid displacement fields found")
        log_progress("  Falling back to RIGID-ONLY transforms (affine registration)")
        use_non_rigid = False

    # Create output directory
    ensure_dir(out)

    # Build mapping from slide name to original file path
    slide_name_to_path: Dict[str, str] = {}
    for f in registrar.original_img_list:
        basename = os.path.basename(f)
        slide_name = basename.replace('.ome.tif', '').replace('.ome.tiff', '')
        slide_name_to_path[slide_name] = f

    log_progress(f"\nWarping {len(registrar.slide_dict)} slides to: {out}")
    if parallel_warping:
        log_progress(f"  Mode: Parallel (ThreadPoolExecutor, {n_workers} workers)")
    else:
        log_progress(f"  Mode: Sequential")
    log_progress(f"  Transform: {'rigid + non-rigid' if use_non_rigid else 'rigid-only'}")

    # Initialize progress tracker
    tracker = ProgressTracker(
        total_steps=len(registrar.slide_dict),
        operation_name="Slide Warping"
    )
    tracker.start()

    warped_count = 0
    failed_slides: List[Tuple[str, str]] = []

    if parallel_warping and len(registrar.slide_dict) > 1:
        # Parallel warping using ThreadPoolExecutor
        futures = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for slide_name, slide_obj in registrar.slide_dict.items():
                if slide_name not in slide_name_to_path:
                    failed_slides.append((slide_name, "Path not found"))
                    continue
                if slide_obj is None:
                    failed_slides.append((slide_name, "Slide object is None"))
                    continue

                src_path = slide_name_to_path[slide_name]
                out_path = os.path.join(out, f"{slide_name}_registered.ome.tiff")

                future = executor.submit(
                    warp_single_slide,
                    slide_name, slide_obj, src_path, out_path, use_non_rigid
                )
                futures[future] = slide_name

            for future in as_completed(futures):
                slide_name = futures[future]
                result = future.result()

                if result.success:
                    warped_count += 1
                    tracker.step_complete(slide_name, f"Saved: {result.output_path}")
                else:
                    failed_slides.append((slide_name, result.error or "Unknown error"))
                    tracker.step_complete(slide_name, f"FAILED: {result.error}")
    else:
        # Sequential warping with retry logic
        for slide_name, slide_obj in registrar.slide_dict.items():
            # Validate slide
            if slide_name not in slide_name_to_path:
                log_progress(f"  ERROR: Cannot find path for '{slide_name}'")
                failed_slides.append((slide_name, "Path not found"))
                tracker.step_complete(slide_name, "FAILED: Path not found")
                continue

            if slide_obj is None:
                log_progress(f"  ERROR: slide_obj is None for '{slide_name}'")
                failed_slides.append((slide_name, "Slide object is None"))
                tracker.step_complete(slide_name, "FAILED: Slide object is None")
                continue

            src_path = slide_name_to_path[slide_name]
            out_path = os.path.join(out, f"{slide_name}_registered.ome.tiff")

            # Retry context for transient failures (conservative: 2 attempts, 2s delay)
            retry_ctx = RetryContext(
                max_attempts=2,
                delay_seconds=2.0,
                cleanup_func=default_cleanup,
            )

            warp_succeeded = False
            for attempt in retry_ctx:
                try:
                    slide_obj.warp_and_save_slide(
                        src_f=src_path,
                        dst_f=out_path,
                        level=0,
                        non_rigid=use_non_rigid,
                        crop=True,
                        interp_method="bicubic",
                    )
                    warp_succeeded = True
                    warped_count += 1
                    retry_ctx.succeeded()
                    break
                except (WarpingError, ValisMemoryError, OSError) as e:
                    log_progress(f"  Attempt {attempt} failed: {e}")
                    retry_ctx.failed(e)
                except Exception as e:
                    # Non-retryable error
                    log_progress(f"  ERROR warping {slide_name}: {e}")
                    failed_slides.append((slide_name, str(e)))
                    break

            if warp_succeeded:
                tracker.step_complete(slide_name, f"Saved: {out_path}")
            elif retry_ctx.all_attempts_failed:
                failed_slides.append((slide_name, str(retry_ctx.last_exception)))
                tracker.step_complete(slide_name, f"FAILED after retries: {retry_ctx.last_exception}")

            # Memory cleanup after each slide
            if hasattr(slide_obj, 'slide_reader') and hasattr(slide_obj.slide_reader, 'close'):
                try:
                    slide_obj.slide_reader.close()
                except Exception:
                    pass
            gc.collect()

    # Finish progress tracking
    tracker.finish(success=warped_count > 0)

    # Report results
    log_progress(f"\n{'='*70}")
    log_progress(f"Warping Summary:")
    log_progress(f"  Successfully warped: {warped_count}/{len(registrar.slide_dict)}")
    if failed_slides:
        log_progress(f"  Failed slides: {len(failed_slides)}")
        for slide_name, error in failed_slides:
            log_progress(f"    - {slide_name}: {error}")
    log_progress(f"{'='*70}")

    if warped_count == 0:
        log_progress("All slides failed to warp. Registration cannot proceed.")
    elif failed_slides:
        log_progress(f"WARNING: {len(failed_slides)} slides failed, but {warped_count} succeeded")

    log_progress(f"{warped_count} slides warped and saved to: {out}")

    # ========================================================================
    # Cleanup Phase
    # ========================================================================
    reporter.enter_phase("cleanup")
    gc.collect()
    registration.kill_jvm()
    reporter.finish()

    log_progress("\n" + "=" * 70)
    log_progress("REGISTRATION COMPLETED SUCCESSFULLY!")
    log_progress("=" * 70)

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='VALIS registration for WSI processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing preprocessed files')
    parser.add_argument('--out', required=True,
                        help='Output directory for registered slides')

    # Reference image options
    parser.add_argument('--reference', type=str, default=None,
                        help='Filename of reference image (takes precedence over --reference-markers)')
    parser.add_argument('--reference-markers', nargs='+', default=['DAPI', 'SMA'],
                        help='Markers to identify reference image (legacy fallback)')

    # Registration parameters
    parser.add_argument('--max-processed-dim', type=int, default=512,
                        help='Maximum image dimension for rigid registration')
    parser.add_argument('--max-non-rigid-dim', type=int, default=2048,
                        help='Maximum dimension for non-rigid registration')
    parser.add_argument('--micro-reg-fraction', type=float, default=0.125,
                        help='Fraction of image size for micro-registration')
    parser.add_argument('--num-features', type=int, default=5000,
                        help='Number of SuperPoint features to detect')
    parser.add_argument('--max-image-dim', type=int, default=4000,
                        help='Maximum image dimension for caching (controls RAM usage)')
    parser.add_argument('--skip-micro-registration', action='store_true',
                        help='Skip the micro-rigid registration step')

    # Performance options
    parser.add_argument('--parallel-warping', action='store_true',
                        help='Enable parallel slide warping using ThreadPoolExecutor')
    parser.add_argument('--n-workers', type=int, default=2,
                        help='Number of parallel workers for warping')

    # Advanced registration options
    parser.add_argument('--use-tiled-registration', action='store_true',
                        help='Use NonRigidTileRegistrar for large images (parallel tile processing)')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Tile size for tiled registration')
    parser.add_argument('--no-affine-optimizer', action='store_true',
                        help='Disable AffineOptimizerMattesMI (mutual information refinement)')
    parser.add_argument('--image-type', type=str, default='auto',
                        choices=['auto', 'brightfield', 'fluorescence'],
                        help='Image type for preprocessing optimization')

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        return valis_registration(
            input_dir=args.input_dir,
            out=args.out,
            reference=args.reference,
            reference_markers=args.reference_markers,
            max_processed_image_dim_px=args.max_processed_dim,
            max_non_rigid_dim_px=args.max_non_rigid_dim,
            micro_reg_fraction=args.micro_reg_fraction,
            num_features=args.num_features,
            max_image_dim_px=args.max_image_dim,
            skip_micro_registration=args.skip_micro_registration,
            parallel_warping=args.parallel_warping,
            n_workers=args.n_workers,
            # New advanced options
            use_tiled_registration=args.use_tiled_registration,
            tile_size=args.tile_size,
            use_affine_optimizer=not args.no_affine_optimizer,
            image_type=args.image_type,
        )
    except ValisError as e:
        log_progress(f"ERROR: VALIS registration failed: {e}")
        return 1
    except Exception as e:
        log_progress(f"ERROR: Registration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
