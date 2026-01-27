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
from logger import get_logger
from metadata import get_channel_names

# Module-level logger
logger = get_logger(__name__)
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
from valis.non_rigid_registrars import OpticalFlowWarper, NonRigidTileRegistrar

# Note: AffineOptimizerMattesMI removed - requires SimpleITK with Elastix bindings
# which is not available in this environment. Registration still works via
# SuperPoint/SuperGlue feature matching without the affine optimizer refinement.




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
        # Minimum 8GB, maximum 64GB
        recommended = max(8, min(64, int(total_size_gb * 2 + 4)))
        logger.info(f"Input files total: {total_size_gb:.1f} GB, recommending {recommended} GB JVM heap")
        return recommended
    except Exception as e:
        logger.info(f"Could not estimate JVM memory: {e}, using default {default_gb} GB")
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
                    logger.warning(f"  [WARN] {f} has no pages, skipping")
                    invalid_slides.append((fpath, "No pages"))
                    continue

                # Check if image is essentially empty
                page = tif.pages[0]
                if page.shape[0] < 10 or page.shape[1] < 10:
                    logger.warning(f"  [WARN] {f} is too small ({page.shape}), skipping")
                    invalid_slides.append((fpath, f"Too small: {page.shape}"))
                    continue

            valid_slides.append(fpath)
        except Exception as e:
            logger.warning(f"  [WARN] Cannot read {f}: {e}")
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
    
    logger.info(f"Found {len(image_files)} image files in {directory}")
    
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
        logger.info(f"[OK] Found reference image: {matching_files[0]}")
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
    logger.info("Validating input slides...")
    valid_slides, invalid_slides = validate_input_slides(input_dir)
    if not valid_slides:
        raise FileNotFoundError(f"No valid slides found in {input_dir}")
    logger.info(f"  Valid: {len(valid_slides)}, Invalid: {len(invalid_slides)}")

    # Initialize JVM with adaptive memory sizing
    jvm_mem_gb = estimate_jvm_memory(input_dir, default_gb=16)
    logger.info(f"Initializing JVM with {jvm_mem_gb}GB heap...")
    registration.init_jvm(mem_gb=jvm_mem_gb)
    logger.info(f"JVM initialized with {jvm_mem_gb}GB heap")

    # Configuration
    if reference_markers is None:
        reference_markers = ['DAPI', 'SMA']

    ensure_dir(os.path.dirname(out) or '.')

    # Use output directory as results directory for VALIS internal files
    results_dir = os.path.dirname(out)

    # ========================================================================
    # VALIS Parameters - Use provided values or defaults
    # ========================================================================
    logger.info("=" * 70)
    logger.info("VALIS Registration Configuration")
    logger.info("=" * 70)
    logger.info(f"Rigid resolution: {max_processed_image_dim_px}px")
    logger.info(f"Non-rigid resolution: {max_non_rigid_dim_px}px")
    logger.info(f"Micro-registration fraction: {micro_reg_fraction}")
    logger.info(f"Feature detector: SuperPoint with {num_features} features")
    logger.info("=" * 70)

    # Find reference image
    if reference:
        # Modern approach: use specified reference filename
        logger.info(f"Using specified reference image: {reference}")
        ref_image_path = os.path.join(input_dir, reference)
        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(f"Specified reference image not found: {ref_image_path}")
        ref_image = reference
    else:
        # Legacy approach: search by markers
        logger.info(f"Searching for reference image with markers: {reference_markers}")
        try:
            ref_image = find_reference_image(input_dir, required_markers=reference_markers)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"[FAIL] {e}")
            logger.info("Falling back to first available image")
            files = sorted(glob.glob(os.path.join(input_dir, '*.ome.tif')))
            if not files:
                raise FileNotFoundError(f"No .ome.tif files in {input_dir}")
            ref_image = os.path.basename(files[0])

    logger.info(f"Using reference image: {ref_image}")

    # ========================================================================
    # Initialize VALIS Registrar with Memory Optimization
    # ========================================================================
    logger.info("\nInitializing VALIS registration...")
    # Note: pyvips cache is already disabled at module level in valis_lib/registration.py

    logger.info(f"Memory optimization parameters:")
    logger.info(f"  max_processed_image_dim_px: {max_processed_image_dim_px} (controls analysis resolution)")
    logger.info(f"  max_non_rigid_registration_dim_px: {max_non_rigid_dim_px} (controls non-rigid accuracy)")
    logger.info(f"  max_image_dim_px: {max_image_dim_px} (limits cached image size for RAM control)")

    # ========================================================================
    # Configure Non-Rigid Registration Strategy
    # ========================================================================
    # NonRigidTileRegistrar: Processes tiles in parallel, same accuracy as OpticalFlowWarper
    # but better performance for large images. Each tile is registered independently and
    # displacement fields are stitched together.
    if use_tiled_registration:
        logger.info(f"  Non-rigid registrar: NonRigidTileRegistrar (tile_size={tile_size}px)")
        non_rigid_registrar = NonRigidTileRegistrar(tile_wh=tile_size, tile_buffer=100)
    else:
        logger.info(f"  Non-rigid registrar: OpticalFlowWarper (default)")
        non_rigid_registrar = OpticalFlowWarper()

    # ========================================================================
    # Affine Optimizer - Disabled (requires SimpleITK with Elastix bindings)
    # ========================================================================
    # Note: AffineOptimizerMattesMI is not used because it requires SimpleITK
    # with Elastix bindings (SimpleElastix) which is not available.
    # Registration still works well via SuperPoint/SuperGlue feature matching.
    logger.info(f"  Affine optimizer: None (feature-based alignment only)")

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

        # Affine optimizer - disabled (requires SimpleITK with Elastix)
        "affine_optimizer_cls": None,

        # Micro-rigid registration - high-resolution local alignment
        "micro_rigid_registrar_cls": None if skip_micro_registration else MicroRigidRegistrar,

        # Registration behavior
        "create_masks": True,
    }

    registrar = registration.Valis(input_dir, results_dir, **registrar_kwargs)

    # ========================================================================
    # Perform Registration
    # ========================================================================
    reporter.enter_phase("rigid")
    logger.info("Starting rigid and non-rigid registration...")
    logger.info("This may take 15-45 minutes...")

    try:
        _, _, error_df = registrar.register()
        logger.info("Initial registration completed")
        logger.info(f"\nRegistration errors:\n{error_df}")
    except MemoryError as e:
        logger.error(f"\n[FAIL] Memory exhausted during registration: {e}")
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "unable to write to memory" in error_msg or "tifffillstrip" in error_msg:
            logger.info("\n" + "=" * 70)
            logger.error("[FAIL] pyvips memory allocation failure")
            logger.info("=" * 70)
            logger.info("VALIS cannot load the TIFF files into memory.")
            logger.info("\nPossible causes:")
            logger.info("  1. Files are too large for available RAM")
            logger.info("  2. TIFF files may be corrupted or have format issues")
            logger.info("  3. Files need to be saved with tiling/compression")
            logger.info("\nSuggested fixes:")
            logger.info(f"  1. Increase max_image_dim_px parameter (currently {max_image_dim_px})")
            logger.info("  2. Re-save TIFF files with compression='zlib' and tile=(256,256)")
            logger.info("  3. Ensure preprocessing saves tiles with proper TIFF structure")
            logger.info("=" * 70)
            raise RuntimeError(
                f"VALIS registration failed due to memory/TIFF issue: {e}"
            ) from e
        logger.error(f"\n[FAIL] Registration failed: {e}")
        raise

    # ========================================================================
    # Validate Registration Results
    # ========================================================================
    # Check that all slides have valid transformation matrices (M)
    # If registration failed silently, M will be None and warping will fail
    slides_without_M = []
    for name, slide_obj in registrar.slide_dict.items():
        if slide_obj.M is None:
            slides_without_M.append(name)
        elif not isinstance(slide_obj.M, np.ndarray):
            slides_without_M.append(name)
            logger.warning(f"  Slide {name} has invalid M type: {type(slide_obj.M)}")

    if slides_without_M:
        logger.error(f"\n[FAIL] Registration incomplete - {len(slides_without_M)} slides have no transformation matrix:")
        for name in slides_without_M:
            logger.error(f"    - {name}")
        raise RuntimeError(
            f"Registration failed: {len(slides_without_M)} slides have no transformation matrix (M is None). "
            "This usually indicates the feature matching or rigid registration failed. "
            "Check that input images have sufficient overlap and features."
        )

    logger.info(f"  All {len(registrar.slide_dict)} slides have valid transformation matrices")

    # ========================================================================
    # Micro-registration - Try with error handling
    # ========================================================================
    if skip_micro_registration:
        logger.info("\nSkipping micro-registration (--skip-micro-registration flag set)")
    else:
        reporter.enter_phase("micro")
        logger.info("Attempting micro-registration...")
        logger.info("NOTE: This may fail if SimpleElastix is not properly installed")

        try:
            img_dims = np.array([s.slide_dimensions_wh[0] for s in registrar.slide_dict.values()])
            min_max_size = np.min([np.max(d) for d in img_dims])
            micro_reg_size = int(np.floor(min_max_size * micro_reg_fraction))

            logger.info(f"Micro-registration size: {micro_reg_size}px")
            logger.info("Starting micro-registration (may take 30-120 minutes)...")

            _, micro_error = registrar.register_micro(
                max_non_rigid_registration_dim_px=micro_reg_size,
                reference_img_f=ref_image,
                align_to_reference=True,
            )

            logger.info("Micro-registration completed")
            logger.info(f"\nMicro-registration errors:\n{micro_error}")

        except Exception as e:
            logger.warning(f"\n[WARN] Micro-registration failed: {e}")
            logger.info("Continuing without micro-registration...")
            logger.info("(This is usually caused by SimpleElastix not being available)")
    
    # ========================================================================
    # Warp and Save Phase
    # ========================================================================
    reporter.enter_phase("warp")
    logger.info("Preparing to warp slides...")

    # Log registrewar state
    logger.info(f"\nRegistrar state:")
    logger.info(f"  - Number of slides: {len(registrar.slide_dict)}")
    logger.info(f"  - Slide dict keys: {list(registrar.slide_dict.keys())}")

    # Check if non-rigid registration succeeded by examining displacement fields
    non_rigid_available = False
    for slide_name, slide_obj in registrar.slide_dict.items():
        has_bk = hasattr(slide_obj, 'bk_dxdy') and slide_obj.bk_dxdy is not None
        has_fwd = hasattr(slide_obj, 'fwd_dxdy') and slide_obj.fwd_dxdy is not None
        has_stored = hasattr(slide_obj, 'stored_dxdy') and slide_obj.stored_dxdy

        if has_bk or has_fwd or has_stored:
            non_rigid_available = True
            logger.info(f"  Non-rigid displacement fields found for: {slide_name}")
            break

    if non_rigid_available:
        logger.info("  Non-rigid registration succeeded - will apply full transforms")
        use_non_rigid = True
    else:
        logger.info("  No non-rigid displacement fields found")
        logger.info("  Falling back to RIGID-ONLY transforms (affine registration)")
        use_non_rigid = False

    # Create output directory
    ensure_dir(out)

    # ========================================================================
    # Check JVM Status Before Warping
    # ========================================================================
    # VALIS may have killed JVM during registration (e.g., in error handlers)
    # Once killed, JVM cannot be restarted in the same Python process
    try:
        import jpype
        if not jpype.isJVMStarted():
            logger.error("\n" + "=" * 70)
            logger.error("[FAIL] JVM is not running!")
            logger.error("=" * 70)
            logger.error("The Java Virtual Machine was killed during registration.")
            logger.error("This prevents warping slides because BioFormats requires JVM.")
            logger.error("\nThis typically happens when VALIS encounters an internal error")
            logger.error("during registration and calls kill_jvm() in its exception handler.")
            logger.error("\nThe transformation matrices WERE computed successfully, but")
            logger.error("we cannot warp the slides without JVM for BioFormats I/O.")
            logger.error("\nSuggested workarounds:")
            logger.error("  1. Try --skip-micro-registration flag (micro-reg may be killing JVM)")
            logger.error("  2. Reduce --max-image-dim to lower memory usage")
            logger.error("  3. Check logs above for specific errors that triggered JVM kill")
            logger.error("=" * 70)
            raise RuntimeError(
                "JVM was killed during registration. Warping cannot proceed. "
                "Try --skip-micro-registration or check for errors above."
            )
        logger.info("  JVM is running - proceeding with warping")
    except ImportError:
        logger.warning("  Cannot check JVM status (jpype not available)")

    # Build mapping from slide name to original file path
    slide_name_to_path: Dict[str, str] = {}
    for f in registrar.original_img_list:
        basename = os.path.basename(f)
        slide_name = basename.replace('.ome.tif', '').replace('.ome.tiff', '')
        slide_name_to_path[slide_name] = f

    logger.info(f"\nWarping {len(registrar.slide_dict)} slides to: {out}")
    if parallel_warping:
        logger.info(f"  Mode: Parallel (ThreadPoolExecutor, {n_workers} workers)")
    else:
        logger.info(f"  Mode: Sequential")
    logger.info(f"  Transform: {'rigid + non-rigid' if use_non_rigid else 'rigid-only'}")

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
                if slide_obj.M is None:
                    failed_slides.append((slide_name, "No transformation matrix (M is None)"))
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
                logger.error(f"  [FAIL] Cannot find path for '{slide_name}'")
                failed_slides.append((slide_name, "Path not found"))
                tracker.step_complete(slide_name, "FAILED: Path not found")
                continue

            if slide_obj is None:
                logger.error(f"  [FAIL] slide_obj is None for '{slide_name}'")
                failed_slides.append((slide_name, "Slide object is None"))
                tracker.step_complete(slide_name, "FAILED: Slide object is None")
                continue

            if slide_obj.M is None:
                logger.error(f"  [FAIL] slide '{slide_name}' has no transformation matrix (M is None)")
                failed_slides.append((slide_name, "No transformation matrix (M is None)"))
                tracker.step_complete(slide_name, "FAILED: No M matrix")
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
                except (MemoryError, OSError) as e:
                    logger.info(f"  Attempt {attempt} failed: {e}")
                    retry_ctx.failed(e)
                except Exception as e:
                    # Non-retryable error
                    logger.info(f"  ERROR warping {slide_name}: {e}")
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
    logger.info(f"\n{'='*70}")
    logger.info(f"Warping Summary:")
    logger.info(f"  Successfully warped: {warped_count}/{len(registrar.slide_dict)}")
    if failed_slides:
        logger.info(f"  Failed slides: {len(failed_slides)}")
        for slide_name, error in failed_slides:
            logger.info(f"    - {slide_name}: {error}")
    logger.info(f"{'='*70}")

    if warped_count == 0:
        logger.error("All slides failed to warp. Registration cannot proceed.")
        # Cleanup before exit
        reporter.enter_phase("cleanup")
        gc.collect()
        try:
            registration.kill_jvm()
        except Exception:
            pass  # JVM may already be dead
        reporter.finish()

        logger.error("\n" + "=" * 70)
        logger.error("REGISTRATION FAILED - No slides were warped")
        logger.error("=" * 70)
        return 1
    elif failed_slides:
        logger.warning(f"[WARN] {len(failed_slides)} slides failed, but {warped_count} succeeded")

    logger.info(f"{warped_count} slides warped and saved to: {out}")

    # ========================================================================
    # Cleanup Phase
    # ========================================================================
    reporter.enter_phase("cleanup")
    gc.collect()
    try:
        registration.kill_jvm()
    except Exception:
        pass  # JVM may already be dead
    reporter.finish()

    logger.info("\n" + "=" * 70)
    logger.info("REGISTRATION COMPLETED SUCCESSFULLY!")
    logger.info(f"  {warped_count}/{len(registrar.slide_dict)} slides warped")
    logger.info("=" * 70)

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
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Number of parallel workers for warping')

    # Advanced registration options
    parser.add_argument('--use-tiled-registration', action='store_true',
                        help='Use NonRigidTileRegistrar for large images (parallel tile processing)')
    parser.add_argument('--tile-size', type=int, default=2048,
                        help='Tile size for tiled registration')
    parser.add_argument('--image-type', type=str, default='fluorescence',
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
            # Advanced options
            use_tiled_registration=args.use_tiled_registration,
            tile_size=args.tile_size,
            image_type=args.image_type,
        )
    except Exception as e:
        logger.error(f"[FAIL] Registration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
