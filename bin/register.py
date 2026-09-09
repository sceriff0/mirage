#!/usr/bin/env python3
"""VALIS registration script for WSI processing pipeline.

This script performs multi-modal image registration using VALIS (Virtual Alignment
of pathoLogy Image Series). It aligns multiple preprocessed OME-TIFF files and
creates registered outputs for each slide.

Features:
- SuperPoint feature detection with SuperGlue matching
- Optional micro-rigid registration for high-resolution alignment
- Structured error handling with retry logic for transient failures
- Memory-optimized processing for large images
- Progress tracking with ETA estimation

Usage:
    python register.py --input-dir /path/to/preprocessed --out /path/to/output

Example:
    python register.py \\
        --input-dir ./preprocessed \\
        --out ./registered \\
        --reference panel1.ome.tif \\
        --max-image-dim 6000
"""

from __future__ import annotations

# Standard library
import argparse
import gc
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np
import tifffile

# Add utils directory to path before local imports
sys.path.insert(0, str(Path(__file__).parent / "utils"))

# Local utilities
from image_utils import ensure_dir
from logger import get_logger

# Module-level logger
logger = get_logger(__name__)

__all__ = ["main"]

from progress import PhaseReporter, ProgressTracker  # noqa: E402
from retry import RetryContext, default_cleanup  # noqa: E402

# Environment configuration (must be before VALIS imports that use numba)
os.environ["NUMBA_DISABLE_JIT"] = "0"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["LD_LIBRARY_PATH"] = (
    "/usr/local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)
# matplotlib (pulled in transitively by valis) builds a font cache under $HOME by default; on a
# read-only cluster $HOME that warns/stalls. Redirect its config + generic XDG cache to /tmp.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

# VALIS library imports
from valis import registration  # noqa: E402
from valis import warp_tools as valis_warp_tools  # noqa: E402

# AffineOptimizerMattesMI refinement is not used: it requires SimpleITK with Elastix bindings,
# which is not available in this environment. Registration works via SuperPoint/SuperGlue feature
# matching without the affine optimizer refinement.
# Memory mode presets + registrar-kwargs builder, and the JVM-heap sizer, live in the shared
# single-source-of-truth module so there is one heap formula. See bin/utils/valis_config.py.
from valis_config import (  # noqa: E402
    MEMORY_PRESETS,
    build_registrar_kwargs,
    init_jvm,
    resolve_memory_mode,
)

# VALIS-free (tifffile only), but named for what it guards; it sits below the
# numba guard with the other valis_* imports so tests/test_numba_guard.py's
# `^from valis` rule reads it the same way it reads valis_config.
from valis_preflight import (  # noqa: E402
    level0_max_dims,
    refusal_message,
    slides_too_small_for_non_rigid,
)


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
        if not f.lower().endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
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
                    logger.warning(
                        f"  [WARN] {f} is too small ({page.shape}), skipping"
                    )
                    invalid_slides.append((fpath, f"Too small: {page.shape}"))
                    continue

            valid_slides.append(fpath)
        except Exception as e:
            logger.warning(f"  [WARN] Cannot read {f}: {e}")
            invalid_slides.append((fpath, str(e)))

    return valid_slides, invalid_slides


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
        Valid file extensions. Default: ['.tif', '.tiff', '.ome.tif', '.ome.tiff']

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
        valid_extensions = [".tif", ".tiff", ".ome.tif", ".ome.tiff"]

    all_files = os.listdir(directory)
    image_files = [
        f for f in all_files if any(f.lower().endswith(ext) for ext in valid_extensions)
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
    memory_mode: str = "high",
    max_processed_dim: Optional[int] = None,
    max_non_rigid_dim: Optional[int] = None,
    micro_reg_fraction: float = 0.125,
    max_image_dim_px: int = 4000,
    micro_reg: int = 1,
    interp_method: str = "bicubic",
    jvm_heap_gb: Optional[int] = None,
    stage_checkpoint_dir: Optional[str] = None,
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
        Legacy fallback for standalone runs: marker names that must all appear in the
        reference image's FILENAME. No default -- the pipeline always passes
        ``--reference`` (see modules/local/register.nf), and a hardcoded marker pair
        here would be both cohort-specific and filename-based, which the rest of the
        pipeline forbids (channel identity comes from metadata, never the filename).
        When neither ``reference`` nor ``reference_markers`` is given, the
        alphabetically first slide is used.
    memory_mode : str, optional
        Memory preset, from bin/utils/valis_config.py's MEMORY_PRESETS: "high"
        (2048/2048 px), "medium" (1024/1024 px) or "low" (512/512 px). All three use
        SuperPoint + SuperGlue with 5000 features. Default: "high"
    micro_reg_fraction : float, optional
        Fraction of image size for micro-registration. Default: 0.125
    max_image_dim_px : int, optional
        Maximum image dimension for caching (controls RAM usage). Default: 4000
    micro_reg : int, optional
        Micro-registration depth (nested ordinal). 0 = neither micro pass;
        1 = micro-rigid only (``MicroRigidRegistrar`` refines ``slide.M`` inside
        ``register()``, the default — matches the pipeline's ``reg_micro_reg``
        default); 2 = also the micro non-rigid pass (``register_micro()``). VALIS
        controls the two independently, so this ordinal is the single knob that gates both.
    stage_checkpoint_dir : str, optional
        Where to snapshot each slide's forward displacement field after the non-rigid stage
        and before micro-registration. Consumed by WARP_SEG_QC (reg_qc=2) to report the
        'non_rigid' stage separately from 'micro'; VALIS composes the two destructively, so
        the snapshot is the only way to tell them apart. None (default) writes nothing.

    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Unpack memory preset. 'custom' draws its base values from 'high' and then takes whichever
    # per-knob overrides were passed; resolve_memory_mode encodes that and rejects unknown tiers.
    # Explicit `is not None`, never `or`: a deliberate 0 must not be coalesced away.
    preset = MEMORY_PRESETS[resolve_memory_mode(memory_mode)]
    max_processed_image_dim_px = (
        max_processed_dim
        if max_processed_dim is not None
        else preset["max_processed_image_dim_px"]
    )
    max_non_rigid_dim_px = (
        max_non_rigid_dim
        if max_non_rigid_dim is not None
        else preset["max_non_rigid_registration_dim_px"]
    )
    num_features = preset["num_features"]
    feature_detector_cls = preset["feature_detector_cls"]
    matcher = preset["matcher"]
    # New preset keys with fallbacks to CLI args/defaults
    preset_max_image_dim = preset.get("max_image_dim_px", max_image_dim_px)

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

    # Initialize JVM with adaptive memory sizing (shared formula: bin/utils/valis_config.py).
    # jvm_heap_gb, when set, overrides the auto-estimate exactly as before.
    logger.info("Initializing JVM...")
    jvm_mem_gb = init_jvm(input_dir, override_gb=jvm_heap_gb)
    logger.info(f"JVM initialized with {jvm_mem_gb}GB heap")

    ensure_dir(os.path.dirname(out) or ".")

    # Use output directory as results directory for VALIS internal files
    results_dir = os.path.dirname(os.path.abspath(out)) if os.path.dirname(out) else "."

    # ========================================================================
    # VALIS Parameters - Determined by memory_mode preset
    # ========================================================================
    logger.info("=" * 70)
    logger.info("VALIS Registration Configuration")
    logger.info("=" * 70)
    logger.info(f"Memory mode: {memory_mode}")
    if max_processed_dim is not None or max_non_rigid_dim is not None:
        logger.info(
            "  Custom overrides: "
            f"max_processed_dim={max_processed_dim}, max_non_rigid_dim={max_non_rigid_dim}"
        )
    logger.info(f"  Feature detector: {feature_detector_cls.__name__}")
    logger.info(f"  Matcher: {type(matcher).__name__}")
    logger.info(f"  Rigid resolution: {max_processed_image_dim_px}px")
    logger.info(f"  Non-rigid resolution: {max_non_rigid_dim_px}px")
    logger.info(f"  Number of features: {num_features}")
    logger.info(f"Micro-registration fraction: {micro_reg_fraction}")
    logger.info("=" * 70)

    # Find reference image
    if reference:
        # Modern approach: use specified reference filename
        ref_basename = os.path.basename(reference)
        logger.info(f"Using specified reference image: {ref_basename}")
        ref_image_path = os.path.join(input_dir, ref_basename)
        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(
                f"Specified reference image not found: {ref_image_path}"
            )
        ref_image = ref_basename
    elif not reference_markers:
        # Neither a reference filename nor a marker list: take the alphabetically first
        # slide. Deliberately NOT a marker guess -- find_reference_image matches on the
        # FILENAME, so any default marker list here would both hardcode one cohort's
        # panel naming and contradict the pipeline's metadata-only channel identity rule.
        files = sorted(
            set(glob.glob(os.path.join(input_dir, "*.ome.tif")))
            | set(glob.glob(os.path.join(input_dir, "*.ome.tiff")))
        )
        if not files:
            raise FileNotFoundError(f"No .ome.tif or .ome.tiff files in {input_dir}")
        ref_image = os.path.basename(files[0])
        logger.info(
            "No --reference and no --reference-markers given; using the "
            f"alphabetically first slide as reference: {ref_image}"
        )
    else:
        # Legacy approach: search by markers in the filename
        logger.info(f"Searching for reference image with markers: {reference_markers}")
        try:
            ref_image = find_reference_image(
                input_dir, required_markers=reference_markers
            )
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"[FAIL] {e}")
            logger.info("Falling back to first available image")
            files = sorted(
                set(glob.glob(os.path.join(input_dir, "*.ome.tif")))
                | set(glob.glob(os.path.join(input_dir, "*.ome.tiff")))
            )
            if not files:
                raise FileNotFoundError(
                    f"No .ome.tif or .ome.tiff files in {input_dir}"
                )
            ref_image = os.path.basename(files[0])

    logger.info(f"Using reference image: {ref_image}")

    # ========================================================================
    # Initialize VALIS Registrar with Memory Optimization
    # ========================================================================
    logger.info("\nInitializing VALIS registration...")
    # Note: pyvips cache is already disabled at module level in valis_lib/registration.py

    logger.info("Memory optimization parameters:")
    logger.info(
        f"  max_processed_image_dim_px: {max_processed_image_dim_px} (controls analysis resolution)"
    )
    logger.info(
        f"  max_non_rigid_registration_dim_px: {max_non_rigid_dim_px} (controls non-rigid accuracy)"
    )
    logger.info(
        f"  max_image_dim_px: {preset_max_image_dim} (limits cached image size for RAM control)"
    )

    # Non-rigid registrar: OpticalFlowWarper, set via non_rigid_registrar_cls in
    # build_registrar_kwargs below. VALIS internally auto-switches to
    # NonRigidTileRegistrar when estimated memory > 10GB (see TILER_THRESH_GB
    # in valis_lib/registration.py). Explicitly passing NonRigidTileRegistrar
    # triggers a bug: its fwd_dxdy is always a pyvips.Image, but the Slide
    # setter silently rejects pyvips, leaving fwd_dxdy=None. This causes
    # measure_error() to report identical rigid/non-rigid errors.

    # ========================================================================
    # Affine Optimizer - Disabled (requires SimpleITK with Elastix bindings)
    # ========================================================================
    # Note: AffineOptimizerMattesMI is not used because it requires SimpleITK
    # with Elastix bindings (SimpleElastix) which is not available.
    # Registration still works well via SuperPoint/SuperGlue feature matching.
    logger.info("  Affine optimizer: None (feature-based alignment only)")

    # Build registrar kwargs via the shared single-source-of-truth builder (bin/utils/valis_config.py).
    registrar_kwargs = build_registrar_kwargs(
        reference_img_f=ref_image,
        memory_mode=memory_mode,
        micro_reg=micro_reg,
        max_image_dim_px=max_image_dim_px,
        max_processed_dim=max_processed_dim,
        max_non_rigid_dim=max_non_rigid_dim,
    )

    # ------------------------------------------------------------------
    # Preflight: refuse input VALIS 1.0.0-1.2.0 cannot register, BEFORE the
    # JVM starts and the rigid stage runs. A slide whose full resolution is no
    # larger than max_non_rigid_registration_dim_px makes VALIS read it at
    # pyramid level -1; Bio-Formats rejects that, the reader swallows the
    # error, and register() kills the JVM and returns None 158 s later. VALIS's
    # own clamp of that parameter does not prevent it. bin/utils/valis_preflight.py
    # carries the full chain and the reproduction.
    # ------------------------------------------------------------------
    slide_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff"))
    )
    non_rigid_dim = registrar_kwargs["max_non_rigid_registration_dim_px"]
    too_small = slides_too_small_for_non_rigid(
        level0_max_dims(slide_files), non_rigid_dim
    )
    if too_small:
        raise RuntimeError(refusal_message(too_small, non_rigid_dim=non_rigid_dim))
    logger.info(
        f"  Preflight: every slide's full resolution exceeds the non-rigid size ({non_rigid_dim}px)"
    )

    registrar = registration.Valis(input_dir, results_dir, **registrar_kwargs)

    # ========================================================================
    # Perform Registration
    # ========================================================================
    reporter.enter_phase("rigid")
    logger.info("Starting rigid and non-rigid registration...")
    logger.info("This may take 15-45 minutes...")

    try:
        _, _, error_df = registrar.register()
        if error_df is None:
            # Valis.register() never raises: on any exception it prints the traceback,
            # warns with the message, kills the JVM and returns (None, None, None). The
            # real error is on stdout above this line -- if it reads "local variable
            # 'tile' referenced before assignment", a slide was read at pyramid level -1
            # (see bin/utils/valis_preflight.py) and the Bio-Formats exception it hides
            # is a few lines further up.
            raise RuntimeError(
                "VALIS registration failed inside Valis.register(), which swallowed the "
                "exception, killed the JVM and returned None. The traceback it printed "
                "is on stdout just above. A 'local variable 'tile' referenced before "
                "assignment' warning means a slide was read at pyramid level -1 because "
                "its full resolution is smaller than the source size the non-rigid stage "
                "needs (slides only slightly larger than --max-non-rigid-dim can still "
                "hit this through the tissue-mask term); lower --max-non-rigid-dim or use "
                "registration_method = 'tiled'."
            )
        logger.info("Initial registration completed")
        logger.info(f"\nRegistration errors:\n{error_df}")

        # ---- Safety net: repair missing fwd_dxdy ----
        # NonRigidTileRegistrar produces fwd_dxdy as pyvips.Image, but the
        # Slide setter silently rejects pyvips types. Recompute from bk_dxdy.
        repaired_count = 0
        for slide_name, slide_obj in registrar.slide_dict.items():
            if slide_obj.bk_dxdy is not None and slide_obj.fwd_dxdy is None:
                bk = slide_obj.bk_dxdy
                if isinstance(bk, np.ndarray):
                    logger.warning(
                        f"  [{slide_name}] fwd_dxdy is None — recomputing from bk_dxdy inverse"
                    )
                    slide_obj.fwd_dxdy = valis_warp_tools.get_inverse_field(bk)
                    repaired_count += 1
                else:
                    logger.warning(
                        f"  [{slide_name}] fwd_dxdy is None and bk_dxdy is pyvips — cannot repair in-place"
                    )
        if repaired_count > 0:
            logger.info(
                f"  Repaired fwd_dxdy for {repaired_count} slides — re-measuring error"
            )
            error_df = registrar.measure_error()
            logger.info(f"\nCorrected registration errors:\n{error_df}")

        # ---- Displacement field diagnostics ----
        # Diagnose why rigid and non-rigid errors may be identical
        logger.info("\n" + "=" * 70)
        logger.info("DISPLACEMENT FIELD DIAGNOSTICS")
        logger.info("=" * 70)
        ref_slide = registrar.get_ref_slide()
        logger.info(f"Reference slide: {ref_slide.name}")
        logger.info(
            f"Non-rigid bbox: {getattr(registrar, '_non_rigid_bbox', 'NOT SET')}"
        )
        logger.info(
            f"Full displacement shape: {getattr(registrar, '_full_displacement_shape_rc', 'NOT SET')}"
        )

        import pyvips

        def _report_dxdy(name, field_name, dxdy, stored):
            """Report displacement field type, shape, and magnitude."""
            if dxdy is None:
                level = "warning" if field_name == "fwd_dxdy" else "info"
                msg = f"  [{name}] {field_name}: None"
                if field_name == "fwd_dxdy":
                    msg += " *** NON-RIGID POINT WARPING WILL BE SKIPPED ***"
                getattr(logger, level)(msg)
            elif isinstance(dxdy, np.ndarray):
                logger.info(
                    f"  [{name}] {field_name}: numpy shape={dxdy.shape}, "
                    f"max_abs_dx={np.abs(dxdy[0]).max():.4f}, max_abs_dy={np.abs(dxdy[1]).max():.4f}, "
                    f"mean_abs_dx={np.abs(dxdy[0]).mean():.4f}, mean_abs_dy={np.abs(dxdy[1]).mean():.4f}"
                )
            elif isinstance(dxdy, pyvips.Image):
                # pyvips.Image — extract stats without loading entire field into memory
                logger.info(
                    f"  [{name}] {field_name}: pyvips.Image {dxdy.width}x{dxdy.height} bands={dxdy.bands} "
                    f"(stored_dxdy={stored})"
                )
                try:
                    stats = dxdy.stats()
                    # pyvips stats() returns (cols=bands+1, rows=7):
                    # Row 0=min, 1=max, 2=sum, 3=sum^2, 4=mean, 5=sd, 6=count
                    # Col 0=all, Col 1=band0(dx), Col 2=band1(dy)
                    stats_np = np.array(
                        [
                            [stats(x, y)[0] for y in range(stats.height)]
                            for x in range(stats.width)
                        ]
                    )
                    # stats_np shape: (bands+1, 7)  — col 0=all, col 1=dx, col 2=dy; row 0=min,1=max,4=mean
                    if stats_np.shape[0] >= 3:
                        logger.info(
                            f"           dx: min={stats_np[1, 0]:.4f}, max={stats_np[1, 1]:.4f}, mean={stats_np[1, 4]:.4f}"
                        )
                        logger.info(
                            f"           dy: min={stats_np[2, 0]:.4f}, max={stats_np[2, 1]:.4f}, mean={stats_np[2, 4]:.4f}"
                        )
                    else:
                        logger.info(
                            f"           stats shape: {stats_np.shape} (unexpected)"
                        )
                except Exception as e:
                    logger.info(f"           Could not read stats: {e}")
            else:
                logger.info(
                    f"  [{name}] {field_name}: type={type(dxdy).__name__} (stored_dxdy={stored})"
                )

        for slide_name, slide_obj in registrar.slide_dict.items():
            is_ref = slide_name == ref_slide.name
            stored = getattr(slide_obj, "stored_dxdy", False)

            # Check bk_dxdy
            try:
                bk = slide_obj.bk_dxdy
            except Exception as e:
                bk = None
                logger.info(f"  [{slide_name}] bk_dxdy: ERROR reading: {e}")

            # Check fwd_dxdy
            try:
                fwd = slide_obj.fwd_dxdy
            except Exception as e:
                fwd = None
                logger.info(f"  [{slide_name}] fwd_dxdy: ERROR reading: {e}")

            if is_ref:
                logger.info(
                    f"  [{slide_name}] REFERENCE — bk_dxdy={'set' if bk is not None else 'None'}, fwd_dxdy={'set' if fwd is not None else 'None'}"
                )
                continue

            _report_dxdy(slide_name, "bk_dxdy", bk, stored)
            _report_dxdy(slide_name, "fwd_dxdy", fwd, stored)

        logger.info("=" * 70)
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
            logger.info(
                f"  1. Increase max_image_dim_px parameter (currently {max_image_dim_px})"
            )
            logger.info(
                "  2. Re-save TIFF files with compression='zlib' and tile=(256,256)"
            )
            logger.info(
                "  3. Ensure preprocessing saves tiles with proper TIFF structure"
            )
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
        logger.error(
            f"\n[FAIL] Registration incomplete - {len(slides_without_M)} slides have no transformation matrix:"
        )
        for name in slides_without_M:
            logger.error(f"    - {name}")
        raise RuntimeError(
            f"Registration failed: {len(slides_without_M)} slides have no transformation matrix (M is None). "
            "This usually indicates the feature matching or rigid registration failed. "
            "Check that input images have sufficient overlap and features."
        )

    logger.info(
        f"  All {len(registrar.slide_dict)} slides have valid transformation matrices"
    )

    # ========================================================================
    # Staged-QC checkpoint (reg_qc >= 2) - MUST be taken here
    # ========================================================================
    # This is the only instant at which the post-non-rigid, pre-micro state exists.
    # register_micro() updates each slide with fwd_dxdy = fwd_dxdy + micro_residual and
    # writes the result back onto the same attribute, so afterwards the registrar holds one
    # composed field and the intermediate is gone. WARP_SEG_QC needs it to separate the
    # 'non_rigid' stage from the 'micro' stage. slide.M is already final here — the
    # MicroRigidRegistrar runs inside register(), before the non-rigid stage — so only the
    # displacement field has to be saved.
    #
    # Never allowed to fail a registration: this is QC input, and QC here is non-gating.
    if stage_checkpoint_dir:
        try:
            from stage_checkpoint import write_checkpoint

            manifest = write_checkpoint(
                registrar,
                stage_checkpoint_dir,
                micro_registration=(micro_reg >= 2),
            )
            n_fields = sum(1 for e in manifest["slides"].values() if e.get("field"))
            logger.info(
                f"\nWrote staged-QC checkpoint to {stage_checkpoint_dir} "
                f"({n_fields}/{len(manifest['slides'])} slides carry a displacement field)"
            )
            for err in manifest.get("errors", []):
                logger.warning(f"  [WARN] stage checkpoint: {err}")
        except Exception as e:
            logger.warning(f"\n[WARN] Could not write the staged-QC checkpoint: {e}")
            logger.info(
                "Continuing; reg_qc=2 will fall back to reporting rigid vs final only."
            )

    # ========================================================================
    # (B) Persist the PRE-micro feature-TRE before register_micro() clobbers it
    # ========================================================================
    # register_micro() re-runs measure_error() and OVERWRITES {name}_summary.csv with post-micro
    # numbers, and it composes the micro residual into bk_dxdy — so the rewritten non_rigid_D no
    # longer isolates the non-rigid stage. Save the pre-micro error_df now, so the QC report can
    # line VALIS's own non_rigid feature-TRE up against WARP_SEG_QC's non_rigid cell displacement.
    # Only meaningful at level 2 (below it register_micro never runs, so the summary already IS
    # the pre-micro one). Never fail a registration over a QC artifact.
    if micro_reg >= 2 and error_df is not None:
        try:
            premicro_f = os.path.join(
                registrar.data_dir, f"{registrar.name}_summary_premicro.csv"
            )
            error_df.to_csv(premicro_f, index=False)
            logger.info(f"Wrote pre-micro registration summary to {premicro_f}")
        except Exception as e:
            logger.warning(f"[WARN] Could not write pre-micro summary: {e}")

    # ========================================================================
    # Micro-registration (non-rigid, register_micro) - level 2 only
    # ========================================================================
    micro_registration_ran = False
    if micro_reg < 2:
        logger.info(
            f"\nSkipping micro non-rigid registration (micro_reg={micro_reg} < 2)"
        )
    else:
        reporter.enter_phase("micro")
        logger.info("Attempting micro-registration...")
        logger.info("NOTE: This may fail if SimpleElastix is not properly installed")

        try:
            img_dims = np.array(
                [s.slide_dimensions_wh[0] for s in registrar.slide_dict.values()]
            )
            min_max_size = np.min([np.max(d) for d in img_dims])
            micro_reg_size = int(np.floor(min_max_size * micro_reg_fraction))

            logger.info(f"Micro-registration size: {micro_reg_size}px")
            logger.info("Starting micro-registration (may take 30-120 minutes)...")

            _, micro_error = registrar.register_micro(
                max_non_rigid_registration_dim_px=micro_reg_size,
                reference_img_f=ref_image,
                align_to_reference=True,
                tile_wh=2048,
            )

            micro_registration_ran = True
            logger.info("Micro-registration completed")
            logger.info(f"\nMicro-registration errors:\n{micro_error}")

        except Exception as e:
            logger.warning(f"\n[WARN] Micro-registration failed: {e}")
            logger.info("Continuing without micro-registration...")
            logger.info("(This is usually caused by SimpleElastix not being available)")

    # Micro-registration is caught-and-continued above, so whether it actually ran is only
    # knowable here. Correct the checkpoint: if it did not run, the field saved before this
    # block is also the final field, and the QC must not report a 'micro' stage that would be
    # a byte-for-byte duplicate of 'non_rigid'.
    if stage_checkpoint_dir:
        from stage_checkpoint import set_micro_registration

        if (
            set_micro_registration(stage_checkpoint_dir, micro_registration_ran)
            and not micro_registration_ran
        ):
            logger.info(
                "  Staged-QC checkpoint marked: no distinct micro stage to report"
            )

    # ========================================================================
    # Warp and Save Phase
    # ========================================================================
    reporter.enter_phase("warp")
    logger.info("Preparing to warp slides...")

    # Log registrewar state
    logger.info("\nRegistrar state:")
    logger.info(f"  - Number of slides: {len(registrar.slide_dict)}")
    logger.info(f"  - Slide dict keys: {list(registrar.slide_dict.keys())}")

    # Check if non-rigid registration succeeded by examining displacement fields
    # Only check moving slides (reference may have identity displacement fields)
    non_rigid_available = False
    ref_name = registrar.get_ref_slide().name
    for slide_name, slide_obj in registrar.slide_dict.items():
        if slide_name == ref_name:
            continue
        has_bk = hasattr(slide_obj, "bk_dxdy") and slide_obj.bk_dxdy is not None
        has_fwd = hasattr(slide_obj, "fwd_dxdy") and slide_obj.fwd_dxdy is not None
        has_stored = hasattr(slide_obj, "stored_dxdy") and slide_obj.stored_dxdy

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
            logger.error(
                "This prevents warping slides because BioFormats requires JVM."
            )
            logger.error("")
            logger.error(
                "VALIS kills the JVM from Valis.register()'s catch-all, after printing"
            )
            logger.error(
                "the real exception and its traceback to stdout. Look there, not here."
            )
            logger.error("")
            logger.error(
                "Suggested workarounds. The known cause is a slide whose full resolution"
            )
            logger.error(
                "is smaller than the source size the non-rigid stage needs, which VALIS"
            )
            logger.error(
                "then reads at pyramid level -1 (bin/utils/valis_preflight.py):"
            )
            logger.error(
                "  1. Lower --max-non-rigid-dim below the smallest slide, with a margin"
            )
            logger.error(
                "     (pipeline: memory_mode = 'custom', reg_valis_max_non_rigid_dim = N)."
            )
            logger.error("  2. Use the tiled backend: registration_method = 'tiled'.")
            logger.error("=" * 70)
            raise RuntimeError(
                "JVM was killed during registration. Warping cannot proceed. The exception "
                "VALIS swallowed is printed on stdout above the 'JVM is not running' banner."
            )
        logger.info("  JVM is running - proceeding with warping")
    except ImportError:
        logger.warning("  Cannot check JVM status (jpype not available)")

    # Build mapping from slide name to original file path
    slide_name_to_path: Dict[str, str] = {}
    for f in registrar.original_img_list:
        basename = os.path.basename(f)
        slide_name = (
            basename.replace(".ome.tiff", "")
            .replace(".ome.tif", "")
            .replace(".tiff", "")
            .replace(".tif", "")
        )
        slide_name_to_path[slide_name] = f

    logger.info(f"\nWarping {len(registrar.slide_dict)} slides to: {out}")
    logger.info(
        f"  Transform: {'rigid + non-rigid' if use_non_rigid else 'rigid-only'}"
    )

    # Initialize progress tracker
    tracker = ProgressTracker(
        total_steps=len(registrar.slide_dict), operation_name="Slide Warping"
    )
    tracker.start()

    warped_count = 0
    failed_slides: List[Tuple[str, str]] = []

    # Warp each slide sequentially (parallel warping removed).
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
            logger.error(
                f"  [FAIL] slide '{slide_name}' has no transformation matrix (M is None)"
            )
            failed_slides.append((slide_name, "No transformation matrix (M is None)"))
            tracker.step_complete(slide_name, "FAILED: No M matrix")
            continue

        src_path = slide_name_to_path[slide_name]
        if slide_name.endswith("_corrected"):
            out_name = slide_name[: -len("_corrected")]
        else:
            out_name = slide_name
        out_path = os.path.join(out, f"{out_name}_registered.ome.tiff")

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
                    interp_method=interp_method,
                )
                warp_succeeded = True
                warped_count += 1
                # Note: interp_method defaults to "bicubic" (VALIS's own default for
                # warp_and_save_slide/warp_slide/warp_img). Bicubic interpolation can
                # produce small negative overshoot near sharp edges; these are clipped
                # to 0 downstream by clip_negative_values() in split_multichannel.py.
                retry_ctx.succeeded()
                break
            except (MemoryError, OSError) as e:
                logger.info(f"  Attempt {attempt} failed: {e}")
                retry_ctx.failed(e)
            except Exception as e:
                # Non-retryable error
                logger.info(f"  ERROR warping {slide_name}: {e}")
                failed_slides.append((slide_name, str(e)))
                tracker.step_complete(slide_name, f"FAILED (non-retryable): {e}")
                break

        if warp_succeeded:
            tracker.step_complete(slide_name, f"Saved: {out_path}")
        elif retry_ctx.all_attempts_failed:
            # Per-slide loop inside a multi-slide warp pass (`for slide_name,
            # slide_obj in registrar.slide_dict.items():` above): log and
            # continue to the next slide rather than raising. This branch was
            # ALREADY written this way -- it is the partial-failure design --
            # but bin/utils/retry.py's __next__ used to raise last_exception
            # on exhaustion instead of StopIteration, so the `for attempt in
            # retry_ctx:` loop above never completed normally and this elif
            # was unreachable dead code. The whole-task decision still
            # happens after the outer loop: `warped_count == 0` returns 1
            # (total failure), and a non-empty `failed_slides` with at least
            # one success only warns. Do NOT re-raise here -- that would
            # regress a 400+ GB REGISTER task to dying on one bad slide,
            # which is exactly the defect this fix closes.
            failed_slides.append((slide_name, str(retry_ctx.last_exception)))
            tracker.step_complete(
                slide_name, f"FAILED after retries: {retry_ctx.last_exception}"
            )

        # Memory cleanup after each slide
        if hasattr(slide_obj, "slide_reader") and hasattr(
            slide_obj.slide_reader, "close"
        ):
            try:
                slide_obj.slide_reader.close()
            except Exception:
                pass
        gc.collect()

    # Finish progress tracking
    tracker.finish(success=warped_count > 0)

    # Report results
    logger.info(f"\n{'=' * 70}")
    logger.info("Warping Summary:")
    logger.info(f"  Successfully warped: {warped_count}/{len(registrar.slide_dict)}")
    if failed_slides:
        logger.info(f"  Failed slides: {len(failed_slides)}")
        for slide_name, error in failed_slides:
            logger.info(f"    - {slide_name}: {error}")
    logger.info(f"{'=' * 70}")

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
        logger.warning(
            f"[WARN] {len(failed_slides)} slides failed, but {warped_count} succeeded"
        )

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
        description="VALIS registration for WSI processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing preprocessed files"
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for registered slides"
    )

    # Reference image options
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Filename of reference image (takes precedence over --reference-markers)",
    )
    parser.add_argument(
        "--reference-markers",
        nargs="+",
        default=None,
        help=(
            "Legacy fallback for standalone runs: marker names that must all appear in "
            "the reference image's FILENAME. No default -- the pipeline always passes "
            "--reference. Without either flag the alphabetically first slide is used."
        ),
    )

    # Registration parameters
    parser.add_argument(
        "--memory-mode",
        type=str,
        default="high",
        choices=["high", "medium", "low", "custom"],
        help=(
            "Memory preset, from MEMORY_PRESETS in bin/utils/valis_config.py. All three "
            "use SuperPoint + SuperGlue with 5000 features and differ only in the "
            "processed / non-rigid registration dimensions: "
            '"high" 2048/2048 px, "medium" 1024/1024 px, "low" 512/512 px. '
            '"custom" starts from "high" and applies whichever of --max-processed-dim / '
            "--max-non-rigid-dim are given, leaving the rest at the high values."
        ),
    )
    parser.add_argument(
        "--max-processed-dim",
        type=int,
        default=None,
        help=(
            "Override the preset's feature detection/matching working size (px). "
            "Unset = take it from --memory-mode."
        ),
    )
    parser.add_argument(
        "--max-non-rigid-dim",
        type=int,
        default=None,
        help=(
            "Override the preset's non-rigid registration size (px). Unset = take it from "
            "--memory-mode. Must be SMALLER than the full resolution of every slide: VALIS "
            "1.0.0-1.2.0 reads a slide no larger than this at pyramid level -1 and dies "
            "(its own clamp does not prevent it), so register.py refuses such input up "
            "front. Lower it below the smallest slide on small-format input such as TMA "
            "cores, with a margin."
        ),
    )
    parser.add_argument(
        "--micro-reg-fraction",
        type=float,
        default=0.125,
        help="Fraction of image size for micro-registration",
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=4000,
        help="Maximum image dimension for caching (controls RAM usage)",
    )
    parser.add_argument(
        "--micro-reg",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Micro-registration depth (nested): 0=none, 1=micro-rigid only "
        "(refines slide.M) [default], 2=+micro non-rigid (register_micro)",
    )

    # Advanced registration options
    parser.add_argument(
        "--interp-method",
        type=str,
        default="bicubic",
        choices=["bilinear", "bicubic", "nearest"],
        help="Interpolation method for warping. Default is bicubic (VALIS's own "
        "default); any negative overshoot near sharp edges is clipped downstream "
        "by clip_negative_values() in split_multichannel.py.",
    )
    parser.add_argument(
        "--jvm-heap-gb",
        type=int,
        default=None,
        help="Explicit JVM heap size in GB (overrides auto-estimation). "
        "Useful for scaling on retries.",
    )
    parser.add_argument(
        "--stage-checkpoint-dir",
        type=str,
        default=None,
        help="Snapshot each slide's displacement field after the non-rigid "
        "stage and before micro-registration. Needed by WARP_SEG_QC "
        "(reg_qc=2) to score the non_rigid and micro stages separately; "
        "VALIS composes them destructively, so nothing downstream can "
        "recover the intermediate. Omit to write nothing.",
    )

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
            memory_mode=args.memory_mode,
            max_processed_dim=args.max_processed_dim,
            max_non_rigid_dim=args.max_non_rigid_dim,
            micro_reg_fraction=args.micro_reg_fraction,
            max_image_dim_px=args.max_image_dim,
            micro_reg=args.micro_reg,
            # Advanced options
            interp_method=args.interp_method,
            jvm_heap_gb=args.jvm_heap_gb,
            stage_checkpoint_dir=args.stage_checkpoint_dir,
        )
    except Exception as e:
        logger.error(f"[FAIL] Registration failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
