#!/usr/bin/env python3
"""VALIS registration for the WSI pipeline: valis-wsi 1.2.0, its default or its hires example.

MIRAGE's VALIS mode is the benchmark's ``valis120`` call (``benchmarks/regbench/methods/
valis.py`` on the benchmarking branch)::

    registrar = registration.Valis(src_dir, dst_dir, img_list=<the patient's slides>,
                                   reference_img_f=<the reference slide>)
    registrar.register(if_processing_kwargs={"channel": 0, "adaptive_eq": True})
    # then every slide is warped at level 0, crop="reference", VALIS's bicubic default

Every constructor and ``register()`` argument is VALIS's default (rigid 512 px, non-rigid
2048 px, ``align_to_reference=False``, VALIS's own detector/matcher, no micro-registration)
with ONE exception: the registration channel. VALIS's default fluorescence processor picks
it by a fuzzy NAME match on ``"dapi"`` (``slide_io.get_channel_index``), falling back to
index 0 with only a warning. The pipeline's conversion puts the nuclear channel at index 0
under whatever name the samplesheet gives it, so the index is passed explicitly and a
decoy channel named like "DAPI_x" can never win. ``adaptive_eq=True`` is VALIS's default.

Not settings, and kept: the pyramid-level -1 reader clamp (``bin/utils/valis_preflight.py``;
the defect is still present in 1.2.0, registration.py:3952-3954), the read-only-$HOME JVM
cache guard, the fixed JVM heap (started through ``slide_io.init_jvm`` because 1.2.0's
``registration.init_jvm`` ignores the size it is given), the staged-QC checkpoint, the
registrar pickle WARP_SEG_QC reads, and writing every channel.

``--valis-mode hires`` (``params.reg_valis_mode``) is valis-wsi 1.2.0's OWN high-resolution
example, as shipped (``docs/examples.rst:122-149`` at v1.2.0; the benchmark's
``valis120_hires`` row, ``benchmarks/regbench/methods/valis.py`` on ``regbench-paper``)::

    registrar = registration.Valis(src_dir, dst_dir, img_list=..., reference_img_f=...,
                                   micro_rigid_registrar_cls=MicroRigidRegistrar)
    registrar.register(if_processing_kwargs={"channel": 0, "adaptive_eq": True})
    registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size,
                             if_processing_kwargs={"channel": 0, "adaptive_eq": True})

with ``micro_reg_size = floor(0.25 x the smallest slide's longest side)``, computed as the
example computes it (``micro_reg_size()``). Nothing else changes: the same channel by index
(passed to ``register_micro`` too, which picks its channel by the same name match), the same
warp in the reference frame. VALIS 1.2.0's known micro-pass defect (on the non-tiled path every
slide's micro image is prepared with the LAST slide's field) is NOT fixed: ruling 2026-11-05,
hires is upstream's example as shipped. ``default`` runs no micro pass at all.

The tuned configuration this replaced (VALIS 1.0.0, SuperPoint/SuperGlue, tiers, micro-
registration at 0.125) is archived at the git tag ``archive/valis-mirage-tuned``.

Usage:
    python register.py --input-dir ./preprocessed --out ./registered --reference panel1.ome.tif
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

__all__ = [
    "main",
    "REGISTRATION_CHANNEL",
    "VALIS_MODES",
    "constructor_kwargs",
    "micro_reg_size",
    "register_kwargs",
    "register_micro_kwargs",
]

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
from valis import registration, slide_io  # noqa: E402
from valis import warp_tools as valis_warp_tools  # noqa: E402
from valis.micro_rigid_registrar import MicroRigidRegistrar  # noqa: E402

# VALIS-free (tifffile only), but named for what it guards; it sits below the
# numba guard with the other valis imports so tests/test_numba_guard.py's
# `^from valis` rule reads it the same way.
from valis_preflight import (  # noqa: E402
    clamp_negative_levels,
    level0_max_dims,
    slides_too_small_for_non_rigid,
    upsample_notice,
)

# The registration channel, by INDEX: the pipeline's conversion always writes the nuclear
# channel first. The one argument that differs from VALIS's default (which matches the
# NAME "dapi"). An int reaches slide_io.get_channel_index unchanged (1.2.0, slide_io.py:1076).
REGISTRATION_CHANNEL = 0


def register_kwargs() -> dict:
    """The exact kwargs ``Valis.register()`` is called with: VALIS's own fluorescence
    processing arguments (``{"channel": "dapi", "adaptive_eq": True}``) with the channel
    replaced by the nuclear channel's index. Nothing else is passed."""
    return {
        "if_processing_kwargs": {"channel": REGISTRATION_CHANNEL, "adaptive_eq": True}
    }


# reg_valis_mode: VALIS's own two configurations, and nothing of MIRAGE's. Mirrored by the
# schema's reg_valis_mode enum (tests/test_register_valis_modes.py).
VALIS_MODES = ("default", "hires")

# valis-wsi 1.2.0 docs/examples.rst:134, the high-resolution example: the micro pass runs at a
# quarter of the smallest slide's longest side.
MICRO_REG_FRACTION = 0.25


def _check_mode(valis_mode: str) -> None:
    if valis_mode not in VALIS_MODES:
        raise ValueError(f"valis mode {valis_mode!r} is not one of {VALIS_MODES}")


def micro_reg_size(slide_dimensions_wh) -> int:
    """The hires example's micro-registration size, statement for statement.

    ``slide_dimensions_wh`` is each slide's ``Slide.slide_dimensions_wh`` (one (w, h) row per
    pyramid level, level 0 first). examples.rst:141-146 at v1.2.0::

        img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in
                             registrar.slide_dict.values()])
        min_max_size = np.min([np.max(d) for d in img_dims])
        micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)
    """
    img_dims = np.array([dims[0] for dims in slide_dimensions_wh])
    min_max_size = np.min([np.max(d) for d in img_dims])
    return int(np.floor(min_max_size * MICRO_REG_FRACTION).astype(int))


def constructor_kwargs(valis_mode: str) -> dict:
    """``Valis(...)``'s arguments beyond the slides and the reference: none at ``default``;
    the hires example's ``micro_rigid_registrar_cls=MicroRigidRegistrar`` at ``hires``."""
    _check_mode(valis_mode)
    if valis_mode == "hires":
        return {"micro_rigid_registrar_cls": MicroRigidRegistrar}
    return {}


def register_micro_kwargs(valis_mode: str, slide_dimensions_wh) -> Optional[dict]:
    """``register_micro()``'s arguments, or None: ``default`` runs no micro pass.

    At ``hires``: the example's ``max_non_rigid_registration_dim_px=micro_reg_size`` plus the
    same registration channel ``register()`` gets (``register_micro`` re-processes the slides
    and would otherwise pick its channel by VALIS's name match on "dapi")."""
    _check_mode(valis_mode)
    if valis_mode != "hires":
        return None
    return {
        "max_non_rigid_registration_dim_px": micro_reg_size(slide_dimensions_wh),
        **register_kwargs(),
    }


def repair_fwd_dxdy(registrar) -> int:
    """Recompute a missing ``fwd_dxdy`` from ``bk_dxdy``; returns how many slides it repaired.

    NonRigidTileRegistrar produces fwd_dxdy as pyvips.Image, but the Slide setter silently
    rejects pyvips types. Run after every non-rigid pass (``register()``, and
    ``register_micro()`` at hires)."""
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
    return repaired_count


def start_jvm(heap_gb: int) -> int:
    """Start Bio-Formats' JVM with ``heap_gb`` before VALIS can start it.

    valis-wsi 1.2.0's ``registration.init_jvm(jar=None, mem_gb=10)`` calls
    ``slide_io.init_jvm(jar=None, mem_gb=10)`` with the LITERAL 10 (registration.py:132-135),
    so a heap handed to it never arrives. Starting it here through ``slide_io.init_jvm`` makes
    ``--jvm-heap-gb`` real; VALIS's later calls find the JVM running and do nothing. The JVM
    cache is pointed off a read-only $HOME first (bin/utils/jvm_cache.py).
    """
    from jvm_cache import point_jvm_cache_off_readonly_home

    point_jvm_cache_off_readonly_home()
    slide_io.init_jvm(mem_gb=int(heap_gb))
    return int(heap_gb)


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
    jvm_heap_gb: int = 10,
    stage_checkpoint_dir: Optional[str] = None,
    valis_mode: str = "default",
) -> int:
    """Register the slides in ``input_dir`` with valis-wsi 1.2.0: its defaults, or its
    high-resolution example (``valis_mode="hires"``).

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
        reference image's FILENAME. The pipeline always passes ``--reference``. When
        neither is given, the alphabetically first slide is used.
    jvm_heap_gb : int, optional
        Bio-Formats JVM heap, GB. Default 10, VALIS's own value.
    stage_checkpoint_dir : str, optional
        Where to snapshot each slide's displacement field after ``register()`` and before
        any micro pass. Consumed by WARP_SEG_QC (reg_qc=2), which reads it to report the
        'non_rigid' stage separately from 'rigid' and, at ``hires``, from 'micro'. At
        ``default`` no micro pass runs, so the checkpoint records that there is no distinct
        'micro' stage. None (default) writes nothing.
    valis_mode : str, optional
        ``"default"`` (``register()`` only) or ``"hires"`` (upstream's high-resolution
        example: ``MicroRigidRegistrar`` in the constructor, then ``register_micro()`` at
        ``micro_reg_size()``). See the module docstring.

    Returns
    -------
    int
        Exit code (0 for success)
    """
    _check_mode(valis_mode)
    reporter = PhaseReporter()

    # ========================================================================
    # Phase 1: Initialization
    # ========================================================================
    reporter.enter_phase("init")

    logger.info("Validating input slides...")
    valid_slides, invalid_slides = validate_input_slides(input_dir)
    if not valid_slides:
        raise FileNotFoundError(f"No valid slides found in {input_dir}")
    logger.info(f"  Valid: {len(valid_slides)}, Invalid: {len(invalid_slides)}")

    logger.info("Initializing JVM...")
    jvm_mem_gb = start_jvm(jvm_heap_gb)
    logger.info(f"JVM initialized with {jvm_mem_gb}GB heap")

    ensure_dir(os.path.dirname(out) or ".")

    # Use output directory as results directory for VALIS internal files
    results_dir = os.path.dirname(os.path.abspath(out)) if os.path.dirname(out) else "."

    logger.info("=" * 70)
    logger.info(
        "VALIS Registration Configuration: "
        + (
            "valis-wsi defaults"
            if valis_mode == "default"
            else "valis-wsi high-resolution example (hires)"
        )
    )
    logger.info("=" * 70)
    logger.info(
        f"  Rigid resolution: {registration.DEFAULT_MAX_PROCESSED_IMG_SIZE}px (default)"
    )
    logger.info(
        f"  Non-rigid resolution: {registration.DEFAULT_MAX_NON_RIGID_REG_SIZE}px (default)"
    )
    if valis_mode == "default":
        logger.info("  align_to_reference: False (default); micro-registration: none")
    else:
        logger.info(
            "  align_to_reference: False (default); MicroRigidRegistrar inside register(), "
            f"then register_micro() at {MICRO_REG_FRACTION} x the smallest slide's longest side"
        )
    logger.info(f"  Valis() extra kwargs: {constructor_kwargs(valis_mode)}")
    logger.info(f"  register() kwargs: {register_kwargs()}")
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
    # Initialize the VALIS registrar: the patient's slides and the reference, nothing else
    # ========================================================================
    logger.info("\nInitializing VALIS registration...")

    # ------------------------------------------------------------------
    # VALIS 1.0.0-1.2.0 asks Bio-Formats for pyramid level -1 when a slide's full
    # resolution is smaller than the source size the non-rigid stage needs -- which
    # depends on the tissue-mask extent, so no size check predicts it. The clamp maps
    # a negative level to 0 at the reader (upstream's one-line fix, applied here), so
    # the slide is read at full resolution and upsampled instead of killing the JVM.
    # It changes nothing on a slide where the level is >= 0.
    # bin/utils/valis_preflight.py carries the chain and why the clamp is safe.
    # ------------------------------------------------------------------
    clamped = clamp_negative_levels(slide_io.BioFormatsSlideReader)
    if clamped:
        logger.info(
            f"  Pyramid-level clamp applied to BioFormatsSlideReader: {clamped}"
        )
    elif not getattr(
        slide_io.BioFormatsSlideReader.slide2image, "_mirage_level_clamp", False
    ):
        logger.warning(
            "  Pyramid-level clamp NOT applied: BioFormatsSlideReader's slide2vips/"
            "slide2image signatures differ from VALIS 1.2.0. A slide smaller than the "
            "source size the non-rigid stage needs will crash VALIS again."
        )
    slide_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff"))
    )
    non_rigid_dim = registration.DEFAULT_MAX_NON_RIGID_REG_SIZE
    too_small = slides_too_small_for_non_rigid(
        level0_max_dims(slide_files), non_rigid_dim
    )
    if too_small:
        logger.warning(upsample_notice(too_small, non_rigid_dim=non_rigid_dim))

    registrar = registration.Valis(
        input_dir,
        results_dir,
        img_list=slide_files,
        reference_img_f=os.path.join(input_dir, ref_image),
        **constructor_kwargs(valis_mode),
    )

    # ========================================================================
    # Perform Registration
    # ========================================================================
    reporter.enter_phase("rigid")
    logger.info("Starting rigid and non-rigid registration...")

    try:
        _, _, error_df = registrar.register(**register_kwargs())
        if error_df is None:
            # Valis.register() never raises: on any exception it prints the traceback,
            # warns with the message, kills the JVM and returns (None, None, None). The
            # real error is on stdout above this line.
            raise RuntimeError(
                "VALIS registration failed inside Valis.register(), which swallowed the "
                "exception, killed the JVM and returned None. The traceback it printed "
                "is on stdout just above. If it reads 'local variable 'tile' referenced "
                "before assignment' next to 'Invalid resolution: -1', the pyramid-level "
                "clamp did not reach the reader -- look for 'Pyramid-level clamp NOT "
                "applied' earlier in this log (bin/utils/valis_preflight.py)."
            )
        logger.info("Registration completed")
        logger.info(f"\nRegistration errors:\n{error_df}")
        # ---- Safety net: repair missing fwd_dxdy ----
        repaired_count = repair_fwd_dxdy(registrar)
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
                "  1. Re-save TIFF files with compression='zlib' and tile=(256,256)"
            )
            logger.info(
                "  2. Ensure preprocessing saves tiles with proper TIFF structure"
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

    # The micro pass, if this mode runs one: None at default. Sized now, from the
    # registrar's slides, so the checkpoint below can record whether one is about to run.
    micro_kwargs = register_micro_kwargs(
        valis_mode, [s.slide_dimensions_wh for s in registrar.slide_dict.values()]
    )

    # ========================================================================
    # Staged-QC checkpoint (reg_qc >= 2)
    # ========================================================================
    # WARP_SEG_QC reads it to report 'non_rigid' separately from 'rigid' (and, at hires,
    # from 'micro'). This is the only instant the post-non-rigid, pre-micro state exists:
    # register_micro() composes its residual into the same fields destructively. At default
    # no micro pass runs, so the field saved here IS the final field, and the checkpoint
    # says so (micro_registration=False): the QC then reports no 'micro' stage that would be
    # a byte-for-byte duplicate of 'non_rigid'. At hires it says a micro pass is about to
    # run, and is corrected after the fact if register_micro() turns out to be a no-op.
    # slide.M is final here at both modes: MicroRigidRegistrar runs INSIDE register().
    #
    # Never allowed to fail a registration: this is QC input, and QC here is non-gating.
    if stage_checkpoint_dir:
        try:
            from stage_checkpoint import write_checkpoint

            manifest = write_checkpoint(
                registrar,
                stage_checkpoint_dir,
                micro_registration=micro_kwargs is not None,
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
    # Micro-registration: hires only (upstream's high-resolution example)
    # ========================================================================
    if micro_kwargs is not None:
        # register_micro() re-runs measure_error() and OVERWRITES {name}_summary.csv with
        # post-micro numbers, so the pre-micro feature TRE is saved first: the QC report
        # reads *_summary_premicro.csv to isolate VALIS's own non-rigid TRE. A QC artifact:
        # never fails the registration.
        try:
            premicro_f = os.path.join(
                registrar.data_dir, f"{registrar.name}_summary_premicro.csv"
            )
            error_df.to_csv(premicro_f, index=False)
            logger.info(f"Saved the pre-micro error summary to {premicro_f}")
        except Exception as e:
            logger.warning(f"[WARN] Could not save the pre-micro error summary: {e}")

        reporter.enter_phase("micro")
        logger.info(
            "Starting register_micro("
            + ", ".join(f"{k}={v!r}" for k, v in micro_kwargs.items())
            + ")"
        )
        # Not caught: a failure here fails the task (and climbs REGISTER's retry ramp)
        # rather than shipping a hires run that silently lacks its micro pass.
        micro_registrar, micro_error = registrar.register_micro(**micro_kwargs)
        # An early return hands back (None, error_df): VALIS's non-rigid pass already ran at
        # this size or larger (slides under 4 x 2048 px), so nothing was registered.
        micro_ran = micro_registrar is not None
        if micro_ran:
            logger.info(f"Micro-registration completed\n{micro_error}")
            repaired = repair_fwd_dxdy(registrar)
            if repaired:
                logger.info(
                    f"  Repaired fwd_dxdy for {repaired} slides after register_micro"
                )
        else:
            logger.warning(
                "register_micro() returned without registering: the first non-rigid pass "
                "already ran at this size or larger, so the result is register()'s."
            )
        if stage_checkpoint_dir:
            from stage_checkpoint import set_micro_registration

            set_micro_registration(stage_checkpoint_dir, micro_ran)

    # ========================================================================
    # Warp and Save Phase
    # ========================================================================
    reporter.enter_phase("warp")
    logger.info("Preparing to warp slides...")

    # Log registrar state
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
                "The pyramid-level -1 crash (a slide smaller than the source size the"
            )
            logger.error(
                "non-rigid stage needs) is clamped at the reader; if this log says the"
            )
            logger.error(
                "clamp was NOT applied, that is the cause (bin/utils/valis_preflight.py)."
            )
            logger.error("Otherwise: the tiled backend, registration_method = 'tiled'.")
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
                    # explicit, not crop=True (which resolves to the registrar's
                    # "reference"): the frame every warp of this registrar must also use
                    crop="reference",
                )
                warp_succeeded = True
                warped_count += 1
                # Interpolation is VALIS's default (bicubic). Any small negative
                # overshoot near sharp edges is clipped to 0 downstream by
                # clip_negative_values() in split_multichannel.py.
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
        description="VALIS (valis-wsi 1.2.0: defaults or its hires example) registration for WSI processing",
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
    parser.add_argument(
        "--jvm-heap-gb",
        type=int,
        default=10,
        help="Bio-Formats JVM heap in GB (VALIS's own value is 10).",
    )
    parser.add_argument(
        "--valis-mode",
        choices=VALIS_MODES,
        default="default",
        help="'default': register() at valis-wsi's defaults. 'hires': valis-wsi's own "
        "high-resolution example (MicroRigidRegistrar, then register_micro at 0.25 x the "
        "smallest slide's longest side), as shipped.",
    )
    parser.add_argument(
        "--stage-checkpoint-dir",
        type=str,
        default=None,
        help="Snapshot each slide's displacement field after registration, for "
        "WARP_SEG_QC (reg_qc=2) to score the rigid and non-rigid stages separately. "
        "Omit to write nothing.",
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
            jvm_heap_gb=args.jvm_heap_gb,
            stage_checkpoint_dir=args.stage_checkpoint_dir,
            valis_mode=args.valis_mode,
        )
    except Exception as e:
        logger.error(f"[FAIL] Registration failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
