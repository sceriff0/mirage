"""STARE stage 1/4 (``stare coarse``): global rigid anchor (M0) + tile plan.

Estimates the whole-slide rigid ``M0`` (moving -> reference) from the DAPI channels and writes the
tile grid the downstream per-tile registration fans out over. One cheap task per moving slide.

The estimate runs on a **thumbnail**, as docs/parallel_registration_design.md always specified.
Both slides are read lazily (zarr region reads, in row bands) and decimated by one shared integer
factor, so peak memory is a band plus the thumbnail rather than the full-resolution plane.

The anchor itself is ``stare.coarse_align.estimate_anchor``: a brute-force rotation sweep with
normalised cross-correlation at 256 px, refined at the ``--max-dim`` thumbnail, with an ORB
fallback, and a loud REFUSAL when neither is trustworthy (see that module's docstring). It is
FFT-based and its memory is a handful of thumbnail-sized canvases: measured peak RSS well under
1 GB at 1024 px (the numbers are in conf/modules.config's TILED_COARSE note). It replaced a DISK +
LightGlue matcher whose U-Net needed ~1.1 + 7.3 * Mpx GB (~32 GB at 2048 px).

``--max-dim`` is the REFINE resolution. The anchor only has to land the moving slide inside the
per-tile read halo (``--halo``, reference-frame px); the per-tile step refines the residual from
there, and SOLVE's robust affine absorbs any global rotation left over. The sweep's quantisation
bound (half a 0.25 deg step at the thumbnail's half-diagonal) is reported as ``coarse_tre`` in
full-res px, so keep ``factor`` small enough that it stays comfortably under ``halo``. Raising it
costs CPU (each of ~50 refine evaluations is a thumbnail warp + two FFTs), not meaningful memory.

``--max-dim`` is REQUIRED and has no default at the stage level. The mirage pipeline always passes
it explicitly, resolved from ``RegPresets.STARE``; a default here would be a fourth, unpinned copy
of that tier value, and ``tests/test_reg_presets_inlined_in_config.py`` pins only the three
Groovy/config copies. ``stare register`` (``stare.cli``) supplies the package's own default. It
also has a floor -- see the argument's own help.

The mirage pipeline invokes this stage through ``bin/tiled_coarse.py``, a shim over ``main``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from stare.coarse_align import (
    CoarseRefused,
    estimate_anchor,
    scale_transform_to_full_res,
)
from stare.log import configure_logging, get_logger
from stare.slide_io import band_rows_for, decimation_factor, open_lazy, read_decimated
from stare.tile_grid import tile_grid

logger = get_logger(__name__)


def _size_gb(path) -> float:
    try:
        return os.path.getsize(path) / 1e9
    except OSError:
        return float("nan")


def _timed_read(src, index, factor, which):
    """Read one decimated DAPI plane, logging how long it took and what came back.

    The intensity summary is not decoration: a blank or saturated nuclear plane (wrong
    ``--nuclear-index``, an empty cycle) is the usual reason the anchor is REFUSED, and printing
    the observed range here puts that diagnosis in `.command.out` next to the refusal.
    """
    t0 = time.perf_counter()
    plane = read_decimated(src, index, factor)
    lo, med, hi = (float(v) for v in np.percentile(plane, (1.0, 50.0, 99.9)))
    logger.info(
        f"coarse: read {which} DAPI in {time.perf_counter() - t0:.1f}s -> "
        f"{plane.shape[1]}x{plane.shape[0]} {plane.dtype} "
        f"({plane.nbytes / 1e6:.0f} MB), intensity p1/p50/p99.9 = "
        f"{lo:.1f}/{med:.1f}/{hi:.1f}"
    )
    return plane


def _finite_or_none(x):
    """JSON has no Infinity: a sweep with no competing peak reports its ratio as null."""
    x = float(x)
    return x if np.isfinite(x) else None


def main(argv=None) -> int:
    """CLI entry point: estimate STARE's global anchor M0 and emit the tile plan.

    Reads the nuclear/fiducial channel of both slides at a thumbnail bounded by
    ``--max-dim``, estimates the rigid anchor (NCC rotation sweep, ORB fallback, or a
    refusal that fails the task), and writes the M0 JSON
    plus the tile-plan CSV that ``stare reg-tile`` fans out over.

    Returns
    -------
    int
        0 on success.
    """
    configure_logging()
    ap = argparse.ArgumentParser(description="STARE coarse anchor + tile plan.")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--moving", required=True)
    ap.add_argument(
        "--nuclear-index",
        "--dapi-index",  # deprecated alias, kept so hand-run commands keep working
        dest="nuclear_index",
        type=int,
        default=0,
        help=(
            "Index of the nuclear/fiducial channel the transform is estimated from. "
            "The pipeline resolves this from channel metadata (MarkerUtils) and passes "
            "it explicitly; 0 is CONVERT_IMAGE's promoted position."
        ),
    )
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--halo", type=int, default=256)
    ap.add_argument(
        "--max-dim",
        type=int,
        # REQUIRED, no default: the pipeline resolves this from RegPresets.STARE and always
        # passes it. A default here would be a fourth copy of the tier value that nothing pins.
        required=True,
        help=(
            "longest thumbnail side (px) the anchor is refined on. Must be >= 16. There is "
            "deliberately NO 'disable decimation' value: every refine step warps and FFTs the "
            "whole thumbnail, so a full-resolution plane is hours of CPU and tens of GB."
        ),
    )
    # Rigid only: SOLVE's robust affine absorbs any residual scale or shear, and a sweep over
    # scale as well as angle would multiply the cost for nothing the per-tile stage needs.
    ap.add_argument("--model", default="euclidean", choices=["euclidean"])
    ap.add_argument("--out-m0", required=True, help="output M0 JSON (+ reference dims)")
    ap.add_argument("--out-tiles", required=True, help="output tile-plan CSV")
    a = ap.parse_args(argv)

    # The hazard gate for a HAND-RUN. On the pipeline path ParamUtils.validateRegPresets
    # enforces a 256 px floor before any task is instantiated; nothing guards this stage when
    # it is invoked directly. slide_io.decimation_factor() reads `max_dim <= 0` as "no
    # decimation" and returns factor 1, handing the anchor the full-resolution plane. 16 rather
    # than 256 because a small thumbnail is a legitimate thing for a test or a probe to ask for.
    if a.max_dim < 16:
        ap.error(
            f"--max-dim {a.max_dim} is below the 16 px floor. There is no value that disables "
            "decimation: the anchor warps and FFTs the whole plane it is handed, so the "
            "full-resolution plane is not a slow run, it is an unschedulable one."
        )

    t_start = time.perf_counter()
    logger.info(
        f"coarse: start | reference={Path(a.reference).name} "
        f"({_size_gb(a.reference):.2f} GB) moving={Path(a.moving).name} "
        f"({_size_gb(a.moving):.2f} GB) | nuclear_index={a.nuclear_index} model={a.model} "
        f"max_dim={a.max_dim} tile={a.tile} halo={a.halo}"
    )

    # Nested acquisition (not two opens then one try/finally) so a failure opening the moving
    # slide still closes the already-open reference handle -- the reg_tile stage's pattern.
    ref_src, _ref_dtype, ref_close = open_lazy(a.reference)
    try:
        mov_src, _mov_dtype, mov_close = open_lazy(a.moving)
        try:
            _c, h, w = (
                ref_src.shape
            )  # FULL-resolution reference dims: the tile plan's frame
            _mc, mh, mw = mov_src.shape
            # ONE factor for both slides: the anchor correlates the two thumbnails pixel for
            # pixel, so a per-slide factor would introduce a scale change a rigid M0 cannot
            # represent at all.
            factor = decimation_factor([(h, w), (mh, mw)], a.max_dim)
            logger.info(
                f"coarse: reference {w}x{h} C={_c} {_ref_dtype} | "
                f"moving {mw}x{mh} C={_mc} {_mov_dtype} | "
                f"decimation 1/{factor} -> ~{w // factor}x{h // factor} thumbnail, "
                f"streamed in {band_rows_for(w, factor)}-row bands"
            )
            # `not 0 <= i < C`, not `i >= C`: a negative index passes the upper-bound
            # test and then silently reads the LAST channel via Python's wraparound.
            if not 0 <= a.nuclear_index < min(_c, _mc):
                # read_decimated raises on this, but naming BOTH channel counts up front turns
                # "index out of range for C=3" into an actionable message.
                logger.warning(
                    f"coarse: --nuclear-index {a.nuclear_index} is out of range for "
                    f"reference C={_c} / moving C={_mc}"
                )
            ref_nuc = _timed_read(ref_src, a.nuclear_index, factor, "reference")
            mov_nuc = _timed_read(mov_src, a.nuclear_index, factor, "moving")
        finally:
            mov_close()
    finally:
        ref_close()

    t0 = time.perf_counter()
    try:
        anchor = estimate_anchor(ref_nuc, mov_nuc, model=a.model)
    except CoarseRefused as exc:
        # A wrong anchor fails NOTHING downstream -- the per-tile reads land in the wrong place
        # and the slide is published mis-registered with exit 0 -- so an anchor nobody can
        # vouch for fails the task here, naming both slides.
        raise CoarseRefused(
            f"coarse: REFUSED to anchor moving={Path(a.moving).name} onto "
            f"reference={Path(a.reference).name} (nuclear_index={a.nuclear_index}, "
            f"thumbnail 1/{factor}): {exc}"
        ) from exc
    logger.info(
        f"coarse: anchor estimated in {time.perf_counter() - t0:.1f}s via {anchor.method}"
    )
    m0_ds, coarse_tre_ds, n_inliers = anchor.M, anchor.residual_px, anchor.n_inliers
    # The fit lives in thumbnail pixels; everything downstream (tile plan, per-tile source
    # regions, the stitch) is full-resolution, so lift both the map and its residual here.
    m0 = scale_transform_to_full_res(m0_ds, factor)
    coarse_tre = float(coarse_tre_ds) * factor

    Path(a.out_m0).write_text(
        json.dumps(
            {
                "M0": m0.tolist(),
                "ref_h": int(h),
                "ref_w": int(w),
                "ref_name": Path(a.reference).stem,
                "coarse_tre": float(coarse_tre),
                "n_inliers": int(n_inliers),
                "coarse_factor": int(factor),
                # How the anchor was obtained and how sure it is. `coarse_tre` is the ORB
                # inlier RMS for "orb", and the search grid's quantisation bound for
                # "ncc_sweep" (which has no correspondences; n_inliers is 0 there).
                "coarse_method": anchor.method,
                "coarse_peak_ncc": float(anchor.peak_ncc),
                "coarse_peak_ratio": _finite_or_none(anchor.peak_ratio),
                "coarse_angle_deg": float(anchor.angle_deg),
            },
            indent=2,
        )
    )

    logger.info(
        f"coarse: M0 dx={m0[0, 2]:+.1f}px dy={m0[1, 2]:+.1f}px "
        f"rot={np.degrees(np.arctan2(m0[1, 0], m0[0, 0])):+.2f}deg | "
        f"method={anchor.method} peak_ncc={anchor.peak_ncc:.3f} "
        f"ratio={anchor.peak_ratio:.2f} n_inliers={n_inliers} "
        f"residual={coarse_tre:.2f}px full-res "
        f"({coarse_tre_ds:.2f} thumbnail px x {factor})"
    )
    # The anchor's only job is to land the moving slide inside the per-tile read halo; the
    # per-tile step refines from there. Nothing downstream checks that, and a residual wider
    # than the halo does not fail -- it silently produces tiles whose true match lies outside
    # the region that was read, i.e. a mis-registered slide that still exits 0.
    if not np.isfinite(coarse_tre):
        logger.warning("coarse: residual not finite -- the anchor fit did not converge")
    elif coarse_tre >= a.halo:
        logger.warning(
            f"coarse: residual {coarse_tre:.1f}px >= halo {a.halo}px -- the per-tile step may "
            f"not recover this. Raise --halo (reg_tiled_halo) or --max-dim "
            f"(reg_tiled_coarse_max_dim, currently decimating 1/{factor})"
        )
    # Only meaningful for the ORB fallback: the sweep has no correspondences and reports 0.
    if anchor.method == "orb" and n_inliers < 10:
        logger.warning(
            f"coarse: only {n_inliers} inliers support M0 -- treat this slide's anchor as "
            f"unverified (low tissue texture, wrong --nuclear-index, or a failed match)"
        )

    tiles = tile_grid(w, h, a.tile, a.halo)
    with open(a.out_tiles, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            ["ix", "iy", "cx", "cy", "x0", "y0", "x1", "y1", "rx0", "ry0", "rx1", "ry1"]
        )
        for t in tiles:
            wr.writerow([t.ix, t.iy, t.cx, t.cy, *t.core, *t.read])

    # The tile count IS the downstream fan-out width: TILED_REG_TILE runs once per row here,
    # so this number is what a reader needs to size the rest of the patient's registration.
    logger.info(
        f"coarse: done in {time.perf_counter() - t_start:.1f}s | "
        f"{len(tiles)} tiles ({a.tile}px core + {a.halo}px halo) over {w}x{h} "
        f"-> {len(tiles)} TILED_REG_TILE tasks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
