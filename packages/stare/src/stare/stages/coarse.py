"""STARE stage 1/4 (``stare coarse``): global rigid anchor (M0) + tile plan.

Estimates the whole-slide rigid ``M0`` (moving -> reference) from the DAPI channels and writes the
tile grid the downstream per-tile registration fans out over. One cheap task per moving slide.

The estimate runs on a **thumbnail**, as docs/parallel_registration_design.md always specified.
Both slides are read lazily (zarr region reads, in row bands) and decimated by one shared integer
factor, so peak memory is a band plus the thumbnail rather than the full-resolution plane. This
matters more than it looks: the anchor's matcher (DISK, a U-Net) allocates activations over the
WHOLE plane it is handed, at roughly ``GB ~= 1.1 + 7.3 * Mpx`` -- a 26k x 26k slide at native
resolution would need thousands of GB, so the thumbnail bound is not a tuning knob but the thing
that makes this step runnable at all.

``--max-dim`` is the accuracy/cost knob. The anchor only has to land the moving slide inside the
per-tile read halo (``--halo``, reference-frame px); the per-tile step refines the residual from
there. A coarse fit residual of ``r`` thumbnail px becomes ``r * factor`` full-res px, so keep
``max_dim`` large enough that ``factor`` stays comfortably under ``halo``.

Raising it costs MEMORY, linearly in area, and that is the thing to know before tuning: DISK's
activation memory measures at ``GB ~= 1.1 + 7.3 * Mpx`` on the pinned stack (3.03 GB at 512 px,
8.78 GB at 1024 px), and TILED_COARSE's `memory` request in conf/modules.config is derived from
this same bound. So ``--max-dim`` is not free: 4096 px NEEDS ~123 GB by that fit, and the
request conf/modules.config derives carries a 1.5x headroom factor, so it would ASK for ~185 GB
-- which is why the STARE tier column tops out at 2048. Lower ``--max-dim`` when the step OOMs; raise it only if the
logged residual approaches ``--halo``.

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

from stare.coarse_align import estimate_rigid, scale_transform_to_full_res
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

    The intensity summary is not decoration. DISK is a learned matcher trained on images in
    [0, 1], so a plane arriving in raw uint16 counts is ~4 orders of magnitude outside its
    trained input range and matches far worse than it should, with no error raised (see the
    note in bin/utils/coarse_align.py). Printing the observed range here is what makes that
    class of regression visible in `.command.out` instead of presenting as a silently bad M0.
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


def main(argv=None) -> int:
    """CLI entry point: estimate STARE's global anchor M0 and emit the tile plan.

    Reads the nuclear/fiducial channel of both slides at a thumbnail bounded by
    ``--coarse-max-dim``, matches with DISK+LightGlue, and writes the M0 JSON
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
            "longest thumbnail side (px) the anchor is estimated on. Must be >= 16. There is "
            "deliberately NO 'disable decimation' value: the matcher is a U-Net whose memory "
            "is linear in thumbnail area, so a full-resolution plane needs thousands of GB."
        ),
    )
    ap.add_argument(
        "--model", default="euclidean", choices=["euclidean", "similarity", "affine"]
    )
    ap.add_argument("--out-m0", required=True, help="output M0 JSON (+ reference dims)")
    ap.add_argument("--out-tiles", required=True, help="output tile-plan CSV")
    a = ap.parse_args(argv)

    # The hazard gate for a HAND-RUN. On the pipeline path ParamUtils.validateRegPresets
    # enforces a 256 px floor before any task is instantiated; nothing guards this stage when
    # it is invoked directly. slide_io.decimation_factor() reads `max_dim <= 0` as "no
    # decimation" and returns factor 1, handing DISK the full-resolution plane -- thousands of
    # GB on a whole slide. 16 rather than 256 because DISK's U-Net downsamples by 16, and a
    # small thumbnail is a legitimate thing for a test or a probe to ask for.
    if a.max_dim < 16:
        ap.error(
            f"--max-dim {a.max_dim} is below the 16 px floor. There is no value that disables "
            "decimation: the anchor's matcher is a U-Net whose activation memory is linear in "
            "thumbnail area, so the full-resolution plane is not a slow run, it is an OOM."
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
            # ONE factor for both slides: the matcher matches descriptors across the two
            # thumbnails, so a per-slide factor would introduce a scale change M0 cannot
            # represent honestly.
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
    m0_ds, coarse_tre_ds, n_inliers = estimate_rigid(ref_nuc, mov_nuc, model=a.model)
    logger.info(f"coarse: anchor estimated in {time.perf_counter() - t0:.1f}s")
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
            },
            indent=2,
        )
    )

    logger.info(
        f"coarse: M0 dx={m0[0, 2]:+.1f}px dy={m0[1, 2]:+.1f}px "
        f"rot={np.degrees(np.arctan2(m0[1, 0], m0[0, 0])):+.2f}deg | "
        f"n_inliers={n_inliers} residual={coarse_tre:.2f}px full-res "
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
    if n_inliers < 10:
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
