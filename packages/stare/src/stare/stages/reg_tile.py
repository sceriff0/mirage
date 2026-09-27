"""STARE stage 2/4 (``stare reg-tile``): one tile's residual (the embarrassingly-parallel part).

Given the global M0 and a tile's core, rigid-warps the reference-frame read box of the moving
DAPI (core + 3 x stride, see ``vector_grid.read_box``) and measures a GRID of window vectors on
the slide-global lattice (``stare.vector_grid``: two passes, foreground-masked, one vector per
owned node with its peak ratio and sharpness). The control JSON keeps the one-point-per-tile
keys (their values are the median of the tile's valid vectors, for the per-tile TRE heatmap)
and adds ``lattice`` and ``vectors``, which SOLVE consumes. One task per tile — this is the little-process fan-out, so unlike
the stitch stage (one process for the whole slide) this runs N times over. It reads through the
same lazy zarr-region primitives (``open_lazy`` + ``source_region``) the stitch uses, so each
invocation decodes only the reference tile and the small moving crop the tile's inverse map draws
from — never the whole slide.

Two ways to name the tile, producing the identical control JSON:

* the explicit geometry (``--ix --iy --cx --cy --rx0 --ry0 --rx1 --ry1`` plus the core
  ``--x0 --y0 --x1 --y1``), which is what the mirage pipeline's TILED_REG_TILE renders from one
  row of the tile plan; or
* ``--plan tiles.csv --row N``, row ``N`` (0-based, header excluded) of the tile plan
  ``stare coarse`` wrote -- for a SLURM array job or any engine that only has an integer index.

The mirage pipeline invokes this stage through ``bin/tiled_reg_tile.py``, a shim over ``main``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from stare.log import configure_logging, get_logger
from stare.slide_io import open_lazy
from stare.tile_residual import foreground_fraction
from stare.vector_grid import estimate_tile_vectors, read_box
from stare.warp import source_region, warp_image

logger = get_logger(__name__)


def _check_nuclear_index(index, c_n):
    """Same out-of-range check + message ``slide_io.nuclear_channel`` raised, for a lazy source."""
    if not 0 <= index < c_n:
        raise ValueError(f"--nuclear-index {index} out of range for C={c_n}")


TILE_FIELDS = ("ix", "iy", "cx", "cy", "rx0", "ry0", "rx1", "ry1")
# The tile's CORE (the half-open region it owns; cores partition the slide). Optional on the
# explicit form so a hand-run command written before the vector grid keeps working -- without
# it the read box stands in for the core and neighbouring tiles emit the nodes in their shared
# halo twice, which SOLVE averages. The plan CSV has always carried these columns.
CORE_FIELDS = ("x0", "y0", "x1", "y1")


def plan_rows(plan_path):
    """The tile plan ``stare coarse`` wrote, as a list of dicts in file order.

    Returns
    -------
    list of dict
        One dict per tile with the CSV's string values; ``plan_row`` types them.
    """
    with open(plan_path, newline="") as f:
        return list(csv.DictReader(f))


def plan_row(plan_path, row):
    """Row ``row`` of the tile plan, typed the way the explicit CLI flags are.

    Parameters
    ----------
    plan_path : str or Path
        The tile-plan CSV.
    row : int
        0-based index, header excluded -- the value a SLURM array task holds.

    Returns
    -------
    dict
        ``{"ix", "iy", "rx0", "ry0", "rx1", "ry1"}`` as ints, ``{"cx", "cy"}`` as floats.
    """
    rows = plan_rows(plan_path)
    if not 0 <= row < len(rows):
        raise IndexError(
            f"--row {row} is out of range for the {len(rows)}-tile plan {plan_path}"
        )
    r = rows[row]
    out = {k: int(r[k]) for k in ("ix", "iy", "rx0", "ry0", "rx1", "ry1")}
    out.update({k: float(r[k]) for k in ("cx", "cy")})
    for k in CORE_FIELDS:
        if r.get(k) not in (None, ""):
            out[k] = int(r[k])
    return out


def _resolve_tile(ap, a):
    """Fill the tile geometry on ``a`` from ``--plan/--row``, or require it explicit."""
    explicit = [
        name for name in TILE_FIELDS + CORE_FIELDS if getattr(a, name) is not None
    ]
    if a.plan is not None or a.row is not None:
        if a.plan is None or a.row is None:
            ap.error("--plan and --row go together")
        if explicit:
            ap.error(
                "--plan/--row and the explicit tile geometry are alternatives; got both "
                f"(explicit: {', '.join('--' + n for n in explicit)})"
            )
        for k, v in plan_row(a.plan, a.row).items():
            setattr(a, k, v)
        return
    missing = [name for name in TILE_FIELDS if getattr(a, name) is None]
    if missing:
        ap.error(
            "the tile geometry is required: either --plan tiles.csv --row N, or all of "
            f"{', '.join('--' + n for n in TILE_FIELDS)} (missing "
            f"{', '.join('--' + n for n in missing)})"
        )


# Both halves of the phase correlation are read at THIS precision. They used to differ -- the
# reference tile float32, the moving crop `dtype=float` (= float64), twelve lines apart -- so one
# measurement was assembled from two precisions and scikit-image promoted the pair internally.
#
# float32, not float64: the source data is uint16 and float32 carries 24 mantissa bits, so the
# extra precision bought nothing while doubling the bytes of the two largest arrays in the task
# (the moving crop and the warped tile). Measured over 6 seeded tiles with a known shift, the
# recovered displacement is identical to every printed digit and the correlation error moves by
# 7e-11 to 1.3e-09. Guarded by
# tests/test_dtype_rounding_contract.py.
TILE_DTYPE = np.float32


def main(argv=None) -> int:
    """CLI entry point: measure one tile's residual displacement against the reference.

    Writes a per-tile control-point JSON carrying the tile's window vectors (what SOLVE
    consumes) and the ``error``/``ref_fg``/``mov_fg`` values -- the only on-disk record
    of those, which is why the artifact is published.

    Returns
    -------
    int
        0 on success.
    """
    configure_logging()
    ap = argparse.ArgumentParser(description="STARE per-tile residual.")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--moving", required=True)
    ap.add_argument("--m0", required=True, help="M0 JSON from tiled_coarse")
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
    # The tile's geometry: EITHER every one of these explicitly (the pipeline's form) OR
    # `--plan tiles.csv --row N`. Not `required=True` on the explicit ones any more, because
    # argparse cannot express "this group or that one"; `_resolve_tile` below enforces it.
    for name in ("--ix", "--iy", "--rx0", "--ry0", "--rx1", "--ry1"):
        ap.add_argument(name, type=int, default=None)
    for name in ("--cx", "--cy"):
        ap.add_argument(name, type=float, default=None)
    ap.add_argument(
        "--plan",
        default=None,
        help="tile-plan CSV written by `stare coarse`; with --row, replaces the explicit "
        "--ix/--iy/--cx/--cy/--rx0/--ry0/--rx1/--ry1 geometry",
    )
    ap.add_argument(
        "--row",
        type=int,
        default=None,
        help="0-based row of --plan (header excluded) naming this tile",
    )
    for name in CORE_FIELDS:
        ap.add_argument(
            f"--{name}",
            type=int,
            default=None,
            help="tile core bound (half-open, reference px); the plan's x0/y0/x1/y1",
        )
    ap.add_argument(
        "--stride",
        type=int,
        default=128,
        help="vector-lattice stride (px); window = 2 x stride. Node k is centred at "
        "W/2 + k*stride in reference-frame pixels, so every tile shares one lattice.",
    )
    ap.add_argument("--out", required=True, help="output control-point JSON")
    a = ap.parse_args(argv)
    _resolve_tile(ap, a)
    if a.stride < 1:
        ap.error(f"--stride must be positive, got {a.stride}")
    core = tuple(getattr(a, k) for k in CORE_FIELDS)
    if any(v is None for v in core):
        if any(v is not None for v in core):
            ap.error("--x0/--y0/--x1/--y1 go together")
        core = (a.rx0, a.ry0, a.rx1, a.ry1)
        logger.warning(
            f"tile ({a.ix},{a.iy}): no core bounds given; using the read box {core} as the "
            "core, so nodes in the halo shared with a neighbour are emitted by both tiles "
            "(SOLVE averages them)"
        )

    m0 = np.asarray(json.loads(Path(a.m0).read_text())["M0"], dtype=float)

    # Lazy zarr-region reads (the tiled_stitch.py pattern): decode only the reference read box
    # and the small moving crop its inverse map draws from, never either whole slide. The
    # acquisitions are nested (not two opens followed by one try/finally) so that if opening the
    # moving slide raises, the already-open reference handle is still closed.
    ref_src, _ref_dtype, ref_close = open_lazy(a.reference)
    try:
        mov_src, _mov_dtype, mov_close = open_lazy(a.moving)
        try:
            _check_nuclear_index(a.nuclear_index, ref_src.shape[0])
            _check_nuclear_index(a.nuclear_index, mov_src.shape[0])
            # The read box is DERIVED from the core and the stride, not the plan's halo: the
            # core plus W/2 (an owned window's reach) plus W (the pass-1 capture margin),
            # clamped to the slide -- see vector_grid.read_box for the memory bound.
            bx0, by0, bx1, by1 = read_box(core, a.stride, ref_src.shape[1:])
            out_h, out_w = by1 - by0, bx1 - bx0
            ref_tile = np.asarray(
                ref_src[a.nuclear_index, slice(by0, by1), slice(bx0, bx1)],
                dtype=TILE_DTYPE,
            )
            _c, mh, mw = mov_src.shape
            sx0, sy0, sx1, sy1 = source_region(
                m0, None, (bx0, by0), (out_h, out_w), src_shape=(mh, mw)
            )
            if sx1 > sx0 and sy1 > sy0:
                crop = np.asarray(
                    mov_src[a.nuclear_index, slice(sy0, sy1), slice(sx0, sx1)],
                    dtype=TILE_DTYPE,
                )
                # warp_image -> resample_bilinear force intensities to float64 internally
                # (mesh_field.py:112), so the warped tile comes back float64 whatever went in.
                # Cast it back so BOTH halves of the correlation below are at TILE_DTYPE.
                mov_tile = warp_image(
                    crop,
                    m0,
                    None,
                    (out_h, out_w),
                    out_origin=(bx0, by0),
                    src_origin=(sx0, sy0),
                ).astype(TILE_DTYPE, copy=False)
            else:
                mov_tile = np.zeros((out_h, out_w), dtype=TILE_DTYPE)
        finally:
            mov_close()
    finally:
        ref_close()

    # A GRID of window vectors on the global lattice, two passes (quarter-res capture, then
    # full-res with a 3-point Gaussian peak), foreground-masked, each with its peak ratio --
    # research/stare-optimal-design-2026-09-27.md §2. The manufactured all-zeros mov_tile
    # (moving crop outside the slide) yields no valid vector: its correlation has no positive
    # peak.
    vec = estimate_tile_vectors(ref_tile, mov_tile, (bx0, by0), core, a.stride)
    vectors = [
        [
            int(v[0]),
            int(v[1]),
            v[2],
            v[3],
            round(v[4], 4),
            round(v[5], 4),
            round(v[6], 3),
            round(v[7], 3),
            v[8],
        ]
        for v in vec["vectors"]
    ]

    # Per-tile summary at the top level: the MEDIAN of the valid vectors, which tre_report's
    # per-tile heatmap reads (`tre`). `error` is the median normalised correlation error
    # (1 - ncc^2, scikit-image's quantity for the same kernel) of those vectors -- recorded,
    # not gated on; with no valid vector the tile reports a zero displacement and error NaN.
    # SOLVE reads only `lattice` and `vectors`.
    if vectors:
        dx = float(np.median([v[4] for v in vectors]))
        dy = float(np.median([v[5] for v in vectors]))
        error = float(np.median(vec["errors"]))
    else:
        dx, dy, error = 0.0, 0.0, float("nan")
    tre = float(np.hypot(dx, dy))

    # PHASE 1 of the foreground work: EMIT, do not gate (tests/test_foreground_fraction.py).
    # Measured on the tile's CORE, the region the tile speaks for; the per-window foreground
    # the vector grid gates on is in each vector's last element.
    cx0, cy0, cx1, cy1 = core
    core_sl = (
        slice(max(0, cy0 - by0), max(0, cy1 - by0)),
        slice(max(0, cx0 - bx0), max(0, cx1 - bx0)),
    )
    ref_fg = foreground_fraction(ref_tile[core_sl])
    mov_fg = foreground_fraction(mov_tile[core_sl])

    Path(a.out).write_text(
        json.dumps(
            {
                "ix": a.ix,
                "iy": a.iy,
                "cx": a.cx,
                "cy": a.cy,
                "dx": dx,
                "dy": dy,
                "tre": tre,
                "error": error,
                "ref_fg": ref_fg,
                "mov_fg": mov_fg,
                "lattice": vec["lattice"],
                "vectors": vectors,
                "rejected": vec["rejected"],
                "pass1": vec["pass1"],
                # fraction of pass-2 windows whose 3-point Gaussian sub-pixel fit fell back
                # to the parabola (a sample <= the local minimum); null: nothing correlated
                "gauss_fallback_rate": vec.get("gauss_fallback_rate"),
            }
        )
    )
    logger.info(
        f"tile ({a.ix},{a.iy}): {len(vectors)}/{len(vectors) + len(vec['rejected'])} vectors "
        f"(stride {a.stride}); median dxy=({dx:.2f},{dy:.2f}) error={error:.4f} "
        f"pass1 {vec['pass1'].get('n_valid', 0)}/{vec['pass1'].get('n', 0)} "
        f"ref_fg={ref_fg:.4f} mov_fg={mov_fg:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
