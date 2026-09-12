"""STARE stage 2/4 (``stare reg-tile``): one tile's residual (the embarrassingly-parallel part).

Given the global M0 and a tile's read box, rigid-warps just that reference-frame window of the
moving DAPI, phase-correlates it against the reference window, and emits the tile's control-point
displacement + local TRE. One task per tile — this is the little-process fan-out, so unlike
the stitch stage (one process for the whole slide) this runs N times over. It reads through the
same lazy zarr-region primitives (``open_lazy`` + ``source_region``) the stitch uses, so each
invocation decodes only the reference tile and the small moving crop the tile's inverse map draws
from — never the whole slide.

Two ways to name the tile, producing the identical control JSON:

* the explicit geometry (``--ix --iy --cx --cy --rx0 --ry0 --rx1 --ry1``), which is what the
  mirage pipeline's TILED_REG_TILE renders from one row of the tile plan; or
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
from stare.tile_residual import foreground_fraction, residual_displacement
from stare.warp import source_region, warp_image

logger = get_logger(__name__)


def _check_nuclear_index(index, c_n):
    """Same out-of-range check + message ``slide_io.nuclear_channel`` raised, for a lazy source."""
    if not 0 <= index < c_n:
        raise ValueError(f"--nuclear-index {index} out of range for C={c_n}")


TILE_FIELDS = ("ix", "iy", "cx", "cy", "rx0", "ry0", "rx1", "ry1")


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
    return out


def _resolve_tile(ap, a):
    """Fill the tile geometry on ``a`` from ``--plan/--row``, or require it explicit."""
    explicit = [name for name in TILE_FIELDS if getattr(a, name) is not None]
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
# 7e-11 to 1.3e-09 -- against a gate threshold of 0.99. Guarded by
# tests/test_dtype_rounding_contract.py.
TILE_DTYPE = np.float32


def main(argv=None) -> int:
    """CLI entry point: measure one tile's residual displacement against the reference.

    Writes a per-tile control-point JSON carrying the recovered displacement and
    the ``error``/``ref_fg``/``mov_fg`` values behind STARE's accept/reject gate --
    the only on-disk record of those, which is why the artifact is published.

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
    ap.add_argument("--upsample", type=int, default=10)
    ap.add_argument("--out", required=True, help="output control-point JSON")
    a = ap.parse_args(argv)
    _resolve_tile(ap, a)

    m0 = np.asarray(json.loads(Path(a.m0).read_text())["M0"], dtype=float)

    # Lazy zarr-region reads (the tiled_stitch.py pattern): decode only the reference tile and
    # the small moving crop the tile's inverse map draws from, never either whole slide. The
    # acquisitions are nested (not two opens followed by one try/finally) so that if opening the
    # moving slide raises, the already-open reference handle is still closed.
    ref_src, _ref_dtype, ref_close = open_lazy(a.reference)
    try:
        mov_src, _mov_dtype, mov_close = open_lazy(a.moving)
        try:
            _check_nuclear_index(a.nuclear_index, ref_src.shape[0])
            _check_nuclear_index(a.nuclear_index, mov_src.shape[0])
            out_h, out_w = a.ry1 - a.ry0, a.rx1 - a.rx0
            ref_tile = np.asarray(
                ref_src[a.nuclear_index, slice(a.ry0, a.ry1), slice(a.rx0, a.rx1)],
                dtype=TILE_DTYPE,
            )
            _c, mh, mw = mov_src.shape
            sx0, sy0, sx1, sy1 = source_region(
                m0, None, (a.rx0, a.ry0), (out_h, out_w), src_shape=(mh, mw)
            )
            if sx1 > sx0 and sy1 > sy0:
                crop = np.asarray(
                    mov_src[a.nuclear_index, slice(sy0, sy1), slice(sx0, sx1)],
                    dtype=TILE_DTYPE,
                )
                # warp_image -> resample_bilinear force intensities to float64 internally
                # (mesh_field.py:112), so the warped tile comes back float64 whatever went in.
                # Cast it back so BOTH halves of the correlation below are at TILE_DTYPE.
                # Pushing float32 intensities all the way through the resampler is the right
                # end state -- coordinates want float64, intensities do not -- but that changes
                # the stitch's output path too, so it belongs with the slide_io/stitch seam
                # work rather than here.
                mov_tile = warp_image(
                    crop,
                    m0,
                    None,
                    (out_h, out_w),
                    out_origin=(a.rx0, a.ry0),
                    src_origin=(sx0, sy0),
                ).astype(TILE_DTYPE, copy=False)
            else:
                mov_tile = np.zeros((out_h, out_w), dtype=TILE_DTYPE)
        finally:
            mov_close()
    finally:
        ref_close()

    # `error` is the correlation's confidence, not decoration: the tile is emitted
    # unconditionally (including the manufactured all-zeros mov_tile above, where the moving crop
    # fell outside the slide), and tiled_solve.py's --max-error gate is what keeps a peak found in
    # background or across the section edge out of the deformation mesh.
    dx, dy, tre, error = residual_displacement(ref_tile, mov_tile, upsample=a.upsample)

    # PHASE 1 of the foreground work: EMIT, do not gate. Measured over the dense band sweep
    # (21 blanking fractions x 4 geometries x 3 seeds; 200 accepted configs, 68 of them wrong),
    # a threshold on mov_fg that loses no correct tile still rejects 41/68 of the wrong ones,
    # and the mov_fg/ref_fg ratio 44/68 -- a real, partial separator. It does NOT close the
    # band: the distributions overlap, so about a third survives any per-tile cut and
    # neighbourhood consistency is still required. Nothing reads these keys yet, deliberately:
    # gating changes registration output and needs the dense-sweep acceptance plus a real-slide
    # before/after. Both crops are measured because the ratio separates better than the moving
    # crop alone. Guarded by tests/test_foreground_fraction.py.
    ref_fg = foreground_fraction(ref_tile)
    mov_fg = foreground_fraction(mov_tile)

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
            }
        )
    )
    logger.info(
        f"tile ({a.ix},{a.iy}): dxy=({dx:.2f},{dy:.2f}) tre={tre:.2f}px error={error:.4f} "
        f"ref_fg={ref_fg:.4f} mov_fg={mov_fg:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
