#!/usr/bin/env python3
"""Solve ASHLAR's cross-cycle alignment and emit it as a STARE transform manifest.

ASHLAR is a cross-cycle registrar like VALIS and STARE, but it expresses its answer as a
per-tile PLACEMENT rather than a transform object: ``LayerAligner.positions`` is one
``(y, x)`` pair per tile of the moving cycle. A per-tile displacement sampled on a
rectangular grid is exactly a control-grid field -- which is what
``bin/utils/mesh_field.MeshField`` consumes and what the STARE manifest serializes. So
ASHLAR's output is convertible into the manifest the reg_qc=2 scorer already reads, and it
can be ranked against VALIS and STARE on the SAME segmentation-overlap metric instead of a
parallel table in a different metric family.

(An earlier design concluded reg_qc=2 was out of scope for ASHLAR "because Ashlar does not
produce a VALIS pickle or a STARE manifest". It does produce one -- it just never
serializes it.)

THE DERIVATION. A point ``p`` in the moving slide's native frame lies in tile ``t`` at
offset ``p - mov_meta.positions[t]``; ASHLAR places that offset at
``layer.positions[t] + (p - mov_meta.positions[t])`` in the mosaic frame. The reference
tile ``s = reference_idx[t]`` maps its own native frame into the mosaic by
``ref_aligner.positions[s] - ref_meta.positions[s]``. Undoing that gives the moving point
in the REFERENCE NATIVE frame, and the whole thing collapses to a per-tile translation::

    D(t) = layer.positions[t] - mov_meta.positions[t]
           - (ref_aligner.positions[s] - ref_meta.positions[s])

Algebraically ``D(t) == cycle_offset + shifts[t]`` for accepted tiles, because
``calculate_positions`` builds ``positions`` from exactly those terms. The general form is
used anyway, because ``constrain_positions`` OVERWRITES ``positions`` for discarded tiles
with ``lr.predict(...) + offset`` while leaving ``shifts`` at its raw value -- so reading
``shifts`` directly would report a displacement the mosaic never used.

Decomposed into the manifest's two stages:

  rigid    M0 = translation by the median of D over non-discarded tiles
  refined  mesh residual D(t) - median(D), one control point per tile centre

which is precisely ``lib/WarpBackends.groovy``'s ``tiled`` stage vocabulary
(``native / rigid / refined``), so no new stage names enter the QC.

AXIS ORDER IS A TRAP. Everything in ashlar is ``(y, x)`` -- ``tile_position`` returns
``[row, col] * ...``. ``MeshField`` and ``M0`` are ``(x, y)``. Every crossing is done in
:func:`_yx_to_xy` and nowhere else.

Runs inside ``labsyspharm/ashlar:1.20.0`` (amd64-only). Imports ashlar + numpy; the manifest
assembly is reused from ``bin/utils/tiled_manifest.py`` so this cannot drift from the schema
``tiled_stage_warp.make_warper`` reads.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import warnings
from pathlib import Path

import numpy as np

# RELOCATED from bin/ to benchmarks/: ashlar is a BENCHMARK COMPARATOR, not a pipeline
# backend. dev removed the backend at :fire: 6a54479 ("v1.0.0 ships valis and tiled"),
# and tests/test_ashlar_backend_removed.py keeps it out. The comparator still needs a
# driver, so it lives here, harness-owned, and reaches back into the pipeline's
# bin/utils/ for the SHARED manifest/pixel-size code -- reused rather than copied, so
# the manifest ashlar emits cannot drift from the one STARE emits and the single
# predict_from_manifest reads both.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "bin" / "utils"))

from tiled_manifest import build_manifest, slide_entry  # noqa: E402

logger = logging.getLogger(__name__)

# Fraction of tiles ASHLAR may discard before the result is not worth trusting.
# constrain_positions() replaces a discarded tile's position with the reference aligner's
# linear-model prediction; it never raises. A --maximum-shift set below the true
# cross-cycle drift therefore degrades the registration SILENTLY, and this is the only
# signal that it happened.
DEFAULT_MAX_DISCARD_FRACTION = 0.5


def _yx_to_xy(a):
    """The one and only ``(y, x)`` -> ``(x, y)`` crossing. ``a`` is ``(..., 2)``."""
    return np.asarray(a, dtype=float)[..., ::-1]


def tile_displacements(
    layer_positions,
    moving_meta_positions,
    ref_aligner_positions,
    ref_meta_positions,
    reference_idx,
):
    """Per-tile ``D(t)`` in ``(y, x)``, moving-native -> reference-native.

    Pure array maths, so the derivation above is unit-testable without ashlar, a container
    or an image. See the module docstring for why this uses ``positions`` and not
    ``shifts``.
    """
    layer_positions = np.asarray(layer_positions, dtype=float)
    moving_meta_positions = np.asarray(moving_meta_positions, dtype=float)
    ref_aligner_positions = np.asarray(ref_aligner_positions, dtype=float)
    ref_meta_positions = np.asarray(ref_meta_positions, dtype=float)
    idx = np.asarray(reference_idx, dtype=int)

    ref_frame_shift = ref_aligner_positions[idx] - ref_meta_positions[idx]
    return layer_positions - moving_meta_positions - ref_frame_shift


def decompose(displacements_yx, discard=None):
    """Split ``D`` into a global translation and a per-tile residual, both ``(x, y)``.

    The translation is the MEDIAN over non-discarded tiles, mirroring what
    ``constrain_positions`` itself does to derive its fallback offset. Median rather than
    mean because a handful of tiles landing on a fold or a tear should not drag the rigid
    anchor.
    """
    d_xy = _yx_to_xy(displacements_yx)
    keep = (
        np.ones(len(d_xy), dtype=bool)
        if discard is None
        else ~np.asarray(discard, dtype=bool)
    )
    if not keep.any():
        keep = np.ones(
            len(d_xy), dtype=bool
        )  # every tile discarded: no better estimate exists
    translation = np.median(d_xy[keep], axis=0)
    return translation, d_xy - translation


def translation_affine(translation_xy):
    """3x3 forward affine for a pure translation, in the manifest's ``(x, y)`` convention."""
    tx, ty = (float(v) for v in translation_xy)
    return [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]


def control_grid(n_rows, n_cols, tile_size, stride, translation_xy):
    """Control-node coordinates, in the REFERENCE frame, for a uniform tile grid.

    ``tiled_stage_warp`` samples the mesh at the RIGID position ``M0 . xy``, so the nodes
    are the tile centres pushed through ``M0`` -- here a pure translation. Centres use the
    full ``tile_size``, not each tile's valid extent, which keeps ``grid_x``/``grid_y``
    uniformly spaced and strictly ascending as ``MeshField`` requires.
    """
    tx, ty = (float(v) for v in translation_xy)
    half = tile_size / 2.0
    grid_x = [c * stride + half + tx for c in range(n_cols)]
    grid_y = [r * stride + half + ty for r in range(n_rows)]
    return grid_x, grid_y


def ashlar_entry(displacements_yx, discard, grid, out_shape):
    """One manifest slide entry (``M0`` + ``mesh`` + ``out_shape``) from ASHLAR's positions."""
    n_rows, n_cols = int(grid["n_rows"]), int(grid["n_cols"])
    n = len(displacements_yx)
    if n != n_rows * n_cols:
        raise ValueError(
            f"ashlar returned {n} tile positions but the grid is {n_rows}x{n_cols}="
            f"{n_rows * n_cols}. The manifest reshape is row-major and would silently "
            "mis-place every control point."
        )
    translation, residual_xy = decompose(displacements_yx, discard)
    grid_x, grid_y = control_grid(
        n_rows, n_cols, float(grid["tile_size"]), float(grid["stride"]), translation
    )
    entry = slide_entry(
        translation_affine(translation),
        grid_x,
        grid_y,
        residual_xy.reshape(n_rows, n_cols, 2),
    )
    entry["out_shape"] = [int(out_shape[0]), int(out_shape[1])]
    return entry, translation


def _reader(tile_dir, grid, pixel_size):
    """A FilePatternReader over one retiled cycle.

    The reader is selected by CONSTRUCTOR here rather than by ashlar's
    ``reader|path|key=value`` command-line prefix syntax, since this module drives the
    library directly. ``pixel_size`` is passed explicitly and is NOT optional: it defaults
    to 1.0, and ``max_shift_pixels = max_shift / pixel_size``, so leaving it unset silently
    reinterprets a micron budget as a pixel budget.
    """
    from ashlar.filepattern import FilePatternReader

    return FilePatternReader(
        str(tile_dir),
        grid["pattern"],
        float(grid["overlap"]),
        pixel_size=float(pixel_size),
    )


def _exact_reference_aligner(reg, reader, channel, max_shift_um):
    """An EdgeAligner holding the reference grid's NOMINAL positions, never registered.

    retile.py cuts every tile from one already-stitched slide, so the reference positions
    are exact by construction -- there is nothing for EdgeAligner to find -- and adjacent
    tiles overlap on IDENTICAL pixels. Registering those anyway trips ashlar.utils.nccw,
    which raises when correlation exceeds total amplitude by more than an ABSOLUTE 1e-5,
    as identical bright overlaps do through float rounding (job 6844139: "correlation >
    total_amplitude (diff=1.1444091796875e-05)", tile pair 627/628).

    LayerAligner reads from its reference aligner: ``reader`` (with the thumbnail),
    ``channel``, ``metadata``, ``positions``, ``lr`` (constrain_positions predicts with it)
    and ``centers``; fit_model would also set ``origin``. All are filled here as EdgeAligner
    itself would for a perfect alignment: identity model, origin (0, 0).
    """
    import sklearn.linear_model

    edge = reg.EdgeAligner(
        reader, channel=channel, max_shift=max_shift_um, verbose=False
    )
    edge.make_thumbnail()
    positions = np.asarray(edge.metadata.positions, dtype=float).copy()
    edge.positions = positions
    edge.lr = sklearn.linear_model.LinearRegression().fit(positions, positions)
    edge.lr.coef_, edge.lr.intercept_ = np.eye(2), np.zeros(2)  # exactly the identity
    edge.origin = np.zeros(2)
    edge.centers = positions + np.asarray(edge.metadata.size, dtype=float) / 2
    return edge


def _run_aligners(
    ref_dir,
    ref_grid,
    mov_dir,
    mov_grid,
    channel,
    max_shift_um,
    pixel_size,
    reference_edges="exact",
):
    """Place the reference cycle, then run LayerAligner on the moving cycle.

    ``reference_edges='exact'`` (default) places the reference tiles at their known positions
    (see _exact_reference_aligner); ``'register'`` runs ASHLAR's EdgeAligner on them, as for
    raw microscope tiles.

    ``DataWarning`` is captured rather than printed: ``EdgeAligner.fit_model`` warns
    "Could not align enough edges, proceeding anyway with original stage positions" and
    substitutes the IDENTITY transform when its fitted matrix is degenerate. That path
    produces a confident-looking, meaningless alignment, so the caller must be able to see
    it happened.
    """
    from ashlar import reg

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if reference_edges == "exact":
            edge = _exact_reference_aligner(
                reg, _reader(ref_dir, ref_grid, pixel_size), channel, max_shift_um
            )
        else:
            edge = reg.EdgeAligner(
                _reader(ref_dir, ref_grid, pixel_size),
                channel=channel,
                max_shift=max_shift_um,
                verbose=False,
            )
            edge.run()
        layer = reg.LayerAligner(
            _reader(mov_dir, mov_grid, pixel_size),
            edge,
            channel=channel,
            max_shift=max_shift_um,
            verbose=False,
        )
        layer.run()
    messages = [
        str(w.message) for w in caught if issubclass(w.category, reg.DataWarning)
    ]
    return edge, layer, messages


def _check(edge, layer, max_discard_fraction, warnings_seen):
    """Fail or warn on the ways ASHLAR degrades without raising. Returns a diagnostics dict."""
    n = len(layer.positions)
    discard = np.asarray(getattr(layer, "discard", np.zeros(n, dtype=bool)), dtype=bool)
    discarded = int(discard.sum())
    fraction = discarded / n if n else 0.0

    degenerate = [m for m in warnings_seen if "Could not align enough edges" in m]
    if degenerate:
        raise SystemExit(
            "ASHLAR's EdgeAligner could not fit a model on the reference grid and fell "
            "back to the identity transform, so its stage positions are the fabricated "
            "ones and the cross-cycle result is meaningless. Original warning: "
            f"{degenerate[0]}"
        )

    identity = np.arange(n)
    off_diagonal = int((np.asarray(layer.reference_idx, dtype=int) != identity).sum())
    if off_diagonal:
        logger.warning(
            "%d/%d moving tiles matched a non-corresponding reference tile. The two grids "
            "should be congruent; a large count means the cycles were not padded to a "
            "common canvas.",
            off_diagonal,
            n,
        )

    if fraction > max_discard_fraction:
        raise SystemExit(
            f"ASHLAR discarded {discarded}/{n} tiles ({fraction:.0%} > "
            f"{max_discard_fraction:.0%}). constrain_positions() replaced them with model "
            "predictions rather than erroring, so this run would have looked successful. "
            "The usual cause is --maximum-shift set below the true cross-cycle drift."
        )
    if discarded:
        logger.warning(
            "ASHLAR discarded %d/%d tiles (%.0f%%); their positions are model "
            "predictions, not measurements.",
            discarded,
            n,
            100 * fraction,
        )

    errors = np.asarray(layer.errors, dtype=float)
    finite = errors[np.isfinite(errors)]
    return {
        "n_tiles": int(n),
        "n_discarded": discarded,
        "discard_fraction": float(fraction),
        "n_reference_idx_off_diagonal": off_diagonal,
        "cycle_offset_yx": [
            float(v) for v in np.asarray(layer.cycle_offset, dtype=float)
        ],
        "error_p50": float(np.median(finite)) if finite.size else None,
        "error_p90": float(np.percentile(finite, 90)) if finite.size else None,
        "n_error_infinite": int((~np.isfinite(errors)).sum()),
        "data_warnings": warnings_seen,
    }, discard


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run ASHLAR cross-cycle alignment and emit a STARE transform manifest."
    )
    ap.add_argument(
        "--ref-tiles", required=True, help="retiled reference cycle directory"
    )
    ap.add_argument(
        "--moving-tiles", required=True, help="retiled moving cycle directory"
    )
    ap.add_argument("--reference-name", required=True)
    ap.add_argument("--moving-name", required=True)
    ap.add_argument(
        "--nuclear-index",
        type=int,
        default=0,
        help="nuclear/fiducial channel index ashlar aligns on",
    )
    ap.add_argument(
        "--maximum-shift",
        type=float,
        required=True,
        dest="max_shift_um",
        help="microns; must exceed the true cross-cycle drift (see module docstring)",
    )
    ap.add_argument(
        "--max-discard-fraction", type=float, default=DEFAULT_MAX_DISCARD_FRACTION
    )
    ap.add_argument(
        "--reference-edges",
        choices=("exact", "register"),
        default="exact",
        help="exact: the reference tiles keep the positions they were cut at (they come from "
        "one stitched slide); register: ASHLAR's EdgeAligner re-registers them, as for raw "
        "microscope tiles",
    )
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-tre", required=True)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ref_dir, mov_dir = Path(a.ref_tiles), Path(a.moving_tiles)
    ref_grid = json.loads((ref_dir / "grid.json").read_text())
    mov_grid = json.loads((mov_dir / "grid.json").read_text())

    for key in ("n_rows", "n_cols", "tile_size", "overlap"):
        if ref_grid[key] != mov_grid[key]:
            raise SystemExit(
                f"reference and moving grids disagree on {key} "
                f"({ref_grid[key]!r} vs {mov_grid[key]!r}). LayerAligner matches tiles "
                "one-for-one against the reference grid, so the cycles must be padded to a "
                "common canvas and retiled with identical settings."
            )

    pixel_size = float(ref_grid["pixel_size_um"])
    edge, layer, warned = _run_aligners(
        ref_dir,
        ref_grid,
        mov_dir,
        mov_grid,
        a.nuclear_index,
        a.max_shift_um,
        pixel_size,
        reference_edges=a.reference_edges,
    )
    diagnostics, discard = _check(edge, layer, a.max_discard_fraction, warned)

    d_yx = tile_displacements(
        layer.positions,
        layer.metadata.positions,
        edge.positions,
        edge.metadata.positions,
        layer.reference_idx,
    )
    entry, translation = ashlar_entry(d_yx, discard, mov_grid, ref_grid["orig_shape"])

    manifest = build_manifest(
        a.reference_name,
        {
            a.reference_name: slide_entry(np.eye(3)),
            a.moving_name: entry,
        },
    )
    Path(a.out_manifest).write_text(json.dumps(manifest, indent=2))

    residual = np.linalg.norm(_yx_to_xy(d_yx) - translation, axis=1)
    diagnostics.update(
        {
            "method": "ashlar",
            "reference": a.reference_name,
            "moving": a.moving_name,
            "pixel_size_um": pixel_size,
            "maximum_shift_um": a.max_shift_um,
            "rigid_translation_xy": [float(v) for v in translation],
            # Spread of the per-tile field about the rigid anchor -- ASHLAR's analogue of
            # STARE's rigid-stage TRE. Not a registration accuracy: reg_qc=2 measures that.
            "mesh_residual_px": {
                "p50": float(np.median(residual)),
                "p90": float(np.percentile(residual, 90)),
                "max": float(residual.max()),
            },
        }
    )
    Path(a.out_tre).write_text(json.dumps(diagnostics, indent=2))

    logger.info(
        "ashlar: %d tiles, %d discarded, rigid translation (x,y)=(%.2f, %.2f) px, "
        "mesh residual p50=%.2f px",
        diagnostics["n_tiles"],
        diagnostics["n_discarded"],
        translation[0],
        translation[1],
        diagnostics["mesh_residual_px"]["p50"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
