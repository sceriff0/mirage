#!/usr/bin/env python3
"""STARE fan-out step 3/4: assemble the transform manifest from per-tile control points.

Gathers the control points emitted by every tile task, lays them on the grid, and writes the
self-contained manifest (reference identity + the moving slide's M0 + mesh) the warp and the
reg_qc=2 scorer consume. One cheap per-slide reduction — kilobytes, no image data.

Three gates, applied in this order, decide what reaches the mesh:

1. **confidence** (``--max-error``) — reject a control point whose correlation found no real
   match. Background and section-edge tiles are the majority of a whole-slide grid and produce
   plausibly *small* displacements, so this, not magnitude, is what rejects them. It rejects
   *pure* background outright; a PARTIALLY on-section tile can still get through with a wrong
   displacement — see the "KNOWN, UNCLOSED EXPOSURE" note on ``_accept``.
2. **range** (``--max-disp``, the read halo) — reject a match that was never inside the read
   window. A cheap secondary net.
3. **TRE** (``--gate-tre``) — of the trustworthy points, refine only those actually misaligned.

The surviving grid is then median-filtered over the *accepted* points only, so smoothing can never
resurrect a cell the confidence gate rejected.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from statistics import median

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ["NUMBA_DISABLE_CACHING"] = "1"

sys.path.insert(0, str(Path(__file__).parent / "utils"))

import numpy as np  # noqa: E402
from logger import configure_logging, get_logger  # noqa: E402
from tiled_manifest import build_manifest, slide_entry  # noqa: E402
from tre_report import build_tre_report  # noqa: E402

logger = get_logger(__name__)


def _accept(control, max_error, max_disp):
    """Is this control point trustworthy enough to place in the mesh?

    Two independent bounds, and the *confidence* one does the real work:

    - ``error`` is scikit-image's normalised correlation error. Phase correlation always returns a
      peak, so a background tile or a tile straddling the section edge produces a displacement
      that a magnitude bound cannot tell from a small real residual -- measured, a
      tissue-vs-background tile lands at |d| = 5.66 px, inside the halo and above the TRE gate,
      and only its error of 0.999961 (against real tissue's 0.040956) exposes it. NaN, which
      scikit-image returns when a crop is empty, is rejected by the same comparison.
    - ``|d| >= max_disp`` means the true match was never inside the read window (``max_disp``
      defaults to ``reg_tiled_halo``), so the peak is an artefact by construction. A cheap
      secondary net, not the primary one.

    A control point with no ``"error"`` key predates this gate. It is accepted -- unknown
    confidence, and the caller warns -- so a run resumed across the change does not silently lose
    every tile written before it.

    KNOWN, UNCLOSED EXPOSURE -- partial-tissue tiles
    ------------------------------------------------
    The confidence gate is decisive at both ends of the "how much of this tile is off-section"
    band and NOT in the middle. Measured over 21 blanking fractions x 4 geometries x 3 seeds
    (``tests/test_tile_residual_confidence.py``):

    - through ~30-35% blanked: accepted, and the true shift is recovered -- correct;
    - roughly **40-90% blanked: the tile is ACCEPTED and the recovered shift is WRONG**, in
      every one of the 12 configurations swept (4 to 9 of the 21 sampled fractions each),
      injecting a bogus displacement of up to **131 px**;
    - from ~95% blanked: rejected.

    The range gate does not help: 131 px is well inside the default ``max_disp`` of 256
    (``reg_tiled_halo``), which is the same reason the range gate rejects nothing on a realistic
    grid. Nothing here distinguishes "correlated against the wrong part of the tile" from "a
    real residual" -- both are confident peaks.

    The real fix is FOREGROUND DETECTION (Otsu or equivalent): skip a tile whose moving crop is
    mostly background instead of trying to score the correlation after the fact. That is
    deliberately out of scope for this change and is tracked as a separate piece of work. Until
    it lands, a section perimeter contributes some wrong control points to the mesh -- fewer and
    smaller than before this gate existed, but not zero.

    Returns ``(accepted, reason)``; ``reason`` is None when accepted.
    """
    if (
        max_disp is not None
        and float(np.hypot(control["dx"], control["dy"])) >= max_disp
    ):
        return False, "disp"
    if "error" not in control:
        return True, None
    error = float(control["error"])
    # written as `not (error <= max_error)` so NaN -- an empty crop, which tiled_reg_tile.py
    # manufactures when the moving region falls outside the slide -- rejects rather than passes.
    if max_error is not None and not (error <= max_error):
        return False, "error"
    return True, None


def _median_filter_accepted(disp, accepted, radius=1):
    """Median-smooth the displacement grid over ACCEPTED control points only.

    Order matters and is deliberate: reject first, then smooth. A whole-grid median would refill a
    cell the confidence gate just rejected from its agreeing neighbours, reintroducing the bug
    through the back door. So a rejected cell is excluded from every median window and keeps the
    mesh's null action, ``[0.0, 0.0]``.
    """
    ny = len(disp)
    nx = len(disp[0]) if ny else 0
    out = [[list(d) for d in row] for row in disp]
    for iy in range(ny):
        for ix in range(nx):
            if not accepted[iy][ix]:
                continue
            xs, ys = [], []
            for jy in range(max(0, iy - radius), min(ny, iy + radius + 1)):
                for jx in range(max(0, ix - radius), min(nx, ix + radius + 1)):
                    if accepted[jy][jx]:
                        xs.append(disp[jy][jx][0])
                        ys.append(disp[jy][jx][1])
            out[iy][ix] = [float(median(xs)), float(median(ys))]
    return out


def _grid_from_controls(
    controls, gate_tre, max_error=None, max_disp=None, median_radius=1
):
    nx = max(c["ix"] for c in controls) + 1
    ny = max(c["iy"] for c in controls) + 1
    grid_x = [0.0] * nx
    grid_y = [0.0] * ny
    disp = [[[0.0, 0.0] for _ in range(nx)] for _ in range(ny)]
    # A cell is "accepted" if it passed the confidence and range gates -- i.e. its measurement is
    # trustworthy. That is not the same as "refined": a tile below `gate_tre` is trustworthily
    # already aligned, so it holds [0.0, 0.0] AND takes part in its neighbours' medians. A
    # rejected cell holds [0.0, 0.0] too, but is excluded from every median window.
    accepted = [[False] * nx for _ in range(ny)]
    n_error, n_disp, n_legacy = 0, 0, 0
    for c in controls:
        grid_x[c["ix"]] = float(c["cx"])
        grid_y[c["iy"]] = float(c["cy"])
        ok, reason = _accept(c, max_error, max_disp)
        if not ok:
            if reason == "error":
                n_error += 1
            else:
                n_disp += 1
            continue
        if "error" not in c:
            n_legacy += 1
        accepted[c["iy"]][c["ix"]] = True
        if float(c["tre"]) >= gate_tre:
            disp[c["iy"]][c["ix"]] = [float(c["dx"]), float(c["dy"])]

    if n_legacy:
        logger.warning(
            f"{n_legacy}/{len(controls)} control point(s) carry no 'error' key -- written before "
            "confidence gating existed. Accepting them with unknown confidence; re-run "
            "TILED_REG_TILE for those tiles to have them gated."
        )
    if n_error or n_disp:
        logger.info(
            f"rejected {n_error + n_disp}/{len(controls)} control point(s): "
            f"{n_error} low-confidence (error > {max_error}), "
            f"{n_disp} out of range (|d| >= {max_disp})"
        )

    disp = _median_filter_accepted(disp, accepted, radius=median_radius)
    return grid_x, grid_y, disp


def main(argv=None) -> int:
    """CLI entry point: fold the per-tile control points into one warp manifest.

    Applies the accept/reject gate, median-filters the accepted displacement
    field, and writes the manifest ``tiled_stitch.py`` warps from, plus the TRE
    JSON the QC report renders.

    Returns
    -------
    int
        0 on success.
    """
    configure_logging()
    ap = argparse.ArgumentParser(
        description="STARE manifest assembly from control points."
    )
    ap.add_argument("--m0", required=True, help="M0 JSON from tiled_coarse")
    ap.add_argument(
        "--controls", required=True, help="glob for the per-tile control JSONs"
    )
    ap.add_argument("--gate-tre", type=float, default=1.0)
    ap.add_argument(
        "--max-error",
        type=float,
        default=0.99,
        help=(
            "reject a control point whose phase-correlation error exceeds this (0 = perfect "
            "match, ~1 = the two crops share no structure). This is the confidence gate that "
            "keeps background and section-edge tiles out of the mesh; a magnitude bound alone "
            "cannot, because those tiles produce plausibly small displacements. NaN (an empty "
            "crop) is always rejected."
        ),
    )
    ap.add_argument(
        "--max-disp",
        type=float,
        default=None,
        help=(
            "reject a control point whose |displacement| is at or beyond this many pixels -- the "
            "match was never inside the read window, so the peak is an artefact. The pipeline "
            "passes reg_tiled_halo. Default None = no range bound."
        ),
    )
    ap.add_argument(
        "--reference-name",
        default=None,
        help="reference slide name for the manifest; defaults to ref_name from the M0 JSON",
    )
    ap.add_argument("--moving-name", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument(
        "--out-tre",
        default=None,
        help="output intrinsic-TRE summary JSON (rigid-stage spatial TRE from the control points)",
    )
    a = ap.parse_args(argv)

    m0_doc = json.loads(Path(a.m0).read_text())
    m0 = np.asarray(m0_doc["M0"], dtype=float)
    reference_name = a.reference_name or m0_doc.get("ref_name") or "reference"

    files = sorted(glob.glob(a.controls))
    if not files:
        raise FileNotFoundError(f"no control-point JSONs matched {a.controls!r}")
    controls = [json.loads(Path(f).read_text()) for f in files]

    grid_x, grid_y, disp = _grid_from_controls(
        controls, a.gate_tre, max_error=a.max_error, max_disp=a.max_disp
    )
    entry = slide_entry(m0, grid_x, grid_y, disp)
    # carry the reference frame so the stitch knows the output size without re-reading the reference
    entry["out_shape"] = [int(m0_doc["ref_h"]), int(m0_doc["ref_w"])]

    manifest = build_manifest(
        reference_name,
        {
            reference_name: {
                "M0": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "mesh": None,
            },
            a.moving_name: entry,
        },
    )
    Path(a.out_manifest).write_text(json.dumps(manifest, indent=2))

    # Intrinsic TRE (fix: the fan-out used to drop this). Built from the same per-tile phase
    # correlations the registration used — coarse rigid TRE + a spatial per-tile heatmap. The
    # post-refinement residual is not measured here (no re-warp in this reduction); the default
    # monolithic path and the reg_benchmark harness provide the final-accuracy number.
    if a.out_tre:
        # Re-ask `_accept` rather than thread the grid's decision out of _grid_from_controls:
        # `_accept` is the single owner of the rule, it is pure and cheap, and re-asking keeps
        # the mesh and the report provably in agreement. A record here that says accepted=False
        # is exactly a control point the mesh did not use.
        records = [
            {
                "ix": int(c["ix"]),
                "iy": int(c["iy"]),
                "cx": float(c["cx"]),
                "cy": float(c["cy"]),
                "tre_rigid": float(c["tre"]),
                "accepted": bool(_accept(c, a.max_error, a.max_disp)[0]),
            }
            for c in controls
        ]
        report = build_tre_report(
            m0_doc.get("coarse_tre", 0.0),
            m0_doc.get("n_inliers", 0),
            records,
            entry["mesh"] is not None,
        )
        report["moving"] = a.moving_name
        report["reference"] = reference_name
        report["note"] = (
            "fan-out: rigid-stage spatial TRE; post-refinement residual not measured here"
        )
        Path(a.out_tre).write_text(json.dumps(report, indent=2))

    n_refined = sum(1 for row in disp for d in row if d != [0.0, 0.0])
    logger.info(
        f"solve: {len(controls)} tiles -> manifest (mesh={'yes' if entry['mesh'] else 'no'}, "
        f"{n_refined} tiles refined)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
