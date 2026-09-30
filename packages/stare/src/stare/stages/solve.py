"""STARE stage 3/4 (``stare solve``): assemble the transform manifest from per-tile control points.

Gathers the control JSONs emitted by every tile task, lays their window vectors on the
slide-global lattice, and writes the self-contained manifest (reference identity + the moving
slide's M0 + mesh) the warp and the reg_qc=2 scorer consume. One cheap per-slide reduction --
kilobytes, no image data.

The solve itself -- the range gate, robust affine, robust DCT-PLS, sigma calibration, the fold
certificate -- lives in ``stare.solve`` (``solve_dctpls``, the only solver since STARE v2). Its
report goes into the ``*_tre.json`` under ``"solve"`` and its name into the moving slide's
manifest entry as ``"solver"``. A control JSON without ``vectors`` (a pre-v2 REG_TILE) is
refused with a message naming the tiles: re-run REG_TILE.

The mirage pipeline invokes this stage through ``bin/tiled_solve.py``, a shim over ``main``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from stare.log import configure_logging, get_logger
from stare.manifest import build_manifest, slide_entry
from stare.solve import solve_dctpls, tile_accepted
from stare.tre_report import build_tre_report

logger = get_logger(__name__)


def main(argv=None) -> int:
    """CLI entry point: fold the per-tile control points into one warp manifest.

    Runs ``stare.solve.solve_dctpls`` and writes the manifest the stitch stage warps
    from, plus the TRE JSON the QC report renders.

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
    ap.add_argument(
        "--max-disp",
        type=float,
        default=None,
        help=(
            "reject a window vector whose |displacement| is at or beyond this many pixels -- the "
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

    grid_x, grid_y, disp, solve_report = solve_dctpls(controls, max_disp=a.max_disp)
    entry = slide_entry(
        m0, grid_x, grid_y, disp, interp=solve_report.get("mesh_interp")
    )
    # carry the reference frame so the stitch knows the output size without re-reading the reference
    entry["out_shape"] = [int(m0_doc["ref_h"]), int(m0_doc["ref_w"])]
    # which solve produced the mesh; the warpers ignore the key, a reader of the manifest wants it
    entry["solver"] = solve_report["solver"]

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
        # accepted = the tile put at least one in-range vector into the lattice
        # (stare.solve.tile_accepted, the same range rule the solve applies). A record that says
        # accepted=False is a tile the mesh took nothing from; the solve may still down-weight
        # more vectors, and says how many in the "solve" report below.
        records = [
            {
                "ix": int(c["ix"]),
                "iy": int(c["iy"]),
                "cx": float(c["cx"]),
                "cy": float(c["cy"]),
                "tre_rigid": float(c["tre"]),
                "accepted": tile_accepted(c, a.max_disp),
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
        report["solve"] = solve_report
        Path(a.out_tre).write_text(json.dumps(report, indent=2))

    n_refined = sum(1 for row in disp for d in row if d != [0.0, 0.0])
    logger.info(
        f"solve[{solve_report['solver']}]: {len(controls)} tiles -> manifest "
        f"(mesh={'yes' if entry['mesh'] else 'no'}, {n_refined} nodes displaced)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
