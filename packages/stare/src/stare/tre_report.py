"""STARE's intrinsic TRE report — the method's own Target Registration Error, à la VALIS.

Built from the same features/correlations the registration itself uses, so it needs no external
ground truth (exactly what makes VALIS's ``error_df`` intrinsic). Two entry points share this
builder so the ``_tre.json`` has one shape:

  * ``coarse_tre_px``     — RMS residual of the DISK matches after the global rigid M0 (rigid-stage
                            TRE; directly comparable to VALIS's rigid error).
  * ``rigid_tre_px``      — percentiles of the per-tile rigid-stage misalignment, plus the full
                            per-tile ``tiles`` list = a *spatial* TRE heatmap VALIS does not give.
  * ``residual_after_px`` — (when measured) percentiles of the per-tile residual AFTER the mesh is
                            applied: STARE's post-registration final-accuracy number, the analogue
                            of VALIS's non-rigid error.

Pure NumPy.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_tre_report"]


def _pct(values):
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    return {
        "mean": float(a.mean()),
        "p50": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def build_tre_report(coarse_tre_px, n_inliers, tile_records, mesh_refined):
    """Assemble the intrinsic-TRE report.

    ``tile_records`` is a list of per-tile dicts carrying at least ``ix, iy, cx, cy, tre_rigid``;
    if every record also has ``tre_after`` the post-refinement residual is summarised too.

    **The percentile summaries cover ACCEPTED records only.** A control point rejected by the
    confidence or range gate never reaches the mesh, and its ``tre_rigid`` is not a conservative
    over-estimate of a real misalignment -- phase correlation always returns a peak, so the
    number is an artefact of correlating against the wrong thing. Averaging it in describes a
    registration that was never performed. ``tiles`` still carries every record, accepted or
    not, because *where* points were dropped is exactly what the spatial heatmap is for.

    A record with no ``accepted`` key counts as accepted -- the same legacy contract
    ``tiled_solve._accept`` applies to a control point written before confidence gating existed.
    """
    kept = [t for t in tile_records if bool(t.get("accepted", True))]
    report = {
        "coarse_tre_px": float(coarse_tre_px),
        "n_inliers": int(n_inliers),
        "n_tiles": len(tile_records),
        "n_accepted": len(kept),
        "n_rejected": len(tile_records) - len(kept),
        "mesh_refined": bool(mesh_refined),
        "rigid_tre_px": _pct(t["tre_rigid"] for t in kept),
        "tiles": tile_records,  # spatial heatmap: one entry per tile, rejected ones included
    }
    if kept and all("tre_after" in t for t in kept):
        report["residual_after_px"] = _pct(t["tre_after"] for t in kept)
    return report
