"""The SOLVE stage: per-tile control points -> a control-grid displacement mesh.

Every comparable method (approximating TPS, elastix FFD, RegWSI's diffusive
solve, PIV) turns sparse, noisy displacement measurements into a dense field by
*reject -> regularise -> densify*. Two solvers live here:

``legacy``
    What STARE shipped until 2026-09: the three gates (confidence, range, TRE),
    then a median filter over the accepted cells. Rejected and unmeasured cells
    hold ``[0, 0]``, the mesh's null action -- a step in the field wherever a
    tile was dropped. Kept byte-for-byte so a prior run reproduces.

``robust``
    The same three gates, then, on the control grid:

    1. **neighbour consistency** -- the normalised median test of Westerweel &
       Scarano (2005): a cell whose displacement disagrees with the median of
       its accepted neighbours by more than ``nmt_threshold`` median absolute
       deviations (plus ``nmt_epsilon`` px of measurement noise) is rejected.
       This is the cross-tile consistency layer ASHLAR gets from its spanning
       tree and STARE dropped; a confident-but-wrong tile (a partly blank crop
       correlated against the wrong structure) is caught here, not by any
       single-tile score.
    2. **in-fill** -- a rejected or unmeasured cell takes the inverse-distance
       weighted mean of accepted cells within ``infill_radius`` grid steps, so a
       dropped tile is bridged from its neighbours instead of stepping to zero.
       Cells with no accepted cell in reach stay at zero (the field decays into
       background rather than extrapolating).
    3. **regularised smoothing** -- a Tikhonov solve
       ``argmin_u  sum_i w_i |u_i - d_i|^2 + lambda * sum |grad u|^2``
       with ``w_i = 1 - error_i`` on measured cells and ``infill_weight`` on
       in-filled ones. ``lambda`` is in grid units, so the same value means the
       same thing at every tile size. Solved sparse; the grid is kilobytes.
    4. **invertibility** -- the stitch inverts the field by fixed-point
       iteration, which converges when the displacement's Lipschitz constant is
       below one (Chen et al. 2008). The Jacobian of ``u`` on the grid is
       reported (its maximum operator norm and the minimum determinant of
       ``I + J``); when the norm reaches ``max_lipschitz`` the field is scaled
       down to it and the report says so, rather than shipping a fold.

Both return the same ``(grid_x, grid_y, disp, report)`` so the stage that writes
the manifest does not care which ran. ``report`` is what ``*_tre.json`` and the
manifest record about the solve.
"""

from __future__ import annotations

import logging
from statistics import median

import numpy as np

logger = logging.getLogger(__name__)

SOLVERS = ("legacy", "robust")

# Westerweel & Scarano (2005), Exp. Fluids 39: universal outlier detection for
# PIV data. Threshold 2.0 and epsilon 0.1 px are the values they report as
# universal across flows; epsilon absorbs the sub-pixel measurement noise.
NMT_THRESHOLD = 2.0
NMT_EPSILON = 0.1
INFILL_RADIUS = 2
INFILL_WEIGHT = 0.25
LAMBDA = 1.0
MAX_LIPSCHITZ = 0.9


def accept(control, max_error, max_disp):
    """Is this control point trustworthy enough to place in the mesh?

    Two independent bounds, and the *confidence* one does the real work:

    - ``error`` is scikit-image's normalised correlation error. Phase
      correlation always returns a peak, so a background tile or a tile
      straddling the section edge produces a displacement that a magnitude
      bound cannot tell from a small real residual -- only its error (~1.0
      against real tissue's ~0.04) exposes it. NaN, which scikit-image returns
      when a crop is empty, is rejected by the same comparison.
    - ``|d| >= max_disp`` means the true match was never inside the read window
      (``max_disp`` defaults to the read halo), so the peak is an artefact.

    A control point with no ``"error"`` key predates confidence gating. It is
    accepted -- unknown confidence -- so a run resumed across that change does
    not silently lose every tile written before it. Returns ``(accepted, reason)``.
    """
    if (
        max_disp is not None
        and float(np.hypot(control["dx"], control["dy"])) >= max_disp
    ):
        return False, "disp"
    if "error" not in control:
        return True, None
    error = float(control["error"])
    # `not (error <= max_error)` so NaN rejects rather than passes.
    if max_error is not None and not (error <= max_error):
        return False, "error"
    return True, None


def _lay_out(controls, gate_tre, max_error, max_disp):
    """Grid coordinates, the raw displacement per cell, and the per-cell state.

    ``measured`` marks cells that passed the gates; ``refined`` the subset of
    those at or above ``gate_tre`` whose displacement is carried into the field
    (a trustworthy tile below the gate is *already aligned*: it contributes
    ``[0, 0]`` as a measurement, not as an absence). ``weight`` is the
    confidence ``1 - error`` (1.0 for a legacy point without one).
    """
    nx = max(c["ix"] for c in controls) + 1
    ny = max(c["iy"] for c in controls) + 1
    grid_x = np.zeros(nx)
    grid_y = np.zeros(ny)
    disp = np.zeros((ny, nx, 2))
    measured = np.zeros((ny, nx), dtype=bool)
    refined = np.zeros((ny, nx), dtype=bool)
    weight = np.zeros((ny, nx))
    counts = {"error": 0, "disp": 0, "legacy": 0, "below_gate": 0}
    for c in controls:
        grid_x[c["ix"]] = float(c["cx"])
        grid_y[c["iy"]] = float(c["cy"])
        ok, reason = accept(c, max_error, max_disp)
        if not ok:
            counts[reason] += 1
            continue
        if "error" not in c:
            counts["legacy"] += 1
            weight[c["iy"], c["ix"]] = 1.0
        else:
            weight[c["iy"], c["ix"]] = float(np.clip(1.0 - float(c["error"]), 0.0, 1.0))
        measured[c["iy"], c["ix"]] = True
        if float(c["tre"]) >= gate_tre:
            refined[c["iy"], c["ix"]] = True
            disp[c["iy"], c["ix"]] = [float(c["dx"]), float(c["dy"])]
        else:
            counts["below_gate"] += 1
    if counts["legacy"]:
        logger.warning(
            f"{counts['legacy']}/{len(controls)} control point(s) carry no 'error' key -- "
            "written before confidence gating existed. Accepting them with unknown "
            "confidence; re-run the tile stage for those tiles to have them gated."
        )
    if counts["error"] or counts["disp"]:
        logger.info(
            f"rejected {counts['error'] + counts['disp']}/{len(controls)} control point(s): "
            f"{counts['error']} low-confidence (error > {max_error}), "
            f"{counts['disp']} out of range (|d| >= {max_disp})"
        )
    return grid_x, grid_y, disp, measured, refined, weight, counts


# ── legacy ────────────────────────────────────────────────────────────────────
def _median_filter_accepted(disp, accepted, radius=1):
    """Median-smooth over ACCEPTED cells only; a rejected cell keeps ``[0, 0]``.

    Reject first, then smooth: a whole-grid median would refill a cell the
    confidence gate just rejected from its agreeing neighbours.
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


def solve_legacy(controls, gate_tre, max_error=None, max_disp=None, median_radius=1):
    """The pre-2026-09 solve, byte-for-byte: gates, then a median over accepted cells."""
    grid_x, grid_y, disp, measured, refined, _, counts = _lay_out(
        controls, gate_tre, max_error, max_disp
    )
    disp_l = disp.tolist()
    accepted = measured.tolist()
    disp_l = _median_filter_accepted(disp_l, accepted, radius=median_radius)
    report = {
        "solver": "legacy",
        "n_controls": len(controls),
        "n_rejected_error": counts["error"],
        "n_rejected_disp": counts["disp"],
        "n_legacy_unscored": counts["legacy"],
        "n_below_gate": counts["below_gate"],
        "n_refined": int(sum(1 for row in disp_l for d in row if d != [0.0, 0.0])),
    }
    return [float(v) for v in grid_x], [float(v) for v in grid_y], disp_l, report


# ── robust ────────────────────────────────────────────────────────────────────
def _neighbours(iy, ix, ny, nx, radius=1):
    for jy in range(max(0, iy - radius), min(ny, iy + radius + 1)):
        for jx in range(max(0, ix - radius), min(nx, ix + radius + 1)):
            if (jy, jx) != (iy, ix):
                yield jy, jx


def normalized_median_test(
    disp, measured, threshold=NMT_THRESHOLD, epsilon=NMT_EPSILON
):
    """Westerweel & Scarano's universal outlier test on the measured cells.

    For each measured cell with at least three measured 8-neighbours: the
    residual ``r = |d - median(neighbours)|`` per component, normalised by the
    neighbours' median absolute residual plus ``epsilon``; the cell is an
    outlier when the larger normalised component exceeds ``threshold``. Cells
    with fewer than three measured neighbours cannot be tested and are kept.
    Returns a boolean grid of outliers.
    """
    ny, nx = measured.shape
    outlier = np.zeros((ny, nx), dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            if not measured[iy, ix]:
                continue
            nb = [
                disp[jy, jx]
                for jy, jx in _neighbours(iy, ix, ny, nx)
                if measured[jy, jx]
            ]
            if len(nb) < 3:
                continue
            nb = np.asarray(nb)
            med = np.median(nb, axis=0)
            resid_nb = np.median(np.abs(nb - med), axis=0)
            ratio = np.abs(disp[iy, ix] - med) / (resid_nb + epsilon)
            if float(ratio.max()) > threshold:
                outlier[iy, ix] = True
    return outlier


def infill(disp, trusted, radius=INFILL_RADIUS):
    """Fill untrusted cells from trusted ones within ``radius`` grid steps (IDW).

    A cell with no trusted cell in reach stays at zero: the field decays into
    unmeasured background rather than extrapolating a guess across it.
    Returns ``(filled_disp, filled_mask)``.
    """
    ny, nx = trusted.shape
    out = disp.copy()
    filled = np.zeros((ny, nx), dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            if trusted[iy, ix]:
                continue
            num = np.zeros(2)
            den = 0.0
            for jy, jx in _neighbours(iy, ix, ny, nx, radius):
                if not trusted[jy, jx]:
                    continue
                w = 1.0 / float(np.hypot(jy - iy, jx - ix))
                num += w * disp[jy, jx]
                den += w
            if den > 0:
                out[iy, ix] = num / den
                filled[iy, ix] = True
            else:
                out[iy, ix] = 0.0
    return out, filled


def tikhonov_smooth(disp, weight, lam=LAMBDA):
    """``argmin_u sum w |u - d|^2 + lam * sum |grad u|^2`` on the grid, per component.

    ``weight`` zero on a cell means "no data here, follow the neighbours";
    ``lam`` in grid units. A 1x1 grid or ``lam == 0`` returns the input.
    """
    ny, nx, _ = disp.shape
    n = ny * nx
    if n == 1 or lam <= 0:
        return disp.copy()
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve

    idx = np.arange(n).reshape(ny, nx)
    rows, cols, vals = [], [], []

    def edge(a, b):
        # the gradient term |u_a - u_b|^2 contributes [[1,-1],[-1,1]] to the normal matrix
        rows.extend([a, a, b, b])
        cols.extend([a, b, a, b])
        vals.extend([1.0, -1.0, -1.0, 1.0])

    for iy in range(ny):
        for ix in range(nx):
            if ix + 1 < nx:
                edge(idx[iy, ix], idx[iy, ix + 1])
            if iy + 1 < ny:
                edge(idx[iy, ix], idx[iy + 1, ix])
    laplacian = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    w = weight.reshape(n)
    a = diags(w) + lam * laplacian
    out = np.empty_like(disp)
    for k in range(2):
        b = w * disp[:, :, k].reshape(n)
        # a is symmetric positive semi-definite; with any positive weight it is
        # non-singular. An all-zero weight field has nothing to fit: return zeros.
        if not np.any(w > 0):
            out[:, :, k] = 0.0
            continue
        out[:, :, k] = spsolve(a.tocsc(), b).reshape(ny, nx)
    return out


def jacobian_report(grid_x, grid_y, disp):
    """Lipschitz and folding diagnostics of the displacement field on the grid.

    Finite differences of ``u`` over the (possibly non-uniform) grid give the
    2x2 Jacobian per cell; ``max_operator_norm`` is the largest spectral norm
    (the field's Lipschitz constant on the grid) and ``min_jacobian_det`` the
    smallest ``det(I + J)`` (negative means a fold). A single-row or
    single-column grid has no gradient along that axis.
    """
    gy = np.asarray(grid_y, dtype=float)
    gx = np.asarray(grid_x, dtype=float)
    ny, nx, _ = disp.shape
    du_dx = np.zeros((ny, nx, 2))
    du_dy = np.zeros((ny, nx, 2))
    if nx > 1:
        du_dx = np.gradient(disp, gx, axis=1)
    if ny > 1:
        du_dy = np.gradient(disp, gy, axis=0)
    # J = [[dux/dx, dux/dy], [duy/dx, duy/dy]] per cell
    j = np.stack(
        [
            np.stack([du_dx[..., 0], du_dy[..., 0]], axis=-1),
            np.stack([du_dx[..., 1], du_dy[..., 1]], axis=-1),
        ],
        axis=-2,
    )
    norms = np.linalg.norm(j, ord=2, axis=(-2, -1))
    dets = np.linalg.det(np.eye(2) + j)
    return {
        "max_operator_norm": float(norms.max()) if norms.size else 0.0,
        "min_jacobian_det": float(dets.min()) if dets.size else 1.0,
    }


def solve_robust(
    controls,
    gate_tre,
    max_error=None,
    max_disp=None,
    nmt_threshold=NMT_THRESHOLD,
    nmt_epsilon=NMT_EPSILON,
    infill_radius=INFILL_RADIUS,
    infill_weight=INFILL_WEIGHT,
    lam=LAMBDA,
    max_lipschitz=MAX_LIPSCHITZ,
):
    """Gates -> neighbour consistency -> in-fill -> Tikhonov -> invertibility."""
    grid_x, grid_y, disp, measured, refined, weight, counts = _lay_out(
        controls, gate_tre, max_error, max_disp
    )
    outlier = normalized_median_test(disp, measured, nmt_threshold, nmt_epsilon)
    trusted = measured & ~outlier
    filled, filled_mask = infill(disp, trusted, infill_radius)
    w = np.where(trusted, weight, 0.0)
    w = np.where(filled_mask, infill_weight, w)
    smooth = tikhonov_smooth(filled, w, lam)
    # cells beyond in-fill reach carry no data and no weight: the solve leaves them
    # following their neighbours, which is the intended decay into background
    jac = jacobian_report(grid_x, grid_y, smooth)
    scaled = 1.0
    if jac["max_operator_norm"] > max_lipschitz > 0:
        scaled = max_lipschitz / jac["max_operator_norm"]
        smooth = smooth * scaled
        jac = jacobian_report(grid_x, grid_y, smooth)
    report = {
        "solver": "robust",
        "n_controls": len(controls),
        "n_rejected_error": counts["error"],
        "n_rejected_disp": counts["disp"],
        "n_legacy_unscored": counts["legacy"],
        "n_below_gate": counts["below_gate"],
        "n_rejected_inconsistent": int(outlier.sum()),
        "n_infilled": int(filled_mask.sum()),
        "n_unfilled": int((~trusted & ~filled_mask).sum()),
        "n_refined": int(np.sum(np.any(smooth != 0.0, axis=-1))),
        "lambda": float(lam),
        "nmt_threshold": float(nmt_threshold),
        "lipschitz_scale": float(scaled),
        **jac,
    }
    return (
        [float(v) for v in grid_x],
        [float(v) for v in grid_y],
        smooth.tolist(),
        report,
    )


def solve_grid(
    controls, gate_tre, max_error=None, max_disp=None, solver="robust", **kw
):
    """Dispatch on ``solver``; see the module docstring for what each does."""
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; expected one of {SOLVERS}")
    if solver == "legacy":
        return solve_legacy(controls, gate_tre, max_error, max_disp, **kw)
    return solve_robust(controls, gate_tre, max_error, max_disp, **kw)
