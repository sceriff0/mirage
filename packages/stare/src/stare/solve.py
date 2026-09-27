"""The SOLVE stage: per-tile control points -> a control-grid displacement mesh.

Every comparable method (approximating TPS, elastix FFD, RegWSI's diffusive
solve, PIV) turns sparse, noisy displacement measurements into a dense field by
*reject -> regularise -> densify*. Three solvers live here:

``dctpls`` (the default)
    No TRE gate, so no dead zone: a small displacement is a measurement, never ``[0, 0]``.
    Invalid controls (``accept``) get weight 0; the rest carry ``1/sigma**2`` when they
    report a ``sigma``, else 1 -- the correlation ``error`` is a gate, never a weight
    (its scale is arbitrary). Then:

    1. **robust affine** -- weighted Huber IRLS of a 6-parameter affine, subtracted.
       DCT-PLS's null space is only a constant, so without this a residual rotation left
       by M0 is charged as roughness and shrunk.
    2. **robust DCT-PLS** of the residual (Garcia 2010, CSDA 54:1167): a thin-plate
       (squared-Laplacian) penalty with reflective boundaries, solved by DCT; one shared
       smoothing parameter ``s`` for x and y; bisquare weights on the vector-residual
       norm. Missing cells are filled by the smoother, not zeroed.
    3. **s chosen from the data** -- h-block cross-validation (5 folds of 3x3-cell patches,
       Burman et al. 1994) when there are >= 25 valid cells, GCV below that.
    4. **fold certificate** -- the field's Lipschitz constant and ``min det(I + J)`` are
       reported, and ``fold_certificate_ok`` when the constant is below 0.5. The field is
       never rescaled.

    Why it replaced ``robust``: on the 16-tile synthetic slide ``robust`` scored 8.7 px
    median against 2.2 px for the raw tile vectors -- its TRE gate zeroes sub-gate
    vectors and the zeros poison the median test, its first-order Tikhonov penalty
    shrinks the affine residual, and ``1 - error`` weights have no fixed scale
    (``research/stare-optimal-design-2026-09-27.md`` §0, §3). The core, ``_dctpls_core``,
    takes a lattice array, not controls, so a finer many-points-per-tile lattice can
    feed it unchanged.

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

All three return the same ``(grid_x, grid_y, disp, report)`` so the stage that writes
the manifest does not care which ran. ``report`` is what ``*_tre.json`` and the
manifest record about the solve.
"""

from __future__ import annotations

import logging
from statistics import median

import numpy as np

logger = logging.getLogger(__name__)

SOLVERS = ("dctpls", "robust", "legacy")

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


# ── dctpls ────────────────────────────────────────────────────────────────────
# Huber (1964) tuning for 95 % Gaussian efficiency; 1.4826 turns a MAD into a sigma.
HUBER_C = 1.345
MAD_TO_SIGMA = 1.4826
# Tukey's bisquare at 95 % efficiency, as Garcia (2010) uses for the robust weights.
BISQUARE_C = 4.685
# The robust scale never drops below this (px): a per-window phase-correlation vector is
# not more precise than ~0.1 px (0.10-0.25 px measured, research peaklock.py). Without a
# floor, a near-exact field -- or any fit at small s, whose residuals collapse toward zero
# -- gives MAD ~ 0 and bisquare then rejects a real feature over a hundredth of a pixel.
# With it, a disagreement under ~0.5 px (4.685 x 0.1) is never called an outlier.
SIGMA_FLOOR_PX = 0.1
AFFINE_ITERATIONS = 10
ROBUST_ITERATIONS = 6
# h-block cross-validation: K folds of 3x3-cell patches (Burman et al. 1994). A 3-cell
# patch is wider than one window's overlap, so a held-out cell's neighbours are held out
# with it and its error is not predicted from its own (correlated) measurement.
CV_FOLDS = 5
CV_BLOCK = 3
# log10 s candidates. GCV scans the fine grid; h-block CV scans the coarse one and then
# refines by +-0.25 around its best (each CV candidate costs K fits, each GCV one fit).
LOG10_S_FINE = np.arange(-4.0, 6.0 + 1e-9, 0.25)
CV_LOG10_S = np.arange(-4.0, 6.0 + 1e-9, 0.5)
CV_REFINE = 0.25
MIN_CV_CELLS = 25
MIN_SMOOTH_CELLS = 6
MIN_AFFINE_CELLS = 3
# The fixed point converges for any s; these bound the work, not the answer.
PLS_TOL = 1e-4
PLS_TOL_CV = 1e-3
PLS_MAX_ITER = 300
FOLD_CERTIFICATE_LIPSCHITZ = 0.5


def _lattice_from_controls(controls, max_error, max_disp):
    """Controls -> a regular lattice of observations with prior weights.

    Lays the controls on the grid exactly as ``_lay_out`` does (``ix``/``iy`` index
    the lattice, ``cx``/``cy`` are the node coordinates). Returns
    ``(grid_x, grid_y, Y, W0, counts)``: ``Y`` is ``(ny, nx, 2)`` with the measured
    ``(dx, dy)``, ``W0`` the prior weight -- ``1/sigma**2`` when a control carries a
    positive ``"sigma"``, else 1.0; 0 for a missing or rejected cell. The phase-
    correlation ``error`` is used ONLY by the ``accept`` gate, never as a weight: its
    scale is arbitrary (0.001-0.05 on one dataset, 0.96 on another), so a weight built
    from it means a different amount of smoothing on every slide.

    Kept separate from the solve so a later phase can feed many points per tile on a
    finer regular lattice straight into ``_dctpls_core``.
    """
    nx = max(c["ix"] for c in controls) + 1
    ny = max(c["iy"] for c in controls) + 1
    grid_x = np.zeros(nx)
    grid_y = np.zeros(ny)
    Y = np.zeros((ny, nx, 2))
    W0 = np.zeros((ny, nx))
    counts = {"error": 0, "disp": 0, "legacy": 0}
    for c in controls:
        grid_x[c["ix"]] = float(c["cx"])
        grid_y[c["iy"]] = float(c["cy"])
        ok, reason = accept(c, max_error, max_disp)
        if not ok:
            counts[reason] += 1
            continue
        if "error" not in c:
            counts["legacy"] += 1
        dx, dy = float(c["dx"]), float(c["dy"])
        if not (np.isfinite(dx) and np.isfinite(dy)):
            counts["error"] += 1
            continue
        sigma = c.get("sigma")
        w = 1.0
        if sigma is not None and np.isfinite(float(sigma)) and float(sigma) > 0:
            w = 1.0 / float(sigma) ** 2
        Y[c["iy"], c["ix"]] = [dx, dy]
        W0[c["iy"], c["ix"]] = w
    if counts["error"] or counts["disp"]:
        logger.info(
            f"rejected {counts['error'] + counts['disp']}/{len(controls)} control point(s): "
            f"{counts['error']} low-confidence (error > {max_error}), "
            f"{counts['disp']} out of range (|d| >= {max_disp})"
        )
    return grid_x, grid_y, Y, W0, counts


def _robust_affine(Y, W0, gx, gy, iterations=AFFINE_ITERATIONS):
    """Weighted Huber IRLS of the 6-parameter affine ``u = a + B (x, y)``.

    Coordinates are centred and scaled for conditioning; the returned field is in
    pixels. ``c = 1.345 * 1.4826 * MAD`` of the vector-residual norms, re-estimated
    every iteration. Returns ``(field (ny, nx, 2), coef (3, 2) in pixel coords,
    huber weight per valid cell)``.
    """
    GX, GY = np.meshgrid(gx, gy)
    m = W0.ravel() > 0
    # centre on the MEASURED cells: when they are collinear (one row of tiles) the
    # unidentifiable slope column is then exactly zero on them, and the minimum-norm
    # solution extrapolates a constant instead of a spurious tilt
    w_m = W0.ravel()[m]
    x0 = float(np.average(GX.ravel()[m], weights=w_m))
    y0 = float(np.average(GY.ravel()[m], weights=w_m))
    sc = max(float(np.ptp(gx)), float(np.ptp(gy)), 1.0)
    A = np.column_stack(
        [np.ones(GX.size), (GX.ravel() - x0) / sc, (GY.ravel() - y0) / sc]
    )
    Am, Ym, w0 = A[m], Y.reshape(-1, 2)[m], W0.ravel()[m]
    wr = np.ones(m.sum())
    coef = np.zeros((3, 2))
    for _ in range(iterations):
        sw = np.sqrt(w0 * wr)
        coef = np.linalg.lstsq(Am * sw[:, None], Ym * sw[:, None], rcond=None)[0]
        r = np.linalg.norm(Ym - Am @ coef, axis=1)
        mad = float(np.median(np.abs(r - np.median(r))))
        c = HUBER_C * max(MAD_TO_SIGMA * mad, SIGMA_FLOOR_PX)
        wr = np.where(r <= c, 1.0, c / np.maximum(r, 1e-300))
    field = (A @ coef).reshape(GX.shape + (2,))
    # back to pixel coordinates: u = a + b (x - x0)/sc + c (y - y0)/sc
    b, cc = coef[1] / sc, coef[2] / sc
    a = coef[0] - b * x0 - cc * y0
    return field, np.stack([a, b, cc]), wr


def _affine_params(coef):
    """``u = a + b x + c y`` -> the map ``x + u`` as translation, rotation, scale, shear.

    ``M = I + [[b_x, c_x], [b_y, c_y]] = R(theta) @ [[scale_x, shear], [0, scale_y]]``.
    """
    a, b, c = coef
    M = np.array([[1.0 + b[0], c[0]], [b[1], 1.0 + c[1]]])
    theta = float(np.arctan2(M[1, 0], M[0, 0]))
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    U = R.T @ M
    return {
        "tx": float(a[0]),
        "ty": float(a[1]),
        "rotation_deg": float(np.degrees(theta)),
        "scale_x": float(U[0, 0]),
        "scale_y": float(U[1, 1]),
        "shear": float(U[0, 1]),
    }


def _dct_eigenvalues(ny, nx, gx, gy):
    """Eigenvalues of the reflective discrete Laplacian on the lattice (Garcia 2010).

    Anisotropic spacing is honoured (Garcia's smoothn): each axis' term is divided by
    its squared step relative to the larger step, so ``s`` is in grid units.
    """
    hx = float(np.median(np.diff(gx))) if nx > 1 else 1.0
    hy = float(np.median(np.diff(gy))) if ny > 1 else 1.0
    hmax = max(abs(hx), abs(hy), 1e-12)
    ly = (-2.0 + 2.0 * np.cos(np.arange(ny) * np.pi / ny)) / (abs(hy) / hmax) ** 2
    lx = (-2.0 + 2.0 * np.cos(np.arange(nx) * np.pi / nx)) / (abs(hx) / hmax) ** 2
    return ly[:, None] + lx[None, :]


def _pls_fit(R, W, Lam, s, Z0=None, tol=PLS_TOL, max_iter=PLS_MAX_ITER):
    """Weighted penalised least squares on the lattice, one shared ``s`` for x and y.

    ``argmin_Z sum W |Z - R|^2 + s |Laplacian Z|^2`` with reflective boundaries, by
    Garcia's fixed point ``Z <- IDCT(G * DCT(W (R - Z) + Z))``, ``G = 1/(1 + s Lam^2)``,
    over-relaxed by 1.75 when the weights are not all one. ``W`` must lie in [0, 1].
    """
    from scipy.fft import dctn, idctn

    G = (1.0 / (1.0 + s * Lam**2))[..., None]
    complete = bool(np.all(W == 1.0))
    rf = 1.0 if complete else 1.75
    Z = R.copy() if Z0 is None else Z0.copy()
    Wv = W[..., None]
    for _ in range(max_iter):
        D = dctn(Wv * (R - Z) + Z, axes=(0, 1), norm="ortho")
        Zn = idctn(G * D, axes=(0, 1), norm="ortho")
        Zn = rf * Zn + (1.0 - rf) * Z
        change = np.linalg.norm(Zn - Z) / max(np.linalg.norm(Zn), 1e-12)
        Z = Zn
        if complete or change < tol:
            break
    return Z


def _gcv_s(R, W, Lam, log10_grid=LOG10_S_FINE):
    """Garcia's GCV score, minimised over ``log10 s`` on ``log10_grid``.

    A grid search, not Brent: on a small lattice the GCV curve has a long flat tail at
    large ``s`` (only the constant survives) where a bracketing minimiser settles on a
    spurious local minimum.

    Used only below ``MIN_CV_CELLS`` valid cells, i.e. a coarse grid of one vector per
    large tile. Deliberately NOT floored at ``s >= 1``: that floor is a patch for
    correlated errors between 50 %-overlapping windows (Altman 1990), which h-block CV
    now handles properly on the grids where it applies. On a coarse grid the residual
    after the affine is mostly real, unresolved deformation, and the floor smooths it
    away -- measured on the 16-tile synthetic slide: 3.7 px median with the floor
    against 2.24 px for GCV's own choice (interpolation, same as raw bilinear).
    """
    from scipy.fft import dctn, idctn

    grid = np.asarray(log10_grid)
    n = Lam.size
    nobs = max(int((W > 0).sum()), 1)
    Z = None
    best, best_s = np.inf, float(10 ** grid[0])
    for p in grid[::-1]:  # large s -> small s, warm-started
        s = float(10**p)
        Z = _pls_fit(R, W, Lam, s, Z0=Z, tol=PLS_TOL_CV)
        # Garcia's GCV at the fixed point: DCT of the pseudo-data, G applied, trace = sum G
        D = dctn(W[..., None] * (R - Z) + Z, axes=(0, 1), norm="ortho")
        G = 1.0 / (1.0 + s * Lam**2)
        Zp = idctn(G[..., None] * D, axes=(0, 1), norm="ortho")
        rss = float((W[..., None] * (R - Zp) ** 2).sum()) / 2 / nobs
        score = rss / max(1.0 - G.sum() / n, 1e-12) ** 2
        if score < best:
            best, best_s = score, s
    return best_s


def _bisquare(R, Z, W0, s):
    """Tukey bisquare weights on the studentised vector-residual norm (Garcia 2010)."""
    t = np.sqrt(1.0 + 16.0 * s)
    h = (np.sqrt(1.0 + t) / np.sqrt(2.0) / t) ** 2  # average leverage, 2-D
    r = np.linalg.norm(R - Z, axis=-1)
    m = W0 > 0
    if not m.any():
        return np.ones_like(W0)
    mad = float(np.median(np.abs(r[m] - np.median(r[m]))))
    # Garcia: u = r / (1.4826 MAD sqrt(1 - h)). The floor bounds that denominator, so a
    # residual under BISQUARE_C * SIGMA_FLOOR_PX is never rejected however small s is.
    scale = max(MAD_TO_SIGMA * mad * np.sqrt(max(1.0 - h, 0.0)), SIGMA_FLOOR_PX)
    u = r / scale
    return np.where(u < BISQUARE_C, (1.0 - (u / BISQUARE_C) ** 2) ** 2, 0.0)


def _robust_pls(R, W0, Lam, s, Wr=None, iterations=ROBUST_ITERATIONS):
    """Robust iterations at a FIXED ``s``: fit, re-weight by bisquare, refit."""
    Wr = np.ones_like(W0) if Wr is None else Wr
    Z = None
    for _ in range(iterations):
        Z = _pls_fit(R, W0 * Wr, Lam, s, Z0=Z)
        Wr = _bisquare(R, Z, W0, s)
    Z = _pls_fit(R, W0 * Wr, Lam, s, Z0=Z)
    return Z, Wr


def _hblock_folds(W, k=CV_FOLDS, block=CV_BLOCK):
    """Fold id per cell: 3x3-cell patches dealt round-robin (deterministic) to ``k`` folds."""
    ny, nx = W.shape
    by, bx = np.meshgrid(np.arange(ny) // block, np.arange(nx) // block, indexing="ij")
    nbx = (nx + block - 1) // block
    bid = by * nbx + bx
    # a fixed permutation so neighbouring patches do not land in the same fold
    nblocks = int(bid.max()) + 1
    order = np.random.default_rng(0).permutation(nblocks)
    return order[bid] % k


def _hblock_cv_errors(R, W, Lam, log10_grid=CV_LOG10_S):
    """Held-out error norm per cell and per ``s``, over h-block folds.

    Returns an array ``(len(log10_grid), ny, nx)``: for each valid cell, the norm of
    ``fit - observation`` where the fit was made WITHOUT that cell's whole 3x3 patch;
    NaN elsewhere. ``None`` when no fold has both held-out and training cells.
    """
    folds = _hblock_folds(W)
    valid = W > 0
    E = np.full((len(log10_grid),) + W.shape, np.nan)
    any_fold = False
    for f in range(CV_FOLDS):
        hold = valid & (folds == f)
        Wt = np.where(hold, 0.0, W)
        if not hold.any() or not (Wt > 0).any():
            continue
        any_fold = True
        Z = None
        # large s -> small s: each fit warm-starts from a smoother neighbour
        for i in range(len(log10_grid) - 1, -1, -1):
            Z = _pls_fit(R, Wt, Lam, 10 ** log10_grid[i], Z0=Z, tol=PLS_TOL_CV)
            E[i][hold] = np.linalg.norm(Z[hold] - R[hold], axis=-1)
    return E if any_fold else None


def _huber_losses(E, W, c):
    """Weighted mean Huber loss of held-out errors, one value per row of ``E``."""
    held = ~np.isnan(E[0])
    w = W[held]
    out = []
    for e in E:
        x = e[held]
        rho = np.where(x <= c, 0.5 * x**2, c * (x - 0.5 * c))
        out.append(float(np.sum(w * rho) / np.sum(w)))
    return np.asarray(out)


def _hblock_select(R, W, Lam):
    """``log10 s`` by h-block CV: coarse scan, then a +-``CV_REFINE`` refinement.

    The score is the weighted mean Huber loss of the held-out errors. A squared loss
    would let one wildly wrong vector (large held-out error at every ``s``) steer the
    choice, so the loss is Huber with ONE scale for all candidates: ``1.345 x`` the
    robust scale of the errors at the coarse ``s`` with the smallest median held-out
    error. Returns ``(s, held-out errors at s)`` or ``None`` when there is no fold.
    """
    E = _hblock_cv_errors(R, W, Lam, CV_LOG10_S)
    if E is None:
        return None
    held = ~np.isnan(E[0])
    med = np.array([np.median(e[held]) for e in E])
    c = HUBER_C * max(MAD_TO_SIGMA * float(med[int(np.argmin(med))]), SIGMA_FLOOR_PX)
    i = int(np.argmin(_huber_losses(E, W, c)))
    p = float(CV_LOG10_S[i])
    fine = np.array([p - CV_REFINE, p + CV_REFINE])
    fine = fine[(fine >= LOG10_S_FINE[0]) & (fine <= LOG10_S_FINE[-1])]
    cand, errs = [p], [E[i]]
    if fine.size:
        Ef = _hblock_cv_errors(R, W, Lam, fine)
        cand += list(fine)
        errs += list(Ef)
    errs = np.asarray(errs)
    j = int(np.argmin(_huber_losses(errs, W, c)))
    return float(10 ** cand[j]), errs[j]


def _dctpls_core(Y, W0, gx, gy):
    """Robust affine + robust DCT-PLS on a regular lattice.

    ``Y`` is ``(ny, nx, 2)`` observations, ``W0`` prior weights (0 = no data),
    ``gx``/``gy`` the node coordinates in pixels. Returns ``(field, info)``: the
    displacement at EVERY node (holes filled by the smoother, never zeroed) and a
    JSON-serialisable dict of what was done. Independent of how the lattice was
    built, so a later phase can hand it many points per tile on a finer lattice.
    """
    Y = np.asarray(Y, dtype=float)
    W0 = np.asarray(W0, dtype=float)
    ny, nx = W0.shape
    valid = W0 > 0
    n_valid = int(valid.sum())
    info = {
        "n_valid": n_valid,
        "n_downweighted": 0,
        "affine": None,
        "smoothing_s": None,
        "smoothing_selection": "none",
        "holdout_rmse_px": None,
    }
    if n_valid == 0:
        return np.zeros((ny, nx, 2)), info
    W0n = np.where(valid, W0 / W0[valid].max(), 0.0)
    Y = np.where(valid[..., None], Y, 0.0)
    if n_valid < MIN_AFFINE_CELLS:
        t = np.median(Y[valid], axis=0)
        field = np.broadcast_to(t, (ny, nx, 2)).copy()
        info["smoothing_selection"] = "translation_only"
        info["affine"] = _affine_params(np.stack([t, np.zeros(2), np.zeros(2)]))
        return field, info
    aff, coef, huber_w = _robust_affine(Y, W0n, np.asarray(gx), np.asarray(gy))
    info["affine"] = _affine_params(coef)
    if n_valid < MIN_SMOOTH_CELLS:
        info["smoothing_selection"] = "affine_only"
        info["n_downweighted"] = int(np.sum(huber_w < 0.1))
        return aff, info
    R = np.where(valid[..., None], Y - aff, 0.0)
    Lam = _dct_eigenvalues(ny, nx, gx, gy)
    # Choose s ONCE, on the prior weights, then re-weight (bisquare) at that fixed s.
    # Alternating the two spirals: dropping the cells a fit disagrees with makes the rest
    # look cleaner, the selector then smooths less, more cells get dropped -- measured
    # on a 63x63 synthetic lattice, 28 % of vectors downweighted and 0.35 px against
    # 0.28 px for one selection. Robustness in the selection comes from its loss instead.
    picked = _hblock_select(R, W0n, Lam) if n_valid >= MIN_CV_CELLS else None
    if picked is not None:
        s, e_held = picked
        selection = "hblock_cv"
    else:
        s, e_held = _gcv_s(R, W0n, Lam), None
        selection = "gcv"
    Z, Wr = _robust_pls(R, W0n, Lam, s)
    rmse = None
    if e_held is not None:
        held = ~np.isnan(e_held)
        # outliers the robust fit rejected do not count against the field's accuracy
        wh = (W0n * Wr)[held]
        if wh.sum() > 0:
            rmse = float(np.sqrt(np.sum(wh * e_held[held] ** 2) / wh.sum()))
    info["smoothing_s"] = float(s)
    info["smoothing_selection"] = selection
    info["holdout_rmse_px"] = rmse
    info["n_downweighted"] = int(np.sum(Wr[valid] < 0.1))
    return aff + Z, info


def solve_dctpls(controls, max_error=None, max_disp=None, **kw):
    """Robust affine, then robust DCT-PLS of the residual; no dead zone, no rescale.

    There is no TRE gate here -- a caller's ``gate_tre`` never reaches this solver.
    A displacement below any threshold is a measurement, not a reason to write
    ``[0, 0]``: hard-zeroing it poisons every neighbour-based estimate and puts a step
    in the field. Invalid controls (NaN / low-confidence / out-of-range, ``accept``)
    carry weight 0 and their nodes are filled by the smoother.

    Stages: (1) weighted Huber IRLS affine, subtracted -- DCT-PLS's null space is only a
    constant, so without this a residual rotation from M0 would be charged as
    roughness and shrunk; (2) Garcia's (2010) robust DCT-PLS (thin-plate penalty,
    reflective boundaries, bisquare weights) on the residual, with ``s`` chosen by
    h-block cross-validation (>= 25 valid cells) or floored GCV (6-24); fewer cells
    fall back to affine-only (3-5), translation-only (1-2) or no mesh (0).

    Extra keyword arguments are accepted and ignored so the dispatcher can pass one
    set. Returns ``(grid_x, grid_y, disp, report)`` like the other solvers.
    """
    del kw
    grid_x, grid_y, Y, W0, counts = _lattice_from_controls(
        controls, max_error, max_disp
    )
    field, info = _dctpls_core(Y, W0, grid_x, grid_y)
    jac = jacobian_report(grid_x, grid_y, field)
    report = {
        "solver": "dctpls",
        "n_controls": len(controls),
        "n_rejected_error": counts["error"],
        "n_rejected_disp": counts["disp"],
        "n_legacy_unscored": counts["legacy"],
        "n_valid": info["n_valid"],
        "n_downweighted": info["n_downweighted"],
        "affine": info["affine"],
        "smoothing_s": info["smoothing_s"],
        "smoothing_selection": info["smoothing_selection"],
        "holdout_rmse_px": info["holdout_rmse_px"],
        "lipschitz": jac["max_operator_norm"],
        "min_det_jacobian": jac["min_jacobian_det"],
        "fold_certificate_ok": bool(
            jac["max_operator_norm"] < FOLD_CERTIFICATE_LIPSCHITZ
        ),
        "measured": (W0 > 0).astype(int).tolist(),
    }
    return (
        [float(v) for v in grid_x],
        [float(v) for v in grid_y],
        field.tolist(),
        report,
    )


def solve_grid(
    controls, gate_tre, max_error=None, max_disp=None, solver="dctpls", **kw
):
    """Dispatch on ``solver``; see the module docstring for what each does.

    ``gate_tre`` is honoured by ``legacy`` and ``robust`` and IGNORED by ``dctpls``,
    which never zeroes a measured displacement however small.
    """
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; expected one of {SOLVERS}")
    if solver == "legacy":
        return solve_legacy(controls, gate_tre, max_error, max_disp, **kw)
    if solver == "dctpls":
        return solve_dctpls(controls, max_error, max_disp, **kw)
    return solve_robust(controls, gate_tre, max_error, max_disp, **kw)
