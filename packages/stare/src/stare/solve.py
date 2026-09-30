"""The SOLVE stage: REG_TILE's window vectors -> a control-grid displacement mesh.

Every comparable method (approximating TPS, elastix FFD, RegWSI's diffusive solve, PIV)
turns sparse, noisy displacement measurements into a dense field by
*reject -> regularise -> densify*. STARE does it with one solver, ``dctpls``, on the
slide-global vector lattice REG_TILE measures (``stare.vector_grid``):

0. **the lattice** -- every tile's window vectors laid on one lattice (node ``k`` at
   ``origin + k * stride``). A vector is valid when it is finite and ``|d| < max_disp`` (the
   range gate: a peak further than the read window is an artefact). There is no
   correlation-error gate -- it drops good window vectors -- and no TRE gate: a small
   displacement is a measurement, never ``[0, 0]``. Background is kept out upstream:
   REG_TILE emits only foreground-masked vectors whose peak ratio clears its floor.
1. **robust affine** -- weighted Huber IRLS of a 6-parameter affine, subtracted. DCT-PLS's
   null space is only a constant (an affine is penalised through the boundary rows), so
   without this a residual rotation left by M0 is charged as roughness and shrunk.
2. **robust DCT-PLS** of the residual (Garcia 2010, CSDA 54:1167): a thin-plate
   (squared-Laplacian) penalty with reflective boundaries, solved by DCT; one shared
   smoothing parameter ``s`` for x and y; bisquare weights on the vector-residual norm.
   Missing nodes are filled by the smoother, not zeroed. Both robust steps scale the
   residual NORM by its Rayleigh median (``median |r| / 1.1774``), not by the 1-D
   ``1.4826 MAD``, and take their cutoffs from chi-square with 2 dof (``HUBER_C``,
   ``BISQUARE_C``).
3. **s chosen once, from the data** -- spatial block cross-validation (5 folds of 3x3-node
   patches) when there are >= 25 valid nodes, GCV below that; fewer nodes degrade to
   affine-only (3-5), translation-only (1-2) or no mesh (0). Whether the held-out patches
   get a 1-cell buffer ring (true h-block CV, Burman, Chow & Nolan 1994; ``hblock_cv``) or
   not (``block_cv``) is decided from the data, as Valavi et al. size the buffer from the
   residuals' spatial autocorrelation: the ring is used only when the lag-1 noise
   correlation estimated from the plain block-CV fit's robust-weighted residuals exceeds
   ``CV_BUFFER_RHO`` (reported as ``residual_lag1_rho`` and ``cv_buffer``; see the
   constants).
4. **sigma calibration** -- per-vector sigma from block held-out residuals binned by peak
   ratio, calibrated on folds 0/2/4 and scored (``coverage_1sigma``,
   ``rms_error_over_rms_sigma``, over the vectors the robust fit kept) on the disjoint
   folds 1/3; then one re-solve at ``w = 1/sigma^2`` at the SAME ``s`` (in mean-weight
   units), not a re-scan.
5. **re-index** -- the field is re-indexed to the moving frame the stitch evaluates it in,
   ``F(g) = D(g + F(g))``, iterated to a 1e-3 px fixed-point residual, which is reported
   (``reindex_residual_px``).
6. **fold certificate** -- the Lipschitz constant and ``min det(I + J)`` of the field as
   the mesh INTERPOLATES it (sub-grid at stride/4) are reported, and
   ``fold_certificate_ok`` when the constant is below 0.5. The field is never rescaled.

History: until STARE v2 (2026-09) SOLVE also carried a ``legacy`` solver (three gates, then a
median filter) and a ``robust`` one (gates, normalised median test, in-fill, Tikhonov), both
fed one control point per tile. On the 16-tile synthetic slide ``robust`` scored 8.7 px
median against 2.2 px for the raw tile vectors (``research/stare-optimal-design-2026-09-27.md``
§0, §3). Both were removed with the one-point-per-tile path, so a control JSON without
``vectors`` (written by REG_TILE before the vector grid) cannot be re-solved: re-run REG_TILE.

``solve_dctpls`` returns ``(grid_x, grid_y, disp, report)``; ``report`` is what ``*_tre.json``
and the manifest record about the solve.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def _in_range(dx, dy, max_disp):
    """A vector is usable when it is finite and inside the range gate (``None`` = no bound)."""
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return False
    return max_disp is None or float(np.hypot(dx, dy)) < max_disp


def _require_vectors(controls):
    """Refuse a control JSON written before the vector grid, with the remedy in the message."""
    missing = [c for c in controls if "vectors" not in c or "lattice" not in c]
    if missing:
        names = ", ".join(f"({c.get('ix')},{c.get('iy')})" for c in missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise ValueError(
            f"{len(missing)}/{len(controls)} control point(s) carry no 'vectors'/'lattice' "
            f"(tiles {names}{more}): they were written by a REG_TILE that predates the "
            "window-vector grid (STARE v1, one point per tile). STARE v2's SOLVE solves only "
            "the vector lattice and cannot re-solve them -- re-run REG_TILE for these tiles "
            "(a -resume after the upgrade re-runs it, the task script changed)."
        )


def tile_accepted(control, max_disp):
    """Did this tile contribute at least one vector to the mesh?

    The per-tile ``accepted`` flag of the ``*_tre.json`` report: ``True`` when the tile
    carries a finite window vector inside the range gate. It is a report of what SOLVE used,
    not a gate of its own -- the vectors are the unit SOLVE accepts or rejects.
    """
    return any(
        _in_range(float(v[4]), float(v[5]), max_disp)
        for v in control.get("vectors", [])
    )


# The fold certificate samples the interpolated field every ``stride / JACOBIAN_SUBSTEPS``
# (the cubic B-spline can overshoot BETWEEN nodes, where node differences never look), with
# central differences of +-JACOBIAN_FD_FRACTION of a lattice step.
JACOBIAN_SUBSTEPS = 4
JACOBIAN_FD_FRACTION = 0.01


def _axis_samples(grid, substeps):
    """Points every ``1/substeps`` of a node step along a 1-D ascending ``grid``."""
    g = np.asarray(grid, dtype=float)
    if g.size < 2:
        return g.copy()
    t = np.linspace(0.0, g.size - 1.0, (g.size - 1) * substeps + 1)
    return np.interp(t, np.arange(g.size, dtype=float), g)


def jacobian_report(grid_x, grid_y, disp, interp="bilinear"):
    """Lipschitz and folding diagnostics of the field AS THE MESH INTERPOLATES IT.

    The 2x2 Jacobian ``J = du/dx`` is evaluated on the ``interp`` interpolant
    (``MeshField``) at a sub-grid of ``JACOBIAN_SUBSTEPS`` points per node step, by central
    differences clamped to the lattice. ``max_operator_norm`` is the largest spectral norm
    (the field's Lipschitz constant) and ``min_jacobian_det`` the smallest ``det(I + J)``
    (negative means a fold); ``L < 1`` guarantees ``det(I + J) >= (1 - L)^2 > 0``. A
    single-row or single-column grid has no gradient along that axis.
    """
    from stare.mesh_field import MeshField

    gx = np.asarray(grid_x, dtype=float)
    gy = np.asarray(grid_y, dtype=float)
    disp = np.asarray(disp, dtype=float)
    if disp.size == 0:
        return {"max_operator_norm": 0.0, "min_jacobian_det": 1.0}
    mesh = MeshField(gx, gy, disp, interp=interp)
    xs, ys = _axis_samples(gx, JACOBIAN_SUBSTEPS), _axis_samples(gy, JACOBIAN_SUBSTEPS)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()

    def partial(coord, other, grid, along_x):
        if grid.size < 2:
            return np.zeros((coord.size, 2))
        h = JACOBIAN_FD_FRACTION * float(np.min(np.diff(grid)))
        hi = np.minimum(coord + h, grid[-1])
        lo = np.maximum(coord - h, grid[0])
        if along_x:
            a, b = np.column_stack([hi, other]), np.column_stack([lo, other])
        else:
            a, b = np.column_stack([other, hi]), np.column_stack([other, lo])
        return (mesh.displacement(a) - mesh.displacement(b)) / (hi - lo)[:, None]

    du_dx = partial(X, Y, gx, True)
    du_dy = partial(Y, X, gy, False)
    # J = [[dux/dx, dux/dy], [duy/dx, duy/dy]] per sample
    j = np.stack(
        [
            np.stack([du_dx[:, 0], du_dy[:, 0]], axis=-1),
            np.stack([du_dx[:, 1], du_dy[:, 1]], axis=-1),
        ],
        axis=-2,
    )
    norms = np.linalg.norm(j, ord=2, axis=(-2, -1))
    dets = np.linalg.det(np.eye(2) + j)
    return {
        "max_operator_norm": float(norms.max()) if norms.size else 0.0,
        "min_jacobian_det": float(dets.min()) if dets.size else 1.0,
    }


# ── dctpls ────────────────────────────────────────────────────────────────────
# The robust scale of a 2-D residual. Every robust step below works on the NORM ``|r|`` of a
# vector residual, and for an isotropic Gaussian residual with per-component sigma that norm
# is Rayleigh-distributed, not Gaussian: 1.4826 x MAD (the 1-D Gaussian rule) of Rayleigh
# norms is ~0.66 sigma, and a Huber threshold built on it down-weighted ~67 % of clean data.
# The Rayleigh median is sigma sqrt(2 ln 2), so sigma_hat = median(|r|) / 1.1774.
RAYLEIGH_MEDIAN = float(np.sqrt(2.0 * np.log(2.0)))  # 1.17741
# Huber threshold on the NORM, in sigma_hat units: the 95 % point of chi-square(2 dof),
# sqrt(-2 ln 0.05) = 2.4477, so exactly 5 % of clean isotropic Gaussian residuals are
# down-weighted (the role 1.345 plays for a 1-D residual, whose 95 % efficiency point it is).
HUBER_C = float(np.sqrt(-2.0 * np.log(0.05)))  # 2.44775
# Tukey's bisquare cutoff, as Garcia (2010) uses for the robust weights, carried from 1-D to
# the norm at the SAME tail probability: 4.685 sigma leaves P(|z| > 4.685) = 2.80e-6 of a 1-D
# Gaussian; the chi-square(2) norm with that tail is sqrt(-2 ln 2.80e-6) = 5.0569 sigma.
BISQUARE_C_1D = 4.685
BISQUARE_C = float(
    np.sqrt(-2.0 * np.log(math.erfc(BISQUARE_C_1D / math.sqrt(2.0))))
)  # 5.0569


def _norm_scale(r):
    """Robust per-component sigma of 2-D residuals from their norms: ``median(|r|) / 1.1774``."""
    r = np.asarray(r, dtype=float)
    return float(np.median(r)) / RAYLEIGH_MEDIAN if r.size else 0.0


# The robust scale never drops below this (px): a per-window phase-correlation vector is
# not more precise than ~0.1 px (0.10-0.25 px measured, research peaklock.py). Without a
# floor, a near-exact field -- or any fit at small s, whose residuals collapse toward zero
# -- gives a scale ~ 0 and bisquare then rejects a real feature over a hundredth of a pixel.
# With it, a disagreement under ~0.5 px (5.06 x 0.1) is never called an outlier.
SIGMA_FLOOR_PX = 0.1
AFFINE_ITERATIONS = 10
ROBUST_ITERATIONS = 6
# Block cross-validation: K folds of 3x3-cell patches; a held-out patch is left out of the
# training fit and its cells are scored. Around each patch there can be a 1-cell buffer ring
# that is ALSO left out of training but not scored -- true h-block CV (Burman, Chow & Nolan
# 1994), for errors correlated between neighbouring vectors: without the ring the cells just
# outside a patch carry part of the held-out cells' error into the prediction, and CV
# undersmooths (De Brabanter et al. 2011). On noise correlated over ~1 cell the ring picks s
# ~1000x larger and a 35 % smaller error
# (test_hblock_cv_with_a_buffer_ring_does_not_undersmooth_correlated_noise). Where the errors
# do NOT correlate the ring only costs: on the 8192^2 synthetic slide (Hann-windowed, 50 %
# overlap, error lag-1 correlation 0.01-0.07) its 5x5 holes turn the CV into an
# extrapolation problem and the field error rose ~50 % (median 0.13-0.15 -> 0.20-0.22 px).
#
# So the ring is chosen per slide, as Valavi et al. (2019) size the buffer from the
# residuals' spatial autocorrelation. Plain block CV selects s first (the initial fit); the
# lag-1 correlation of the noise between 4-neighbour nodes is estimated from that fit's
# robust-weighted residuals (``_noise_lag1_rho``); above CV_BUFFER_RHO the h-block
# selection (1-cell ring) re-selects s, otherwise the plain choice stands.
#
# The raw residual correlation cannot be thresholded as it is: the residual is (I - H) e, and
# the smoother's high-pass imprint dominates it. Measured on 48x48 lattices, iid noise reads
# -0.13 at plain CV's s, and noise with true lag-1 0.58 reads -0.16 because plain CV then
# interpolates (s at the grid floor) and whitens it -- the very failure the ring exists for.
# So the same statistic is taken on two reference noises pushed through the SAME fit
# (weights, s): iid noise (lag-1 0) and a 2x2 box average of it, which is exactly the linear
# 50 %-overlap window model (lag-1 0.5, diagonal 0.25, 0 beyond; solve NOTES S4); the
# estimate is the observed value placed linearly between the two, scaled to that 0 .. 0.5.
# The statistic is the sign correlation (Sheppard: P(same sign) = 1/2 + arcsin(rho) / pi),
# not the product moment, so a single lattice-scale FEATURE left in the residuals -- a
# 3-cell bump an interpolating fit leaves as a large local residual -- does not read as
# correlated noise: measured on the bump test's 10x10 lattice 0.02-0.12 (product moment
# 0.43-0.55, which would have smoothed the bump away). On 48x48 lattices: iid -0.04, true
# 0.10 -> 0.05, 0.58 -> 0.52, 0.77 -> 1.0.
#
# CV_BUFFER_RHO = 0.2: the linear 50 %-overlap model (research/stare-papers/oa/solve/NOTES.md
# S4) predicts 0.5, far above it, while the
# Hann-windowed vectors REG_TILE actually emits correlate at 0.01-0.07 against the truth,
# far below it -- so the ring stays off on those slides, where it cost ~50 % field error.
#
# The residual cannot tell correlated noise from lattice-scale SIGNAL an interpolating fit
# leaves behind everywhere (a wave of ~6 nodes period read 1.3-1.4 on a 15x15 lattice). The
# ring's own choice can: there it hides a whole period, and h-block CV picks the flattest
# field on the grid (s = 1e6; the wave is lost, field error 0.17 -> 2.1 px on
# tests/test_tiled_pipeline.py's local-deformation slide). So a ring whose s reaches
# CV_RING_MAX_LOG10_S -- a field with no local structure left -- is refused and plain block
# CV stands (logged). On correlated noise the ring's s is interior (0.56 on the 48x48 tests).
CV_BUFFER_RHO = 0.2
CV_RING_MAX_LOG10_S = 5.0
CV_BUFFER_RING = 1
CV_LABELS = {0: "block_cv", CV_BUFFER_RING: "hblock_cv"}
CV_FOLDS = 5
CV_BLOCK = 3
# log10 s candidates. GCV scans the fine grid; block CV scans the coarse one and then
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


def _robust_affine(Y, W0, gx, gy, iterations=AFFINE_ITERATIONS):
    """Weighted Huber IRLS of the 6-parameter affine ``u = a + B (x, y)``.

    Coordinates are centred and scaled for conditioning; the returned field is in
    pixels. ``c = HUBER_C * sigma_hat`` on the vector-residual norms, with
    ``sigma_hat = median(|r|) / 1.1774`` (the Rayleigh median), re-estimated every
    iteration: 5 % of clean isotropic Gaussian residuals are down-weighted. Returns ``(field (ny, nx, 2), coef (3, 2) in pixel coords,
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
        c = HUBER_C * max(_norm_scale(r), SIGMA_FLOOR_PX)
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
    correlated errors between 50 %-overlapping windows (Altman 1990), which block CV
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
    """Tukey bisquare weights on the studentised vector-residual norm (Garcia 2010).

    Garcia studentises a 1-D residual by ``1.4826 MAD sqrt(1 - h)``; here the residual is a
    2-D vector, so the scale is the Rayleigh one (``_norm_scale``) and the cutoff is
    ``BISQUARE_C`` = 5.06, the norm with the same tail probability as 4.685 in 1-D.
    """
    t = np.sqrt(1.0 + 16.0 * s)
    h = (np.sqrt(1.0 + t) / np.sqrt(2.0) / t) ** 2  # average leverage, 2-D
    r = np.linalg.norm(R - Z, axis=-1)
    m = W0 > 0
    if not m.any():
        return np.ones_like(W0)
    # Garcia: u = r / (sigma_hat sqrt(1 - h)). The floor bounds that denominator, so a
    # residual under BISQUARE_C * SIGMA_FLOOR_PX is never rejected however small s is.
    scale = max(_norm_scale(r[m]) * np.sqrt(max(1.0 - h, 0.0)), SIGMA_FLOOR_PX)
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


def _cv_folds(W, k=CV_FOLDS, block=CV_BLOCK):
    """Fold id per cell: 3x3-cell patches dealt round-robin (deterministic) to ``k`` folds."""
    ny, nx = W.shape
    by, bx = np.meshgrid(np.arange(ny) // block, np.arange(nx) // block, indexing="ij")
    nbx = (nx + block - 1) // block
    bid = by * nbx + bx
    # a fixed permutation so neighbouring patches do not land in the same fold
    nblocks = int(bid.max()) + 1
    order = np.random.default_rng(0).permutation(nblocks)
    return order[bid] % k


def _cv_split(folds, f, valid, buffer=0):
    """``(scored, excluded)`` for fold ``f``: the held-out cells and everything not trained on.

    ``scored`` is the fold's valid cells; ``excluded`` adds the ``buffer``-cell ring around
    them (a square dilation), which the training fit ignores and nothing scores.
    """
    from scipy.ndimage import binary_dilation

    in_fold = folds == f
    excluded = in_fold
    if buffer > 0:
        excluded = binary_dilation(
            in_fold, structure=np.ones((2 * buffer + 1, 2 * buffer + 1), dtype=bool)
        )
    return valid & in_fold, excluded


def _block_cv_errors(R, W, Lam, log10_grid=CV_LOG10_S, buffer=0):
    """Held-out error norm per cell and per ``s``, over block-CV folds.

    Returns an array ``(len(log10_grid), ny, nx)``: for each valid cell, the norm of
    ``fit - observation`` where the fit was made WITHOUT that cell's whole 3x3 patch and
    the ``buffer``-cell ring around it; NaN elsewhere. ``None`` when no fold has both
    held-out and training cells.
    """
    folds = _cv_folds(W)
    valid = W > 0
    E = np.full((len(log10_grid),) + W.shape, np.nan)
    any_fold = False
    for f in range(CV_FOLDS):
        hold, excluded = _cv_split(folds, f, valid, buffer)
        Wt = np.where(excluded, 0.0, W)
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


def _block_cv_select(R, W, Lam, coarse=None, buffer=0):
    """``log10 s`` by block CV: coarse scan, then a +-``CV_REFINE`` refinement.

    ``buffer`` is the h-block ring (0: plain spatial-block CV; 1: true h-block CV --
    ``_dctpls_core`` picks it per slide, see the constants).

    The score is the weighted mean Huber loss of the held-out errors. A squared loss
    would let one wildly wrong vector (large held-out error at every ``s``) steer the
    choice, so the loss is Huber with ONE scale for all candidates: ``HUBER_C x`` the
    Rayleigh scale (``median / 1.1774``) of the error norms at the coarse ``s`` with the
    smallest median held-out error. Returns ``(s, held-out errors at s)`` or ``None`` when there is no fold.
    """
    coarse = CV_LOG10_S if coarse is None else np.asarray(coarse, dtype=float)
    E = _block_cv_errors(R, W, Lam, coarse, buffer=buffer)
    if E is None:
        return None
    held = ~np.isnan(E[0])
    med = np.array([np.median(e[held]) for e in E])
    c = HUBER_C * max(float(med.min()) / RAYLEIGH_MEDIAN, SIGMA_FLOOR_PX)
    i = int(np.argmin(_huber_losses(E, W, c)))
    p = float(coarse[i])
    fine = np.array([p - CV_REFINE, p + CV_REFINE])
    fine = fine[(fine >= LOG10_S_FINE[0]) & (fine <= LOG10_S_FINE[-1])]
    cand, errs = [p], [E[i]]
    if fine.size:
        Ef = _block_cv_errors(R, W, Lam, fine, buffer=buffer)
        cand += list(fine)
        errs += list(Ef)
    errs = np.asarray(errs)
    j = int(np.argmin(_huber_losses(errs, W, c)))
    return float(10 ** cand[j]), errs[j]


def _sign_lag1(E, W):
    """Lag-1 sign correlation of residual vectors ``E`` between 4-neighbour nodes.

    Over both components and both lattice directions, pairs of nodes with ``W > 0``:
    ``sin(pi (P(same sign) - 1/2))``, which is the correlation for a Gaussian pair
    (Sheppard) and is bounded by one sign per node however large a residual is. ``None``
    without a pair.
    """
    ok = (np.asarray(W) > 0) & np.all(np.isfinite(E), axis=-1)
    same = total = 0
    for a, b, m in (
        (E[:, :-1], E[:, 1:], ok[:, :-1] & ok[:, 1:]),
        (E[:-1], E[1:], ok[:-1] & ok[1:]),
    ):
        p = np.sign(a[m] * b[m])
        same += int((p > 0).sum())
        total += int((p != 0).sum())
    if total == 0:
        return None
    return float(np.sin(np.pi * (same / total - 0.5)))


def _noise_lag1_rho(R, W, Lam, s, Z, Wr):
    """Lag-1 correlation of the NOISE behind the fit ``Z`` of ``R``, imprint-corrected.

    ``_sign_lag1`` of the residual, placed between the same statistic on iid noise (0) and
    on the 50 %-overlap window model (0.5), each fitted with the same weights ``W * Wr``
    and ``s`` (see ``CV_BUFFER_RHO``). ``None`` when the fit cannot tell the two references
    apart (they read within 0.05 of each other) or a statistic is undefined.
    """
    Wf = W * Wr
    obs = _sign_lag1(R - Z, Wf)
    x = np.random.default_rng(0).normal(0.0, 1.0, (W.shape[0] + 1, W.shape[1] + 1, 2))
    white = x[:-1, :-1]
    box = (x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:]) / 2.0
    r_white = _sign_lag1(white - _pls_fit(white, Wf, Lam, s), Wf)
    r_box = _sign_lag1(box - _pls_fit(box, Wf, Lam, s), Wf)
    if obs is None or r_white is None or r_box is None or r_box - r_white < 0.05:
        return None
    return 0.5 * (obs - r_white) / (r_box - r_white)


def _select_s_adaptive(R, W, Lam):
    """``(s, held-out errors, buffer, rho, (Z, Wr) or None)`` -- plain block CV, then the
    h-block ring only if the noise correlates (``CV_BUFFER_RHO``) and the ring's ``s`` is
    below ``CV_RING_MAX_LOG10_S``.

    The robust fit at the plain ``s`` is returned when that ``s`` stands, so it is not
    recomputed. ``None`` when block CV has no fold.
    """
    plain = _block_cv_select(R, W, Lam, buffer=0)
    if plain is None:
        return None
    s0, e0 = plain
    Z0, Wr0 = _robust_pls(R, W, Lam, s0)
    rho = _noise_lag1_rho(R, W, Lam, s0, Z0, Wr0)
    if rho is not None and rho > CV_BUFFER_RHO:
        ring = _block_cv_select(R, W, Lam, buffer=CV_BUFFER_RING)
        if ring is not None and np.log10(ring[0]) < CV_RING_MAX_LOG10_S:
            return ring[0], ring[1], CV_BUFFER_RING, rho, None
        if ring is not None:
            logger.info(
                f"h-block CV refused: lag-1 noise correlation {rho:.2f} > "
                f"{CV_BUFFER_RHO}, but the ring chose s = {ring[0]:.3g} (>= "
                f"1e{CV_RING_MAX_LOG10_S:g}, a flat field): the residual structure is "
                "lattice-scale signal the ring hides, not noise; plain block CV kept"
            )
    return s0, e0, 0, rho, (Z0, Wr0)


def _dctpls_core(Y, W0, gx, gy, s_fixed=None, internals=None, cv_buffer=0):
    """Robust affine + robust DCT-PLS on a regular lattice.

    ``Y`` is ``(ny, nx, 2)`` observations, ``W0`` prior weights (0 = no data),
    ``gx``/``gy`` the node coordinates in pixels. Returns ``(field, info)``: the
    displacement at EVERY node (holes filled by the smoother, never zeroed) and a
    JSON-serialisable dict of what was done. Independent of how the lattice was
    built (the unit tests feed it synthetic lattices directly).

    ``s_fixed`` skips the selection and solves at that ``s`` -- the re-solve at calibrated
    weights reuses the first solve's choice. ``s`` is in units of the MEAN data weight: the
    objective ``sum W |Z - R|^2 + s |Lap Z|^2`` is unchanged by scaling ``W`` and ``s``
    together, so the fit runs at ``s_fixed * mean(W)`` over the valid cells (the first solve
    has uniform weights, mean 1, where the two agree). ``internals``, when a dict, receives
    the affine field, the residual ``R``, the eigenvalues, the normalised weights and the
    chosen ``s`` -- what the sigma calibration needs to re-fit held-out folds at that ``s``.
    ``cv_buffer`` is the ring the ``s_fixed`` path scores its held-out error with (the one
    the first solve chose); a selecting solve decides it itself (``_select_s_adaptive``).
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
        "residual_lag1_rho": None,
        "cv_buffer": None,
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
    # The sigma re-solve (s_fixed) does not scan again either.
    wbar = float(W0n[valid].mean())
    fitted = None
    if s_fixed is not None:
        s = float(s_fixed) * wbar
        selection = "fixed"
        e_held = None
        if n_valid >= MIN_CV_CELLS:
            E = _block_cv_errors(R, W0n, Lam, np.array([np.log10(s)]), buffer=cv_buffer)
            e_held = None if E is None else E[0]
            info["cv_buffer"] = int(cv_buffer)
    else:
        picked = _select_s_adaptive(R, W0n, Lam) if n_valid >= MIN_CV_CELLS else None
        if picked is not None:
            s, e_held, buffer, rho, fitted = picked
            selection = CV_LABELS[buffer]
            info["cv_buffer"] = int(buffer)
            info["residual_lag1_rho"] = None if rho is None else float(rho)
        else:
            s, e_held = _gcv_s(R, W0n, Lam), None
            selection = "gcv"
    if fitted is not None:
        Z, Wr = fitted
    else:
        Z, Wr = _robust_pls(R, W0n, Lam, s)
    rmse = None
    if e_held is not None:
        held = ~np.isnan(e_held)
        # outliers the robust fit rejected do not count against the field's accuracy
        wh = (W0n * Wr)[held]
        if wh.sum() > 0:
            rmse = float(np.sqrt(np.sum(wh * e_held[held] ** 2) / wh.sum()))
    info["smoothing_s"] = float(s) / wbar
    info["smoothing_s_effective"] = float(s)
    info["smoothing_selection"] = selection
    info["holdout_rmse_px"] = rmse
    info["n_downweighted"] = int(np.sum(Wr[valid] < 0.1))
    if internals is not None:
        internals.update(
            {
                "R": R,
                "Lam": Lam,
                "W0n": W0n,
                "s": float(s),
                "Wr": Wr,
                "cv_buffer": info["cv_buffer"],
            }
        )
    return aff + Z, info


# ── dctpls on the vector lattice ──────────────────────────────────────────────
# Sigma calibration (research/stare-sota-review-2026-09-27.md Part C §4): peak-ratio bins
# of held-out residuals. At most CAL_MAX_BINS bins of at least CAL_MIN_PER_BIN vectors; no
# calibration at all below CAL_MIN_VECTORS (2 bins' worth).
CAL_MAX_BINS = 8
CAL_MIN_PER_BIN = 30
CAL_MIN_VECTORS = 60
CAL_SIGMA_FLOOR_PX = 0.05
# Folds of the block-CV split the sigma bins are CALIBRATED on; the rest score the coverage.
# A coverage computed on the residuals that set sigma is near-tautological (~0.68 by
# construction of a median-based scale); scored on disjoint folds it is a real check.
CAL_FOLDS = (0, 2, 4)
# median of a chi-square with 2 dof: a 2-D isotropic Gaussian residual with per-component
# sigma has median |e|^2 = 2 ln 2 sigma^2
_MEDIAN_CHI2_2 = 2.0 * np.log(2.0)


def _lattice_from_vectors(controls, max_disp):
    """Every tile's window vectors laid on the slide-global lattice.

    Returns ``(grid_x, grid_y, Y, W0, PR, counts, lattice)``. Node ``(kx, ky)`` sits at
    ``origin + k * stride``; the extent covers every node any tile reported, valid or
    rejected, so the field spans the tiles' cores. A node reported twice (a hand-run tile
    without core bounds) takes the mean vector and the larger peak ratio. Controls without
    ``vectors`` are refused by the caller (``_require_vectors``).
    """
    with_v = controls
    lattices = {
        (int(c["lattice"]["stride"]), float(c["lattice"]["origin"])) for c in with_v
    }
    if len(lattices) != 1:
        raise ValueError(
            f"controls disagree on the vector lattice (stride, origin): {sorted(lattices)}"
        )
    stride, origin = lattices.pop()
    nodes = []
    for c in with_v:
        nodes += [(int(v[0]), int(v[1])) for v in c["vectors"]]
        nodes += [(int(r[0]), int(r[1])) for r in c.get("rejected", [])]
    counts = {"disp": 0, "nonfinite": 0, "duplicates": 0, "n_vectors": 0}
    if not nodes:
        return None, None, None, None, None, counts, {"stride": stride}
    kx0 = min(k[0] for k in nodes)
    ky0 = min(k[1] for k in nodes)
    nx = max(k[0] for k in nodes) - kx0 + 1
    ny = max(k[1] for k in nodes) - ky0 + 1
    grid_x = origin + (kx0 + np.arange(nx)) * stride
    grid_y = origin + (ky0 + np.arange(ny)) * stride
    S = np.zeros((ny, nx, 2))
    N = np.zeros((ny, nx))
    PR = np.zeros((ny, nx))
    for c in with_v:
        for v in c["vectors"]:
            counts["n_vectors"] += 1
            dx, dy, pr = float(v[4]), float(v[5]), float(v[6])
            if not (np.isfinite(dx) and np.isfinite(dy)):
                counts["nonfinite"] += 1
                continue
            if not _in_range(dx, dy, max_disp):
                counts["disp"] += 1
                continue
            iy, ix = int(v[1]) - ky0, int(v[0]) - kx0
            if N[iy, ix] > 0:
                counts["duplicates"] += 1
            S[iy, ix] += (dx, dy)
            N[iy, ix] += 1
            PR[iy, ix] = max(PR[iy, ix], pr if np.isfinite(pr) else 0.0)
    W0 = (N > 0).astype(float)
    Y = np.where(N[..., None] > 0, S / np.maximum(N, 1)[..., None], 0.0)
    if counts["disp"] or counts["nonfinite"]:
        logger.info(
            f"rejected {counts['disp'] + counts['nonfinite']}/{counts['n_vectors']} "
            f"vector(s): {counts['disp']} out of range (|d| >= {max_disp}), "
            f"{counts['nonfinite']} non-finite"
        )
    lattice = {
        "stride": stride,
        "origin": origin,
        "k0": [int(kx0), int(ky0)],
        "shape": [int(ny), int(nx)],
    }
    return grid_x, grid_y, Y, W0, PR, counts, lattice


def _block_heldout(R, W, Lam, s, buffer=0):
    """Held-out residual VECTORS ``fit - obs`` per valid node at a fixed ``s`` (block CV).

    Same folds as the ``s`` selection; pass it the selection's ``buffer`` ring.
    Returns ``(E, folds)``.
    """
    folds = _cv_folds(W)
    valid = W > 0
    E = np.full(W.shape + (2,), np.nan)
    for f in range(CV_FOLDS):
        hold, excluded = _cv_split(folds, f, valid, buffer)
        Wt = np.where(excluded, 0.0, W)
        if not hold.any() or not (Wt > 0).any():
            continue
        Z = _pls_fit(R, Wt, Lam, s, tol=PLS_TOL_CV)
        E[hold] = Z[hold] - R[hold]
    return E, folds


def _calibrate_sigma(PR, valid, E, folds=None, cal_folds=CAL_FOLDS, inlier=None):
    """Per-vector sigma from held-out residuals binned by peak ratio, scored on other folds.

    Bins are peak-ratio quantiles, ``min(8, n // 30)`` of them, fitted on the residuals of
    the ``cal_folds`` block-CV folds only. Per bin, sigma is the robust per-component scale
    ``sqrt(median |e|^2 / (2 ln 2))`` (exact for an isotropic Gaussian residual, insensitive
    to the outliers the robust fit rejects), floored at 0.05 px. The remaining folds, which
    set nothing, score it: ``coverage_1sigma`` (fraction of residual components within
    +-sigma; ideal ~0.68 for a Gaussian) and ``rms_error_over_rms_sigma`` (per-component RMS
    residual over RMS sigma; ideal 1). ``inlier`` (optional boolean lattice) restricts the
    SCORING to the vectors the robust fit kept, as ``holdout_rmse_px`` does: ~1 % of window
    vectors are gross outliers the bisquare rejects, and on the synthetic slide they alone
    push the RMS ratio to 2.5-5 while the bulk is calibrated. ``folds`` ``None`` puts every
    node in calibration and scores nothing (``None`` for both scores).

    Returns ``(sigma per node, bins, scores)`` or ``(None, reason, None)``; ``scores`` is
    ``{"coverage_1sigma", "rms_error_over_rms_sigma", "n_scored", "calibration_folds",
    "scoring_folds"}``.

    The held-out residual is measurement error PLUS the smoother's prediction error at a
    node whose patch (and its buffer ring) was withheld, so sigma is an upper bound on
    measurement noise; the scores say how honest it is on this slide.
    """
    ok = valid & np.all(np.isfinite(E), axis=-1)
    if folds is None:
        cal = ok
        score = np.zeros_like(ok)
        scoring = []
    else:
        in_cal = np.isin(folds, list(cal_folds))
        cal, score = ok & in_cal, ok & ~in_cal
        if inlier is not None:
            score = score & inlier
        scoring = sorted(set(range(CV_FOLDS)) - set(cal_folds))
    n = int(cal.sum())
    if n < CAL_MIN_VECTORS:
        return (
            None,
            f"{n} calibration-fold vectors with a held-out residual < {CAL_MIN_VECTORS}: "
            f"too few to calibrate {CAL_MIN_PER_BIN}-vector peak-ratio bins; uniform "
            "weights kept",
            None,
        )
    nb = max(1, min(CAL_MAX_BINS, n // CAL_MIN_PER_BIN))
    pr = PR[cal]
    e2 = np.sum(E[cal] ** 2, axis=-1)
    edges = np.quantile(pr, np.linspace(0.0, 1.0, nb + 1))
    b_of = np.clip(np.searchsorted(edges[1:-1], pr, side="right"), 0, nb - 1)
    sig = np.empty(nb)
    bins = []
    for b in range(nb):
        m = b_of == b
        if not m.any():
            sig[b] = np.nan
            continue
        sig[b] = max(
            float(np.sqrt(np.median(e2[m]) / _MEDIAN_CHI2_2)), CAL_SIGMA_FLOOR_PX
        )
        bins.append(
            {
                "pr_lo": float(edges[b]),
                "pr_hi": float(edges[b + 1]),
                "sigma_px": float(sig[b]),
                "n": int(m.sum()),
            }
        )
    # a quantile tie can leave a bin empty: it takes its neighbour's sigma
    for b in range(nb):
        if not np.isfinite(sig[b]):
            finite = np.flatnonzero(np.isfinite(sig))
            sig[b] = sig[finite[np.argmin(np.abs(finite - b))]]
    node_b = np.clip(np.searchsorted(edges[1:-1], PR, side="right"), 0, nb - 1)
    sigma = np.where(valid, sig[node_b], np.nan)
    scores = {
        "coverage_1sigma": None,
        "rms_error_over_rms_sigma": None,
        "n_scored": int(score.sum()),
        "calibration_folds": sorted(int(f) for f in cal_folds) if scoring else [],
        "scoring_folds": [int(f) for f in scoring],
    }
    if score.any():
        comp = np.abs(E[score])
        sg = sigma[score]
        scores["coverage_1sigma"] = float(np.mean(comp <= sg[:, None]))
        scores["rms_error_over_rms_sigma"] = float(
            np.sqrt(np.mean(comp**2)) / np.sqrt(np.mean(sg**2))
        )
    return sigma, bins, scores


# Fixed-point re-indexing of the field to the stitch's frame. ``F <- D(g + F)`` is a
# contraction when D's Lipschitz constant L is < 1, and its error shrinks by a factor L per
# step -- NOT to "well under 0.01 px" in a fixed 5 steps whatever the field: after k steps
# the error is bounded by L^k / (1 - L) times the first step, which at L = 0.49 and a 100 px
# offset is still ~1.5 px after 5. Real fields have L ~ 0.02, where 3 steps suffice; the loop
# runs until the fixed-point residual max |F - D(g + F)| is below REINDEX_TOL_PX (or the
# iteration cap) and that residual is REPORTED (``reindex_residual_px``), not assumed.
REINDEX_MAX_ITERATIONS = 30
REINDEX_TOL_PX = 1e-3


def _reindex_to_moving_frame(grid_x, grid_y, D, interp="bilinear", info=None):
    """``F(g) = D(g + F(g))``: the lattice field re-indexed from reference to moving points.

    A window vector is MEASURED at a reference-frame node ``x``: ``ref(x) = mov(x - D(x))``.
    The stitch (``warp._invert``) evaluates the mesh at the moving point instead -- it solves
    ``v = u - F(v)`` for the M0-frame moving point ``v`` of reference pixel ``u`` -- so the
    mesh must hold ``F(v) = D(v + F(v))``. The two differ by ``J D``: negligible for a
    sub-pixel field, 0.3 px for a +100 px offset with a 0.15 deg residual rotation, measured
    on the synthetic slide (0.15 px median error at the reference nodes, 0.71 px through the
    stitch until this was added). ``D`` is read with the same ``interp`` the manifest
    records, clamped at the lattice edges.

    ``info``, when a dict, receives ``reindex_residual_px`` (the final
    ``max |F(g) - D(g + F(g))|``) and ``reindex_iterations``.
    """
    from stare.mesh_field import MeshField

    ny, nx, _ = D.shape
    if ny < 2 and nx < 2:
        if info is not None:
            info.update({"reindex_residual_px": 0.0, "reindex_iterations": 0})
        return D.copy()
    # D interpolated exactly as the manifest's reader will interpolate F: one MeshField rule
    mesh_d = MeshField(grid_x, grid_y, D, interp=interp)
    G = np.stack(np.meshgrid(mesh_d.grid_x, mesh_d.grid_y), axis=-1).reshape(-1, 2)
    F = D.reshape(-1, 2).copy()
    residual, it = np.inf, 0
    while it < REINDEX_MAX_ITERATIONS:
        Fn = mesh_d.displacement(G + F)
        residual = float(np.abs(Fn - F).max()) if F.size else 0.0
        F = Fn
        it += 1
        if residual < REINDEX_TOL_PX:
            break
    # the residual of the RETURNED field: max |F - D(g + F)|
    residual = float(np.abs(mesh_d.displacement(G + F) - F).max()) if F.size else 0.0
    if info is not None:
        info.update({"reindex_residual_px": residual, "reindex_iterations": it})
    return F.reshape(ny, nx, 2)


# The interpolant the vector-lattice mesh is written with (the manifest's ``"interp"``).
# Measured on the 8192^2 synthetic e2e (seeds 0, 1; base and +100 px), field error through
# the stitch's inverse, bilinear -> cubic: median 0.153/0.200/0.180/0.203 ->
# 0.130/0.141/0.144/0.140 px and p99 0.435/0.705/0.777/0.611 -> 0.403/0.637/0.748/0.538 px.
VECTOR_MESH_INTERP = "cubic"


def _solve_dctpls_vectors(controls, max_disp, interp=VECTOR_MESH_INTERP):
    """``solve_dctpls`` on REG_TILE's vector lattice, with sigma calibration."""
    import time

    t0 = time.perf_counter()
    grid_x, grid_y, Y, W0, PR, counts, lattice = _lattice_from_vectors(
        controls, max_disp
    )
    base = {
        "solver": "dctpls",
        "input": "vectors",
        "mesh_interp": interp,
        "n_controls": len(controls),
        "n_vectors": counts["n_vectors"],
        "n_rejected_disp": counts["disp"],
        "n_rejected_nonfinite": counts["nonfinite"],
        "n_duplicate_nodes": counts["duplicates"],
        "lattice": lattice,
    }
    if grid_x is None:
        # not one lattice node anywhere (a stride larger than the slide): no mesh
        logger.warning(
            "no tile reported a lattice node; the slide stays rigid (no mesh)"
        )
        report = {
            **base,
            "n_valid": 0,
            "n_downweighted": 0,
            "affine": None,
            "smoothing_s": None,
            "smoothing_selection": "none",
            "holdout_rmse_px": None,
            "residual_lag1_rho": None,
            "cv_buffer": None,
            "smoothing_s_effective": None,
            "sigma_calibration": None,
            "sigma_calibration_skipped": "no lattice node",
            "coverage_1sigma": None,
            "rms_error_over_rms_sigma": None,
            "coverage_n_scored": None,
            "coverage_calibration_folds": None,
            "coverage_scoring_folds": None,
            "reindex_residual_px": None,
            "reindex_iterations": None,
            "lipschitz": 0.0,
            "min_det_jacobian": 1.0,
            "fold_certificate_ok": True,
            "solve_seconds": round(time.perf_counter() - t0, 2),
            "measured": [],
        }
        return [], [], [], report
    internals = {}
    field, info = _dctpls_core(Y, W0, grid_x, grid_y, internals=internals)
    calibration, scores, why = None, None, None
    valid = W0 > 0
    selection = info["smoothing_selection"]
    if "s" in internals and selection in CV_LABELS.values():
        buffer = internals["cv_buffer"]
        E, folds = _block_heldout(
            internals["R"],
            internals["W0n"],
            internals["Lam"],
            internals["s"],
            buffer=buffer,
        )
        sigma, bins, scores = _calibrate_sigma(
            PR, valid, E, folds, inlier=internals["Wr"] >= 0.1
        )
        if sigma is None:
            why = bins
        else:
            calibration = bins
            s_chosen, rho = info["smoothing_s"], info["residual_lag1_rho"]
            W1 = np.where(valid, 1.0 / np.where(valid, sigma, 1.0) ** 2, 0.0)
            # s is chosen ONCE: the re-solve reuses it (in mean-weight units), no re-scan
            field, info = _dctpls_core(
                Y, W1, grid_x, grid_y, s_fixed=s_chosen, cv_buffer=buffer
            )
            info["smoothing_selection"] = selection
            info["residual_lag1_rho"] = rho
    else:
        why = (
            f"smoothing chosen by {info['smoothing_selection']}, not block CV "
            f"({info['n_valid']} valid vectors < {MIN_CV_CELLS}); uniform weights kept"
        )
    if why:
        logger.info(f"sigma calibration skipped: {why}")
    scores = scores or {}
    reindex = {}
    field = _reindex_to_moving_frame(grid_x, grid_y, field, interp=interp, info=reindex)
    jac = jacobian_report(grid_x, grid_y, field, interp=interp)
    report = {
        **base,
        "n_valid": info["n_valid"],
        "n_downweighted": info["n_downweighted"],
        "affine": info["affine"],
        "smoothing_s": info["smoothing_s"],
        "smoothing_selection": info["smoothing_selection"],
        "holdout_rmse_px": info["holdout_rmse_px"],
        "residual_lag1_rho": info["residual_lag1_rho"],
        "cv_buffer": info["cv_buffer"],
        "smoothing_s_effective": info.get("smoothing_s_effective"),
        "sigma_calibration": calibration,
        "sigma_calibration_skipped": why,
        "coverage_1sigma": scores.get("coverage_1sigma"),
        "rms_error_over_rms_sigma": scores.get("rms_error_over_rms_sigma"),
        "coverage_n_scored": scores.get("n_scored"),
        "coverage_calibration_folds": scores.get("calibration_folds"),
        "coverage_scoring_folds": scores.get("scoring_folds"),
        "reindex_residual_px": reindex.get("reindex_residual_px"),
        "reindex_iterations": reindex.get("reindex_iterations"),
        "lipschitz": jac["max_operator_norm"],
        "min_det_jacobian": jac["min_jacobian_det"],
        "fold_certificate_ok": bool(
            jac["max_operator_norm"] < FOLD_CERTIFICATE_LIPSCHITZ
        ),
        "solve_seconds": round(time.perf_counter() - t0, 2),
        "measured": valid.astype(int).tolist(),
    }
    return (
        [float(v) for v in grid_x],
        [float(v) for v in grid_y],
        field.tolist(),
        report,
    )


def solve_dctpls(controls, max_disp=None, interp=VECTOR_MESH_INTERP):
    """Robust affine, then robust DCT-PLS of the residual, on the vector lattice.

    ``controls`` are REG_TILE's per-tile control JSONs; every one must carry ``lattice`` and
    ``vectors`` (``_require_vectors`` refuses one that does not, naming the tiles). Validity
    is "finite and ``|d| < max_disp``"; the prior weights are calibrated from the data
    (``_calibrate_sigma``): one solve at uniform weights, block-CV held-out residuals binned
    by peak ratio, then one re-solve at ``w = 1/sigma^2``. See the module docstring for the
    stages.

    ``interp`` is the interpolant the mesh is re-indexed with and that the manifest records
    (``report["mesh_interp"]``; ``VECTOR_MESH_INTERP``, cubic, by default).

    Returns ``(grid_x, grid_y, disp, report)``; with no lattice node anywhere the three
    arrays are empty and the slide stays rigid.
    """
    if not controls:
        raise ValueError("no control points to solve")
    _require_vectors(controls)
    return _solve_dctpls_vectors(controls, max_disp, interp=interp)
