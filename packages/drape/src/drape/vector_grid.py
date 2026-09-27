"""Per-tile displacement vectors on a GLOBAL lattice: the REG_TILE estimator.

One control point per 2048 px tile cannot describe a field that varies inside the tile: the
single correlation averages it (~1.5 px from the tile's own mean field on the synthetic
slide) and SOLVE then has 16 numbers to reconstruct a slide from. This module replaces it
with a grid of window vectors, one per lattice node, the PIV/SOFIMA estimator
(``research/stare-optimal-design-2026-09-27.md`` §2):

* **Lattice.** Window ``W = 2 * stride`` (50 % overlap). Node ``k`` has its centre at
  ``W/2 + k * stride`` in REFERENCE-frame pixels, on both axes, for the whole slide -- so
  every tile agrees on the node positions and a node is owned by exactly the one tile whose
  CORE contains its centre (cores partition the slide, ``tile_grid``). SOLVE stitches the
  tiles' vectors back into one lattice by ``(kx, ky)``.
* **Foreground.** Otsu on a heavily blurred quarter-resolution reference (the blur merges
  nuclei into tissue), with ``tile_residual``'s bimodality guard. A unimodal tile is all
  tissue or all background; that is decided by how much of its high-pass variance survives
  a 4x area downsample (structure survives, white noise keeps 1/16). A window is estimated
  only when >= 25 % of it is foreground (SOFIMA's ``max_masked = 0.75``).
* **Pass 1** at 1/4 resolution: windows of ``stride`` quarter-px (``2W`` full-res) at stride
  ``stride/2`` over the whole read region. Capture +-``stride`` full-res px (the quarter
  rule). The pass-1 lattice is cleaned by a 3x3 median over valid nodes, with a
  least-squares affine for nodes that have no valid neighbour, interpolated to full
  resolution, and the moving crop is pulled back through it. Two passes are REQUIRED, not an
  option: with +100 px left over from COARSE a single pass keeps 3 % of windows and a 10 px
  median error, two passes 0.67 px (``research/stare-sota-review-2026-09-27.md`` Part C §6).
* **Pass 2** at full resolution on the tile's owned nodes: a DoG high-pass (a sigma=3
  Gaussian subtracted), analogous to ASHLAR's Laplacian/LoG whitening; Hann window, FFT
  cross-correlation, integer peak, then a **3-point Gaussian** sub-pixel fit on the
  surface minus its local minimum (Xue et al. 2014; parabolic when a sample is still not
  positive -- the per-tile ``gauss_fallback_rate`` says how often). Measured less noisy
  than a x10 upsampled DFT and <= 0.08 px biased (``peaklock.py``, Part C §3). Per vector:
  the **peak ratio** (peak over the highest other local maximum more than 3 px away) and
  the **sharpness** (``|peak|`` over ``|minimum|`` within 5 px, SOFIMA's rule).

Sign convention is the control point's, unchanged: ``(dx, dy)`` is what to ADD to a moving
coordinate (in the M0-warped frame) to reach the reference, i.e. ``ref(x) = mov(x - d)``;
``tile_residual.residual_displacement`` returns the same quantity.

Pure NumPy / SciPy / scikit-image -- the tiled container has no OpenCV.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "FG_MIN",
    "PEAK_RATIO_MIN",
    "RATIO_CAP",
    "owned_nodes",
    "read_box",
    "tissue_mask",
    "estimate_tile_vectors",
    "correlate_windows",
    "peak_stats",
]

# A window is estimated only when this fraction of it is foreground (SOFIMA max_masked=0.75).
FG_MIN = 0.25
# The loose floor on the peak ratio; SOLVE does the rest (design §2 step 4).
PEAK_RATIO_MIN = 1.2
# A ratio whose denominator is not positive is "no competing peak at all". JSON has no
# Infinity, so it is written as this finite cap instead of inf.
RATIO_CAP = 1000.0
PEAK_EXCLUDE_RADIUS = 3
SHARPNESS_RADIUS = 5
WHITEN_SIGMA = 3.0
QUARTER = 4
# Blur (in full-res px) that merges nuclei into a tissue silhouette before Otsu: ~10 um at
# 0.325 um/px, a few nuclear spacings.
TISSUE_BLUR_PX = 32.0
# Same cut as tile_residual.BIMODAL_MIN_SEPARATION (between-class gap in pooled sigmas).
BIMODAL_MIN_SEPARATION = 4.0
# Unimodal tile: fraction of its high-pass variance that survives a 4x area downsample.
# White noise keeps 1/16 = 0.0625; nuclei (10-40 px) keep most of theirs.
TEXTURE_SURVIVAL_MIN = 0.25
# The unimodal texture test runs on a central crop of this size, high-passed at this sigma
# (removes illumination gradients; keeps nuclei-scale structure).
TEXTURE_CROP = 1024
TEXTURE_HIGHPASS_PX = 8.0
# Windows correlated per FFT batch -- bounds the transient stack at 64 x W^2 floats.
BATCH = 64


def read_box(core, stride, image_shape):
    """The region a tile must read: its core plus ``W/2 + W`` = ``3 * stride``, clamped.

    ``W/2`` is the reach of an owned node's window past the core; the further ``W`` is the
    pass-1 capture margin, so the moving pixels a node's window is pulled back from are inside
    the read even at the full +-stride capture. Memory is therefore bounded by the core plus
    ``2 x 1.5W`` per axis -- ``(2048 + 768)^2`` = 7.9 Mpx at the defaults -- whatever the
    slide size.
    """
    x0, y0, x1, y1 = (int(v) for v in core)
    h, w = int(image_shape[0]), int(image_shape[1])
    m = 3 * int(stride)
    return (max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m))


def _owned_axis(lo, hi, stride):
    """Lattice indices ``k >= 0`` whose centre ``stride * (k + 1)`` lies in ``[lo, hi)``."""
    k0 = max(0, -(-int(lo) // int(stride)) - 1)  # ceil(lo / stride) - 1
    k1 = -(-int(hi) // int(stride)) - 1  # last k with centre < hi
    return list(range(k0, k1))


def owned_nodes(core, stride):
    """``(kxs, kys)``: the global lattice nodes whose centre lies in this tile's core.

    The centre of node ``k`` is ``W/2 + k * stride = stride * (k + 1)``. Cores partition the
    slide, so across all tiles every node is owned exactly once.
    """
    x0, y0, x1, y1 = (int(v) for v in core)
    return _owned_axis(x0, x1, stride), _owned_axis(y0, y1, stride)


# ── foreground ────────────────────────────────────────────────────────────────


def _area_down(a, f=QUARTER):
    """Block-mean downsample by ``f`` (cropping the remainder rows/cols)."""
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    if h == 0 or w == 0:
        return np.zeros((max(h // f, 0), max(w // f, 0)), dtype=np.float32)
    return (
        a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3), dtype=np.float64)
    ).astype(np.float32)


def _otsu_separated(a):
    """Otsu threshold of ``a`` if its two classes are really separated, else ``None``."""
    from skimage.filters import threshold_otsu

    lo, hi = float(a.min()), float(a.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return None
    t = threshold_otsu(a)
    below, above = a[a <= t], a[a > t]
    if below.size == 0 or above.size == 0:
        return None
    within = np.sqrt((below.var() * below.size + above.var() * above.size) / a.size)
    if (above.mean() - below.mean()) / (within + 1e-12) < BIMODAL_MIN_SEPARATION:
        return None
    return t


def tissue_mask(ref_full, ref_q=None):
    """Boolean tissue mask of ``ref_full`` at QUARTER resolution.

    Bimodal (a tissue edge in view): Otsu on the blurred quarter-res image. Unimodal (the
    bimodality guard of ``tile_residual.foreground_fraction`` refuses the split): the tile is
    all tissue when at least ``TEXTURE_SURVIVAL_MIN`` of its high-pass variance survives the
    4x area downsample, else all background. A flat tile is background.
    """
    from scipy.ndimage import gaussian_filter

    a = np.asarray(ref_full, dtype=np.float32)
    q = _area_down(a) if ref_q is None else ref_q
    if q.size == 0:
        return np.zeros(q.shape, dtype=bool)
    if float(q.max()) - float(q.min()) < 1e-9:
        return np.zeros(q.shape, dtype=bool)
    blurred = gaussian_filter(q, TISSUE_BLUR_PX / QUARTER)
    t = _otsu_separated(blurred)
    if t is not None:
        return blurred > t
    # the texture test on a central crop: the ratio is a property of the tile's texture, not
    # of where it is measured, and a full-tile sigma-32 blur cost ~1.8 s of a ~3 s tile
    h, w = a.shape
    cy0, cx0 = max(0, h // 2 - TEXTURE_CROP // 2), max(0, w // 2 - TEXTURE_CROP // 2)
    c = a[cy0 : cy0 + TEXTURE_CROP, cx0 : cx0 + TEXTURE_CROP]
    hp = c - gaussian_filter(c, TEXTURE_HIGHPASS_PX)
    v_full = float(hp.var())
    if v_full <= 1e-12:
        return np.zeros(q.shape, dtype=bool)
    survival = float(_area_down(hp).var()) / v_full
    return np.full(q.shape, survival >= TEXTURE_SURVIVAL_MIN, dtype=bool)


class _Integral:
    """Summed-area table of a quarter-res mask, for per-window foreground fractions."""

    def __init__(self, mask):
        m = np.asarray(mask, dtype=np.float64)
        self.s = np.zeros((m.shape[0] + 1, m.shape[1] + 1))
        self.s[1:, 1:] = m.cumsum(0).cumsum(1)
        self.h, self.w = m.shape

    def frac(self, y0, y1, x0, x1, area):
        """Mask sum over ``[y0,y1) x [x0,x1)`` (clamped) divided by ``area``."""
        y0, y1 = min(max(y0, 0), self.h), min(max(y1, 0), self.h)
        x0, x1 = min(max(x0, 0), self.w), min(max(x1, 0), self.w)
        if y1 <= y0 or x1 <= x0 or area <= 0:
            return 0.0
        s = self.s
        tot = s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]
        return float(tot / area)


# ── correlation ───────────────────────────────────────────────────────────────


def _subpixel(cm, c0, cp, floor=0.0):
    """3-point Gaussian peak offset on ``c - floor``; parabolic when a sample is not positive.

    ``floor`` is the local correlation minimum (Xue et al. 2014): subtracting it before the
    log makes the three samples positive on a whitened (zero-mean, negative-lobed)
    correlation, where the raw samples beside the peak can dip below zero and force the
    parabolic fallback. Returns ``(offset, fell_back)``.
    """
    am, a0, ap = cm - floor, c0 - floor, cp - floor
    if am > 0 and a0 > 0 and ap > 0:
        lm, l0, lp = math.log(am), math.log(a0), math.log(ap)
        den = 2.0 * lm - 4.0 * l0 + 2.0 * lp
        off = (lm - lp) / den if den < 0 else 0.0
        fell_back = False
    else:
        den = cm - 2.0 * c0 + cp
        off = 0.5 * (cm - cp) / den if den < 0 else 0.0
        fell_back = True
    return float(min(max(off, -1.0), 1.0)), fell_back


def peak_stats(c, lmax=None):
    """Sub-pixel peak and validity statistics of one ``fftshift``-ed correlation surface.

    Returns ``(dx, dy, peak_ratio, sharpness, fell_back)`` or ``None`` for a non-positive
    peak. ``dx``/``dy`` are relative to the surface centre. The sub-pixel fit is the
    3-point Gaussian on the surface minus its minimum within ``SHARPNESS_RADIUS`` of the
    peak (``_subpixel``); ``fell_back`` is ``True`` when either axis needed the parabola.
    ``peak_ratio`` is the peak over the highest other local maximum more than
    ``PEAK_EXCLUDE_RADIUS`` away; ``sharpness`` is ``|peak| / |min within
    SHARPNESS_RADIUS|`` -- on ABSOLUTE values, as SOFIMA filters, so a negative minimum
    (a whitened surface) cannot flip its sign or turn it into the cap.
    """
    from scipy.ndimage import maximum_filter

    h, w = c.shape
    if lmax is None:
        lmax = c == maximum_filter(c, size=3, mode="wrap")
    flat = int(np.argmax(c))
    py, px = divmod(flat, w)
    peak = float(c[py, px])
    if not (peak > 0) or not np.isfinite(peak):
        return None
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    # circular distance to the peak
    dyy = np.minimum(np.abs(yy - py), h - np.abs(yy - py))
    dxx = np.minimum(np.abs(xx - px), w - np.abs(xx - px))
    d2 = dyy**2 + dxx**2
    lo = float(c[d2 <= SHARPNESS_RADIUS**2].min())
    oy, fy = _subpixel(float(c[(py - 1) % h, px]), peak, float(c[(py + 1) % h, px]), lo)
    ox, fx = _subpixel(float(c[py, (px - 1) % w]), peak, float(c[py, (px + 1) % w]), lo)
    others = c[lmax & (d2 > PEAK_EXCLUDE_RADIUS**2)]
    second = float(others.max()) if others.size else 0.0
    ratio = peak / second if second > 0 else RATIO_CAP
    sharp = abs(peak) / abs(lo) if lo != 0 else RATIO_CAP
    return (
        (px - w // 2) + ox,
        (py - h // 2) + oy,
        min(ratio, RATIO_CAP),
        min(sharp, RATIO_CAP),
        fx or fy,
    )


def correlate_windows(R, M):
    """Shift, peak ratio, sharpness, normalised error and fallback flag per window pair.

    ``R``, ``M`` are ``(n, h, w)`` stacks already whitened and Hann-windowed. The circular
    cross-correlation ``c(t) = sum_x R(x) M(x - t)`` peaks at the ``t`` with
    ``R(x) = M(x - t)``, i.e. at the control displacement ``d`` itself. Returns an ``(n, 6)``
    array ``[dx, dy, peak_ratio, sharpness, error, gauss_fallback]`` (``peak_stats``);
    ``error = 1 - ncc^2`` is scikit-image's phase-correlation error for the same kernel and
    ``gauss_fallback`` is 1.0 when the sub-pixel fit fell back to the parabola. A
    non-positive peak gives NaN everywhere.
    """
    from scipy.fft import irfft2, rfft2
    from scipy.ndimage import maximum_filter

    R = np.asarray(R, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)
    n, h, w = R.shape
    out = np.full((n, 6), np.nan)
    if n == 0:
        return out
    C = irfft2(
        rfft2(R, axes=(1, 2)) * np.conj(rfft2(M, axes=(1, 2))), s=(h, w), axes=(1, 2)
    )
    C = np.fft.fftshift(C, axes=(1, 2))
    energy = np.sqrt(
        (R.astype(np.float64) ** 2).sum(axis=(1, 2))
        * (M.astype(np.float64) ** 2).sum(axis=(1, 2))
    )
    lmax = C == maximum_filter(C, size=(1, 3, 3), mode="wrap")
    for i in range(n):
        st = peak_stats(C[i], lmax[i])
        if st is None:
            continue
        dx, dy, ratio, sharp, fell_back = st
        ncc = float(C[i].max()) / energy[i] if energy[i] > 0 else 0.0
        out[i] = [dx, dy, ratio, sharp, 1.0 - ncc * ncc, float(fell_back)]
    return out


def _whiten(a, sigma=WHITEN_SIGMA):
    from scipy.ndimage import gaussian_filter

    a = np.asarray(a, dtype=np.float32)
    return a - gaussian_filter(a, sigma)


def _correlate_boxes(ref, mov, boxes, shape):
    """Correlate the ``(y0, x0)``-anchored ``shape`` windows of two whitened arrays."""
    from skimage.filters import window

    h, w = shape
    han = window("hann", (h, w)).astype(np.float32)
    res = []
    for b in range(0, len(boxes), BATCH):
        chunk = boxes[b : b + BATCH]
        R = np.stack([ref[y : y + h, x : x + w] for y, x in chunk]) * han
        M = np.stack([mov[y : y + h, x : x + w] for y, x in chunk]) * han
        res.append(correlate_windows(R, M))
    return np.concatenate(res) if res else np.zeros((0, 6))


# ── pass 1 ────────────────────────────────────────────────────────────────────


def _window_starts(n, win, step):
    """Starts of ``win``-wide windows at ``step`` covering ``[0, n)``; the last hugs the end."""
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, step))
    if starts[-1] + win < n:
        starts.append(n - win)
    return starts


def _clean_lattice(V, valid, cy, cx):
    """3x3 median over valid nodes; an affine LSQ fit where no valid neighbour exists."""
    ny, nx, _ = V.shape
    out = np.zeros_like(V)
    nv = int(valid.sum())
    if nv == 0:
        return out, "none"
    if nv >= 3:
        GX, GY = np.meshgrid(cx, cy)
        A = np.column_stack([np.ones(nv), GX[valid], GY[valid]])
        coef = np.linalg.lstsq(A, V[valid], rcond=None)[0]
        fallback = np.column_stack([np.ones(GX.size), GX.ravel(), GY.ravel()]) @ coef
        fallback = fallback.reshape(ny, nx, 2)
    else:
        fallback = np.broadcast_to(np.median(V[valid], axis=0), (ny, nx, 2))
    for iy in range(ny):
        for ix in range(nx):
            sl = (slice(max(0, iy - 1), iy + 2), slice(max(0, ix - 1), ix + 2))
            nb = valid[sl]
            if nb.any():
                out[iy, ix] = np.median(V[sl][nb], axis=0)
            else:
                out[iy, ix] = fallback[iy, ix]
    return out, "median3x3+affine"


def _interp_matrix(targets, centres):
    """Dense linear-interpolation weights of ``targets`` on sorted ``centres`` (clamped)."""
    targets = np.asarray(targets, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64)
    A = np.zeros((targets.size, centres.size))
    if centres.size == 1:
        A[:, 0] = 1.0
        return A
    t = np.clip(targets, centres[0], centres[-1])
    j = np.clip(np.searchsorted(centres, t, side="right") - 1, 0, centres.size - 2)
    f = (t - centres[j]) / (centres[j + 1] - centres[j])
    rows = np.arange(targets.size)
    A[rows, j] = 1.0 - f
    A[rows, j + 1] = f
    return A


def _pass1(ref_q, mov_q, mask_q, stride):
    """Quarter-res window vectors over the read region and the cleaned lattice.

    Returns ``(field, cy, cx, info)`` with ``field`` ``(ny, nx, 2)`` in FULL-res px (control
    convention) at node centres ``cy``/``cx`` in full-res px relative to the read origin.
    """
    hq, wq = ref_q.shape
    win = max(int(stride), 8)
    step = max(int(stride) // 2, 4)
    info = {"window_q": win, "stride_q": step, "n": 0, "n_valid": 0, "cleaning": "none"}
    if hq < 8 or wq < 8:
        return np.zeros((1, 1, 2)), np.array([0.0]), np.array([0.0]), info
    wy, wx = min(win, hq), min(win, wq)
    ys, xs = _window_starts(hq, wy, step), _window_starts(wq, wx, step)
    rq, mq = _whiten(ref_q), _whiten(mov_q)
    fg = _Integral(mask_q)
    boxes, keep = [], []
    for y in ys:
        for x in xs:
            ok = fg.frac(y, y + wy, x, x + wx, wy * wx) >= FG_MIN
            keep.append(ok)
            if ok:
                boxes.append((y, x))
    res = _correlate_boxes(rq, mq, boxes, (wy, wx))
    V = np.zeros((len(ys), len(xs), 2))
    valid = np.zeros((len(ys), len(xs)), dtype=bool)
    cap = min(wy, wx) / 4.0
    it = iter(res)
    for n, ok in enumerate(keep):
        if not ok:
            continue
        dx, dy, pr, _sh, _err, _fb = next(it)
        iy, ix = divmod(n, len(xs))
        if np.isfinite(dx) and pr >= PEAK_RATIO_MIN and max(abs(dx), abs(dy)) <= cap:
            V[iy, ix] = [dx * QUARTER, dy * QUARTER]
            valid[iy, ix] = True
    # quarter px j covers full px [4j, 4j+4): a window's centre maps to 4*start + 2*win
    cy = np.array([QUARTER * y + QUARTER * wy / 2.0 for y in ys])
    cx = np.array([QUARTER * x + QUARTER * wx / 2.0 for x in xs])
    field, how = _clean_lattice(V, valid, cy, cx)
    info.update({"n": len(keep), "n_valid": int(valid.sum()), "cleaning": how})
    return field, cy, cx, info


# ── the tile ─────────────────────────────────────────────────────────────────


def _crop_padded(a, origin, box):
    """``a`` (anchored at global ``origin = (x, y)``) cropped to global ``box``, zero-padded."""
    ox, oy = origin
    x0, y0, x1, y1 = box
    out = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    sx0, sy0 = max(x0, ox), max(y0, oy)
    sx1, sy1 = min(x1, ox + a.shape[1]), min(y1, oy + a.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = a[
            sy0 - oy : sy1 - oy, sx0 - ox : sx1 - ox
        ]
    return out


def estimate_tile_vectors(ref, mov, origin, core, stride, two_pass=True):
    """All owned-node vectors of one tile.

    Parameters
    ----------
    ref, mov : 2-D arrays
        The reference and the M0-warped moving nuclear channel over the tile's READ region
        (``read_box``), both in the reference frame.
    origin : (int, int)
        ``(x, y)`` of the read region's top-left in reference-frame pixels.
    core : (x0, y0, x1, y1)
        The tile's owned core, half-open.
    stride : int
        Lattice stride; window ``W = 2 * stride``.
    two_pass : bool
        ``False`` skips pass 1 (a zero pull-back) -- for tests that must show pass 1 is needed.

    Returns
    -------
    dict
        ``lattice`` (``stride``, ``window``, ``origin``), ``vectors`` -- one
        ``[kx, ky, cx, cy, dx, dy, peak_ratio, sharpness, fg]`` per VALID node --,
        ``rejected`` -- ``[kx, ky, cx, cy, fg, peak_ratio]`` per owned node that was not
        emitted (``peak_ratio`` null when the window was never correlated) --, ``errors``
        (the valid vectors' normalised correlation error, same order), ``pass1`` info and
        ``gauss_fallback_rate`` -- the fraction of pass-2 windows with a positive peak whose
        sub-pixel fit fell back to the parabola (``None`` when no window was correlated).
    """
    from scipy.ndimage import map_coordinates

    stride = int(stride)
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")
    W = 2 * stride
    ref = np.asarray(ref, dtype=np.float32)
    mov = np.asarray(mov, dtype=np.float32)
    ox, oy = int(origin[0]), int(origin[1])
    kxs, kys = owned_nodes(core, stride)
    lattice = {"stride": stride, "window": W, "origin": W / 2.0}
    result = {
        "lattice": lattice,
        "vectors": [],
        "rejected": [],
        "errors": [],
        "pass1": {"n": 0, "n_valid": 0, "cleaning": "skipped"},
        "gauss_fallback_rate": None,
    }
    if not kxs or not kys:
        return result

    ref_q = _area_down(ref)
    mask_q = tissue_mask(ref, ref_q)
    fg = _Integral(mask_q)

    # pass 1: coarse field over the read region, in control convention, full-res px
    if two_pass:
        field1, c1y, c1x, info1 = _pass1(ref_q, _area_down(mov), mask_q, stride)
        result["pass1"] = info1
    else:
        field1, c1y, c1x = np.zeros((1, 1, 2)), np.array([0.0]), np.array([0.0])

    # pass-2 region: the union of the owned windows (global px)
    box = (kxs[0] * stride, kys[0] * stride, kxs[-1] * stride + W, kys[-1] * stride + W)
    bx0, by0, bx1, by1 = box
    # pixel centres, read-relative (edge coordinates, as the pass-1 centres are)
    ys_rel = np.arange(by0, by1, dtype=np.float64) - oy + 0.5
    xs_rel = np.arange(bx0, bx1, dtype=np.float64) - ox + 0.5
    Ay, Ax = _interp_matrix(ys_rel, c1y), _interp_matrix(xs_rel, c1x)
    d1x = (Ay @ field1[..., 0] @ Ax.T).astype(np.float32)
    d1y = (Ay @ field1[..., 1] @ Ax.T).astype(np.float32)

    ref2 = _crop_padded(ref, (ox, oy), box)
    # pull-back: movw(x) = mov(x - d1(x)) (ref(x) = mov(x - d))
    yy = (np.arange(by0, by1, dtype=np.float32) - oy)[:, None] - d1y
    xx = (np.arange(bx0, bx1, dtype=np.float32) - ox)[None, :] - d1x
    movw = map_coordinates(
        mov, [yy, xx], order=1, mode="constant", cval=0.0, prefilter=False
    )
    movw = movw.astype(np.float32, copy=False)
    del yy, xx
    r2, m2 = _whiten(ref2), _whiten(movw)
    del ref2, movw

    area_q = (W / QUARTER) ** 2
    boxes, meta = [], []
    for ky in kys:
        for kx in kxs:
            wx0, wy0 = kx * stride, ky * stride
            f = fg.frac(
                (wy0 - oy) // QUARTER,
                -(-(wy0 + W - oy) // QUARTER),
                (wx0 - ox) // QUARTER,
                -(-(wx0 + W - ox) // QUARTER),
                area_q,
            )
            f = min(f, 1.0)
            c = (float(stride * (kx + 1)), float(stride * (ky + 1)))
            if f < FG_MIN:
                result["rejected"].append([kx, ky, c[0], c[1], round(f, 4), None])
                continue
            boxes.append((wy0 - by0, wx0 - bx0))
            meta.append((kx, ky, c, f, wy0 - by0, wx0 - bx0))
    res = _correlate_boxes(r2, m2, boxes, (W, W))
    # Pass 2 measures the Hann-weighted mean of (d - d1) over its window, so the pass-1 term
    # added back is d1's Hann-weighted mean over the SAME window, not its value at the centre:
    # a noisy pass-1 lattice is piecewise linear, and centre-vs-mean put up to ~1 px on a
    # uniform 100 px shift (measured), against ~0.2 px this way.
    from skimage.filters import window

    han = window("hann", (W, W))
    han = han / han.sum()
    d1_mean = [
        (
            float((han * d1x[y : y + W, x : x + W]).sum()),
            float((han * d1y[y : y + W, x : x + W]).sum()),
        )
        for y, x in boxes
    ]
    fb = res[:, 5] if len(res) else np.zeros(0)
    fb = fb[np.isfinite(fb)]
    result["gauss_fallback_rate"] = float(fb.mean()) if fb.size else None
    for (kx, ky, c, f, _ry, _rx), (sx, sy, pr, sh, err, _fb), (m1x, m1y) in zip(
        meta, res, d1_mean
    ):
        if not (np.isfinite(sx) and np.isfinite(sy)) or not (pr >= PEAK_RATIO_MIN):
            result["rejected"].append(
                [
                    kx,
                    ky,
                    c[0],
                    c[1],
                    round(f, 4),
                    None if not np.isfinite(pr) else float(pr),
                ]
            )
            continue
        dx = m1x + float(sx)
        dy = m1y + float(sy)
        result["vectors"].append(
            [kx, ky, c[0], c[1], dx, dy, float(pr), float(sh), round(f, 4)]
        )
        result["errors"].append(float(err))
    return result
