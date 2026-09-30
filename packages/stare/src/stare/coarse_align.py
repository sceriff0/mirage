"""Global rigid anchor (M0) estimation for the STARE registration method.

The COARSE step aligns a whole moving slide to the reference with one rigid ``M0``, on a pair of
nuclear-channel thumbnails that share one decimation factor. Its only job is to be ROBUST: the
per-tile stage refines the residual, and SOLVE's robust affine absorbs any global rotation, scale
or shear left in M0. So a precise-but-fragile matcher is the wrong tool, and this module does not
use one.

THE METHOD (research/stare-optimal-design-2026-09-27.md, section 1):

1. **Preprocess** both thumbnails: robust [0, 1] intensity window, Gaussian blur (sigma 1 px),
   Otsu tissue mask on a tissue-scale blur (background set to 0, tissue keeps its intensity).
2. **Sweep** at 256 px (longest side): rotate the moving image about its centre over 0-360 deg in
   3 deg steps and, for each angle, take the peak of the zero-mean, GLOBALLY normalised
   cross-correlation of the zero-padded pair (FFT; divided by the two whole canvases' norms,
   not by the local norms under each shift as in Lewis 1995's locally normalised NCC; "NCC"
   below means this global form). At 256 px this matches tissue SHAPE, which
   survives nuclei turnover, tissue loss and a gamma change -- the cases that broke feature
   matching and log-polar spectrum registration in the scout's benchmark.
3. **Refine** the best angle and the best DISTINCT runner-up at the ``--max-dim`` thumbnail over
   +-3 deg in 0.25 deg steps, pick the higher correlation peak, then sub-pixel translation with
   scikit-image's ``phase_cross_correlation`` (upsample 10).
4. **Accept** when the peak NCC >= :data:`MIN_PEAK_NCC` and the best / second-distinct peak ratio
   >= :data:`MIN_PEAK_RATIO`. Otherwise run the **ORB fallback** (scikit-image ORB, 5000
   keypoints, cross-checked matches, RANSAC Euclidean) and take it with >= :data:`ORB_MIN_INLIERS`
   inliers when it agrees with a sweep candidate within 1 deg (or the sweep had no candidate at
   all); a disagreeing fallback is scored by the same correlation and the better of the two wins.
5. **Refuse loudly** (:class:`CoarseRefused`) when neither is acceptable. A wrong anchor does not
   fail anything downstream -- the per-tile reads simply land in the wrong place and the slide
   comes out mis-registered with exit 0 -- so an unverifiable anchor is an error, not a warning.

NumPy + SciPy + scikit-image only: no torch, no kornia, no OpenCV, no JVM. The tiled container
already carries all three. Memory is a few FFT canvases of the thumbnail (well under 1 GB at the
1024 px tier) instead of the ~48 GB the retired learned matcher needed at 2048 px.

**Sweep step vs blur, measured (2026-09-27).** The nearest sweep angle is at most step/2 from
the truth. The coarse NOTES' derivation (mean tissue radius ~85 px at 256 px) says a 3 deg step
needs >= ~1.33 px of correlation length, which the sigma = 1 blur alone does not guarantee, so
the correlation-vs-angle curve was measured on the ten hard synthetic geometries of
``tests/test_coarse_anchor.py`` (5 rotations x {80 % nuclei turnover, same nuclei}; peak NCC
0.63-0.78). Each cell is the worst, over the ten, of the NCC at that offset from the
geometry's own peak as a fraction of the peak::

    blur sigma | 0.5 deg | 1.0 deg (half of 2 deg) | 1.5 deg (half of 3 deg)
    1.0        | 0.993   | 0.981                   | 0.963
    1.5        | 0.996   | 0.985                   | 0.971
    2.0        | 0.997   | 0.991                   | 0.981

Every pair keeps >= 90 % of the peak at its half-step, so the cheapest stands: sigma 1 px
(:data:`BLUR_SIGMA`) and 3 deg (:data:`SWEEP_STEP_DEG`; 120 angles against 180 at 2 deg, and
the blur costs nothing either way). Tissue-scale structure does supply the correlation length
the derivation asked for. Pinned by
``test_half_a_sweep_step_off_the_peak_keeps_90_percent_of_the_correlation``.

:func:`estimate_transform_from_matches` (points -> transform + RMS residual) stays the
deterministic, unit-tested core of the ORB fallback.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from stare.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "Anchor",
    "CoarseRefused",
    "estimate_anchor",
    "estimate_rigid",
    "estimate_transform_from_matches",
    "normalize_intensity",
    "scale_transform_to_full_res",
]

_MIN_SAMPLES = {"euclidean": 2, "similarity": 2, "affine": 3}

# ------------------------------------------------------------------------------------------
# Tunables. Module constants, not CLI flags: they encode the validated recipe, and a knob
# nobody has measured is a way to ship a silently worse anchor.
# ------------------------------------------------------------------------------------------
SWEEP_SIDE = 256  # px, longest side of the rotation sweep's images
# 3 deg at sigma 1: the cheapest step/blur pair whose half-step keeps >= 90 % of the peak
# correlation on every test geometry -- the measured table is in the module docstring.
SWEEP_STEP_DEG = 3.0
REFINE_HALF_DEG = 3.0  # refine each sweep candidate over +-this ...
REFINE_STEP_DEG = 0.25  # ... in this step
SUBPIXEL_UPSAMPLE = 10
BLUR_SIGMA = 1.0  # px at the scale being correlated
# Two sweep peaks closer than this are one basin (the correlation of a blurred tissue shape is
# several steps wide); the "second distinct peak" is the best LOCAL maximum outside it.
DISTINCT_DEG = 10.0
MIN_PEAK_NCC = 0.3
MIN_PEAK_RATIO = 1.15
ORB_KEYPOINTS = 5000
ORB_MIN_INLIERS = 30
ORB_AGREE_DEG = 1.0
ORB_RESIDUAL_PX = 3.0


class CoarseRefused(RuntimeError):
    """Neither the rotation sweep nor the ORB fallback produced an anchor worth trusting."""


@dataclass(frozen=True)
class Anchor:
    """The COARSE result, in THUMBNAIL pixels (``scale_transform_to_full_res`` lifts ``M``).

    ``residual_px``: for ``method == "orb"`` the RMS of the RANSAC inliers; for
    ``"ncc_sweep"`` the search grid's quantisation bound -- half a refine step of rotation at the
    moving thumbnail's half-diagonal plus half a sub-pixel translation step. The sweep has no
    point correspondences to take a residual over, so this is a resolution bound, not a fit
    residual, and it is documented as such rather than reported as NaN (NaN would reach the TRE
    report's JSON as a non-standard literal).
    ``n_inliers``: RANSAC inliers for ``"orb"``; 0 for ``"ncc_sweep"`` (no correspondences).
    """

    M: np.ndarray
    residual_px: float
    n_inliers: int
    method: str
    peak_ncc: float
    peak_ratio: float
    angle_deg: float


def _model_class(model):
    from skimage.transform import (
        AffineTransform,
        EuclideanTransform,
        SimilarityTransform,
    )

    return {
        "euclidean": EuclideanTransform,
        "similarity": SimilarityTransform,
        "affine": AffineTransform,
    }[model]


def _ransac(data, cls, min_samples, residual_threshold, max_trials, random_state):
    """Call skimage.measure.ransac across the rng/random_state kwarg rename."""
    from skimage.measure import ransac

    kw = dict(
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
    )
    try:
        return ransac(data, cls, rng=random_state, **kw)
    except TypeError:
        return ransac(data, cls, random_state=random_state, **kw)


def _rms(residuals_xy):
    if len(residuals_xy) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(np.asarray(residuals_xy) ** 2, axis=1))))


def estimate_transform_from_matches(
    src,
    dst,
    model="euclidean",
    robust=True,
    residual_threshold=3.0,
    max_trials=1000,
    random_state=0,
):
    """Estimate the transform mapping ``src`` -> ``dst`` and its residual TRE.

    Returns ``(M, residual_px, n_inliers)`` where ``M`` is the 3x3 forward affine, ``residual_px``
    is the RMS distance between the transformed inlier ``src`` and ``dst`` (the fit's Target
    Registration Error), and ``n_inliers`` is how many correspondences the fit used.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if model not in _MIN_SAMPLES:
        raise ValueError(
            f"unknown model {model!r}; expected one of {list(_MIN_SAMPLES)}"
        )

    cls = _model_class(model)
    if robust and len(src) > _MIN_SAMPLES[model]:
        tform, inliers = _ransac(
            (src, dst),
            cls,
            _MIN_SAMPLES[model],
            residual_threshold,
            max_trials,
            random_state,
        )
        if tform is None:  # ransac failed to find a consensus set
            inliers = np.ones(len(src), dtype=bool)
            tform = cls()
            tform.estimate(src, dst)
    else:
        inliers = np.ones(len(src), dtype=bool)
        tform = cls()
        tform.estimate(src, dst)

    m = np.asarray(tform.params, dtype=float)
    residual = _rms(tform(src[inliers]) - dst[inliers])
    return m, residual, int(np.count_nonzero(inliers))


def scale_transform_to_full_res(m, factor):
    """Lift a transform estimated on ``factor``-decimated images back to full-resolution pixels.

    A decimated coordinate relates to the full-resolution one by ``p_ds = p_full / factor``, so
    the full-resolution map is ``diag(f, f, 1) @ M_ds @ diag(1/f, 1/f, 1)``. The two scalings
    cancel across the linear 2x2 block and survive only on the translation column -- i.e. the
    rotation/scale part is invariant and only the offsets grow by ``factor``. Returning ``M_ds``
    unchanged is the natural mistake, and it under-translates by exactly ``factor``.
    """
    m = np.array(m, dtype=float, copy=True)
    factor = float(factor)
    m[:2, 2] *= factor
    return m


# The planes reaching this module are raw microscopy counts (uint16, 0..65535) widened to
# float32 by `slide_io.read_decimated`. Reference and moving come from different imaging cycles
# with different exposures, so two planes of the same tissue can arrive at wildly different
# dynamic ranges; the Otsu mask and ORB's FAST threshold both want a common [0, 1] scale first.
#
# A percentile window, not `img / img.max()`: one hot pixel (routine in fluorescence) would
# otherwise compress the whole tissue into the bottom of the range. p99.9 rather than p99
# because DAPI nuclei ARE the bright minority -- clipping at p99 saturates the very structure
# being matched.
_NORM_PCT = (1.0, 99.9)


def normalize_intensity(img):
    """Rescale ``img`` to [0, 1] over a robust percentile window (see the note above)."""
    a = np.asarray(img, dtype=float)
    lo, hi = (float(v) for v in np.percentile(a, _NORM_PCT))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.zeros_like(a)  # constant/all-NaN plane: nothing to match on
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


# ------------------------------------------------------------------------------------------
# Geometry. Everything below is in (x, y) = (col, row) for transforms and (row, col) for
# arrays; the two meet only in _rotate_canvas and _canvas_to_m0.
# ------------------------------------------------------------------------------------------


def _rot(theta_deg):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def _wrap_deg(a):
    """Angle in (-180, 180]."""
    a = (float(a) + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _angle_dist(a, b):
    return abs(_wrap_deg(a - b))


def _preprocess(img, sigma=BLUR_SIGMA):
    """[0, 1] window -> blur -> tissue mask (background 0). Float32."""
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu

    a = normalize_intensity(img).astype(np.float32)
    b = gaussian_filter(a, sigma)
    # Tissue scale: ~8 px at 256, i.e. the mask follows the tissue outline, not the nuclei, at
    # any resolution this is called at.
    tissue_sigma = max(2.0, max(a.shape) / 32.0)
    t_img = gaussian_filter(a, tissue_sigma)
    if float(np.ptp(t_img)) > 1e-6:
        try:
            mask = t_img > threshold_otsu(t_img)
        except ValueError:  # skimage raises on a single-valued image
            mask = np.ones_like(b, dtype=bool)
    else:
        mask = np.ones_like(b, dtype=bool)
    return np.where(mask, b, 0.0).astype(np.float32)


@dataclass(frozen=True)
class _Canvas:
    """A square canvas large enough to hold either image at any rotation, plus its FFT size."""

    side: int
    fft: tuple

    @classmethod
    def for_shapes(cls, *shapes):
        from scipy.fft import next_fast_len

        side = int(math.ceil(max(math.hypot(h, w) for h, w in shapes))) + 2
        n = next_fast_len(2 * side - 1, real=True)
        return cls(side=side, fft=(n, n))

    def offset(self, shape):
        """(x, y) offset of an image of ``shape`` placed centred on the canvas."""
        h, w = shape
        return np.array([(self.side - w) // 2, (self.side - h) // 2], dtype=float)

    @property
    def centre(self):
        c = (self.side - 1) / 2.0
        return np.array([c, c])

    def place(self, img):
        out = np.zeros((self.side, self.side), np.float32)
        ox, oy = (int(v) for v in self.offset(img.shape))
        out[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img
        return out


def _warp_xy(img, m_out_to_in, output_shape):
    """Bilinear resample: output pixel (x, y) takes ``img`` at ``m_out_to_in @ (x, y, 1)``.

    scikit-image's ``warp`` rather than ``scipy.ndimage.affine_transform``: for an affine
    matrix it dispatches to a compiled fast path that measured ~2x quicker on these sizes, and
    it takes the map in (x, y) directly.
    """
    from skimage.transform import AffineTransform, warp

    return warp(
        img,
        AffineTransform(matrix=np.asarray(m_out_to_in, float)),
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=0.0,
        preserve_range=True,
    ).astype(np.float32)


def _rotate_canvas(canvas_img, theta_deg, canvas):
    """Rotate about the canvas centre: the content at canvas point q moves to R(q - c) + c."""
    if theta_deg % 360.0 == 0.0:
        return canvas_img
    r = _rot(theta_deg)
    c = canvas.centre
    inv = np.eye(3)
    inv[:2, :2] = r.T
    inv[:2, 2] = c - r.T @ c
    return _warp_xy(canvas_img, inv, canvas_img.shape)


def _zero_mean(x):
    x = x - np.float32(x.mean())
    return x, float(np.linalg.norm(x))


class _Correlator:
    """Globally normalised FFT cross-correlation of a fixed reference canvas against rotated
    moving canvases.

    Both canvases are zero-mean over the WHOLE canvas (the prototype's normalisation): the
    masked background and the padding then carry the same constant, so neither the image
    rectangle nor the rotated corners become a feature. The correlation is linear (the FFT is at
    least twice the canvas), and its value at a shift is divided by the two canvases' norms, so a
    peak of 1 means identical content in full overlap.
    """

    def __init__(self, ref_img, canvas):
        from scipy.fft import rfft2

        self.canvas = canvas
        a, self.norm_a = _zero_mean(canvas.place(ref_img))
        self.ref_canvas = a
        self.fa = rfft2(a, s=canvas.fft, workers=-1)

    def peak(self, mov_canvas_rot):
        """(peak NCC, integer shift t (x, y)) with rotated-moving point u <-> reference u + t."""
        from scipy.fft import irfft2, rfft2

        b, norm_b = _zero_mean(mov_canvas_rot)
        if self.norm_a <= 0 or norm_b <= 0:
            return 0.0, np.zeros(2)
        fb = rfft2(b, s=self.canvas.fft, workers=-1)
        c = irfft2(self.fa * np.conj(fb), s=self.canvas.fft, workers=-1)
        i = np.unravel_index(int(np.argmax(c)), c.shape)
        n = np.array(c.shape)
        k = np.array(i, dtype=float)
        k[k > n / 2] -= n[k > n / 2]  # circular index -> signed shift
        return float(c[i]) / (self.norm_a * norm_b), k[::-1]  # (row, col) -> (x, y)


def _canvas_to_m0(theta_deg, t_xy, canvas, ref_shape, mov_shape):
    """Compose placement + rotation + shift into the thumbnail M0 (moving -> reference).

    Reference pixel r sits at canvas r + o_r; moving pixel q at q + o_m, which the rotation
    sends to R(q + o_m - c) + c and the correlation shift to that + t. So
    r = R q + [R(o_m - c) + c + t - o_r].
    """
    r = _rot(theta_deg)
    o_r, o_m, c = canvas.offset(ref_shape), canvas.offset(mov_shape), canvas.centre
    m = np.eye(3)
    m[:2, :2] = r
    m[:2, 2] = r @ (o_m - c) + c + np.asarray(t_xy, float) - o_r
    return m


@dataclass(frozen=True)
class _SweepResult:
    angles: np.ndarray
    peaks: np.ndarray
    shifts: list
    canvas: _Canvas
    ref_shape: tuple
    mov_shape: tuple
    k: int

    def m_thumb(self, i):
        """The sweep's M0 at angle index ``i``, lifted from sweep px to thumbnail px.

        ``downscale_local_mean`` by ``k`` puts sweep pixel p at thumbnail ``k p + (k - 1) / 2``.
        """
        m_s = _canvas_to_m0(
            self.angles[i], self.shifts[i], self.canvas, self.ref_shape, self.mov_shape
        )
        s = np.array(
            [[self.k, 0, (self.k - 1) / 2], [0, self.k, (self.k - 1) / 2], [0, 0, 1]]
        )
        return s @ m_s @ np.linalg.inv(s)


def _sweep(ref, mov):
    """Rotation sweep at SWEEP_SIDE: the correlation peak and shift of every SWEEP_STEP_DEG."""
    from skimage.transform import downscale_local_mean

    k = max(1, int(math.ceil(max(*ref.shape, *mov.shape) / SWEEP_SIDE)))
    rs = _preprocess(downscale_local_mean(ref, (k, k)) if k > 1 else ref)
    ms = _preprocess(downscale_local_mean(mov, (k, k)) if k > 1 else mov)
    canvas = _Canvas.for_shapes(rs.shape, ms.shape)
    corr = _Correlator(rs, canvas)
    mc = canvas.place(ms)
    angles = np.arange(0.0, 360.0, SWEEP_STEP_DEG)
    res = [corr.peak(_rotate_canvas(mc, a, canvas)) for a in angles]
    return _SweepResult(
        angles=angles,
        peaks=np.array([r[0] for r in res]),
        shifts=[r[1] for r in res],
        canvas=canvas,
        ref_shape=rs.shape,
        mov_shape=ms.shape,
        k=k,
    )


def _distinct_runner_up(angles, peaks, best_i):
    """Index of the best LOCAL maximum at least DISTINCT_DEG from the best angle.

    A local maximum, not merely the best angle outside the exclusion: the flank of the main
    basin just past DISTINCT_DEG is not a competing hypothesis. Falls back to the flank maximum
    (conservative: a smaller ratio) when the curve has no other local maximum.
    """
    n = len(peaks)
    far = np.array([_angle_dist(a, angles[best_i]) >= DISTINCT_DEG for a in angles])
    local = np.array(
        [
            peaks[i] >= peaks[(i - 1) % n] and peaks[i] >= peaks[(i + 1) % n]
            for i in range(n)
        ]
    )
    cand = np.flatnonzero(far & local)
    if cand.size == 0:
        cand = np.flatnonzero(far)
    if cand.size == 0:
        return None
    return int(cand[np.argmax(peaks[cand])])


class _Refiner:
    """Correlation at the full thumbnail, in the REFERENCE frame, around a predicted anchor.

    The sweep already fixed the translation to within a few thumbnail px per degree of angle
    change, so the refine step does not need the sweep's global search: the moving thumbnail is
    warped into the reference frame by the predicted transform and correlated over shifts of at
    most ``window`` px (zero padding of that width keeps the correlation linear). That keeps each
    of the ~50 evaluations to one reference-sized warp and two FFTs.
    """

    def __init__(self, ref_p, mov_p):
        from scipy.fft import next_fast_len, rfft2

        self.ref_p, self.mov_p = ref_p, mov_p
        h, w = ref_p.shape
        self.window = int(max(16, math.ceil(0.08 * max(h, w))))
        self.fft = (
            next_fast_len(h + self.window, real=True),
            next_fast_len(w + self.window, real=True),
        )
        a, self.norm_a = _zero_mean(ref_p)
        self.ref_zm = a
        self.fa = rfft2(a, s=self.fft, workers=-1)
        self.c_mov = np.array([(mov_p.shape[1] - 1) / 2.0, (mov_p.shape[0] - 1) / 2.0])

    def m_at(self, angle, p):
        """Rotate by ``angle`` about the moving centre, then put that centre at ``p``."""
        m = np.eye(3)
        r = _rot(angle)
        m[:2, :2] = r
        m[:2, 2] = np.asarray(p, float) - r @ self.c_mov
        return m

    def warp(self, m):
        """The moving thumbnail resampled into the reference frame through ``m`` (mov -> ref)."""
        return _warp_xy(self.mov_p, np.linalg.inv(m), self.ref_p.shape)

    def peak(self, m):
        """(peak NCC, shift t (x, y)) with ``T(t) @ m`` the better anchor; |t| <= window."""
        from scipy.fft import irfft2, rfft2

        b, norm_b = _zero_mean(self.warp(m))
        if self.norm_a <= 0 or norm_b <= 0:
            return 0.0, np.zeros(2)
        c = irfft2(
            self.fa * np.conj(rfft2(b, s=self.fft, workers=-1)), s=self.fft, workers=-1
        )
        wn = self.window
        # shifts in [-wn, wn] on both axes: rows/cols 0..wn and n-wn..n-1 of the circular map
        rows = np.r_[0 : wn + 1, self.fft[0] - wn : self.fft[0]]
        cols = np.r_[0 : wn + 1, self.fft[1] - wn : self.fft[1]]
        sub = c[np.ix_(rows, cols)]
        i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
        dy = rows[i] - (self.fft[0] if rows[i] > wn else 0)
        dx = cols[j] - (self.fft[1] if cols[j] > wn else 0)
        return float(sub[i, j]) / (self.norm_a * norm_b), np.array([dx, dy], float)

    def refine(self, m_pred):
        """Best (peak, angle, M) over the predicted angle +- REFINE_HALF_DEG, to REFINE_STEP_DEG.

        Coarse-to-fine rather than every 0.25 deg: a 1 deg pass over the whole +-3 deg window,
        then REFINE_STEP_DEG steps over +-0.75 deg around its best. Same final resolution and
        the same window, 13 evaluations instead of 25 -- at 1024 px the correlation surface is
        one smooth basin at this scale (the sweep already chose the basin).
        """
        a0 = _m_angle(m_pred)
        p = m_pred[:2, :2] @ self.c_mov + m_pred[:2, 2]  # where the moving centre lands
        best = (-np.inf, a0, m_pred)
        seen = set()

        def visit(a):
            nonlocal best
            key = round(a / REFINE_STEP_DEG)
            if key in seen:
                return
            seen.add(key)
            m = self.m_at(a, p)
            pk, t = self.peak(m)
            if pk > best[0]:
                m_t = m.copy()
                m_t[:2, 2] += t
                best = (pk, a, m_t)

        coarse = 1.0
        n = int(round(REFINE_HALF_DEG / coarse))
        for j in range(-n, n + 1):
            visit(a0 + j * coarse)
        centre = best[1]
        n = int(round((coarse - REFINE_STEP_DEG) / REFINE_STEP_DEG))
        for j in range(-n, n + 1):
            visit(centre + j * REFINE_STEP_DEG)
        return best

    def subpixel(self, m):
        """Sub-pixel translation of ``m`` (upsampled phase correlation, unnormalised)."""
        from skimage.registration import phase_cross_correlation

        b, norm_b = _zero_mean(self.warp(m))
        if self.norm_a <= 0 or norm_b <= 0:
            return m  # nothing to correlate (a blank plane); the refusal gate decides
        pad = [(0, self.fft[0] - b.shape[0]), (0, self.fft[1] - b.shape[1])]
        shift, _err, _ph = phase_cross_correlation(
            np.pad(self.ref_zm, pad),
            np.pad(b, pad),
            upsample_factor=SUBPIXEL_UPSAMPLE,
            normalization=None,
        )
        t = np.asarray(shift, float)[::-1]  # (row, col) -> (x, y)
        if np.max(np.abs(t)) > 1.0:
            logger.warning(
                f"coarse: sub-pixel shift {t.round(2).tolist()} is not a sub-pixel "
                "correction of the integer NCC peak; keeping the integer peak"
            )
            return m
        out = m.copy()
        out[:2, 2] += t
        return out


def _m_angle(m):
    return math.degrees(math.atan2(m[1, 0], m[0, 0]))


def _orb_fallback(ref, mov, model):
    """scikit-image ORB + cross-checked matches + RANSAC. Returns (M, rms, n_inliers)."""
    from scipy.ndimage import gaussian_filter
    from skimage.feature import ORB, match_descriptors

    def features(img):
        a = gaussian_filter(normalize_intensity(img), BLUR_SIGMA)
        det = ORB(n_keypoints=ORB_KEYPOINTS)
        try:
            det.detect_and_extract(a)
        except RuntimeError:  # skimage raises when no keypoint survives
            return np.empty((0, 2)), np.empty((0, 0), bool)
        return det.keypoints[:, ::-1], det.descriptors  # (row, col) -> (x, y)

    kr, dr = features(ref)
    km, dm = features(mov)
    if len(kr) < _MIN_SAMPLES[model] or len(km) < _MIN_SAMPLES[model]:
        return None, float("nan"), 0
    matches = match_descriptors(dm, dr, cross_check=True, max_ratio=0.8)
    if len(matches) <= _MIN_SAMPLES[model]:
        return None, float("nan"), 0
    return estimate_transform_from_matches(
        km[matches[:, 0]],
        kr[matches[:, 1]],
        model=model,
        residual_threshold=ORB_RESIDUAL_PX,
        max_trials=2000,
    )


def estimate_anchor(ref, mov, model="euclidean"):
    """Estimate ``M0`` mapping ``mov`` thumbnail coordinates onto ``ref``; see the module doc.

    Returns an :class:`Anchor`. Raises :class:`CoarseRefused` when no candidate passes, with the
    scores in the message; the caller adds the slide names.
    """
    if model != "euclidean":
        raise ValueError(
            f"COARSE estimates a rigid (euclidean) anchor only, got model={model!r}. "
            "SOLVE's robust affine absorbs any residual scale or shear."
        )
    ref = np.asarray(ref, dtype=np.float32)
    mov = np.asarray(mov, dtype=np.float32)
    t0 = time.perf_counter()

    # 1. sweep
    sw = _sweep(ref, mov)
    i1 = int(np.argmax(sw.peaks))
    i2 = _distinct_runner_up(sw.angles, sw.peaks, i1)
    idx = [i1] + ([i2] if i2 is not None else [])
    t_sweep = time.perf_counter() - t0

    # 2. refine each candidate at the thumbnail
    rf = _Refiner(_preprocess(ref), _preprocess(mov))
    refined = sorted((rf.refine(sw.m_thumb(i)) for i in idx), key=lambda r: -r[0])
    peak, _angle, m_best = refined[0]
    runner = refined[1][0] if len(refined) > 1 else 0.0
    ratio = peak / runner if runner > 0 else float("inf")
    m_sweep = rf.subpixel(m_best)
    angle = _m_angle(m_sweep)
    half_diag = 0.5 * math.hypot(*mov.shape)
    quant_px = math.hypot(
        math.radians(REFINE_STEP_DEG / 2.0) * half_diag, 0.5 / SUBPIXEL_UPSAMPLE
    )
    logger.info(
        f"coarse: NCC sweep {len(sw.angles)} angles @1/{sw.k} in {t_sweep:.1f}s -> "
        f"candidates {[round(float(sw.angles[i]), 1) for i in idx]} (sweep peaks "
        f"{[round(float(sw.peaks[i]), 3) for i in idx]}); refined @{max(ref.shape)}px -> "
        f"angle {angle:+.2f} deg, peak NCC {peak:.3f}, ratio {ratio:.2f} "
        f"({time.perf_counter() - t0:.1f}s total)"
    )
    sweep = Anchor(
        M=m_sweep,
        residual_px=quant_px,
        n_inliers=0,
        method="ncc_sweep",
        peak_ncc=float(peak),
        peak_ratio=float(ratio),
        angle_deg=_wrap_deg(angle),
    )
    if peak >= MIN_PEAK_NCC and ratio >= MIN_PEAK_RATIO:
        return sweep

    # 3. fallback
    why = (
        f"NCC sweep: peak {peak:.3f} (need >= {MIN_PEAK_NCC}), best/second-distinct ratio "
        f"{ratio:.2f} (need >= {MIN_PEAK_RATIO}) at {_wrap_deg(angle):+.2f} deg"
    )
    logger.warning(f"coarse: {why} -- trying the ORB fallback")
    m_orb, rms, n_in = _orb_fallback(ref, mov, model)
    if m_orb is None or n_in < ORB_MIN_INLIERS:
        raise CoarseRefused(
            f"no trustworthy rigid anchor. {why}; ORB fallback: {n_in} RANSAC inliers "
            f"(need >= {ORB_MIN_INLIERS}). Check --nuclear-index, that both slides show the "
            "same tissue, and that the nuclear channel is not blank."
        )
    a_orb = _m_angle(m_orb)
    orb_peak, _t = rf.peak(m_orb)
    orb = Anchor(
        M=m_orb,
        residual_px=float(rms),
        n_inliers=int(n_in),
        method="orb",
        peak_ncc=float(orb_peak),
        peak_ratio=float(ratio),
        angle_deg=_wrap_deg(a_orb),
    )
    sweep_has_candidate = peak >= MIN_PEAK_NCC
    agrees = any(_angle_dist(a_orb, r[1]) <= ORB_AGREE_DEG for r in refined)
    logger.info(
        f"coarse: ORB fallback {n_in} inliers, rms {rms:.2f}px, angle {a_orb:+.2f} deg, "
        f"NCC at that angle {orb_peak:.3f}; agrees with a sweep candidate: {agrees}"
    )
    if not sweep_has_candidate or agrees or orb_peak >= peak:
        return orb
    logger.warning(
        "coarse: the ORB fallback disagrees with the sweep and scores a lower NCC "
        f"({orb_peak:.3f} < {peak:.3f}); keeping the sweep's anchor"
    )
    return sweep


def estimate_rigid(ref, mov, model="euclidean", **_ignored):
    """``(M0, residual_px, n_inliers)`` -- the 3-tuple view of :func:`estimate_anchor`."""
    a = estimate_anchor(ref, mov, model=model)
    return a.M, a.residual_px, a.n_inliers
