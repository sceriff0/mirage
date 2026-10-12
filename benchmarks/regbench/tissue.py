"""Tissue-level agreement of the WRITTEN images: ACCREDIT's score and its pieces. Secondary.

For each variant that has an image on disk, the nuclear channel of that image is compared
with the nuclear channel of the reference slide at 4000 px (longest edge of the reference),
with the pieces and the composite of ACCREDIT's quality score (Zhou et al. 2026, bioRxiv,
doi 10.64898/2026.08.08.743602). An ``initial`` row (the unregistered moving slide placed on
the reference canvas by the identity) gives the scale; on known-truth cases a ``truth`` row
(the moving slide resampled by the generating map, drawn by the benchmark) and the same
plane shifted by known amounts say what the block can resolve.

**This block is never ranked and never enters the primary tables.** ACCREDIT's score was
designed and reported for an H&E reference and a fluorescence moving image; here both are
the same fluorescence channel, which is outside what it was validated for. Its weights,
ranges and thresholds were not calibrated on this data, and the source reports no check of
the score against landmark error.

**The definition is the RELEASED CODE**, re-implemented here (nothing is vendored; the
repository is MIT-licensed): github.com/LeeZhou-bearway/ACCREDIT at commit ``9293473``,
``quality/quality_metrics.py`` ``[FULL]`` and
``benchmark/evaluation/compute_qcs_4methods.py`` ``[FULL]`` (the script that scored the
other methods' images in the preprint), with ``benchmark/evaluation/qcs_formula_v2.py``
``[FULL]``. Line numbers below are of those files at that commit.

=====================  ====================================================================
piece                  as released
=====================  ====================================================================
evaluation size        longest edge 4000 px, area mean, never enlarged; the output size is
                       ``(int(w * s), int(h * s))`` (``compute_qcs_4methods.py:52-56``)
8-bit image            1st..99th percentile of the POSITIVE pixels mapped to 0..255; all
                       zero when fewer than 100 pixels are positive (``:44-49``)
tissue mask            Otsu, a 15 px elliptical closing twice, connected components under
                       1e-4 of the frame dropped (``:59-70``)
crop                   every piece below except the two penalties is computed inside the
                       bounding box of the WRITTEN image's mask; under 100 mask pixels
                       there is no score (``:86-92``)
tissue Dice            ``2 |A & B| / (|A| + |B| + 1e-10)`` (``quality_metrics.py:26-29``)
boundary Dice          band = dilate - erode with a 17 px ellipse (r = 8) of each mask;
                       Dice of the two MASKS restricted to the union of the two bands
                       (``:34-54``)
contour distance       the single longest external contour of each mask, at most 2000
                       evenly taken points each; the MEAN (and the 95th percentile) of the
                       pooled nearest-neighbour distances, both ways; 999 when a mask has
                       no contour (``:86-119``). The composite uses the mean (``:260``).
inlier ratio           ORB, 5000 features, on the two 8-bit crops, no mask; brute-force
                       Hamming matching with cross-check; a four-parameter similarity by
                       RANSAC at 5.0 px; inliers / matches (``:124-154``)
NMI                    ``(H(A) + H(B)) / H(A, B)``, 64 x 64 bins over the data range,
                       natural log, on the pixels where BOTH 8-bit images are above 0
                       (``:233-250``); 1.0 when under 100 such pixels
NCC                    on the overlap of the two masks; stored, NOT in the composite
                       (``:59-81``, ``qcs_formula_v2.py:9``)
coverage               written-image mask / ALL pixels of the frame
                       (``compute_qcs_4methods.py:99``)
background             written-image mask on the reference's background / the reference's
                       BACKGROUND (``:100``)
composite              ``(0.30 TD + 0.30 BD + 0.15 clamp(1 - d / 500) + 0.10 clamp(IR /
                       0.15) + 0.15 clamp((NMI - 1) / 0.15)) * min(1, coverage / 0.20) *
                       max(0, 1 - background)``, rounded to 4 places (``:101-105``)
=====================  ====================================================================

Where the preprint's Methods text and the released code disagree, the code is implemented
(ruling R3.3); the disagreements are listed in ``benchmarks/regbench/README.md``.

**What is the benchmark's, not ACCREDIT's.** (1) The reference is the nuclear channel of
the reference slide, not an inverted H&E. (2) Both images are 16-bit, so both go through
the released ``norm8`` after the reduction; in the released script that function is
defined and never called (its inputs were 8-bit already). (3) The written image is read
from the method's own file at full resolution and reduced by an exact area mean (what
``cv2.INTER_AREA`` computes), where the script enlarged 2000 px PNGs. (4) Pixels a method
declares as fill (outside ``valid_box``) are set to 0 before the reduction; the script's
``== 255 -> 0`` rule for one method is not copied. (5) An image that is not on the
reference grid (``origin_xy``: ASHLAR's mosaic) is placed on the reference canvas first.
(6) A written image with under 100 mask pixels is a row with ``status: no_tissue`` and a
composite of 0, where the script stores no score.

**Two steps, two environments.** ``reduce_native`` runs in the driver, right after a
method's writer, with NumPy and tifffile only (every method image has them): one channel is
read in row bands and never the whole image. The pieces need OpenCV (``cv2``), which the
STARE image does not have: ``python -m benchmarks.regbench tissue`` computes them in any
environment that has it and leaves ``tissue.json`` beside each reduced plane; ``score``
reads those, computes what is missing when ``cv2`` is importable, and otherwise lists the
row in ``tissue_missing.csv`` with the reason. The OpenCV version is recorded in each row.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .cases import result_dir, write_json

SOURCE = "LeeZhou-bearway/ACCREDIT@9293473"
DEFINITION = "accredit-9293473-regbench-1"  # bump when anything below changes
LABEL = (
    "ACCREDIT's composite (Zhou et al. 2026, bioRxiv, released code), designed for H&E "
    "references; weights, ranges and thresholds not calibrated on this data; secondary, "
    "not part of any ranking."
)
# The frozen values (compute_qcs_4methods.py:18-22, :59-62; quality_metrics.py:34, :86,
# :124, :219-229, :233).
EVAL_PX = 4000
MIN_POSITIVE = (
    100  # norm8, NMI and the written mask: fewer pixels than this is "nothing"
)
CLOSE_PX = 15
CLOSE_ITERATIONS = 2
MIN_AREA_FRAC = 0.0001
BAND_WIDTH = 8
MAX_CONTOUR_POINTS = 2000
NO_CONTOUR = 999.0
ORB_FEATURES = 5000
RANSAC_PX = 5.0
NMI_BINS = 64
WEIGHTS = {"td": 0.30, "bd": 0.30, "hd": 0.15, "ir": 0.10, "nmi": 0.15}
HD_MAX = 500.0
IR_MAX = 0.15
NMI_MIN = 1.0
NMI_RANGE = 0.15
COV_MIN = 0.20
FROZEN = {
    "eval_px": EVAL_PX,
    "band_width": BAND_WIDTH,
    "nmi_bins": NMI_BINS,
    "orb_features": ORB_FEATURES,
    "ransac_px": RANSAC_PX,
    "close_px": CLOSE_PX,
    "close_iterations": CLOSE_ITERATIONS,
    "min_area_frac": MIN_AREA_FRAC,
    "max_contour_points": MAX_CONTOUR_POINTS,
    "weights": WEIGHTS,
    "hd_max": HD_MAX,
    "ir_max": IR_MAX,
    "nmi_min": NMI_MIN,
    "nmi_range": NMI_RANGE,
    "cov_min": COV_MIN,
}
# The count under this is printed; the source's good / acceptable / poor classes are not.
FAILED_BELOW = 0.25
DOSE_UM = (1.0, 3.0, 10.0, 30.0, 100.0)  # shifts of the truth plane, along x
TRUTH_STEP = (
    4  # the true map is inverted on every 4th evaluation pixel, then interpolated
)
BAND_ROWS = 1024  # native rows held at once while reducing
SUM_ROWS = 64  # of which this many are summed at a time
MOVING_PLANE = "moving_eval.tif"
RESULT = "tissue.json"
TABLES = (
    "tissue",
    "tissue_summary",
    "tissue_missing",
    "tissue_dose",
    "tissue_dose_summary",
)
PIECES = (
    "qcs",
    "tissue_dice",
    "boundary_dice",
    "hausdorff_mean_um",
    "hausdorff_p95_um",
    "inlier_ratio",
    "n_matches",
    "n_inliers",
    "nmi",
    "ncc",
    "coverage",
    "bg_coverage",
)
NO_OPENCV = (
    "OpenCV (cv2) is not importable where this was scored: run `python -m "
    "benchmarks.regbench tissue` in an environment that has it, then score again"
)


# ── the reduced plane: NumPy only, in row bands ───────────────────────────────
def eval_shape(ref_hw, eval_px=EVAL_PX):
    """``(h, w)`` of the evaluation plane: the released rule, its truncation included."""
    h, w = (int(v) for v in ref_hw)
    scale = eval_px / max(h, w)
    if scale >= 1.0:
        return h, w
    return int(h * scale), int(w * scale)


def box_sums(a, t):
    """Integrals of a piecewise-constant signal along the LAST axis.

    Sample ``i`` of ``a`` fills ``[i, i + 1)``; the result's entry ``j`` is the integral
    over ``[t[j], t[j + 1])``. ``t`` is non-decreasing; what lies outside ``[0, n]``
    counts as zero. Exact for fractional boundaries: an area mean, not a sampling."""
    a = np.asarray(a)
    n = a.shape[-1]
    t = np.clip(np.asarray(t, dtype=float), 0.0, n)
    if n == 0 or len(t) < 2:
        return np.zeros((*a.shape[:-1], max(len(t) - 1, 0)))
    lo = np.floor(t).astype(np.int64)
    frac = t - lo
    i0, i1 = lo[:-1], lo[1:]
    idx = np.minimum(lo, n - 1)
    # reduceat sums a[idx[j] : idx[j + 1]], and returns a[idx[j]] for an empty range
    out = np.add.reduceat(a, idx, axis=-1, dtype=np.float64)[..., :-1]
    out[..., idx[:-1] >= idx[1:]] = 0.0
    tail = (i1 == n) & (i0 < n)  # the range reaches the end, the clipped index did not
    out[..., tail] += a[..., n - 1 : n]
    out -= np.where(i0 < n, frac[:-1], 0.0) * a[..., np.minimum(i0, n - 1)]
    out += np.where(i1 < n, frac[1:], 0.0) * a[..., np.minimum(i1, n - 1)]
    return out


def reduce_plane(
    arr,
    channel,
    out_hw,
    fx,
    fy,
    origin_xy=(0.0, 0.0),
    valid_box=None,
    band_rows=BAND_ROWS,
):
    """One channel of a lazy ``(C, H, W)`` image as an area-mean plane of ``out_hw``.

    Output pixel ``(k, j)`` is the mean over the canvas rectangle ``[j fx, (j + 1) fx) x
    [k fy, (k + 1) fy)``; the image's pixel ``(y, x)`` fills ``[x + ox, x + ox + 1) x
    [y + oy, y + oy + 1)`` of that canvas (``origin_xy``: where the image's first pixel
    sits in the canvas's frame). What the image does not cover, and what lies outside
    ``valid_box`` (``[x0, y0, x1, y1]`` in the image's own pixels), is 0. At most
    ``band_rows`` (+2) image rows of ONE channel are in memory at a time."""
    eh, ew = (int(v) for v in out_hw)
    hn, wn = (int(v) for v in arr.shape[1:])
    ox, oy = (float(v) for v in origin_xy)
    tx = np.clip(np.arange(ew + 1) * float(fx) - ox, 0.0, wn)
    ty = np.clip(np.arange(eh + 1) * float(fy) - oy, 0.0, hn)
    x_lo, x_hi = int(math.floor(tx[0])), int(math.ceil(tx[-1]))
    out = np.zeros((eh, ew), dtype=np.float64)
    box = None
    if valid_box is not None:
        box = [int(v) for v in valid_box]
    per_band = max(1, int(band_rows // max(float(fy), 1.0)))
    for k0 in range(0, eh, per_band):
        k1 = min(eh, k0 + per_band)
        y_lo, y_hi = int(math.floor(ty[k0])), int(math.ceil(ty[k1]))
        if y_hi <= y_lo or x_hi <= x_lo:
            continue
        block = np.asarray(arr[int(channel), slice(y_lo, y_hi), slice(x_lo, x_hi)])
        if box is not None:
            block = np.array(block, copy=True)
            block[:, : max(box[0] - x_lo, 0)] = 0
            block[:, max(box[2] - x_lo, 0) :] = 0
            block[: max(box[1] - y_lo, 0)] = 0
            block[max(box[3] - y_lo, 0) :] = 0
        # a few rows at a time: the sums are taken in float64, eight bytes a pixel
        cols = np.concatenate(
            [
                box_sums(block[i : i + SUM_ROWS], tx - x_lo)
                for i in range(0, len(block), SUM_ROWS)
            ]
        )
        del block
        out[k0:k1] = box_sums(cols.T, ty[k0 : k1 + 1] - y_lo).T
    out /= float(fx) * float(fy)
    dtype = np.dtype(arr.dtype)
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(dtype)
    return out.astype(np.float32)


def _factors(ref_hw, eval_px=EVAL_PX):
    """``(eval (h, w), fx, fy)``: full-resolution pixels per evaluation pixel, per axis."""
    eh, ew = eval_shape(ref_hw, eval_px)
    return (eh, ew), int(ref_hw[1]) / ew, int(ref_hw[0]) / eh


def reduce(path, channel, ref_hw, origin_xy=(0.0, 0.0), valid_box=None, eval_px=None):
    """The evaluation plane of one channel of an image file, on the reference canvas."""
    from stare.slide_io import open_lazy

    (eh, ew), fx, fy = _factors(ref_hw, eval_px or EVAL_PX)
    arr, _, close = open_lazy(path)
    try:
        return reduce_plane(arr, int(channel), (eh, ew), fx, fy, origin_xy, valid_box)
    finally:
        close()


def write_plane(path, plane):
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"tmp{time.time_ns()}_{path.name}")
    try:
        tifffile.imwrite(tmp, np.asarray(plane), compression="zlib")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_plane(path):
    import tifffile

    return tifffile.imread(str(path))


def native_channel(native, case, n_channels):
    """The nuclear channel of a written image: the writer's own statement, else the moving
    slide's index (a one-channel file is its only channel)."""
    channel = native.get("nuclear_channel")
    if channel is None:
        channel = 0 if n_channels == 1 else case.extra.get("mov_channel", 0)
    return int(channel)


# why a brightfield case (HyReCo) has no tissue-level row
NOT_BRIGHTFIELD = (
    "the tissue-level block compares the nuclear channel of a written image with the "
    "reference's; a brightfield case has no nuclear channel, and what each method "
    "writes for it differs in kind (RGB, or a one-channel proxy)"
)


def reduce_native(case, native, variant_dir, eval_px=None):
    """Reduce a method's written image right after its writer: ``<variant_dir>/tissue/
    moving_eval.tif`` and the record the driver stores as ``run.json: tissue``."""
    from stare.slide_io import open_lazy

    t0 = time.perf_counter()
    if not case.ref_hw:
        return {"error": "the case records no reference size (ref_hw)"}
    path = native.get("nuclear_file") or native["files"][0]
    arr, _, close = open_lazy(path)
    try:
        channel = native_channel(native, case, int(arr.shape[0]))
        (eh, ew), fx, fy = _factors(case.ref_hw, eval_px or EVAL_PX)
        origin = native.get("origin_xy") or (0.0, 0.0)
        plane = reduce_plane(
            arr, channel, (eh, ew), fx, fy, origin, native.get("valid_box")
        )
        shape = [int(v) for v in arr.shape]
    finally:
        close()
    dest = write_plane(Path(variant_dir) / "tissue" / MOVING_PLANE, plane)
    return {
        "file": str(dest),
        "from": str(path),
        "channel": channel,
        "image_shape": shape,
        "origin_xy": [float(v) for v in origin],
        "valid_box": native.get("valid_box"),
        "eval_hw": [eh, ew],
        "px_per_eval_px": [fx, fy],
        "seconds": time.perf_counter() - t0,
    }


# ── the pieces: the released code, re-implemented (needs OpenCV) ──────────────
def have_cv2():
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def norm8(a):
    """``compute_qcs_4methods.py:44-49``."""
    f = np.asarray(a).astype(np.float32)
    pos = f[f > 0]
    if len(pos) < MIN_POSITIVE:
        return np.zeros(f.shape, dtype=np.uint8)
    p1, p99 = np.percentile(pos, [1, 99])
    return np.clip((f - p1) / (p99 - p1 + 1e-10) * 255, 0, 255).astype(np.uint8)


def make_mask(g, with_threshold=False):
    """``compute_qcs_4methods.py:59-70``: the tissue mask of an 8-bit image."""
    import cv2

    thr, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_PX, CLOSE_PX))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERATIONS)
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_lbl <= 1:
        keep = m > 0
    else:
        min_area = int(MIN_AREA_FRAC * m.size)
        big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_area)
        keep = np.isin(lbls, big[big > 0])
    return (keep, float(thr)) if with_threshold else keep


def tissue_dice(mask_a, mask_b):
    """``quality_metrics.py:26-29``."""
    a, b = np.asarray(mask_a) > 0, np.asarray(mask_b) > 0
    return float(2.0 * np.sum(a & b) / (np.sum(a) + np.sum(b) + 1e-10))


def boundary_dice(mask_a, mask_b, band_width=BAND_WIDTH):
    """``quality_metrics.py:34-54``: Dice of the two masks inside the union of their
    boundary bands (dilate minus erode)."""
    import cv2

    size = band_width * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    a = (np.asarray(mask_a) > 0).astype(np.uint8)
    b = (np.asarray(mask_b) > 0).astype(np.uint8)
    a_band = cv2.dilate(a, kernel) - cv2.erode(a, kernel)
    b_band = cv2.dilate(b, kernel) - cv2.erode(b, kernel)
    region = (a_band > 0) | (b_band > 0)
    if not region.any():
        return 0.0
    a_in, b_in = (a > 0) & region, (b > 0) & region
    return float(2.0 * np.sum(a_in & b_in) / (np.sum(a_in) + np.sum(b_in) + 1e-10))


def ncc(img_a, img_b, mask=None):
    """``quality_metrics.py:59-81``."""
    a, b = np.asarray(img_a, dtype=np.float64), np.asarray(img_b, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask) > 0
        if not m.any():
            return 0.0
        a, b = a[m], b[m]
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    denom = math.sqrt(float(np.sum(a**2))) * math.sqrt(float(np.sum(b**2)))
    return 0.0 if denom < 1e-10 else float(np.sum(a * b) / denom)


def contour_distance(mask_a, mask_b, max_pts=MAX_CONTOUR_POINTS):
    """``quality_metrics.py:86-119`` (the released ``hausdorff_distance``): ``(mean, 95th
    percentile)`` of the nearest-neighbour distances between the longest external contour
    of each mask, pooled over both directions, in pixels. Not a maximum."""
    import cv2

    pts = []
    for mask in (mask_a, mask_b):
        contours, _ = cv2.findContours(
            (np.asarray(mask) > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            return NO_CONTOUR, NO_CONTOUR
        p = max(contours, key=len).reshape(-1, 2).astype(np.float32)
        if len(p) > max_pts:
            p = p[np.linspace(0, len(p) - 1, max_pts, dtype=int)]
        pts.append(p)
    d = np.sqrt(np.sum((pts[0][:, None, :] - pts[1][None, :, :]) ** 2, axis=2))
    both = np.concatenate([d.min(axis=1), d.min(axis=0)])
    return float(np.mean(both)), float(np.percentile(both, 95))


def orb_inlier_ratio(img_a, img_b, n_features=ORB_FEATURES, reproj=RANSAC_PX):
    """``quality_metrics.py:124-154``: ``(inlier ratio, inliers, matches)``."""
    import cv2

    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)
    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return 0.0, 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    if len(matches) < 4:
        return 0.0, 0, len(matches)
    p1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    p2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    _, inliers = cv2.estimateAffinePartial2D(
        p1, p2, method=cv2.RANSAC, ransacReprojThreshold=reproj
    )
    if inliers is None:
        return 0.0, 0, len(matches)
    n = int(inliers.sum())
    return n / len(matches), n, len(matches)


def nmi(img_a, img_b, bins=NMI_BINS):
    """``quality_metrics.py:233-250``: ``(H(A) + H(B)) / H(A, B)``; 1.0 is independence."""
    a, b = np.asarray(img_a), np.asarray(img_b)
    mask = (a > 0) & (b > 0)
    if mask.sum() < MIN_POSITIVE:
        return 1.0
    h, _, _ = np.histogram2d(
        a[mask].ravel().astype(np.float64), b[mask].ravel().astype(np.float64), bins
    )
    h = h / (h.sum() + 1e-10)

    def ent(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p + 1e-10))

    ha, hb, hab = ent(h.sum(1)), ent(h.sum(0)), ent(h.ravel())
    return float((ha + hb) / hab) if hab > 1e-10 else 1.0


def _clamp(v):
    return max(0.0, min(1.0, float(v)))


def terms(m):
    """The normalised terms and the two penalties (``compute_qcs_4methods.py:101-105``)."""
    return {
        "s_td": _clamp(m["tissue_dice"]),
        "s_bd": _clamp(m["boundary_dice"]),
        "s_hd": _clamp(1.0 - m["hausdorff_mean"] / HD_MAX),
        "s_ir": _clamp(m["inlier_ratio"] / IR_MAX),
        "s_nmi": _clamp((m["nmi"] - NMI_MIN) / NMI_RANGE),
        "cov_factor": min(1.0, m["coverage"] / COV_MIN),
        "bg_penalty": max(0.0, 1.0 - m["bg_coverage"]),
    }


def composite(m):
    """ACCREDIT's composite from its pieces (``hausdorff_mean`` in evaluation pixels)."""
    t = terms(m)
    base = (
        WEIGHTS["td"] * t["s_td"]
        + WEIGHTS["bd"] * t["s_bd"]
        + WEIGHTS["hd"] * t["s_hd"]
        + WEIGHTS["ir"] * t["s_ir"]
        + WEIGHTS["nmi"] * t["s_nmi"]
    )
    return round(base * t["cov_factor"] * t["bg_penalty"], 4)


def components(ref8, ref_mask, mov8, mov_mask):
    """Every piece and the composite for one written image
    (``compute_qcs_4methods.py:86-109``). The inputs are the 8-bit evaluation images and
    their masks; ``None`` values and ``status: no_tissue`` when the written image's mask
    has under 100 pixels (the script stores no score there; here the composite is 0)."""
    ref_mask, mov_mask = np.asarray(ref_mask, bool), np.asarray(mov_mask, bool)
    cov = float(mov_mask.sum() / mov_mask.size)
    bg = float((mov_mask & ~ref_mask).sum() / max(1, int((~ref_mask).sum())))
    out = {"status": "ok", "coverage": cov, "bg_coverage": bg}
    ys, xs = np.where(mov_mask)
    if len(ys) < MIN_POSITIVE:
        return {**out, "status": "no_tissue", "qcs": 0.0}
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    hm = ref_mask[y1:y2, x1:x2].astype(np.uint8) * 255
    wm = mov_mask[y1:y2, x1:x2].astype(np.uint8) * 255
    hi, wf = ref8[y1:y2, x1:x2], mov8[y1:y2, x1:x2]
    hd_mean, hd_p95 = contour_distance(hm, wm)
    ir, n_inliers, n_matches = orb_inlier_ratio(hi, wf)
    out.update(
        tissue_dice=tissue_dice(hm, wm),
        boundary_dice=boundary_dice(hm, wm),
        hausdorff_mean=hd_mean,
        hausdorff_p95=hd_p95,
        inlier_ratio=float(ir),
        n_inliers=int(n_inliers),
        n_matches=int(n_matches),
        nmi=nmi(hi, wf),
        ncc=ncc(hi, wf, mask=(hm > 0) & (wm > 0)),
        crop_box=[int(x1), int(y1), int(x2), int(y2)],
    )
    out.update(terms(out))
    out["qcs"] = composite(out)
    return out


class Reference:
    """The reference's evaluation plane, 8-bit image and mask: made once per case."""

    def __init__(self, plane, eval_um=None):
        self.plane = np.asarray(plane)
        self.eval_um = eval_um
        self.img8 = norm8(self.plane)
        self.mask, self.otsu = make_mask(self.img8, with_threshold=True)

    def score(self, moving_plane):
        """The pieces of one written image against this reference, with what a reader
        needs beside them: micrometres, the masks' sizes, the Otsu thresholds."""
        import cv2

        mov8 = norm8(moving_plane)
        if mov8.shape != self.img8.shape:
            raise ValueError(
                f"the written image's plane is {mov8.shape}, the reference's "
                f"{self.img8.shape}"
            )
        mov_mask, mov_otsu = make_mask(mov8, with_threshold=True)
        out = components(self.img8, self.mask, mov8, mov_mask)
        um = self.eval_um
        for key in ("hausdorff_mean", "hausdorff_p95"):
            v = out.get(key)
            none = v is None or um is None or v == NO_CONTOUR
            out[f"{key}_um"] = None if none else v * um
        out.update(
            eval_h=int(mov8.shape[0]),
            eval_w=int(mov8.shape[1]),
            eval_um_per_px=um,
            ref_tissue_frac=float(self.mask.mean()),
            mov_tissue_frac=float(mov_mask.mean()),
            ref_otsu=self.otsu,
            mov_otsu=mov_otsu,
            opencv=cv2.__version__,
            definition=DEFINITION,
        )
        return out


def score_planes(ref_plane, mov_plane, eval_um=None):
    return Reference(ref_plane, eval_um).score(mov_plane)


# ── benchmark-side planes: the reference, the unregistered slide, the truth ───
def eval_um(case, eval_px=None):
    """Micrometres per evaluation pixel (mean of the two axes), or None."""
    if not case.pixel_size_um or not case.ref_hw:
        return None
    _, fx, fy = _factors(case.ref_hw, eval_px or EVAL_PX)
    return float(case.pixel_size_um) * (fx + fy) / 2.0


def _side_channel(case, side):
    from . import imageio

    path = getattr(case, f"{side}_nuclear")
    c = imageio.shape_of(path)[0]
    return path, (0 if c == 1 else int(case.extra.get(f"{side}_channel", 0)))


def cache_dir(out, case):
    return Path(out) / "cache" / "tissue" / case.dataset / str(case.case_id)


def code_sha256():
    """sha256 of this file: the code every plane and every tissue record is made by."""
    import hashlib

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def plane_key(case, side, eval_px=None, **more):
    """What a benchmark-side plane is made FROM and BY: the image file it reduces (its
    ``cases.file_fingerprint``: size, time and sampled bytes), the channel, the
    reference grid, the evaluation size, this file's bytes and ``DEFINITION``."""
    from .cases import file_fingerprint

    path, channel = _side_channel(case, side)
    return {
        "definition": DEFINITION,
        "code_sha256": code_sha256(),
        "image": file_fingerprint(path),
        "channel": int(channel),
        "ref_hw": [int(v) for v in case.ref_hw],
        "eval_px": int(eval_px or EVAL_PX),
        **more,
    }


def plane_record(path):
    """Where a cached plane says what it was made from: ``<plane>.json``."""
    path = Path(path)
    return path.with_name(path.name + ".json")


def _cached_plane(path, make, key):
    """The plane at ``path``. It is read only when the record beside it
    (``plane_record``) holds exactly ``key`` (``plane_key``); a plane without a record,
    or made from another image or by other code, is made again and replaced."""
    path = Path(path)
    record = plane_record(path)
    try:
        held = json.loads(record.read_text()) if path.exists() else None
    except (OSError, ValueError):
        held = None
    wanted = json.loads(json.dumps(key))
    if held != wanted:
        record.unlink(missing_ok=True)  # never a record of a plane that is not there
        write_plane(path, make())
        write_json(record, wanted)
    return read_plane(path)


def reference_plane(out, case, eval_px=None):
    path, channel = _side_channel(case, "ref")
    return _cached_plane(
        cache_dir(out, case) / "reference_eval.tif",
        lambda: reduce(path, channel, case.ref_hw, eval_px=eval_px),
        plane_key(case, "ref", eval_px),
    )


def initial_plane(out, case, eval_px=None):
    """The moving slide as it is, its first pixel on the reference's first pixel."""
    path, channel = _side_channel(case, "mov")
    return _cached_plane(
        cache_dir(out, case) / "initial_eval.tif",
        lambda: reduce(path, channel, case.ref_hw, eval_px=eval_px),
        plane_key(case, "mov", eval_px),
    )


def has_known_map(case):
    spec = case.extra.get("spec")
    return isinstance(spec, dict) and all(k in spec for k in ("theta", "shift", "n"))


def truth_plane(out, case, eval_px=None):
    """The moving slide resampled onto the reference by the GENERATING map: drawn by the
    benchmark (bilinear, at the evaluation size), not by any method. The inverse of the
    generator's map is taken on every ``TRUTH_STEP``-th evaluation pixel and interpolated
    between them, so a discontinuous family is smoothed over that step."""

    def make():
        import cv2
        from stare.slide_io import open_lazy

        from . import selfcheck

        (eh, ew), fx, fy = _factors(case.ref_hw, eval_px or EVAL_PX)
        path, channel = _side_channel(case, "mov")
        arr, _, close = open_lazy(path)
        try:
            mh, mw = (int(v) for v in arr.shape[1:])
            small_hw = (int(math.ceil(mh / fy)), int(math.ceil(mw / fx)))
            small = reduce_plane(arr, channel, small_hw, fx, fy).astype(np.float32)
        finally:
            close()
        inverse = selfcheck.inverse_of(case.extra["spec"], tol=0.01)
        # lattice of evaluation-pixel centres, in full-resolution reference pixels
        jy = np.unique(np.append(np.arange(0, eh, TRUTH_STEP), eh - 1)).astype(float)
        jx = np.unique(np.append(np.arange(0, ew, TRUTH_STEP), ew - 1)).astype(float)
        gx, gy = np.meshgrid((jx + 0.5) * fx - 0.5, (jy + 0.5) * fy - 0.5)
        q = np.empty((gx.size, 2))
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        for i in range(0, len(pts), 250_000):
            q[i : i + 250_000] = inverse(pts[i : i + 250_000])
        # moving full-resolution pixel -> index into the reduced moving plane
        qx = ((q[:, 0] + 0.5) / fx - 0.5).reshape(gx.shape).astype(np.float32)
        qy = ((q[:, 1] + 0.5) / fy - 0.5).reshape(gx.shape).astype(np.float32)
        # the lattice is regular but for its last line: interpolate it per axis
        ax, ay = np.arange(ew, dtype=float), np.arange(eh, dtype=float)

        def dense(coarse):
            rows = np.stack([np.interp(ax, jx, r) for r in coarse])
            return np.stack(
                [np.interp(ay, jy, rows[:, j]) for j in range(ew)], axis=1
            ).astype(np.float32)

        drawn = cv2.remap(
            small,
            dense(qx),
            dense(qy),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return np.clip(np.rint(drawn), 0, 65535).astype(np.uint16)

    return _cached_plane(
        cache_dir(out, case) / "truth_eval.tif",
        make,
        # the generating map and the lattice it is inverted on decide this plane too
        plane_key(case, "mov", eval_px, spec=case.extra["spec"], truth_step=TRUTH_STEP),
    )


def shifted(plane, dx_px):
    """``plane`` moved by ``dx_px`` evaluation pixels along x (bilinear, 0 beyond)."""
    import cv2

    h, w = plane.shape
    m = np.float32([[1, 0, dx_px], [0, 1, 0]])
    return cv2.warpAffine(
        np.asarray(plane, dtype=np.float32),
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ── rows ──────────────────────────────────────────────────────────────────────
def _cached(path, inputs, make, compute):
    """``tissue.json`` when it is of this definition, was made by this file as it is
    (``code_sha256``) and is not older than its planes; else ``make()`` (stored) when
    ``compute``; else None."""
    path = Path(path)
    code = code_sha256()
    try:
        rec = json.loads(path.read_text())
        fresh = all(path.stat().st_mtime >= Path(p).stat().st_mtime for p in inputs)
        if (
            rec.get("definition") == DEFINITION
            and rec.get("code_sha256") == code
            and fresh
        ):
            return rec
    except (OSError, ValueError):
        pass
    if not compute:
        return None
    rec = {**make(), "code_sha256": code}
    write_json(path, rec)
    return rec


def _row(case, variant, source, rec, **extra):
    return {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "group": case.group,
        "family": case.family,
        "pose": case.pose,
        "variant": variant,
        "source": source,
        **extra,
        **{k: v for k, v in rec.items() if k not in ("crop_box", "code_sha256")},
        "crop_box": " ".join(str(v) for v in rec.get("crop_box") or []),
    }


def _missing(case, variant, reason):
    return {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "group": case.group,
        "variant": variant,
        "reason": reason,
    }


def _variant_plane(out, case, variant):
    """``(path of the reduced plane, None)`` or ``(None, why there is none)``."""
    d = result_dir(out, variant, case)
    try:
        run = json.loads((d / "run.json").read_text())
    except (OSError, ValueError):
        return None, "not run"
    from . import score

    if not score.of_this_submission(run):
        return None, score.STALE
    if not run.get("ok"):
        return None, f"the run failed ({run.get('cause') or 'exception'}): no image"
    native = run.get("native")
    if not isinstance(native, dict):
        return None, "the method writes no image for this variant"
    if native.get("error") or not native.get("files"):
        return None, f"the method's writer failed: {native.get('error') or 'no file'}"
    plane = d / "tissue" / MOVING_PLANE
    if plane.exists():
        return plane, None
    source = Path(native.get("nuclear_file") or native["files"][0])
    if not source.exists():  # nothing to back-fill from
        reduced = run.get("tissue") or {}
        why = reduced.get("error") or "it was not reduced when it was written"
        return None, f"no reduced plane ({why}) and the written image is gone"
    try:
        rec = reduce_native(case, native, d)
    except Exception as exc:
        return (
            None,
            f"the written image could not be reduced: {type(exc).__name__}: {exc}",
        )
    if rec.get("error"):
        return None, rec["error"]
    return plane, None


def score_case(job):
    """``(rows, missing, dose)`` of one case: a row per variant with a written image, the
    ``initial`` row, and on a known-truth case the ``truth`` row and its shifted copies."""
    out, case, variants, compute = job
    compute = bool(compute) and have_cv2()
    why_not = NO_OPENCV if not have_cv2() else "not computed"
    rows, missing, dose = [], [], []
    names = [v for v in variants if v not in ("initial", "truth")]
    if not case.ref_nuclear or not Path(case.ref_nuclear).exists() or not case.ref_hw:
        why = "the case has no reference image"
        return [], [_missing(case, v, why) for v in ["initial", *names]], []
    home = cache_dir(out, case)
    try:
        ref_path = home / "reference_eval.tif"
        reference_plane(out, case)
    except Exception as exc:
        why = f"the reference could not be reduced: {type(exc).__name__}: {exc}"
        return [], [_missing(case, v, why) for v in ["initial", *names]], []
    ref = []  # made only when something has to be computed

    def reference():
        if not ref:
            ref.append(Reference(read_plane(ref_path), eval_um(case)))
        return ref[0]

    def add(variant, source, plane_path, result_path):
        rec = _cached(
            result_path,
            [ref_path, plane_path],
            lambda: reference().score(read_plane(plane_path)),
            compute,
        )
        if rec is None:
            missing.append(_missing(case, variant, why_not))
        else:
            rows.append(_row(case, variant, source, rec))
        return rec

    try:
        initial_plane(out, case)
        add("initial", "benchmark", home / "initial_eval.tif", home / "initial.json")
    except Exception as exc:
        missing.append(_missing(case, "initial", f"{type(exc).__name__}: {exc}"))
    if has_known_map(case):
        try:
            truth_path = home / "truth_eval.tif"
            if compute or truth_path.exists():
                plane = truth_plane(out, case)
                add("truth", "benchmark", truth_path, home / "truth.json")
                um = eval_um(case)

                def doses():
                    got = []
                    for d_um in (0.0, *DOSE_UM):
                        rec = reference().score(shifted(plane, d_um / um))
                        got.append({"shift_um": d_um, **rec})
                    return {"definition": DEFINITION, "rows": got}

                recs = _cached(
                    home / "dose.json", [ref_path, truth_path], doses, compute
                )
                for rec in (recs or {}).get("rows", []):
                    dose.append(_row(case, "truth", "benchmark", rec))
            else:
                missing.append(_missing(case, "truth", why_not))
        except Exception as exc:
            missing.append(_missing(case, "truth", f"{type(exc).__name__}: {exc}"))
    for variant in names:
        plane_path, why = _variant_plane(out, case, variant)
        if plane_path is None:
            missing.append(_missing(case, variant, why))
            continue
        try:
            add(variant, "method", plane_path, plane_path.with_name(RESULT))
        except Exception as exc:
            missing.append(_missing(case, variant, f"{type(exc).__name__}: {exc}"))
    return rows, missing, dose


def _num(rows, key):
    vals = [r.get(key) for r in rows]
    return np.array([float(v) for v in vals if v is not None], dtype=float)


def summarise(rows, missing, by=("dataset", "variant")):
    """One row per (dataset, variant): the median and the minimum over the scored cases of
    every piece (``hausdorff``: the maximum is the worst), how many cases were scored, and
    how many composites are under ``FAILED_BELOW``."""
    n_missing = {}
    for m in missing:
        key = tuple(m[k] for k in by)
        n_missing[key] = n_missing.get(key, 0) + 1
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in by), []).append(r)
    out = []
    for key, mine in groups.items():
        qcs = _num(mine, "qcs")
        rec = {
            **dict(zip(by, key)),
            "source": mine[0]["source"],
            "n_scored": len(mine),
            "n_cases": len(mine) + n_missing.get(key, 0),
            "n_no_tissue": sum(r.get("status") == "no_tissue" for r in mine),
            "n_below_025": int((qcs < FAILED_BELOW).sum()),
        }
        for piece in PIECES:
            v = _num(mine, piece)
            worst = np.max if piece.startswith("hausdorff") else np.min
            rec[f"{piece}_median"] = float(np.median(v)) if len(v) else float("nan")
            rec[f"{piece}_worst"] = float(worst(v)) if len(v) else float("nan")
        out.append(rec)
    return out


def score_all(jobs, workers=1, compute=True):
    """``jobs``: ``(out, case, variants)`` per case. Returns the four tables' rows."""
    jobs = [(str(out), case, list(variants), compute) for out, case, variants in jobs]
    if workers > 1 and len(jobs) > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            done = pool.map(score_case, jobs)
    else:
        done = [score_case(j) for j in jobs]
    rows = [r for d in done for r in d[0]]
    missing = [r for d in done for r in d[1]]
    dose = [r for d in done for r in d[2]]
    return {
        "tissue": rows,
        "tissue_summary": summarise(rows, missing),
        "tissue_missing": missing,
        "tissue_dose": dose,
        "tissue_dose_summary": summarise(dose, [], by=("dataset", "shift_um")),
    }


SUMMARY_COLUMNS = (
    "dataset",
    "variant",
    "source",
    "n_scored",
    "n_cases",
    "n_below_025",
    "qcs_median",
    "qcs_worst",
    "tissue_dice_median",
    "boundary_dice_median",
    "hausdorff_mean_um_median",
    "inlier_ratio_median",
    "n_matches_median",
    "nmi_median",
    "coverage_median",
    "bg_coverage_median",
)
DOSE_COLUMNS = (
    "dataset",
    "shift_um",
    "n_scored",
    "qcs_median",
    "tissue_dice_median",
    "boundary_dice_median",
    "hausdorff_mean_um_median",
    "inlier_ratio_median",
    "nmi_median",
)


def markdown(tables, table):
    """The block of ``summary.md``: apart from every other table, and saying so.
    ``table(rows, columns)`` is the scorer's Markdown writer."""
    if not (tables["tissue_summary"] or tables["tissue_missing"]):
        return []
    parts = [
        "## Tissue-level agreement of the written images (secondary; not ranked)\n",
        f"{LABEL}\n",
        "Each row compares the nuclear channel of the image a method WROTE with the "
        "reference's, at 4000 px. Computed here on same-modality fluorescence, which is "
        "outside what the score was validated for; it does not measure accuracy at "
        "nuclear scale and the source reports no check of it against landmark error. "
        "`initial` (the unregistered slide) and `truth` (known-truth cases; drawn by the "
        "benchmark) are for scale. Median over the scored cases; `n_below_025` counts "
        "composites under 0.25. A variant without a written image has no row "
        "(`tissue_missing.csv` says why).\n",
        table(tables["tissue_summary"], list(SUMMARY_COLUMNS)),
        "",
    ]
    if tables["tissue_dose_summary"]:
        parts += [
            "### What the block resolves: the truth plane shifted by a known amount\n",
            "Known-truth cases only, whose evaluation pixel is NOT the cohort's (an "
            "8192 px window at 4000 px).\n",
            table(tables["tissue_dose_summary"], list(DOSE_COLUMNS)),
            "",
        ]
    reasons = {}
    for m in tables["tissue_missing"]:
        key = (m["dataset"], m["variant"], m["reason"].split(":")[0])
        reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        rows = [
            {"dataset": d, "variant": v, "n_cases": n, "reason": why}
            for (d, v, why), n in sorted(reasons.items())
        ]
        parts += [
            "### No tissue row\n",
            table(rows, ["dataset", "variant", "n_cases", "reason"]),
            "",
        ]
    return parts
