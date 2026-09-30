"""Registration accuracy harness for the STARE method (and any other).

Quantifies how well a *registered* image aligns to the *reference* with two ground-truth-free
measures, so it works on real WSIs exactly as on synthetic data:

  * **residual TRE** — the median/p90 distance between ORB features matched between the reference
    and the registered image. Both are in the reference frame, so a well-registered pair has
    near-coincident matches (small TRE) and a misaligned pair large ones. This is the same
    quantity you would compute to compare a tiled output against a VALIS output on real slides:
    run :func:`benchmark` on each method's registered image and compare the numbers.
  * **intensity correlation** over the interior (borders excluded).

:func:`benchmark` reports both before (reference vs moving) and after (reference vs registered),
so the improvement is explicit.

Pure NumPy + scikit-image.
"""

from __future__ import annotations

import numpy as np

__all__ = ["feature_residual", "benchmark"]


def _orb_xy(img, n_keypoints, fast_threshold):
    # This ORB is an INDEPENDENT ACCURACY ORACLE, not a registration front-end. COARSE's own
    # front-ends were reduced to DISK+LightGlue for v1.0.0; scoring the result with the very
    # matcher that produced it would measure nothing, so the oracle deliberately stays
    # classical. tests/test_no_legacy_frontends.py allow-lists this file for that reason.
    #
    # normalize_intensity is imported from coarse_align rather than duplicated so the unit-scale
    # contract has ONE owner: `fast_threshold` is an absolute intensity difference, so raw uint16
    # counts make FAST fire on ~38% of pixels and corner_peaks (quadratic in that count) turns
    # minutes into hours. This path is the more exposed of the two -- feature_residual scores
    # FULL-RESOLUTION registered slides, not the bounded thumbnail COARSE works on.
    from coarse_align import normalize_intensity
    from skimage.feature import ORB

    orb = ORB(n_keypoints=n_keypoints, fast_threshold=fast_threshold)
    orb.detect_and_extract(normalize_intensity(img))
    return orb.keypoints[:, ::-1], orb.descriptors  # (x, y), descriptors


def feature_residual(
    reference, registered, n_keypoints=800, fast_threshold=0.05, outlier_px=50.0
):
    """Residual TRE between ``reference`` and ``registered`` (both in the reference frame).

    Returns ``{"tre_px_median", "tre_px_p90", "n_matches"}`` — the distances between matched ORB
    features, robust to descriptor mismatches (matches beyond ``outlier_px`` are dropped).
    """
    from skimage.feature import match_descriptors

    kp_a, d_a = _orb_xy(reference, n_keypoints, fast_threshold)
    kp_b, d_b = _orb_xy(registered, n_keypoints, fast_threshold)
    matches = match_descriptors(d_a, d_b, cross_check=True)
    if len(matches) == 0:
        return {
            "tre_px_median": float("nan"),
            "tre_px_p90": float("nan"),
            "n_matches": 0,
        }

    dists = np.linalg.norm(kp_a[matches[:, 0]] - kp_b[matches[:, 1]], axis=1)
    good = dists[dists <= outlier_px]
    if (
        good.size == 0
    ):  # everything looked like a mismatch — report what we have, honestly
        good = dists
    return {
        "tre_px_median": float(np.median(good)),
        "tre_px_p90": float(np.percentile(good, 90)),
        "n_matches": int(good.size),
    }


def _interior_corr(a, b, inner_frac):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    h, w = a.shape[:2]
    my, mx = int(h * inner_frac), int(w * inner_frac)
    aa = a[my : h - my, mx : w - mx].ravel()
    bb = b[my : h - my, mx : w - mx].ravel()
    if aa.size == 0 or np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def benchmark(reference, moving, registered, inner_frac=0.15, **residual_kwargs):
    """Before/after accuracy report for one registration.

    ``before`` scores reference vs the un-registered moving image, ``after`` reference vs the
    registered image; ``improvement_tre_px`` is the drop in median residual TRE.
    """
    before = feature_residual(reference, moving, **residual_kwargs)
    after = feature_residual(reference, registered, **residual_kwargs)
    return {
        "before": before,
        "after": after,
        "correlation_before": _interior_corr(reference, moving, inner_frac),
        "correlation_after": _interior_corr(reference, registered, inner_frac),
        "improvement_tre_px": float(before["tre_px_median"] - after["tre_px_median"]),
    }
