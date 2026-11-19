"""Registration accuracy measures for STARE (or any method), without ground truth.

  * **residual TRE**: median/p90 distance between ORB features matched between the
    reference and the registered image. Both are in the reference frame, so good
    registration gives near-coincident matches.
  * **intensity correlation** over the interior (borders excluded).

:func:`benchmark` reports both before (reference vs moving) and after (reference vs
registered). Pure NumPy + scikit-image.
"""

from __future__ import annotations

import numpy as np

__all__ = ["feature_residual", "benchmark"]


def _orb_xy(img, n_keypoints, fast_threshold):
    # ORB here is an accuracy oracle, not a registration front-end
    # (tests/test_no_legacy_frontends.py allow-lists this file).
    #
    # Normalise first: `fast_threshold` is an absolute intensity difference, and on raw
    # uint16 counts FAST fires on a large share of pixels, turning minutes into hours on
    # full-resolution slides.
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
