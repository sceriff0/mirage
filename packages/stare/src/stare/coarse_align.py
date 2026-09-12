"""Global rigid anchor (M0) estimation for the STARE registration method.

The COARSE step aligns a whole moving slide to the reference with a single affine ``M0``, cheaply,
on downsampled images. Learned feature matching (DISK + LightGlue) handles the inter-cycle
rotation/translation that a translation-only phase correlation could not; the per-tile step later
refines the small residual.

The numerically load-bearing piece is :func:`estimate_transform_from_matches` — given point
correspondences it solves for the transform and reports a residual Target Registration Error. It
is deterministic and unit-tested. :func:`estimate_rigid` wraps it with the learned front-end.

NumPy + scikit-image for the geometry, torch + kornia for the matcher. Still no VALIS and no JVM.
"""

from __future__ import annotations

import time

import numpy as np

from stare.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "estimate_transform_from_matches",
    "estimate_rigid",
    "normalize_intensity",
    "scale_transform_to_full_res",
]

_MIN_SAMPLES = {"euclidean": 2, "similarity": 2, "affine": 3}


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
# float32 by `tiled_io.read_decimated`. Every consumer here wants them in [0, 1] first, and for
# a LEARNED matcher that is not a nicety: DISK was trained on images in [0, 1], so raw counts
# are ~4 orders of magnitude outside its trained input range.
#
# Normalising is also a CORRECTNESS fix, not only a domain-range one: reference and moving come
# from different imaging cycles with different exposures, so two planes of the same tissue can
# arrive at wildly different dynamic ranges and match each other far worse than they should --
# a starved or bogus M0, with no error raised.
#
# A percentile window, not `img / img.max()`: one hot pixel (routine in fluorescence) would
# otherwise compress the whole tissue into the bottom of the range and leave the matcher almost
# nothing to work with. p99.9 rather than p99 because DAPI nuclei ARE the bright minority --
# clipping at p99 saturates the very structure the anchor matches on.
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


# DISK + LightGlue weights are ~52 MB and building both models dominates a single call.
# COARSE runs once per slide per task, so this cache only matters for the in-process
# oracle (bin/utils/tiled_pipeline.py) and the test suite -- but it is free there.
#
# The MODEL CLASSES are passed IN rather than imported here, and that is load-bearing:
# tests/test_container_harmonisation.py::
# test_tiled_container_torch_kornia_imports_are_confined_to_disk_lightglue asserts that
# every `import torch` / `from kornia...` in this module and in bin/tiled_coarse.py sits
# inside a function named literally `_frontend_disk_lightglue`. A `from kornia.feature
# import ...` in this helper is an offender and turns that guard RED.
_DISK_MODELS = {}


def _disk_models(device, disk_cls, lightglue_cls):
    key = str(device)
    if key not in _DISK_MODELS:
        _DISK_MODELS[key] = (
            disk_cls.from_pretrained("depth").to(device).eval(),
            lightglue_cls("disk").to(device).eval(),
        )
    return _DISK_MODELS[key]


def _to_disk_tensor(img, torch):
    """One plane -> the 1x3xHxW float32 batch DISK expects.

    Normalising is not optional for a LEARNED matcher: DISK was trained on images in
    [0, 1], and the planes reaching this module are raw microscopy counts widened to
    float32 (0..65535). Feeding those in raw is far outside the trained input range.
    """
    a = normalize_intensity(img).astype(np.float32)
    return torch.from_numpy(a)[None, None].repeat(1, 3, 1, 1)


def _frontend_disk_lightglue(ref, mov, model="euclidean", n_keypoints=2048, **kwargs):
    """DISK + LightGlue -- the learned matcher VALIS and the ACROBAT winners use.

    Returns ``(M0, info)`` with ``residual_px`` and ``n_inliers`` from
    :func:`estimate_transform_from_matches`, exactly like the classical feature front-ends
    it replaced, so the reporting contract downstream is unchanged.

    MEMORY. DISK is a U-Net: its activation memory scales with image AREA, not with
    keypoint count (measured on the pinned stack: 3.03 GB at 512 px, 8.78 GB at 1024 px ->
    GB ~= 1.1 + 7.3*Mpx). The thumbnail size this runs on is governed by the STARE
    `coarse_max_dim` tier (see lib/RegPresets.groovy), and TILED_COARSE's memory request is
    now sized off that SAME bound with that same formula -- the tier table is inlined into
    the `memory` closure in conf/modules.config because conf/*.config cannot see
    lib/*.groovy. So raising the thumbnail bound raises the reservation too; it is still a
    memory decision, not a free accuracy knob, but it is no longer a silent one.
    """
    # BOTH imports live here, inside this function, for two reasons: a torch-present /
    # kornia-absent environment must still get the actionable RuntimeError rather than a
    # bare ImportError, and the confinement guard named above scans for exactly this
    # function. Do not hoist either import to module scope or into a helper.
    try:
        import torch
        from kornia.feature import DISK, LightGlue
    except ImportError as exc:
        raise RuntimeError(
            "frontend='disk_lightglue' needs torch + kornia. They ship in the "
            "bolt3x/mirage-tiled image; install torch==2.3.1 (CPU wheel) + "
            "kornia==0.7.3 to run this outside a container."
        ) from exc

    device = torch.device("cpu")
    disk, matcher = _disk_models(device, DISK, LightGlue)

    t0 = time.perf_counter()
    with torch.inference_mode():
        t_ref = _to_disk_tensor(ref, torch).to(device)
        t_mov = _to_disk_tensor(mov, torch).to(device)
        # pad_if_not_divisible: DISK's U-Net downsamples by 16 and raises on a thumbnail
        # whose dimensions are not multiples of 16. Thumbnail sizes come from
        # decimation_factor(), which produces arbitrary sizes -- so this is not optional.
        feat_kw = dict(
            n=n_keypoints, window_size=5, score_threshold=0.0, pad_if_not_divisible=True
        )
        f_ref = disk(t_ref, **feat_kw)[0]
        f_mov = disk(t_mov, **feat_kw)[0]
        t_feat = time.perf_counter() - t0
        logger.info(
            f"coarse: DISK reference -> {len(f_ref.keypoints)} keypoints | "
            f"moving -> {len(f_mov.keypoints)} keypoints in {t_feat:.1f}s "
            f"({ref.shape[1]}x{ref.shape[0]} / {mov.shape[1]}x{mov.shape[0]})"
        )

        def _side(f, t):
            return {
                "keypoints": f.keypoints[None],
                "descriptors": f.descriptors[None],
                # kornia's own LightGlue fallback for this key is
                # ``data0["image"].shape[-2:][::-1]`` -- i.e. (W, H), NOT (H, W).
                # normalize_keypoints() divides (x, y) keypoints by this size, so a
                # reversed size puts H on the x axis and silently sends every
                # normalised keypoint out of [-1, 1], which _forward()'s internal
                # KORNIA_CHECK then rejects. A SQUARE fixture cannot catch this --
                # (H, W) and (W, H) are the same tuple when H == W -- which is why
                # this shipped broken once already. Do not "tidy" the [::-1] away.
                "image_size": torch.tensor(t.shape[-2:][::-1], device=device)[None],
            }

        out = matcher({"image0": _side(f_ref, t_ref), "image1": _side(f_mov, t_mov)})
        matches = out["matches"][0].cpu().numpy()

    kp_ref = f_ref.keypoints.cpu().numpy()
    kp_mov = f_mov.keypoints.cpu().numpy()
    logger.info(
        f"coarse: {len(matches)} LightGlue matches -> {model} fit "
        f"(DISK+LightGlue total {time.perf_counter() - t0:.1f}s)"
    )
    if len(matches) < _MIN_SAMPLES[model]:
        # estimate_transform_from_matches would fall through to an unconstrained
        # .estimate() and hand back a silently meaningless M0; say so while the slide
        # names are still in the surrounding log context.
        logger.warning(
            f"coarse: only {len(matches)} matches for a {model} fit "
            f"(needs >= {_MIN_SAMPLES[model]}) -- M0 will be unreliable"
        )

    # LightGlue's `matches` column 0 indexes image0 (reference), column 1 indexes image1
    # (moving). DISK keypoints are ALREADY (x, y) -- unlike skimage's classical detectors,
    # whose keypoints are (row, col) and needed a [:, ::-1] reversal. This module used to
    # carry that reversal for its old front-ends; applying it here silently transposes M0.
    src = kp_mov[matches[:, 1]]
    dst = kp_ref[matches[:, 0]]
    m, residual, n_inliers = estimate_transform_from_matches(
        src, dst, model=model, **kwargs
    )
    return m, {"residual_px": residual, "n_inliers": n_inliers}


def estimate_rigid(ref, mov, model="euclidean", **kwargs):
    """Estimate ``M0`` mapping ``mov`` image coordinates onto ``ref``.

    Returns ``(M0, residual_px, n_inliers)``. The module's single entry point: a thin
    reshape over :func:`_frontend_disk_lightglue`, which holds the real implementation.
    """
    m, info = _frontend_disk_lightglue(ref, mov, model=model, **kwargs)
    return m, info["residual_px"], info["n_inliers"]
