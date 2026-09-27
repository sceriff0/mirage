"""Synthetic nuclear-channel slides with a KNOWN displacement field, SciPy only.

The generator of ``research/stare-optimal-design-2026-09-27/stress.py`` without OpenCV (the
tiled container has none): Gaussian nuclei (sigma 5 px, one per ~900 px^2), an optional
tissue silhouette, additive Gaussian noise at 2 % of the peak.

Field convention (the pull-back ``u`` of stress.py): ``mov(p) = base(p - u(p))`` and
``ref = base``, so the moving point of reference pixel ``x`` is ``p = x + u(p)`` and the
control displacement ``d`` (``ref(x) = mov(x - d)``) is ``-(p - x)``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom


def nuclei(n, rng, density=1 / 900, sigma=5.0):
    k = int(n * n * density)
    im = np.zeros((n, n), np.float32)
    ys, xs = rng.integers(0, n, k), rng.integers(0, n, k)
    np.add.at(im, (ys, xs), rng.uniform(0.5, 1.5, k).astype(np.float32))
    return gaussian_filter(im, sigma) * 200


def tissue(n, rng, keep=0.7, cells=32):
    """A blobby tissue silhouette covering ``keep`` of the slide."""
    g = zoom(gaussian_filter(rng.random((cells, cells)), 2), n / cells, order=1)[:n, :n]
    return g > np.percentile(g, 100 * (1 - keep))


def make_pair(n, ufield, seed=0, noise=0.02, mask=None):
    """``(ref, mov)`` float32 with ``mov(p) = base(p - u(p))``; ``ufield(X, Y) -> (ux, uy)``."""
    rng = np.random.default_rng(seed)
    base = nuclei(n, rng)
    if mask is not None:
        base = base * mask
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    ux, uy = ufield(xx, yy)
    mov = map_coordinates(base, [yy - uy, xx - ux], order=3, mode="constant")
    del xx, yy, ux, uy
    sig = noise * float(base.max())
    ref = base + rng.normal(0, sig, base.shape).astype(np.float32)
    mov = mov + rng.normal(0, sig, base.shape).astype(np.float32)
    return ref.astype(np.float32), mov.astype(np.float32)


def moving_point(ufield, X, iters=8):
    """``p = x + u(p)`` by fixed point: the true moving point of reference points ``X``."""
    p = np.array(X, dtype=float)
    for _ in range(iters):
        ux, uy = ufield(p[:, 0], p[:, 1])
        p = X + np.stack([ux, uy], axis=1)
    return p


def true_control(ufield, X):
    """The control displacement ``d(x) = -(p - x)`` at reference points ``X``."""
    return -(moving_point(ufield, X) - np.asarray(X, dtype=float))
